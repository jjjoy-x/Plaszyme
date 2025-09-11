# src/data/loader.py
from __future__ import annotations

import warnings
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData, Batch

from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.builders.gnn_builder import GNNProteinGraphBuilder

Split = Literal["train", "val", "test"]
Mode = Literal["point", "pair", "list"]


# ========= 读取独热矩阵 =========
@dataclass
class MatrixSpec:
    """描述独热矩阵位置与对应根目录"""
    csv_path: str
    pdb_root: str
    sdf_root: str
    # 可扩展: chain 列（如果某些 PDB 需要固定链，可在独热矩阵之外提供一个 {pdb:chain} 的映射）


def _read_onehot_matrix(spec: MatrixSpec) -> Tuple[List[str], List[str], Dict[Tuple[str, str], int]]:
    """
    读取独热矩阵:
      - 行是 PDB 文件名（含/不含后缀都接受，最终会自动补 .pdb）
      - 列是 SDF 文件名（含/不含后缀都接受，最终会自动补 .sdf）
      - 值 ∈ {0,1}
    Return:
      rows: List[pdb_basename]
      cols: List[sdf_basename]
      labels: dict[(pdb_basename, sdf_basename)] -> 0/1
    """
    if not os.path.isfile(spec.csv_path):
        raise FileNotFoundError(spec.csv_path)

    with open(spec.csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("独热矩阵CSV至少需要1列塑料名")

        # 第一列是行名占位，后续列是塑料名
        raw_cols = header[1:]
        cols = [c if os.path.splitext(c)[1].lower() == ".sdf" else f"{c}.sdf" for c in raw_cols]

        rows: List[str] = []
        labels: Dict[Tuple[str, str], int] = {}

        for line in reader:
            if not line:
                continue
            pdb_name = line[0]
            if os.path.splitext(pdb_name)[1].lower() != ".pdb":
                pdb_name = f"{pdb_name}.pdb"
            rows.append(pdb_name)

            # 每一行后续是 0/1
            for j, v in enumerate(line[1:]):
                try:
                    val = int(float(v))  # 兼容 "0","1","0.0","1.0"
                except Exception:
                    val = 0
                labels[(pdb_name, cols[j])] = 1 if val >= 1 else 0

    return rows, cols, labels


# ========= 核心数据集 =========
class PairedPlaszymeDataset(Dataset):
    """
    统一数据集：支持 point / pair / list 三种输出模式。
    - 只保存文件名与标签，真正的“PDB->图 / SDF->特征”在 __getitem__ 时调用外部 builder/featurizer 完成。
    - 你也可以选择在 DataLoader 的 collate_fn 里做（更高效的批内复用/缓存），本实现为简单直接版本。
    """

    def __init__(
        self,
        matrix: MatrixSpec,
        *,
        mode: Mode = "point",
        split: Split = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        # 负样本策略（point/pair/list 通用或部分使用）
        neg_ratio: float = 1.0,          # point: 每个正样本配多少倍负样本（从当前行内0列随机采）
        max_list_len: int = 16,          # list: 每个 enzyme 采样的塑料总数上限（含正+负）
        # 构建器
        enzyme_builder = PlasticFeaturizer,             # 需要实现 .build(pdb_path, chain_id=None) -> PyGData
        plastic_featurizer = GNNProteinGraphBuilder,         # 需要实现 .featurize_file(sdf_path) -> Tensor[D]
        # 链ID映射（可选）
        chain_map: Optional[Dict[str, str]] = None,  # {pdb_basename: chain_id}
    ):
        super().__init__()
        assert mode in ("point", "pair", "list")
        self.mode = mode
        self.matrix_spec = matrix
        self.enzyme_builder = enzyme_builder
        self.plastic_featurizer = plastic_featurizer
        self.neg_ratio = float(neg_ratio)
        self.max_list_len = int(max_list_len)
        self.chain_map = chain_map or {}

        # 读取矩阵
        rows, cols, labels = _read_onehot_matrix(matrix)
        self.rows = rows  # pdb 名
        self.cols = cols  # sdf 名
        self.labels = labels  # (pdb,sdf) -> 0/1

        # 按行切分（基于 pdb）
        rng = random.Random(seed)
        row_ids = list(range(len(self.rows)))
        rng.shuffle(row_ids)
        n = len(row_ids)
        n_tr = int(n * split_ratio[0])
        n_va = int(n * split_ratio[1])
        id_tr = set(row_ids[:n_tr])
        id_va = set(row_ids[n_tr:n_tr + n_va])
        id_te = set(row_ids[n_tr + n_va:])

        if split == "train":
            keep = id_tr
        elif split == "val":
            keep = id_va
        else:
            keep = id_te

        # 生成索引
        if mode == "point":
            # 为 pointwise 提前生成 (pdb,sdf,label) 三元组，并做负采样
            self.samples: List[Tuple[str, str, int]] = []
            for i in range(len(self.rows)):
                if i not in keep:
                    continue
                pdb = self.rows[i]
                # 正样本列
                pos_cols = [s for s in self.cols if self.labels[(pdb, s)] == 1]
                neg_cols = [s for s in self.cols if self.labels[(pdb, s)] == 0]
                # 全加入正样本
                for s in pos_cols:
                    self.samples.append((pdb, s, 1))
                # 负采样
                k_neg = int(len(pos_cols) * self.neg_ratio) if self.neg_ratio > 0 else 0
                if k_neg > 0 and len(neg_cols) > 0:
                    rng.shuffle(neg_cols)
                    for s in neg_cols[:k_neg]:
                        self.samples.append((pdb, s, 0))

        elif mode == "pair":
            # 为 pairwise 生成 (pdb, pos_sdf, neg_sdf) 三元组（每个正样本配一个随机负样本）
            self.samples: List[Tuple[str, str, str]] = []
            for i in range(len(self.rows)):
                if i not in keep:
                    continue
                pdb = self.rows[i]
                pos_cols = [s for s in self.cols if self.labels[(pdb, s)] == 1]
                neg_cols = [s for s in self.cols if self.labels[(pdb, s)] == 0]
                if len(pos_cols) == 0 or len(neg_cols) == 0:
                    continue
                for s_pos in pos_cols:
                    s_neg = rng.choice(neg_cols)
                    self.samples.append((pdb, s_pos, s_neg))

        else:
            # listwise：每行一个样本，返回一个 enzyme + 多个 plastics（含正/随机负），以及 0/1 相关性
            self.row_indices = [i for i in range(len(self.rows)) if i in keep]

        # 简单内存缓存（跨 epoch 复用）
        self._cache_graph: Dict[str, PyGData] = {}
        self._cache_plastic: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        if self.mode in ("point", "pair"):
            return len(self.samples)
        return len(self.row_indices)

    # --- 内部工具 ---
    def _pdb_path(self, pdb_name: str) -> str:
        return os.path.join(self.matrix_spec.pdb_root, pdb_name)

    def _sdf_path(self, sdf_name: str) -> str:
        return os.path.join(self.matrix_spec.sdf_root, sdf_name)

    def _get_graph(self, pdb_name: str) -> PyGData:
        if pdb_name in self._cache_graph:
            return self._cache_graph[pdb_name]
        pdb_path = self._pdb_path(pdb_name)
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(pdb_path)
        chain_id = self.chain_map.get(pdb_name)
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        g, _ = self.enzyme_builder.build_one(pdb_path, name=name)
        if not isinstance(g, PyGData):
            raise TypeError("enzyme_builder.build 必须返回 PyG Data")
        self._cache_graph[pdb_name] = g
        return g

    def _get_plastic(self, sdf_name: str) -> Optional[torch.Tensor]:
        if sdf_name in self._cache_plastic:
            return self._cache_plastic[sdf_name]

        sdf_path = self._sdf_path(sdf_name)
        if not os.path.isfile(sdf_path):
            warnings.warn(f"[PairedPlaszymeDataset] ⚠️ 缺失 SDF 文件: {sdf_path}，跳过该样本")
            return None

        f = self.plastic_featurizer.featurize_file(sdf_path)
        if f is None:
            warnings.warn(f"[PairedPlaszymeDataset] ⚠️ 特征提取失败: {sdf_path}，跳过该样本")
            return None

        f = f.view(1, -1)
        self._cache_plastic[sdf_name] = f
        return f

    # --- 三种模式的 __getitem__ ---
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "point":
            pdb_name, sdf_name, y = self.samples[idx]
            g = self._get_graph(pdb_name)
            f = self._get_plastic(sdf_name)
            return {"mode": "point", "pdb_name": pdb_name, "sdf_name": sdf_name,
                    "enzyme_graph": g, "plastic_feat": f, "label": torch.tensor([y], dtype=torch.float32)}

        if self.mode == "pair":
            pdb_name, sdf_pos, sdf_neg = self.samples[idx]
            g = self._get_graph(pdb_name)
            f_pos = self._get_plastic(sdf_pos)
            f_neg = self._get_plastic(sdf_neg)
            return {"mode": "pair", "pdb_name": pdb_name,
                    "enzyme_graph": g, "plastic_pos": f_pos, "plastic_neg": f_neg}

        # listwise
        row_id = self.row_indices[idx]
        pdb_name = self.rows[row_id]
        g = self._get_graph(pdb_name)

        pos_cols = [s for s in self.cols if self.labels[(pdb_name, s)] == 1]
        neg_cols = [s for s in self.cols if self.labels[(pdb_name, s)] == 0]

        # 裁剪/采样：最多保留 max_list_len，总是优先保留所有正样本
        plastics: List[torch.Tensor] = []
        rels: List[float] = []

        for s in pos_cols:
            plastics.append(self._get_plastic(s))
            rels.append(1.0)

        k_neg = max(0, self.max_list_len - len(plastics))
        if k_neg > 0 and len(neg_cols) > 0:
            random.shuffle(neg_cols)
            for s in neg_cols[:k_neg]:
                plastics.append(self._get_plastic(s))
                rels.append(0.0)

        # [L, D]
        plast_feat = torch.cat(plastics, dim=0) if len(plastics) > 0 else torch.empty(0, 0)
        labels = torch.tensor(rels, dtype=torch.float32)

        return {"mode": "list", "pdb_name": pdb_name, "enzyme_graph": g,
                "plastic_list": plast_feat, "relevance": labels}


# ========= Collate 函数（把多个样本拼成批） =========
def collate_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    通用 collate：
      - 将多个 PyG Data -> Batch
      - 将塑料特征按模式堆叠
    """
    assert len(batch) > 0
    mode: Mode = batch[0]["mode"]

    # 酶图
    g_list = [b["enzyme_graph"] for b in batch]
    enzyme_batch = Batch.from_data_list(g_list)

    if mode == "point":
        feats = torch.cat([b["plastic_feat"] for b in batch], dim=0)  # [B, D]
        labels = torch.cat([b["label"] for b in batch], dim=0)        # [B]
        return {"mode": "point", "enzyme_graph": enzyme_batch, "plastic_feat": feats, "label": labels}

    if mode == "pair":
        pos = torch.cat([b["plastic_pos"] for b in batch], dim=0)     # [B, D]
        neg = torch.cat([b["plastic_neg"] for b in batch], dim=0)     # [B, D]
        return {"mode": "pair", "enzyme_graph": enzyme_batch, "plastic_pos": pos, "plastic_neg": neg}

    # listwise：这里每个样本的 list 长度可能不同，训练脚本里通常会逐条处理或 padding
    # 为保持简单，我们原样返回 list（不做 padding），由训练脚本循环内部样本处理。
    return {"mode": "list", "enzyme_graph": enzyme_batch, "items": batch}