#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base protein graph builder (flat utilities inside).

基础蛋白质图构建器：内部自带 PDB 解析、序列对齐、嵌入器创建等通用能力；
GNN/GVP 子类只需实现节点/边特征的组织即可。

This module defines:
  - `BuilderConfig`: runtime config.
  - `BaseProteinGraphBuilder`: iterate PDBs → parse → align → delegate
    node/edge features → save PyG `Data`.
  - Inline helpers for PDB I/O and a simplified full-vs-atom alignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ---- embedder registry (load from your existing sequence_embedder.py) ----
# 与现有的 sequence_embedder.py 对接；若只使用某些类，也可按需精简导入。
from .sequence_embedder import ESMEmbedder, OneHotEmbedder, PhysChemEmbedder


# =============================================================================
# Config
# =============================================================================

@dataclass
class BuilderConfig:
    """Runtime configuration for builders.

    构建器运行配置。

    Attributes:
        pdb_dir (str): Directory containing standard PDB files (1 chain/file or choose a chain).
            包含标准 PDB 的目录（建议单链/文件或通过 chain_id 指定）。
        out_dir (str): Output directory for `.pt` graphs and `manifest.csv`.
            输出目录，保存 `.pt` 与 `manifest.csv`。
        chain_id (Optional[str]): Chain ID to parse; if None, use the first chain.
            指定链 ID；None 则取首链。
        radius (float): Distance cutoff (Å) for edge building (used by subclasses).
            构边半径阈值（Å），子类使用。
        embedder (Union[Dict[str, Any], List[Dict[str, Any]]]): Sequence embedder spec.
            序列嵌入器配置；可为单个字典（如 {"name":"esm", ...}），
            也可为多个配置的列表（将按顺序拼接特征）。
    """
    pdb_dir: str
    out_dir: str
    chain_id: Optional[str] = None
    radius: float = 10.0
    embedder: Union[Dict[str, Any], List[Dict[str, Any]], None] = None


# =============================================================================
# Internal PDB I/O (single-chain)  内置 PDB 解析
# =============================================================================

def _parse_pdb_single_chain(pdb_path: str, chain_id: Optional[str]) -> Dict[str, Any]:
    """Parse a PDB file for one chain and extract backbone coords and sequences.

    解析单链 PDB：给出 SEQRES（若可得）、ATOM 序列，以及骨架 N/CA/C 坐标。

    Args:
        pdb_path: Path to PDB file.
        chain_id: Target chain ID; if None, pick the first chain.

    Returns:
        Dict[str, Any]: {
            "seq_full": Optional[str],   # SEQRES (or None)
            "seq_atom": str,             # from ATOM records after filtering
            "coords_N": np.ndarray[K,3],
            "coords_CA": np.ndarray[K,3],
            "coords_C": np.ndarray[K,3],
            "res_ids": List[str],        # e.g., "A:123" (with insertion code if present)
        }
    """
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())

    # pick chain
    ch = None
    for _ch in model.get_chains():
        if chain_id is None:
            ch = _ch
            break
        if _ch.id == chain_id:
            ch = _ch
            break
    if ch is None:
        raise ValueError(f"No chain found in {pdb_path} (requested={chain_id})")

    # SEQRES (best effort)
    seq_full = None
    try:
        ppb = PPBuilder()
        for poly in ppb.build_peptides(ch, aa_only=False):
            s = str(poly.get_sequence())
            seq_full = s if seq_full is None else (seq_full + s)
    except Exception:
        seq_full = None

    # ATOM sequence + backbone coords
    seq_atom, coords_N, coords_CA, coords_C, res_ids = [], [], [], [], []
    for res in ch.get_residues():
        # skip hetero/water
        if res.id[0] != " ":
            continue

        # 3-letter to 1-letter
        resname = res.get_resname()
        try:
            aa1 = seq1(
                resname,
                custom_map={
                    "MSE": "M",  # 硒代蛋氨酸常见于晶体结构
                    "SEC": "U",  # Selenocysteine（可按需要映射为 "C" 或 "X"）
                    "PYL": "O",  # Pyrrolysine（也可映射为 "K" 或 "X"）
                },
                undef_code="X",  # 未知或非常规残基 → 'X'
            )
        except Exception:
            aa1 = "X"

        def get_coord(atom_name: str) -> np.ndarray:
            if atom_name in res:
                return res[atom_name].get_coord().astype(float)
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        n = get_coord("N")
        ca = get_coord("CA")
        c = get_coord("C")

        # require CA present
        if not np.isfinite(ca).all():
            continue

        seq_atom.append(aa1)
        coords_N.append(n)
        coords_CA.append(ca)
        coords_C.append(c)

        resseq = res.id[1]
        icode = res.id[2].strip() if isinstance(res.id[2], str) else ""
        res_ids.append(f"{ch.id}:{resseq}{icode}")

    return {
        "seq_full": seq_full,
        "seq_atom": "".join(seq_atom),
        "coords_N": np.asarray(coords_N, dtype=float),
        "coords_CA": np.asarray(coords_CA, dtype=float),
        "coords_C": np.asarray(coords_C, dtype=float),
        "res_ids": res_ids,
    }


# =============================================================================
# Simple embedder factory  内置嵌入器工厂（支持单个或多个）
# =============================================================================

def _create_single_embedder(name: str = "esm", **kwargs) -> Any:
    """Create one residue-level embedder by name.
    根据名称创建单个残基级嵌入器。支持 "esm" / "onehot" / "physchem"

    Args:
        name: Embedder name support list.
        **kwargs: Forwarded to the embedder constructor.

    Returns:
        Any: An object exposing `.embed(seq)->Tensor[L,D]`.
    """
    name = (name or "esm").lower()
    if name == "esm":
        return ESMEmbedder(**kwargs)
    if name == "onehot":
        return OneHotEmbedder(**kwargs)
    if name == "physchem":
        return PhysChemEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder name: {name}")


class _CompositeEmbedder:
    """Concatenate multiple residue embedders along feature dim.

    把多个嵌入器在特征维拼接：单序列输入返回 [L, ΣD_i]；
    多序列输入则返回 List[[L, ΣD_i]]。
    """

    def __init__(self, embedders: Sequence[Any]):
        assert len(embedders) >= 1, "CompositeEmbedder needs at least one sub-embedder."
        self.embedders = list(embedders)
        # 对齐 device 到第一个（假设都已各自处理好）
        self.device = getattr(self.embedders[0], "device", "cpu")

    @property
    def dim(self) -> int:
        return int(sum(getattr(e, "dim", 0) for e in self.embedders))

    def info_str(self) -> str:
        parts = [e.__class__.__name__ for e in self.embedders]
        return "[CompositeEmbedder] " + " + ".join(parts)

    def embed(self, sequences: Union[str, Sequence[str]]):
        # 单条序列：直接拼接
        if isinstance(sequences, str):
            xs = []
            for e in self.embedders:
                x = e.embed(sequences)  # [L, D_i]
                if not torch.is_floating_point(x):
                    x = x.float()  # 统一 float
                x = x.to(self.device, non_blocking=True)  # 统一 device（cuda:0 / cpu）
                xs.append(x)

            L = [t.size(0) for t in xs]
            if len(set(L)) != 1:
                raise RuntimeError(f"[SequenceEmbedder] Length mismatch across embedders: {L}")

            return torch.cat(xs, dim=-1)

        # 多条序列：逐条拼接，保持返回 List[Tensor]
        outs: List[torch.Tensor] = []
        for s in sequences:
            xs = [e.embed(s) for e in self.embedders]
            outs.append(torch.cat(xs, dim=-1))
        return outs


def _create_embedder(embedder_cfg: Union[Dict[str, Any], List[Dict[str, Any]], None]) -> Any:
    """Create residue embedder(s) from config.

    支持：
      - 单个 dict：{"name":"esm", ...}
      - 列表 [ {...}, {...} ]：按顺序创建多个，再拼接
      - None：默认 onehot
    """
    if embedder_cfg is None:
        return _CompositeEmbedder([_create_single_embedder("onehot")])

    # 列表 → 复合
    if isinstance(embedder_cfg, (list, tuple)):
        embedders = []
        for cfg in embedder_cfg:
            cfg = dict(cfg or {})
            name = (cfg.pop("name", "onehot") or "onehot")
            embedders.append(_create_single_embedder(name, **cfg))
        comp = _CompositeEmbedder(embedders)
        try:
            print(comp.info_str())
        except Exception:
            pass
        return comp

    # 单个 dict → 单嵌入器（仍包成 composite，便于 dim/行为一致）
    if isinstance(embedder_cfg, dict):
        cfg = dict(embedder_cfg)
        name = (cfg.pop("name", "onehot") or "onehot")
        emb = _create_single_embedder(name, **cfg)
        comp = _CompositeEmbedder([emb])
        try:
            print(comp.info_str())
        except Exception:
            pass
        return comp

    raise TypeError(f"Unsupported embedder config type: {type(embedder_cfg)}")


# =============================================================================
# Base builder  基类：流程 + 钩子
# =============================================================================

class BaseProteinGraphBuilder:
    """Base protein graph builder with flat utilities inside.

    通用流程（遍历→解析→对齐→委托节点/边→保存），对齐/解析等工具内嵌。
    子类只实现 `_build_node_features(...)` 与 `_build_edges(...)` 两个钩子。
    """

    def __init__(self, cfg: BuilderConfig):
        """Init.

        Args:
            cfg: Runtime configuration. 运行配置。
        """
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        # NEW: 支持单个或多个 embedder，并在内部拼接。
        self.embedder = _create_embedder(self.cfg.embedder)

    # ---------- Public API ----------
    def build_folder(self) -> None:
        """Build graphs for all PDB files in the folder.

        遍历 PDB 目录构建图，并写出 manifest.csv。
        """
        rows: List[Dict[str, Any]] = []
        pdb_files = sorted(f for f in os.listdir(self.cfg.pdb_dir) if f.lower().endswith(".pdb"))

        for fname in tqdm(pdb_files, desc="Building graphs"):
            name = os.path.splitext(fname)[0]
            pdb_path = os.path.join(self.cfg.pdb_dir, fname)
            try:
                data, misc = self.build_one(pdb_path, name=name)
                out_pt = os.path.join(self.cfg.out_dir, f"{name}.pt")
                torch.save(data, out_pt)
                rows.append({**misc, "name": name, "status": "ok", "pt_path": out_pt})
            except Exception as e:
                rows.append({"name": name, "status": f"error:{e}", "pt_path": "", "pdb_path": pdb_path})

        pd.DataFrame(rows).to_csv(os.path.join(self.cfg.out_dir, "manifest.csv"), index=False)

    def build_one(self, pdb_path: str, name: Optional[str] = None) -> Tuple[Data, Dict[str, Any]]:
        """
        ATOM-only 构图：
          - 节点：ATOM 可见且含 CA 的标准残基
          - 序列：直接使用 ATOM 序列（seq_atom），恒等对齐
          - 边：对 kept 的 CA 坐标做半径图（由子类实现细节）
        """
        if name is None:
            name = os.path.splitext(os.path.basename(pdb_path))[0]

        # 1) 解析（单链；已过滤 HET/WAT，只保留标准残基）
        parsed = _parse_pdb_single_chain(pdb_path, self.cfg.chain_id)

        # --- 基本健壮性检查 ---
        ca = parsed["coords_CA"]
        if not (isinstance(ca, np.ndarray) and ca.ndim == 2 and ca.shape[1] == 3):
            raise ValueError(f"[{name}] Bad coords_CA shape: {getattr(ca, 'shape', None)}")

        K = int(ca.shape[0])
        if K == 0:
            # 没有任何可见 CA（如核酸链/异常 PDB）——上游可选择捕获并跳过
            raise ValueError(f"[{name}] no_visible_CA")

        # 2) 节点集合 = 全部 ATOM 可见残基；序列直接用 ATOM
        kept_idx = np.arange(K, dtype=int)  # 0..K-1
        seq_text = parsed.get("seq_atom", "")  # 长度应为 K
        atom_kept_mask = np.ones(K, dtype=bool)  # 记录用

        # 3) 子类生成节点/边特征
        #    节点：传 ATOM 序列 + kept_idx
        node_feats = self._build_node_features(
            seq_full=seq_text,
            kept_idx=kept_idx,
            parsed=parsed,
        )

        #    边：对 kept 的 CA 做半径图
        coords_ca_kept = parsed["coords_CA"][kept_idx]  # [K,3]
        if not (isinstance(coords_ca_kept, np.ndarray) and coords_ca_kept.ndim == 2 and coords_ca_kept.shape[1] == 3):
            coords_ca_kept = np.asarray(coords_ca_kept, dtype=float).reshape(-1, 3)

        edge_feats = self._build_edges(coords_ca_kept, parsed=parsed, aln={
            "L_full": K,
            "kept_idx": kept_idx,
            "seq_to_atom_idx": np.arange(K, dtype=int),  # 恒等对齐
            "atom_kept_mask": atom_kept_mask,
        })

        # 4) 组装 Data
        feats: Dict[str, torch.Tensor] = {}
        feats.update(node_feats)
        feats.update(edge_feats)
        data = Data(**feats)
        data.name = name

        # num_nodes 推断
        if "x_s" in feats and isinstance(feats["x_s"], torch.Tensor):
            data.num_nodes = int(feats["x_s"].size(0))
        elif "x" in feats and isinstance(feats["x"], torch.Tensor):
            data.num_nodes = int(feats["x"].size(0))
        elif "edge_index" in feats and isinstance(feats["edge_index"], torch.Tensor) and feats[
            "edge_index"].numel() > 0:
            data.num_nodes = int(feats["edge_index"].max().item() + 1)
        else:
            data.num_nodes = K

        # res_ids 与 kept 对齐
        all_res_ids: List[str] = parsed.get("res_ids", [])
        if isinstance(all_res_ids, list) and len(all_res_ids) == K:
            data.res_ids = [all_res_ids[i] for i in kept_idx]
        else:
            ch = (all_res_ids[0].split(":")[0] if all_res_ids else (self.cfg.chain_id or "A"))
            data.res_ids = [f"{ch}:{i + 1}" for i in range(K)]

        # 记录元信息（调试用）
        data.L_full = K
        data.kept_idx = torch.as_tensor(kept_idx, dtype=torch.long)
        data.seq_to_atom_idx = torch.arange(K, dtype=torch.long)

        # manifest 信息
        misc = {
            "pdb_path": pdb_path,
            "seq_full_len": 0,  # 不再使用 SEQRES
            "seq_atom_len": len(seq_text),  # 应与 K 一致
            "kept_len": K,
            "warn": "atom_only",
        }
        return data, misc

    # ---------- Hooks for subclass ----------
    def _build_node_features(self, *, seq_full: str, kept_idx: np.ndarray, parsed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build node features (override in subclasses).

        构建节点特征（子类实现）。
        GNN: return {"x": [K, D]}
        GVP: return {"x_s": [K, Ds], "x_v": [K, Dv, 3]}

        Args:
            seq_full: Full sequence used to embed.
            kept_idx: Indices in full sequence to keep (aligned).
            parsed: Parsed PDB dict (coords, seqs, res_ids).

        Returns:
            Dict[str, torch.Tensor]: Node feature dict.
        """
        raise NotImplementedError

    def _build_edges(self, coords_ca_kept: np.ndarray, *, parsed: Dict[str, Any], aln: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build edges (override in subclasses).

        构建边（子类实现）。
        GNN: at least {"edge_index"}
        GVP: typically {"edge_index", "edge_s", "edge_v"}

        Args:
            coords_ca_kept: Kept CA coordinates [K,3].
            parsed: Parsed PDB dict.
            aln: Alignment dict.

        Returns:
            Dict[str, torch.Tensor]: Edge feature dict.
        """
        raise NotImplementedError