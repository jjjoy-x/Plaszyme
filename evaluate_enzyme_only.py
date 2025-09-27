#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_enzyme_topk_to_onehot_flexload.py

目的
----
对测试集进行推理，并把每个样本的预测结果导出为“独热(one-hot)表格”CSV。
同时解决你遇到的 `Unexpected key(s) in state_dict` 问题：
- 先尝试 strict=True；
- 若失败，自动做 **键名对齐**（添加/去掉 "backbone." 前缀，或仅加载交集），
  从而兼容“训练时保存的是 backbone+mlp / 仅backbone / 仅mlp”等不同情况。

使用
----
1) 修改 TEST_CSV / PT_OUT / BEST_CKPT / OUT_CSV 为你的路径；
2) （推荐）先用预处理脚本生成 {PT_OUT}/{pid}.pt；
3) 运行本脚本得到 one-hot CSV。

说明
----
- 默认仅从缓存 .pt 读取图数据；如需测试时临时构图（CPU），把 ALLOW_BUILD_MISSING=True 并设置 PDB_ROOT。
"""

from __future__ import annotations
import os, csv, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# ========= 路径配置（按需修改） =========
TEST_CSV  = "/tmp/pycharm_project_317/dataset/plastics_onehot_testset.csv"                           # 测试集，需含 protein_id 或 pdb_id
PT_OUT    = "/root/autodl-tmp/pdb/pt"                          # 图缓存目录
BEST_CKPT = "/tmp/pycharm_project_317/train_script/runs_enzyme_topk_cls_spawn_safe_2/best.pt"        # 训练保存的 best.pt
OUT_CSV   = "./runs_enzyme_topk_cls_spawn_safe/pred_onehot_test.csv"                           # 导出路径

# 若测试样本缺少缓存，是否允许临时构图（CPU）
ALLOW_BUILD_MISSING = True
PDB_ROOT   = "/root/autodl-tmp/pdb/pdb"
EMBEDDER_DEVICE_FOR_BUILD = "cpu"  # 临时构图统一 CPU

# DataLoader
BATCH_SIZE  = 64
NUM_WORKERS = min(8, os.cpu_count() or 0)
PIN_MEMORY  = torch.cuda.is_available()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 你项目中的模块 =========
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone

if ALLOW_BUILD_MISSING:
    from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
    from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig

# ========= 工具函数 =========
def safe_torch_load(path: str):
    """兼容未来 weights_only=True 的安全加载。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
        return torch.load(path, map_location="cpu")

def resolve_id_column(header: List[str]) -> str:
    for cand in ("protein_id", "pdb_id"):
        if cand in header:
            return cand
    raise ValueError(f"测试 CSV 缺少 id 列（需要 'protein_id' 或 'pdb_id'），实际表头：{header}")

def resolve_pdb_path(pid: str) -> str:
    cands = [os.path.join(PDB_ROOT, f"{pid}.pdb"),
             os.path.join(PDB_ROOT, f"{pid.upper()}.pdb"),
             os.path.join(PDB_ROOT, f"{pid.lower()}.pdb")]
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"PDB not found for pid={pid}. Tried: {cands}")

def cache_path_for(pid: str) -> str:
    return os.path.join(PT_OUT, f"{pid}.pt")

@dataclass
class TestItem:
    pid: str

def read_test_ids(csv_path: str) -> List[TestItem]:
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        id_col = resolve_id_column(header)
        items: List[TestItem] = []
        for row in rd:
            pid = row.get(id_col, "").strip()
            if pid:
                items.append(TestItem(pid=pid))
    if not items:
        raise ValueError("测试 CSV 未读到任何样本（空文件或 id 列为空）")
    return items

# ========= 数据集 =========
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[TestItem], *, allow_build: bool, enz_backbone_name: str):
        super().__init__()
        self.items = items
        self.allow_build = allow_build
        self.enz_backbone_name = enz_backbone_name.upper()
        if self.allow_build:
            if self.enz_backbone_name == "GNN":
                self.builder = GNNProteinGraphBuilder(GNNBuilderConfig(
                    pdb_dir=PDB_ROOT, out_dir=PT_OUT, radius=10.0,
                    embedder=[{"name": "esm", "model_name": "esm2_t12_35M_UR50D", "fp16": False, "device": EMBEDDER_DEVICE_FOR_BUILD}]
                ), edge_mode="none")
            elif self.enz_backbone_name == "GVP":
                self.builder = GVPProteinGraphBuilder(GVPBuilderConfig(
                    pdb_dir=PDB_ROOT, out_dir=PT_OUT, radius=10.0,
                    embedder=[{"name": "esm", "model_name": "esm2_t12_35M_UR50D", "fp16": False, "device": EMBEDDER_DEVICE_FOR_BUILD}]
                ))
            else:
                raise ValueError(f"未知主干类型：{self.enz_backbone_name}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        pid = self.items[idx].pid
        pt_path = cache_path_for(pid)
        data = None
        if os.path.isfile(pt_path):
            data = safe_torch_load(pt_path)
            if not isinstance(data, Data):
                if isinstance(data, (tuple, list)) and isinstance(data[0], Data):
                    data = data[0]
                else:
                    data = None
        if data is None:
            if not self.allow_build:
                raise FileNotFoundError(f"缺少缓存：{pt_path}。如需允许临时构图，请将 ALLOW_BUILD_MISSING=True 并设置 PDB_ROOT。")
            pdb = resolve_pdb_path(pid)
            data, _ = self.builder.build_one(pdb, name=pid)
            try:
                torch.save(data, pt_path)
            except Exception as e:
                print(f"[WARN] 缓存保存失败 {pt_path}: {e}")
        return data, pid

# ========= 模型（酶主干 + MLP）=========
class MLPClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, in_dim: int, num_classes: int,
                 hidden: List[int], dropout: float = 0.1, use_norm: bool = False):
        super().__init__()
        self.backbone = backbone
        layers: List[nn.Module] = [nn.Dropout(dropout)]
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if use_norm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, g):
        z = self.backbone(g)   # [B, D]
        return self.mlp(z)     # [B, C]

# ========= 灵活加载（修复 Unexpected key(s)）=========
def flex_load_state_dict(model: nn.Module, ckpt_state: Dict[str, torch.Tensor]) -> None:
    """
    1) strict=True 尝试；
    2) 若失败：
       - 若 ckpt 键以 'backbone.' 开头而 model 不含该前缀，尝试去前缀；
       - 若 model 需要 'backbone.' 前缀而 ckpt 没有，尝试加前缀；
       - 仍失败则加载交集（strict=False），打印 missing/extra 诊断。
    """
    # 尝试 strict=True
    try:
        model.load_state_dict(ckpt_state, strict=True)
        print("[INFO] Loaded state_dict with strict=True")
        return
    except Exception as e:
        print(f"[WARN] strict=True 加载失败：{e}")

    model_keys = list(model.state_dict().keys())
    ckpt_keys  = list(ckpt_state.keys())
    model_has_backbone = any(k.startswith("backbone.") for k in model_keys)
    ckpt_has_backbone  = any(k.startswith("backbone.") for k in ckpt_keys)

    # 情况 A：ckpt 有前缀，model 没有前缀 -> 去掉前缀
    if ckpt_has_backbone and (not model_has_backbone):
        print("[INFO] 尝试去除 ckpt 的 'backbone.' 前缀后再加载...")
        stripped = {}
        for k, v in ckpt_state.items():
            if k.startswith("backbone."):
                stripped[k[len("backbone."):]] = v
            else:
                stripped[k] = v
        try:
            model.load_state_dict(stripped, strict=True)
            print("[INFO] Loaded after stripping 'backbone.' prefix with strict=True")
            return
        except Exception as e2:
            print(f"[WARN] 去前缀 strict=True 仍失败：{e2}")

    # 情况 B：model 需要前缀，ckpt 没有前缀 -> 添加前缀
    if (not ckpt_has_backbone) and model_has_backbone:
        print("[INFO] 尝试给 ckpt 键添加 'backbone.' 前缀后再加载...")
        added = {}
        for k, v in ckpt_state.items():
            # 粗略判断：mlp.* 不加前缀，其他当作 backbone 参数
            if k.startswith("mlp."):
                added[k] = v
            else:
                added["backbone." + k] = v
        try:
            model.load_state_dict(added, strict=True)
            print("[INFO] Loaded after adding 'backbone.' prefix with strict=True")
            return
        except Exception as e3:
            print(f"[WARN] 加前缀 strict=True 仍失败：{e3}")

    # 最后兜底：交集加载
    print("[INFO] 退化为交集加载（strict=False）")
    model_kset = set(model.state_dict().keys())
    filtered = {k: v for k, v in ckpt_state.items() if k in model_kset}
    msg = model.load_state_dict(filtered, strict=False)
    if getattr(msg, "missing_keys", None):
        print(f"[MISS] missing: {len(msg.missing_keys)} 例如: {msg.missing_keys[:8]}")
    if getattr(msg, "unexpected_keys", None):
        print(f"[EXTRA] unexpected: {len(msg.unexpected_keys)} 例如: {msg.unexpected_keys[:8]}")

# ========= 主流程 =========
def main():
    # 0) 读取测试 id
    test_items = read_test_ids(TEST_CSV)
    print(f"[INFO] Test samples: {len(test_items)}")

    # 1) 读取 checkpoint 配置
    ckpt = safe_torch_load(BEST_CKPT)
    cfg = ckpt.get("cfg", {})
    label_space: List[str] = cfg.get("label_space", None) or ckpt.get("label_columns", None)
    if not label_space:
        raise ValueError("checkpoint 中未找到 label_space/label_columns。")

    enz_backbone_name = str(cfg.get("backbone", "GNN")).upper()
    emb_dim_enz = int(cfg.get("emb_dim_enz", 128))
    hidden_gnn  = cfg.get("hidden_gnn", [128, 128, 128])
    hidden_gvp  = cfg.get("hidden_gvp", [(128, 16), (128, 16), (128, 16)])
    mlp_hidden  = cfg.get("mlp_hidden", [256, 128])
    mlp_norm    = bool(cfg.get("mlp_norm", False))
    dropout     = 0.1
    num_classes = len(label_space)

    print(f"[INFO] Loaded checkpoint: classes={num_classes} | label_space={label_space}")
    print(f"[INFO] Backbone={enz_backbone_name} | D={emb_dim_enz} | MLP={mlp_hidden} | norm={mlp_norm}")

    # 2) 构造模型（与训练同架构：酶主干 + MLP）
    if enz_backbone_name == "GNN":
        backbone = GNNBackbone(conv_type="gcn", hidden_dims=hidden_gnn, out_dim=emb_dim_enz,
                               dropout=dropout, residue_logits=False).to(DEVICE)
    elif enz_backbone_name == "GVP":
        backbone = GVPBackbone(hidden_dims=hidden_gvp, out_dim=emb_dim_enz,
                               dropout=dropout, residue_logits=False).to(DEVICE)
    else:
        raise ValueError(f"未知主干类型：{enz_backbone_name}")

    model = MLPClassifier(backbone, emb_dim_enz, num_classes, mlp_hidden, dropout=dropout, use_norm=mlp_norm).to(DEVICE)

    # 3) 灵活加载权重（修复你遇到的 Unexpected key(s)）
    ckpt_model_state = ckpt.get("model", None)
    if ckpt_model_state is None:
        raise ValueError("checkpoint 不包含 'model' 权重字典。")
    flex_load_state_dict(model, ckpt_model_state)
    model.eval()

    # 4) DataLoader
    ds_test = TestDataset(test_items, allow_build=ALLOW_BUILD_MISSING, enz_backbone_name=enz_backbone_name)
    test_loader = PyGDataLoader(
        ds_test, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        drop_last=False
    )

    # 5) 推理 & 导出 one-hot
    all_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for g, pid in test_loader:
            g = g.to(DEVICE)
            logits = model(g)                    # [B, C]
            pred_idx = logits.argmax(dim=-1)     # [B]
            for i in range(pred_idx.size(0)):
                row = {"id": pid[i]}
                hot = np.zeros(num_classes, dtype=np.int64)
                hot[int(pred_idx[i].item())] = 1
                for j, name in enumerate(label_space):
                    row[name] = int(hot[j])
                row["pred_label"] = label_space[int(pred_idx[i].item())]
                all_rows.append(row)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id"] + list(label_space) + ["pred_label"])
        wr.writeheader()
        wr.writerows(all_rows)

    print(f"[DONE] Wrote one-hot predictions -> {OUT_CSV}")

if __name__ == "__main__":
    main()