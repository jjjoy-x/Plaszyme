#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_enzyme_topk_multiclass_spawn_safe.py

- 仅用“酶主干（GNN/GVP）”输出 + MLP 做塑料多分类
- 支持 Top-K（默认 4），非 Top-K 可丢弃/合并 OTHER
- **CUDA in fork** 防呆：
  (a) 强制 ESM embedder 在 CPU 上跑（避免 worker 里触发 CUDA 初始化）
  (b) DataLoader 使用 spawn 上下文
  (c) 默认 NUM_WORKERS=0（先缓存后提速）
"""

from __future__ import annotations
import os, csv, random, multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# === 你项目里的组件 ===
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone

# =================== 配置 ===================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# 路径
CSV_PATH = "/tmp/pycharm_project_317/dataset/plastics_onehot_trainset.csv"
PDB_ROOT = "/root/autodl-tmp/pdb/pdb"   # {PDB_ROOT}/{pid}.pdb
PT_OUT   = "/root/autodl-tmp/pdb/pt"    # 缓存目录：{PT_OUT}/{pid}.pt
os.makedirs(PT_OUT, exist_ok=True)
OUT_DIR  = "./runs_enzyme_topk_cls_spawn_safe_2"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PATH = os.path.join(OUT_DIR, "best.pt")

# 列定义
ID_COLUMN_CANDIDATES = ["protein_id", "pdb_id"]
SPLIT_COLUMN = "split"             # 可选：train/val
VAL_RATIO = 0.15
LABEL_COLUMNS: Optional[List[str]] = None

# Top-K 设置
TOP_K = 6
NON_TOPK_POLICY = "drop"   # "drop" 或 "other"

# 模型与训练
ENZ_BACKBONE = "GNN"   # "GNN" | "GVP"
EMB_DIM_ENZ  = 128
HIDDEN_GNN   = [128, 128, 128]
HIDDEN_GVP   = [(128,16), (128,16), (128,16)]
DROPOUT      = 0.1

# 分类头（MLP）
MLP_HIDDEN = [256, 128]   # [] -> 仅线性
MLP_NORM   = False

BATCH_SIZE   = 32
EPOCHS       = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-4

# 类别不平衡策略
USE_CLASS_WEIGHTS          = True
USE_WEIGHTED_SAMPLER       = True
CLASS_WEIGHT_SMOOTH_ALPHA  = 1.0

# DataLoader（**关键：默认先 0 工人，避免 CUDA in fork**）
NUM_WORKERS = 2
PIN_MEMORY  = (DEVICE == "cuda")

# =================== 工具 ===================
def detect_id_column(header: List[str]) -> str:
    for cand in ID_COLUMN_CANDIDATES:
        if cand in header: return cand
    raise ValueError(f"未在 CSV 表头中找到 ID 列（候选：{ID_COLUMN_CANDIDATES}）。")

def detect_label_columns(header: List[str], id_col: str) -> List[str]:
    if LABEL_COLUMNS:
        for c in LABEL_COLUMNS:
            if c not in header: raise ValueError(f"指定的标签列 {c} 不在 CSV 表头中")
        return LABEL_COLUMNS
    cols = [c for c in header if c not in [id_col, SPLIT_COLUMN]]
    if len(cols) < 2:
        raise ValueError("自动推断的标签列数 < 2，请手动设置 LABEL_COLUMNS")
    return cols

def argmax_onehot(row: Dict[str, str], label_cols: List[str]) -> int:
    ones = [i for i, c in enumerate(label_cols) if float(row.get(c, "0")) > 0.5]
    if len(ones) >= 1: return ones[0]
    vals = [float(row.get(c, "0")) for c in label_cols]
    idx = int(np.argmax(vals))
    return idx if vals[idx] > 0 else -1

def resolve_pdb_path(pid: str) -> str:
    cands = [os.path.join(PDB_ROOT, f"{pid}.pdb"),
             os.path.join(PDB_ROOT, f"{pid.upper()}.pdb"),
             os.path.join(PDB_ROOT, f"{pid.lower()}.pdb")]
    for p in cands:
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"PDB not found for pid='{pid}'. Tried: {cands}")

def cache_path_for(pid: str) -> str:
    return os.path.join(PT_OUT, f"{pid}.pt")

# =================== 读取CSV & Top-K筛选 ===================
@dataclass
class RawRow:
    pid: str
    cls_idx: int
    split_tag: str

def read_rows(csv_path: str) -> Tuple[List[RawRow], List[str], str]:
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        id_col = detect_id_column(header)
        label_cols = detect_label_columns(header, id_col)
        rows: List[RawRow] = []
        for row in rd:
            pid = row[id_col]
            y = argmax_onehot(row, label_cols)
            if y < 0: continue
            rows.append(RawRow(pid=pid, cls_idx=y, split_tag=row.get(SPLIT_COLUMN, "")))
    return rows, label_cols, id_col

def split_rows(rows: List[RawRow]) -> Tuple[List[RawRow], List[RawRow]]:
    has_split = any(r.split_tag for r in rows)
    if has_split:
        train = [r for r in rows if str(r.split_tag).lower() != "val"]
        val   = [r for r in rows if str(r.split_tag).lower() == "val"]
    else:
        rs = np.random.RandomState(SEED); idx = np.arange(len(rows)); rs.shuffle(idx)
        n_val = int(round(len(rows) * VAL_RATIO)); val_ids = set(idx[:n_val])
        train, val = [], []
        for i, r in enumerate(rows):
            (val if i in val_ids else train).append(r)
    return train, val

def topk_mapping_by_train(train: List[RawRow], label_cols: List[str], k: int, policy: str):
    counts = np.zeros(len(label_cols), dtype=np.int64)
    for r in train: counts[r.cls_idx] += 1
    topk_orig_idx = np.argsort(-counts)[:k].tolist()
    remap = {orig: new for new, orig in enumerate(topk_orig_idx)}
    other_idx = (k if policy == "other" else None)
    return counts, topk_orig_idx, remap, other_idx

@dataclass
class RowItem:
    pid: str
    y_new: int

def apply_topk_policy(rows: List[RawRow], remap: Dict[int, int], other_idx: Optional[int], policy: str) -> List[RowItem]:
    out: List[RowItem] = []
    if policy == "drop":
        for r in rows:
            if r.cls_idx in remap: out.append(RowItem(pid=r.pid, y_new=remap[r.cls_idx]))
    elif policy == "other":
        for r in rows:
            out.append(RowItem(pid=r.pid, y_new=remap.get(r.cls_idx, other_idx)))
    else:
        raise ValueError("NON_TOPK_POLICY 仅支持 'drop' 或 'other'")
    return out

# =================== 数据集（带缓存） ===================
class EnzymeOnlyDataset(Dataset):
    def __init__(self, rows: List[RowItem], enzyme_builder):
        super().__init__()
        self.rows = rows
        self.builder = enzyme_builder

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        rk = self.rows[idx]
        pt_path = cache_path_for(rk.pid)
        data = None
        if os.path.isfile(pt_path):
            data = torch.load(pt_path, map_location="cpu")
            if not isinstance(data, Data):
                if isinstance(data, (tuple, list)) and isinstance(data[0], Data):
                    data = data[0]
                else:
                    data = None
        if data is None:
            pdb = resolve_pdb_path(rk.pid)
            data, _misc = self.builder.build_one(pdb, name=rk.pid)
            try: torch.save(data, pt_path)
            except Exception as e: print(f"[WARN] 缓存保存失败: {pt_path} -> {e}")
        return data, rk.y_new

# =================== 指标 ===================
@torch.no_grad()
def evaluate(model: nn.Module, loader, num_classes: int) -> Dict[str, float]:
    model.eval()
    total, correct = 0, 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for g, y in loader:
        g = g.to(DEVICE); y = y.to(DEVICE)
        logits = model(g)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / max(total, 1)
    eps = 1e-12
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp

    prec_c = tp / (tp + fp + eps)
    rec_c  = tp / (tp + fn + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    macro_p = prec_c.mean().item()
    macro_r = rec_c.mean().item()
    macro_f = f1_c.mean().item()

    tp_micro = tp.sum().item()
    fp_micro = fp.sum().item()
    fn_micro = fn.sum().item()
    micro_p = tp_micro / (tp_micro + fp_micro + eps)
    micro_r = tp_micro / (tp_micro + fn_micro + eps)
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    return {
        "acc": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f),
    }

# =================== 模型 ===================
class MLPClassifier(nn.Module):
    def __init__(self, enz_backbone: nn.Module, in_dim: int, num_classes: int,
                 hidden: List[int], dropout: float = 0.1, use_norm: bool = False):
        super().__init__()
        self.backbone = enz_backbone
        layers: List[nn.Module] = [nn.Dropout(dropout)]
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if use_norm: layers += [nn.LayerNorm(h)]
            layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, g):
        z = self.backbone(g)     # [B, in_dim]
        logits = self.mlp(z)     # [B, C]
        return logits

# =================== 主流程 ===================
def main():
    # 读 CSV & 划分
    raw_rows, label_cols, _ = read_rows(CSV_PATH)
    train_raw, val_raw = split_rows(raw_rows)

    # Top-K 选择
    counts, topk_orig_idx, remap, other_idx = topk_mapping_by_train(train_raw, label_cols, TOP_K, NON_TOPK_POLICY)
    print(f"[INFO] Train counts per class (len={len(label_cols)}): {counts.tolist()}")
    topk_names = [label_cols[i] for i in topk_orig_idx]
    print(f"[INFO] Top-{TOP_K} classes: {topk_names}")

    # 应用策略并重映射
    train_rows = apply_topk_policy(train_raw, remap, other_idx, NON_TOPK_POLICY)
    val_rows   = apply_topk_policy(val_raw,   remap, other_idx, NON_TOPK_POLICY)
    num_classes = TOP_K if NON_TOPK_POLICY == "drop" else (TOP_K + 1)
    if NON_TOPK_POLICY == "other":
        topk_names = topk_names + ["OTHER"]

    print(f"[INFO] After policy='{NON_TOPK_POLICY}': train={len(train_rows)} | val={len(val_rows)} | num_classes={num_classes}")
    print(f"[INFO] New label order: {topk_names}")

    # 构建器 + 主干
    # === 关键：强制 ESM 在 CPU 上嵌入，避免 worker 内 CUDA 初始化 ===
    if ENZ_BACKBONE.upper() == "GNN":
        cfg = GNNBuilderConfig(
            pdb_dir=PDB_ROOT, out_dir=PT_OUT, radius=10.0,
            embedder=[{"name": "esm", "model_name": "esm2_t12_35M_UR50D", "fp16": False, "device": "cpu"}]
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
        enz_backbone = GNNBackbone(conv_type="gcn", hidden_dims=HIDDEN_GNN, out_dim=EMB_DIM_ENZ,
                                   dropout=DROPOUT, residue_logits=False).to(DEVICE)
    elif ENZ_BACKBONE.upper() == "GVP":
        cfg = GVPBuilderConfig(
            pdb_dir=PDB_ROOT, out_dir=PT_OUT, radius=10.0,
            embedder=[{"name": "esm", "model_name": "esm2_t12_35M_UR50D", "fp16": False, "device": "cpu"}]
        )
        enzyme_builder = GVPProteinGraphBuilder(cfg)
        enz_backbone = GVPBackbone(hidden_dims=HIDDEN_GVP, out_dim=EMB_DIM_ENZ,
                                   dropout=DROPOUT, residue_logits=False).to(DEVICE)
    else:
        raise ValueError(f"未知 ENZ_BACKBONE: {ENZ_BACKBONE}")

    # 数据集
    ds_train = EnzymeOnlyDataset(train_rows, enzyme_builder)
    ds_val   = EnzymeOnlyDataset(val_rows,   enzyme_builder)

    # 类别计数（新空间）
    cls_counts = np.zeros(num_classes, dtype=np.int64)
    for r in train_rows: cls_counts[r.y_new] += 1
    print(f"[INFO] Train class counts (new space): {cls_counts.tolist()}")

    # 类别权重（平滑）
    counts_smooth = cls_counts.astype(np.float64) + CLASS_WEIGHT_SMOOTH_ALPHA
    inv = 1.0 / counts_smooth
    class_weights = inv / inv.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE) if USE_CLASS_WEIGHTS else None
    if USE_CLASS_WEIGHTS:
        print(f"[INFO] Class weights (alpha={CLASS_WEIGHT_SMOOTH_ALPHA}): {class_weights_t.tolist()}")

    # 采样器
    if USE_WEIGHTED_SAMPLER:
        sample_weights = [1.0 / counts_smooth[r.y_new] for r in train_rows]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_rows),
            replacement=True
        )
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    # DataLoader（**spawn 上下文**，但默认 NUM_WORKERS=0）
    mp_ctx = mp.get_context("spawn")
    train_loader = PyGDataLoader(
        ds_train, batch_size=BATCH_SIZE, sampler=sampler, shuffle=shuffle_flag,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context=mp_ctx,
        drop_last=False
    )
    val_loader = PyGDataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context=mp_ctx,
        drop_last=False
    )

    # 模型、优化器、损失
    clf = MLPClassifier(
        enz_backbone, in_dim=EMB_DIM_ENZ, num_classes=num_classes,
        hidden=MLP_HIDDEN, dropout=DROPOUT, use_norm=MLP_NORM
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t) if USE_CLASS_WEIGHTS else nn.CrossEntropyLoss()

    # 训练
    best_macro_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        clf.train()
        total_loss, n = 0.0, 0

        for g, y in train_loader:
            g = g.to(DEVICE); y = y.to(DEVICE)
            logits = clf(g)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        metrics = evaluate(clf, val_loader, num_classes)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} | "
              f"val_acc={metrics['acc']:.4f} | "
              f"val_macro_f1={metrics['macro_f1']:.4f} | "
              f"val_micro_f1={metrics['micro_f1']:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "cfg": {
                    "seed": SEED,
                    "backbone": ENZ_BACKBONE,
                    "emb_dim_enz": EMB_DIM_ENZ,
                    "hidden_gnn": HIDDEN_GNN,
                    "hidden_gvp": HIDDEN_GVP,
                    "mlp_hidden": MLP_HIDDEN,
                    "mlp_norm": MLP_NORM,
                    "top_k": TOP_K,
                    "non_topk_policy": NON_TOPK_POLICY,
                    "use_class_weights": USE_CLASS_WEIGHTS,
                    "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                    "class_weight_alpha": CLASS_WEIGHT_SMOOTH_ALPHA,
                    "label_space": topk_names,
                    "embedder_device": "cpu",
                },
                "model": clf.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": metrics,
            }, BEST_PATH)
            print(f"[INFO] Saved BEST -> {BEST_PATH} (macro_f1={best_macro_f1:.4f})")

    print(f"[DONE] Best macro_f1={best_macro_f1:.4f} | path={BEST_PATH}")

if __name__ == "__main__":
    # **关键**：主进程设定 spawn，避免 CUDA 在 fork 后再初始化
    import torch.multiprocessing as tmp
    try:
        tmp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()