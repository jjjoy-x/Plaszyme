#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gnn_with_plastic_binary_from_matrix.py
(从“酶×塑料”的独热矩阵自动生成 (enzyme, plastic, label) 配对；仅硬标签0/1；自动控制正负样本比例)

✅ 你要什么
- 输入是**独热矩阵**（行=酶enzyme，列=塑料；1=已知正样本，0=未知/按负样本处理）
- 不用软标签，直接用 0/1 训练
- 训练/验证时**按酶分组切分**（避免数据泄漏）
- **自动控制正负样本比例**（负样本按每个酶进行下采样/均衡）
- 支持“can_degrade_”等前缀、忽略大小写与 -/_ 差异对齐塑料名称
- 其余保持你原 fuse 模型结构：DeepFRIModel(酶图) + MLP(塑料) + 融合分类器

📦 需要的文件
- enzyme 图数据目录：dataset_dir/{enzyme}.pt  (PyG 的 Data)
- 塑料特征 .pt：由 PlasticFeaturizer.save_features(...) 产出，包含 {"features": {name: Tensor(Dp)}, "feature_names": [...] }
- 独热矩阵 CSV（行=enzyme，列=塑料名或带前缀列名）

用法
- 直接修改下方 CONFIG 或改为 USE_CONFIG=False 用 CLI 参数
"""

import argparse
import os
import random
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as GeoBatch
from torch_geometric.data import Data

from model.gcn_model import DeepFRIModel  # 你的酶图编码器
from utils.visualization import (
    log_confusion_matrix,
    log_per_class_accuracy,
    log_curve,
    log_weights_histogram
)

# sklearn（可选）
try:
    from sklearn.metrics import roc_auc_score
    HAS_SK_METRICS = True
except Exception:
    HAS_SK_METRICS = False

try:
    from sklearn.model_selection import GroupShuffleSplit
    HAS_SK_SPLIT = True
except Exception:
    HAS_SK_SPLIT = False


# ===============================
# Direct configuration
# ===============================
USE_CONFIG = True  # True: 用 CONFIG；False: 用命令行

DEFAULT_NUM_WORKERS = (
    0 if (not torch.cuda.is_available() and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    else 4
)

CONFIG = {
    # --- 路径 ---
    "dataset_dir": "/tmp/pycharm_project_27/dataset/train_graph",                         # 存放 {enzyme}.pt 的目录
    "plastic_feat_pt": "/tmp/pycharm_project_27/test/outputs/all_description_new.pt",       # PlasticFeaturizer.save_features 输出的 .pt
    "matrix_csv": "/tmp/pycharm_project_27/dataset/trainset.csv",               # 独热矩阵 CSV：行=enzyme，列=塑料
    "save_dir": "/tmp/pycharm_project_27/checkpoints/binary_01_1",
    "log_dir": "logs_bin_from_matrix",

    # --- 矩阵索引/前缀 ---
    "enzyme_index_col": 0,                  # enzyme 行索引列（None / "none" 表示没有索引列）
    "label_prefix": "can_degrade_",         # 列名前缀；若列名就是塑料，设为 "" 或 None
    "ignore_case_dash_underscore": True,    # 对齐名称时忽略大小写和 -/_ 差异

    # --- 采样控制（自动控制正负样本比例）---
    "neg_pos_ratio_train": 1.0,             # 训练集：每个酶采样的负样本数 ≈ ratio * 正样本数；<=0 表示用全量负样本
    "neg_pos_ratio_val": 3.0,               # 验证集：同上（一般放宽一点）
    "min_pos_per_enzyme": 1,                # 若某酶正样本数 < 该值，则整个酶跳过（避免无意义负样本）
    "max_pairs_per_enzyme_train": 20000,    # 每个酶最多采样的 pair（训练）
    "max_pairs_per_enzyme_val": 50000,      # 每个酶最多采样的 pair（验证）
    "shuffle_seed": 42,

    # --- 评估 ---
    "pred_threshold": 0.5,                  # 概率→硬标签的阈值（用于 acc/cm）

    # --- 模型结构 ---
    "gnn_type": "gcn",                      # ["gcn", "gat"] 由你的 DeepFRIModel 支持
    "gnn_dims": [128, 128],
    "fc_dims": [128, 64],
    "gnn_embed_dim": 128,
    "plastic_hidden": [256, 128],
    "fusion_hidden": [128, 64],
    "dropout": 0.3,

    # --- 训练 ---
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "lr_step": 10,
    "lr_gamma": 0.5,
    "train_ratio": 0.8,

    # --- DataLoader ---
    "num_workers": DEFAULT_NUM_WORKERS,
    "pin_memory": True,
}


# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------
# Utils: names / features / graphs
# ------------------------------
def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace("-", "").replace("_", "")

def load_plastic_features(plastic_feat_pt: str) -> Tuple[Dict[str, torch.Tensor], int, List[str]]:
    store = torch.load(plastic_feat_pt, map_location="cpu")
    feat_dict: Dict[str, torch.Tensor] = store["features"]
    if not len(feat_dict):
        raise ValueError("Empty plastic features dictionary.")
    any_key = next(iter(feat_dict.keys()))
    feat_dim = int(feat_dict[any_key].numel())
    feature_names = store.get("feature_names", None) or [f"feature_{i}" for i in range(feat_dim)]
    print(f"[INFO] Plastic feature dimension: {feat_dim} | #plastics in features: {len(feat_dict)}")
    return feat_dict, feat_dim, feature_names

def load_graph(graphs_dir: str, enzyme_name: str) -> Data:
    add_safe_globals([Data])
    path = os.path.join(graphs_dir, f"{enzyme_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    data: Data = torch.load(path, weights_only=False)
    return data


# ------------------------------
# 从独热矩阵生成 (enzyme, plastic, label) 配对
# ------------------------------
def build_pairs_from_matrix(
    matrix_csv: str,
    enzyme_index_col,
    label_prefix: str,
    plastic_feat_names: List[str],
    ignore_case_dash_underscore: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    读取独热矩阵（行=enzyme，列=塑料）并对齐到“塑料特征”的名字集合。
    返回：
      - df_long: DataFrame(enzyme, plastic, label)  (未采样，全量0/1)
      - enzymes: 行索引列表
      - aligned_plastics: 与 df_long.plastic 所用的名字集合（来自塑料特征字典的 key）
    """
    # 读取矩阵
    if enzyme_index_col is None:
        mat = pd.read_csv(matrix_csv)
        mat.index = mat.index.astype(str)
    else:
        mat = pd.read_csv(matrix_csv, index_col=enzyme_index_col)
    mat.index = mat.index.astype(str)

    # 提取塑料列：去前缀
    def strip_prefix(col: str, prefix: str) -> str:
        if prefix:
            return col[len(prefix):] if str(col).startswith(prefix) else str(col)
        return str(col)

    col_raw = list(mat.columns)
    col_stripped = [strip_prefix(c, label_prefix) for c in col_raw]

    # 名称规范化映射
    if ignore_case_dash_underscore:
        norm_plastics_features = {normalize_name(k): k for k in plastic_feat_names}
        # 对齐列名到 features 的原名
        aligned = []
        aligned_orig = []
        for c in col_stripped:
            key = normalize_name(c)
            if key in norm_plastics_features:
                aligned.append(key)
                aligned_orig.append(norm_plastics_features[key])
        if not aligned:
            raise ValueError("No plastic columns in matrix can be aligned to plastic features.")
        # 构造一个“矩阵列名(原始) -> features原名”的映射
        raw_to_featname = {}
        for raw, stripped in zip(col_raw, col_stripped):
            key = normalize_name(stripped)
            if key in norm_plastics_features:
                raw_to_featname[raw] = norm_plastics_features[key]
        # 仅保留可对齐列，并重命名为“features原名”
        mat = mat[[c for c in col_raw if c in raw_to_featname]].copy()
        mat.columns = [raw_to_featname[c] for c in mat.columns]
        aligned_plastic_names = list(mat.columns)
    else:
        # 不忽略大小写/符号时，直接按交集对齐
        aligned_plastic_names = [c for c in col_stripped if c in plastic_feat_names]
        rename_map = {raw: stripped for raw, stripped in zip(col_raw, col_stripped) if stripped in aligned_plastic_names}
        mat = mat[[raw for raw in col_raw if raw in rename_map]].copy()
        mat.columns = [rename_map[c] for c in mat.columns]

    # 将矩阵展开为长表 (enzyme, plastic, label)
    df_long = mat.stack().reset_index()
    df_long.columns = ["enzyme", "plastic", "label"]
    df_long["label"] = df_long["label"].astype(float).clip(0.0, 1.0)

    return df_long, list(mat.index), aligned_plastic_names


def sample_pairs_per_enzyme(
    df_long: pd.DataFrame,
    neg_pos_ratio: float,
    min_pos_per_enzyme: int,
    max_pairs_per_enzyme: int,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    """
    按“每个酶”采样：保留所有正样本；负样本按 ratio 下采样（<=0 表示全量负样本）。
    对每个酶限制最多 max_pairs_per_enzyme。
    """
    rng = np.random.default_rng(shuffle_seed)
    result = []
    for enzyme, sub in df_long.groupby("enzyme"):
        pos = sub[sub["label"] >= 0.5]
        neg = sub[sub["label"] < 0.5]
        if len(pos) < min_pos_per_enzyme:
            # 该酶没有有效正样本，整体跳过
            continue
        if neg_pos_ratio is not None and neg_pos_ratio > 0:
            k = int(round(neg_pos_ratio * len(pos)))
            if k < len(neg):
                neg = neg.sample(n=k, random_state=int(rng.integers(0, 1 << 31)))
        # 合并并截断数量上限
        merged = pd.concat([pos, neg], axis=0)
        if len(merged) > max_pairs_per_enzyme:
            merged = merged.sample(n=max_pairs_per_enzyme, random_state=int(rng.integers(0, 1 << 31)))
        result.append(merged)
    if not result:
        raise ValueError("No enzyme retained after per-enzyme sampling. Check your matrix or min_pos_per_enzyme.")
    out = pd.concat(result, axis=0).reset_index(drop=True)
    return out


# ------------------------------
# Dataset / Collate
# ------------------------------
class PairDataset(torch.utils.data.Dataset):
    """
    从 (enzyme, plastic, label) 的 DataFrame 读取样本；只用硬标签 0/1。
    """

    def __init__(self, pairs_df: pd.DataFrame, graphs_dir: str, plastic_feat_dict: Dict[str, torch.Tensor]):
        self.df = pairs_df.reset_index(drop=True)
        self.graphs_dir = graphs_dir
        self.plastic_feats = plastic_feat_dict

        # 过滤：必须存在图与塑料特征
        keep = []
        for i, r in self.df.iterrows():
            e = str(r["enzyme"])
            p = str(r["plastic"])
            gpath = os.path.join(graphs_dir, f"{e}.pt")
            if (p in self.plastic_feats) and os.path.exists(gpath):
                keep.append(i)
        self.df = self.df.loc[keep].reset_index(drop=True)
        print(f"[INFO] PairDataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        enzyme = str(r["enzyme"])
        plastic = str(r["plastic"])
        label = float(r["label"])
        data = load_graph(self.graphs_dir, enzyme)
        plastic_vec = self.plastic_feats[plastic].float()
        return data, plastic_vec, label


def collate_fn(batch):
    data_list, plastics, labels = zip(*batch)
    batch_graph = GeoBatch.from_data_list(data_list)
    plastics = torch.stack(plastics, dim=0)            # [B, Dp]
    labels = torch.tensor(labels, dtype=torch.float32) # [B]
    return batch_graph, plastics, labels


# ------------------------------
# Model (与原版一致)
# ------------------------------
class PlasticEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class FusionBinaryModel(nn.Module):
    def __init__(
        self,
        gnn_type: str,
        gnn_dims: List[int],
        fc_dims: List[int],
        gnn_embed_dim: int,
        plastic_in_dim: int,
        plastic_hidden: List[int],
        fusion_hidden: List[int],
        dropout: float = 0.3
    ):
        super().__init__()
        self.enzyme_enc = DeepFRIModel(
            gnn_type=gnn_type,
            gnn_dims=gnn_dims,
            fc_dims=fc_dims,
            out_dim=gnn_embed_dim,
            dropout=dropout
        )
        self.plastic_enc = PlasticEncoder(plastic_in_dim, plastic_hidden, dropout=dropout)

        in_dim = gnn_embed_dim + (plastic_hidden[-1] if len(plastic_hidden) > 0 else plastic_in_dim)
        dims = [in_dim] + fusion_hidden + [1]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.fusion = nn.Sequential(*layers)

    def forward(self, batch_graph: GeoBatch, plastic_vecs: torch.Tensor):
        enz_embed = self.enzyme_enc(batch_graph)   # [B, gnn_embed_dim]
        pla_embed = self.plastic_enc(plastic_vecs) # [B, H]
        z = torch.cat([enz_embed, pla_embed], dim=1)
        logit = self.fusion(z).squeeze(1)         # [B]
        return logit


# ------------------------------
# Train / Eval
# ------------------------------
def bce_loss_logits(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch_graph, plastics, labels in loader:
        batch_graph = batch_graph.to(device)
        plastics = plastics.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_graph, plastics)
        loss = bce_loss_logits(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, pred_threshold=0.5):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for batch_graph, plastics, labels in loader:
        batch_graph = batch_graph.to(device)
        plastics = plastics.to(device)
        labels = labels.to(device)
        logits = model(batch_graph, plastics)
        loss = bce_loss_logits(logits, labels)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if n == 0:
        return 0.0, 0.0, [], [], float("nan")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits)
    preds = (probs >= pred_threshold).long()
    hard_labels = (all_labels >= 0.5).long()

    acc = float((preds == hard_labels).float().mean().item())

    if HAS_SK_METRICS:
        y_true = hard_labels.numpy()
        y_score = probs.numpy()
        if len(np.unique(y_true)) < 2:
            auroc = float("nan")
            print("[WARN] VAL set has a single class; AUROC is undefined.")
        else:
            auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = float("nan")

    return total_loss / n, acc, preds.tolist(), hard_labels.tolist(), auroc


# ------------------------------
# Group split
# ------------------------------
def group_split_by_enzyme(enzymes: List[str], train_ratio=0.8, seed=42):
    idx = np.arange(len(enzymes))
    groups = np.array(enzymes, dtype=object)
    # 每个 enzyme 只出现一次（按组切分其实就是随机划分 enzyme 集合）
    uniq = np.array(enzymes)
    if HAS_SK_SPLIT:
        gss = GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
        # 这里把样本索引当成“每个酶一个样本”的占位，groups=酶名，本质是按组切
        tr_idx, va_idx = next(gss.split(idx, None, groups))
        return tr_idx.tolist(), va_idx.tolist()
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_train = int(len(uniq) * train_ratio)
        train_set = set(uniq[:n_train])
        tr_idx = [i for i, e in enumerate(enzymes) if e in train_set]
        va_idx = [i for i, e in enumerate(enzymes) if e not in train_set]
        return tr_idx, va_idx


# ------------------------------
# Main
# ------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 载入塑料特征
    plastic_feats, plastic_dim, feature_names = load_plastic_features(args.plastic_feat_pt)
    plastic_feat_names = list(plastic_feats.keys())

    # 从独热矩阵构建全量配对
    enzyme_index_col = None if (args.enzyme_index_col is None or str(args.enzyme_index_col).lower()=="none") else int(args.enzyme_index_col)
    df_long_all, enzyme_list, aligned_plastics = build_pairs_from_matrix(
        matrix_csv=args.matrix_csv,
        enzyme_index_col=enzyme_index_col,
        label_prefix=(args.label_prefix or ""),
        plastic_feat_names=plastic_feat_names,
        ignore_case_dash_underscore=bool(args.ignore_case_dash_underscore)
    )

    # 按酶分组切分（注意，这里是按酶 ID 切分）
    tr_eidx, va_eidx = group_split_by_enzyme(enzyme_list, train_ratio=args.train_ratio, seed=SEED)
    train_enz_set = set([enzyme_list[i] for i in tr_eidx])
    val_enz_set   = set([enzyme_list[i] for i in va_eidx])

    # 拆分后按酶内做负采样/比例控制
    df_train_all_enzyme = df_long_all[df_long_all["enzyme"].isin(train_enz_set)].reset_index(drop=True)
    df_val_all_enzyme   = df_long_all[df_long_all["enzyme"].isin(val_enz_set)].reset_index(drop=True)

    df_train = sample_pairs_per_enzyme(
        df_train_all_enzyme,
        neg_pos_ratio=float(args.neg_pos_ratio_train),
        min_pos_per_enzyme=int(args.min_pos_per_enzyme),
        max_pairs_per_enzyme=int(args.max_pairs_per_enzyme_train),
        shuffle_seed=int(args.shuffle_seed),
    )
    df_val = sample_pairs_per_enzyme(
        df_val_all_enzyme,
        neg_pos_ratio=float(args.neg_pos_ratio_val),
        min_pos_per_enzyme=int(args.min_pos_per_enzyme),
        max_pairs_per_enzyme=int(args.max_pairs_per_enzyme_val),
        shuffle_seed=int(args.shuffle_seed) + 1,
    )

    print(f"[INFO] Train pairs: {len(df_train)} | Val pairs: {len(df_val)}")

    # Dataset / Loader
    train_ds = PairDataset(df_train, graphs_dir=args.dataset_dir, plastic_feat_dict=plastic_feats)
    val_ds   = PairDataset(df_val,   graphs_dir=args.dataset_dir, plastic_feat_dict=plastic_feats)

    pin_memory_flag = torch.cuda.is_available() and bool(args.pin_memory)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )

    # Model
    model = FusionBinaryModel(
        gnn_type=args.gnn_type,
        gnn_dims=args.gnn_dims,
        fc_dims=args.fc_dims,
        gnn_embed_dim=args.gnn_embed_dim,
        plastic_in_dim=plastic_dim,
        plastic_hidden=args.plastic_hidden,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, va_preds, va_labels, va_auc = evaluate(
            model, val_loader, device, pred_threshold=args.pred_threshold
        )
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"val_acc={va_acc:.4f} | val_auc={va_auc:.4f} | lr={lr:.6f}")

        # TensorBoard
        log_curve(writer, "Loss/train", [tr_loss], [epoch])
        log_curve(writer, "Loss/val",   [va_loss], [epoch])
        log_curve(writer, "Acc/val",    [va_acc],  [epoch])
        if not (np.isnan(va_auc) or (isinstance(va_auc, float) and np.isinf(va_auc))):
            log_curve(writer, "AUC/val", [va_auc], [epoch])
        log_curve(writer, "LearningRate", [lr], [epoch])
        log_weights_histogram(writer, model, epoch)

        # 可视化硬标签指标
        class_names = ["0", "1"]
        cm_path = os.path.join(args.save_dir, f"cm_epoch{epoch:02d}.png")
        acc_path = os.path.join(args.save_dir, f"per_class_acc_epoch{epoch:02d}.png")
        log_confusion_matrix(va_labels, va_preds, class_names, cm_path)
        log_per_class_accuracy(va_labels, va_preds, class_names, acc_path)

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

        scheduler.step()

    print(f"\nTraining completed. Best Val Acc: {best_val_acc:.4f}")


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    if USE_CONFIG:
        args = SimpleNamespace(**CONFIG)
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)
    else:
        parser = argparse.ArgumentParser(description="Train GNN+Plastic binary classifier from one-hot matrix (auto negative sampling)")
        # Paths
        parser.add_argument("--dataset_dir", type=str, required=True, help="Folder with enzyme .pt graphs ({enzyme}.pt)")
        parser.add_argument("--plastic_feat_pt", type=str, required=True, help="Plastic features .pt from PlasticFeaturizer.save_features")
        parser.add_argument("--matrix_csv", type=str, required=True, help="Enzyme×Plastic one-hot matrix CSV (rows=enzyme, cols=plastics)")
        parser.add_argument("--save_dir", type=str, default="checkpoints_bin_from_matrix")
        parser.add_argument("--log_dir", type=str, default="logs_bin_from_matrix")
        # Matrix index / prefix
        parser.add_argument("--enzyme_index_col", type=str, default="0", help="Row index column for enzymes (int or 'none')")
        parser.add_argument("--label_prefix", type=str, default="can_degrade_", help="Column prefix; '' if none")
        parser.add_argument("--ignore_case_dash_underscore", action="store_true", help="Normalize names to match features (case-insensitive and strip -/_ )")
        # Sampling ratios
        parser.add_argument("--neg_pos_ratio_train", type=float, default=1.0, help="Train: negatives per positive per-enzyme; <=0 = use all negatives")
        parser.add_argument("--neg_pos_ratio_val", type=float, default=3.0, help="Val: negatives per positive per-enzyme; <=0 = use all negatives")
        parser.add_argument("--min_pos_per_enzyme", type=int, default=1, help="Skip enzyme if positives < this")
        parser.add_argument("--max_pairs_per_enzyme_train", type=int, default=20000)
        parser.add_argument("--max_pairs_per_enzyme_val", type=int, default=50000)
        parser.add_argument("--shuffle_seed", type=int, default=42)
        # Eval
        parser.add_argument("--pred_threshold", type=float, default=0.5)
        # Model
        parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"])
        parser.add_argument("--gnn_dims", nargs='+', type=int, default=[128, 128])
        parser.add_argument("--fc_dims", nargs='+', type=int, default=[128, 64])
        parser.add_argument("--gnn_embed_dim", type=int, default=128)
        parser.add_argument("--plastic_hidden", nargs='+', type=int, default=[256, 128])
        parser.add_argument("--fusion_hidden", nargs='+', type=int, default=[128, 64])
        parser.add_argument("--dropout", type=float, default=0.3)
        # Train
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--lr_step", type=int, default=10)
        parser.add_argument("--lr_gamma", type=float, default=0.5)
        parser.add_argument("--train_ratio", type=float, default=0.8)
        # Loader
        parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
        parser.add_argument("--pin_memory", action="store_true")

        args = parser.parse_args()
        # 处理 enzyme_index_col
        if isinstance(args.enzyme_index_col, str):
            if args.enzyme_index_col.lower() == "none":
                args.enzyme_index_col = None
            else:
                args.enzyme_index_col = int(args.enzyme_index_col)
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)