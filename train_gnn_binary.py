#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gnn_with_plastic_binary_from_matrix.py
(独热矩阵 → (enzyme, plastic, label)；硬标签训练；按酶分组切分；负样本按比例随机下采样)

新增改动（相对上一版）：
1) 正类加权：使用 BCEWithLogitsLoss(pos_weight=neg/pos) 强化正类梯度，抑制“全判负”。
2) 平衡批采样：BalancedBatchSampler，保证每个 mini-batch 内正负比固定（默认 1:1）。
3) 最后一层偏置初始化：按训练集先验 π=pos/(pos+neg)，将输出层 bias 设为 log(π/(1-π))。
4) 验证阶段阈值扫描 + AUPRC：训练结束后在验证集扫阈值，报告 best_F1 / best_MCC / AUPRC，并保存到 txt。

仍保持：
- 输入：独热矩阵（行=酶 enzyme，列=塑料；1=正样本，0=未知按负处理）
- 按酶分组切分，避免数据泄漏
- 负样本按每酶“正:负=1:R”随机下采样（R=neg_pos_ratio_*）
- 可选列前缀对齐（label_prefix）、忽略大小写与 -/_ 差异（ignore_case_dash_underscore）
- 模型：DeepFRIModel(酶图) + MLP(塑料) + 融合二分类
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
from torch_geometric.data import Batch as GeoBatch
from torch_geometric.data import Data

# ========== 你的工程内模块 ==========
from model.gcn_model import DeepFRIModel  # 酶图编码器

# 仅用于最终整体图（避免每轮输出）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn（AUC/PR/混淆矩阵/阈值扫描）
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        confusion_matrix, ConfusionMatrixDisplay,
        average_precision_score, f1_score, matthews_corrcoef
    )
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
    "dataset_dir": "/tmp/pycharm_project_27/dataset/train_graph",                         # {enzyme}.pt
    "plastic_feat_pt": "/tmp/pycharm_project_27/test/outputs/all_description_new.pt",     # PlasticFeaturizer.save_features 输出 .pt
    "matrix_csv": "/tmp/pycharm_project_27/dataset/trainset.csv",                         # 独热矩阵 CSV
    "save_dir": "/tmp/pycharm_project_27/checkpoints/binary_01_balanced_04",

    # --- 矩阵索引/对齐 ---
    "enzyme_index_col": 0,                  # 行索引列；若无索引列，设为 None 或 "none"
    "label_prefix": "can_degrade_",         # 列名前缀；若列名就是塑料名，设为 "" 或 None
    "ignore_case_dash_underscore": True,    # 对齐时忽略大小写和 -/_ 差异

    # --- 负样本采样（按酶）---
    "neg_pos_ratio_train": 1.0,             # 训练：负样本 ≈ ratio * 正样本
    "neg_pos_ratio_val": 1.0,               # 验证：建议先用 1.0（客观评估）
    "min_pos_per_enzyme": 1,                # 某酶正样本数 < 此阈值则跳过该酶
    "max_pairs_per_enzyme_train": 20000,    # 每酶最大样本数（训练）
    "max_pairs_per_enzyme_val": 50000,      # 每酶最大样本数（验证）
    "shuffle_seed": 42,

    # --- 评估 ---
    "pred_threshold": 0.5,                  # 训练日志里的固定阈值；最终会再做阈值扫描

    # --- 模型结构 ---
    "gnn_type": "gat",
    "gnn_dims": [128, 128],
    "fc_dims": [128, 64],
    "gnn_embed_dim": 128,
    "plastic_hidden": [256, 128],
    "fusion_hidden": [128, 64],
    "dropout": 0.1,                         # 建议减小以提升可学性

    # --- 训练 ---
    "batch_size": 64,
    "epochs": 60,
    "lr": 2e-3,
    "weight_decay": 1e-5,
    "lr_step": 15,
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
    # 读矩阵
    if enzyme_index_col is None:
        mat = pd.read_csv(matrix_csv)
        mat.index = mat.index.astype(str)
    else:
        mat = pd.read_csv(matrix_csv, index_col=enzyme_index_col)
    mat.index = mat.index.astype(str)

    # 去前缀
    def strip_prefix(col: str, prefix: str) -> str:
        if prefix:
            return col[len(prefix):] if str(col).startswith(prefix) else str(col)
        return str(col)

    col_raw = list(mat.columns)
    col_stripped = [strip_prefix(c, label_prefix) for c in col_raw]

    # 对齐列名到 features 的原名
    if ignore_case_dash_underscore:
        norm_feat = {normalize_name(k): k for k in plastic_feat_names}
        raw_to_feat = {}
        for raw, stripped in zip(col_raw, col_stripped):
            key = normalize_name(stripped)
            if key in norm_feat:
                raw_to_feat[raw] = norm_feat[key]
        if not raw_to_feat:
            raise ValueError("No plastic columns in matrix can be aligned to plastic features.")
        mat = mat[[c for c in col_raw if c in raw_to_feat]].copy()
        mat.columns = [raw_to_feat[c] for c in mat.columns]
    else:
        cols = [c for c in col_stripped if c in plastic_feat_names]
        rename_map = {raw: stripped for raw, stripped in zip(col_raw, col_stripped) if stripped in cols}
        mat = mat[[raw for raw in col_raw if raw in rename_map]].copy()
        mat.columns = [rename_map[c] for c in mat.columns]

    # 展开为长表
    df_long = mat.stack().reset_index()
    df_long.columns = ["enzyme", "plastic", "label"]
    df_long["label"] = df_long["label"].astype(float).clip(0.0, 1.0)

    return df_long, list(mat.index), list(mat.columns)


def sample_pairs_per_enzyme(
    df_long: pd.DataFrame,
    neg_pos_ratio: float,
    min_pos_per_enzyme: int,
    max_pairs_per_enzyme: int,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(shuffle_seed)
    result = []
    for enzyme, sub in df_long.groupby("enzyme"):
        pos = sub[sub["label"] >= 0.5]
        neg = sub[sub["label"] < 0.5]
        if len(pos) < min_pos_per_enzyme:
            continue
        if neg_pos_ratio is not None and neg_pos_ratio > 0:
            k = int(round(neg_pos_ratio * len(pos)))
            if k < len(neg):
                neg = neg.sample(n=k, random_state=int(rng.integers(0, 1 << 31)))
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
# Model
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
# Balanced Batch Sampler（保证批内正负平衡）
# ------------------------------
class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    以 1:1 采样（默认）返回一个“样本索引序列”，供 DataLoader 使用。
    注意：为了稳定性，batch_size 建议为偶数。
    """
    def __init__(self, pos_idx: List[int], neg_idx: List[int], batch_size: int, pos_ratio: float = 0.5, seed: int = SEED):
        assert 0.0 < pos_ratio < 1.0
        self.pos_idx = list(pos_idx)
        self.neg_idx = list(neg_idx)
        self.bs = int(batch_size)
        self.pos_per_batch = max(int(round(self.bs * pos_ratio)), 1)
        self.neg_per_batch = self.bs - self.pos_per_batch
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        pos = self.pos_idx.copy()
        neg = self.neg_idx.copy()
        self.rng.shuffle(pos)
        self.rng.shuffle(neg)
        pi = ni = 0
        batch = []
        # 以“批”为单位产出索引
        while pi < len(pos) and ni < len(neg):
            cur = []
            for _ in range(self.pos_per_batch):
                if pi >= len(pos): break
                cur.append(pos[pi]); pi += 1
            for _ in range(self.neg_per_batch):
                if ni >= len(neg): break
                cur.append(neg[ni]); ni += 1
            if len(cur) == 0:
                break
            self.rng.shuffle(cur)
            batch.extend(cur)
        return iter(batch)

    def __len__(self):
        # 返回迭代器中“样本索引”的总数（近似）
        return min(len(self.pos_idx), len(self.neg_idx)) // max(self.pos_per_batch, 1) * self.bs


# ------------------------------
# Train / Eval（BCE + pos_weight）
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for batch_graph, plastics, labels in loader:
        batch_graph = batch_graph.to(device)
        plastics = plastics.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_graph, plastics)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, pred_threshold=0.5, criterion=None):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for batch_graph, plastics, labels in loader:
        batch_graph = batch_graph.to(device)
        plastics = plastics.to(device)
        labels = labels.to(device)
        logits = model(batch_graph, plastics)
        if criterion is not None:
            loss = criterion(logits, labels)
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if len(all_logits) == 0:
        return (0.0 if criterion else None), 0.0, float("nan"), np.array([]), np.array([])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    hard_labels = (all_labels.numpy() >= 0.5).astype(int)
    preds = (probs >= pred_threshold).astype(int)

    acc = float((preds == hard_labels).mean())

    if HAS_SK_METRICS and len(np.unique(hard_labels)) >= 2:
        auroc = float(roc_auc_score(hard_labels, probs))
    else:
        auroc = float("nan")

    loss_mean = (total_loss / n) if (criterion and n > 0) else None
    return loss_mean, acc, auroc, probs, hard_labels


# ------------------------------
# Group split
# ------------------------------
def group_split_by_enzyme(enzymes: List[str], train_ratio=0.8, seed=42):
    idx = np.arange(len(enzymes))
    groups = np.array(enzymes, dtype=object)
    if HAS_SK_SPLIT:
        gss = GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
        tr_idx, va_idx = next(gss.split(idx, None, groups))
        return tr_idx.tolist(), va_idx.tolist()
    # 降级：随机切分
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_train = int(len(uniq) * train_ratio)
    train_set = set(uniq[:n_train])
    tr_idx = [i for i, e in enumerate(enzymes) if e in train_set]
    va_idx = [i for i, e in enumerate(enzymes) if e not in train_set]
    return tr_idx, va_idx


# ------------------------------
# Plot helpers (最终一次性输出)
# ------------------------------
def plot_curves(save_dir: str, xs, tr_loss, va_loss, va_acc, va_auc):
    os.makedirs(save_dir, exist_ok=True)

    # Loss
    if all(v is not None for v in va_loss):
        plt.figure(figsize=(6,4))
        plt.plot(xs, tr_loss, label="train loss")
        plt.plot(xs, va_loss, label="val loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curve.png")); plt.close()

    # Acc
    plt.figure(figsize=(6,4))
    plt.plot(xs, va_acc, label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Validation Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png")); plt.close()

    # AUC（若有）
    if np.isfinite(np.nanmean([x for x in va_auc if x == x])):
        plt.figure(figsize=(6,4))
        plt.plot(xs, va_auc, label="val AUROC")
        plt.xlabel("epoch"); plt.ylabel("AUROC"); plt.title("Validation AUROC"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "auc_curve.png")); plt.close()

def plot_final_roc_pr_cm(save_dir: str, y_true: np.ndarray, y_score: np.ndarray, thr: float):
    os.makedirs(save_dir, exist_ok=True)
    if HAS_SK_METRICS and y_true.size > 0 and len(np.unique(y_true)) >= 2:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y_true, y_score):.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (val)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_roc.png")); plt.close()
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(5,5))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (val)"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_pr.png")); plt.close()
        # Confusion Matrix (硬阈值)
        preds = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["0","1"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (val)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cm_val.png")); plt.close()


# ------------------------------
# Main
# ------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 载入塑料特征
    plastic_feats, plastic_dim, _ = load_plastic_features(args.plastic_feat_pt)
    plastic_feat_names = list(plastic_feats.keys())

    # 从独热矩阵构建全量配对
    enzyme_index_col = None if (args.enzyme_index_col is None or str(args.enzyme_index_col).lower()=="none") else int(args.enzyme_index_col)
    df_long_all, enzyme_list, _ = build_pairs_from_matrix(
        matrix_csv=args.matrix_csv,
        enzyme_index_col=enzyme_index_col,
        label_prefix=(args.label_prefix or ""),
        plastic_feat_names=plastic_feat_names,
        ignore_case_dash_underscore=bool(args.ignore_case_dash_underscore)
    )

    # 按酶分组切分
    tr_eidx, va_eidx = group_split_by_enzyme(enzyme_list, train_ratio=args.train_ratio, seed=SEED)
    train_enz_set = set([enzyme_list[i] for i in tr_eidx])
    val_enz_set   = set([enzyme_list[i] for i in va_eidx])

    df_train_all = df_long_all[df_long_all["enzyme"].isin(train_enz_set)].reset_index(drop=True)
    df_val_all   = df_long_all[df_long_all["enzyme"].isin(val_enz_set)].reset_index(drop=True)

    # 负样本随机采样（按酶）
    df_train = sample_pairs_per_enzyme(
        df_train_all,
        neg_pos_ratio=float(args.neg_pos_ratio_train),
        min_pos_per_enzyme=int(args.min_pos_per_enzyme),
        max_pairs_per_enzyme=int(args.max_pairs_per_enzyme_train),
        shuffle_seed=int(args.shuffle_seed),
    )
    df_val = sample_pairs_per_enzyme(
        df_val_all,
        neg_pos_ratio=float(args.neg_pos_ratio_val),
        min_pos_per_enzyme=int(args.min_pos_per_enzyme),
        max_pairs_per_enzyme=int(args.max_pairs_per_enzyme_val),
        shuffle_seed=int(args.shuffle_seed) + 1,
    )

    print(f"[INFO] Train pairs: {len(df_train)} | Val pairs: {len(df_val)}")

    # Dataset / Loader
    train_ds = PairDataset(df_train, graphs_dir=args.dataset_dir, plastic_feat_dict=plastic_feats)
    val_ds   = PairDataset(df_val,   graphs_dir=args.dataset_dir, plastic_feat_dict=plastic_feats)

    # === 统计训练集正负，构造 pos_weight ===
    num_pos = int((train_ds.df["label"] >= 0.5).sum())
    num_neg = int((train_ds.df["label"] <  0.5).sum())
    pos_weight_value = max(num_neg / max(num_pos, 1), 1.0)
    print(f"[INFO] pos_weight for BCEWithLogitsLoss = {pos_weight_value:.3f}")
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))

    # === Balanced batch sampler：保证每批包含固定比例的正样本（默认1:1）===
    pin_memory_flag = torch.cuda.is_available() and bool(args.pin_memory)
    train_idx_pos = np.flatnonzero((train_ds.df["label"].values >= 0.5)).tolist()
    train_idx_neg = np.flatnonzero((train_ds.df["label"].values <  0.5)).tolist()
    sampler = BalancedBatchSampler(train_idx_pos, train_idx_neg, args.batch_size, pos_ratio=0.5, seed=SEED)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag,
        drop_last=False
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

    # === 初始化最后一层 bias 为先验对数几率，避免“强负”起步 ===
    with torch.no_grad():
        last_linear = None
        for m in reversed(model.fusion):
            if isinstance(m, nn.Linear):
                last_linear = m; break
        if last_linear is not None and last_linear.bias is not None:
            pi = max(min(num_pos / max(num_pos + num_neg, 1), 0.99), 0.01)
            init_bias = float(np.log(pi / (1.0 - pi)))
            last_linear.bias.fill_(init_bias)
            print(f"[INFO] init final bias to logit(prior)={init_bias:.3f} (π={pi:.3f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # 记录曲线
    epochs = list(range(1, args.epochs + 1))
    tr_losses, va_losses, va_accs, va_aucs = [], [], [], []

    best_val_key = -1e9
    best_state = None
    for epoch in epochs:
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, bce_criterion)
        va_loss, va_acc, va_auc, _, _ = evaluate(
            model, val_loader, device, pred_threshold=args.pred_threshold, criterion=bce_criterion
        )
        tr_losses.append(tr_loss)
        va_losses.append(va_loss if va_loss is not None else np.nan)
        va_accs.append(va_acc)
        va_aucs.append(va_auc)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={(va_loss if va_loss is not None else float('nan')):.4f} | "
              f"val_acc={va_acc:.4f} | val_auc={va_auc:.4f} | lr={lr:.6f}")

        # 以 AUROC 为主，辅以 acc 作为 tie-break
        key = (0.0 if np.isnan(va_auc) else va_auc) * 1000.0 + va_acc
        if key > best_val_key:
            best_val_key = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        scheduler.step()

    # 保存最佳模型
    if best_state is not None:
        torch.save(best_state, os.path.join(args.save_dir, "best_model.pt"))
    print(f"\nTraining completed. Best key={best_val_key:.3f} (AUROC*1000 + ACC)")

    # 统一绘制总体曲线
    plot_curves(args.save_dir, epochs, tr_losses, va_losses, va_accs, va_aucs)

    # 用最佳模型做最终验证集评估 + ROC/PR/CM + 阈值扫描 + AUPRC
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, _, y_score, y_true = evaluate(model, val_loader, device, pred_threshold=args.pred_threshold, criterion=bce_criterion)
    plot_final_roc_pr_cm(args.save_dir, y_true, y_score, thr=args.pred_threshold)

    # 阈值扫描 + AUPRC 报告
    if HAS_SK_METRICS and y_true.size > 0 and len(np.unique(y_true)) >= 2:
        ap = average_precision_score(y_true, y_score)
        best_thr, best_f1, best_mcc = 0.5, -1.0, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (y_score >= thr).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            mcc = matthews_corrcoef(y_true, preds)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
            best_mcc = max(best_mcc, mcc)
        summary = f"[VAL] AUPRC={ap:.4f} | best_thr={best_thr:.2f} | best_F1={best_f1:.4f} | best_MCC={best_mcc:.4f}\n"
        print(summary)
        with open(os.path.join(args.save_dir, "val_threshold_sweep.txt"), "w") as f:
            f.write(summary)
    else:
        print("[VAL] AUPRC/阈值扫描：验证集类别单一或不可用，已跳过。")


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    if USE_CONFIG:
        args = SimpleNamespace(**CONFIG)
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)
    else:
        parser = argparse.ArgumentParser(description="Train GNN+Plastic binary classifier from one-hot matrix (pos_weight + balanced batch + prior bias + final summary plots).")
        # Paths
        parser.add_argument("--dataset_dir", type=str, required=True, help="Folder with enzyme .pt graphs ({enzyme}.pt)")
        parser.add_argument("--plastic_feat_pt", type=str, required=True, help="Plastic features .pt from PlasticFeaturizer.save_features")
        parser.add_argument("--matrix_csv", type=str, required=True, help="Enzyme×Plastic one-hot matrix CSV (rows=enzyme, cols=plastics)")
        parser.add_argument("--save_dir", type=str, default="checkpoints_bin_from_matrix")

        # Matrix index / prefix
        parser.add_argument("--enzyme_index_col", type=str, default="0", help="Row index column for enzymes (int or 'none')")
        parser.add_argument("--label_prefix", type=str, default="can_degrade_", help="Column prefix; '' if none")
        parser.add_argument("--ignore_case_dash_underscore", action="store_true", help="Normalize names to match features (case-insensitive and strip -/_ )")

        # Sampling ratios
        parser.add_argument("--neg_pos_ratio_train", type=float, default=1.0, help="Train: negatives per positive per-enzyme; <=0 = use all negatives")
        parser.add_argument("--neg_pos_ratio_val", type=float, default=1.0, help="Val: negatives per positive per-enzyme; <=0 = use all negatives")
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
        parser.add_argument("--dropout", type=float, default=0.1)

        # Train
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=60)
        parser.add_argument("--lr", type=float, default=2e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--lr_step", type=int, default=15)
        parser.add_argument("--lr_gamma", type=float, default=0.5)
        parser.add_argument("--train_ratio", type=float, default=0.8)

        # Loader
        parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
        parser.add_argument("--pin_memory", action="store_true")

        args = parser.parse_args()
        if isinstance(args.enzyme_index_col, str):
            if args.enzyme_index_col.lower() == "none":
                args.enzyme_index_col = None
            else:
                args.enzyme_index_col = int(args.enzyme_index_col)
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)