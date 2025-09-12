#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gnn_multilabel.py  (with in-script CONFIG / USE_CONFIG switch + Visualizations)

多标签版 DeepFRI -> 图级多标签分类（只用酶图，不用塑料嵌入）
- 输出层维度 = #classes
- 损失: BCEWithLogitsLoss（按训练集统计 per-class pos_weight 进行不均衡校正）
- 指标: mAP（macro AP）、micro/macro F1@0.5、macro AUROC（可选）

可视化（对齐二分类脚本风格 + 多标签友好补充）：
- TensorBoard 曲线：Loss/train、Val/mAP_macro、Val/micro_F1、Val/macro_F1、Val/macro_AUROC、LearningRate
- 训练集“每类正例占比”条形图：train_label_frequency.png (+ TB)
- 每个 epoch：
  - 验证集“每类 AP”条形图：val_per_class_AP_epochXX.png (+ TB)
  - 展平后的整体混淆矩阵：cm_epochXX.png (+ TB)
  - 按塑料类别的 0.5 阈值准确率：per_label_acc_epochXX.png (+ TB)
"""

# ==============================
# 配置区（可被命令行覆盖）
# ==============================
USE_CONFIG = True
CONFIG = {
    # 必填路径
    "dataset_dir": "/tmp/pycharm_project_27/dataset/train_graph",   # 目录下是 *.pt 图

    # 若图里没有 y，可从单独 CSV 读取多标签（可选）
    "label_csv":   "/tmp/pycharm_project_27/dataset/trainset.csv",  # 含 id 与 can_degrade_* 列；或设为 None
    "id_col": "protein_id",
    "label_prefix": "can_degrade_",

    # 输出
    "save_dir": "/tmp/pycharm_project_27/checkpoints/multilabel/multilabel_1",
    "log_dir":  "/tmp/pycharm_project_27/checkpoints/multilabel/logs_multilabel_1",

    # 模型
    "gnn_type": "gat",          # ["gcn","gat"]
    "gnn_dims": [128, 128],
    "fc_dims":  [128, 64],
    "dropout":  0.1,

    # 训练
    "batch_size": 32,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "lr_step": 10,
    "lr_gamma": 0.5,
    "train_ratio": 0.8,

    # DataLoader
    "num_workers": 0,
    "pin_memory": True,
}

# ==============================
# 代码主体
# ==============================
import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 你的模型
from model.gcn_model import DeepFRIModel

# 与二分类脚本同源的可视化工具
from utils.visualization import (
    log_curve,
    log_weights_histogram,
    log_confusion_matrix,      # 用于展平后的整体二值混淆矩阵
)

# 评估
try:
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    HAS_SK = True
except Exception:
    HAS_SK = False

# Matplotlib（保存 PNG & 也写入 TensorBoard）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------
# I/O & 标签装载
# --------------------------
def list_graph_files(dataset_dir: str) -> List[str]:
    return sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pt")])

def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def load_graph_safe(path: str) -> Data:
    add_safe_globals([Data])
    return torch.load(path, weights_only=False)

def load_label_map(path_json: Optional[str]) -> Optional[Dict[str, int]]:
    if not path_json:
        return None
    if not os.path.isfile(path_json):
        return None
    with open(path_json, "r") as f:
        label2id = json.load(f)
    return {str(k): int(v) for k, v in label2id.items()}

def infer_label_map_from_csv(label_csv: str, label_prefix: str) -> Dict[str, int]:
    import pandas as pd
    df = pd.read_csv(label_csv)
    cols = [c for c in df.columns if str(c).startswith(label_prefix)]
    if not cols:
        raise ValueError(f"在 CSV 中未找到前缀 '{label_prefix}' 的列，无法自动推断 label_map。")
    def strip_prefix(c): return str(c)[len(label_prefix):]
    seen, names = set(), []
    for c in cols:
        name = strip_prefix(c)
        key = name.strip().lower().replace("-", "").replace("_", "")
        if key not in seen:
            seen.add(key)
            names.append(name)
    label2id = {name: i for i, name in enumerate(names)}
    return label2id

def infer_num_classes_from_graphs(graph_paths: List[str]) -> int:
    for p in graph_paths:
        g: Data = load_graph_safe(p)
        if hasattr(g, "y") and isinstance(g.y, torch.Tensor):
            y = g.y.view(-1)
            if y.numel() >= 1:
                return int(y.numel())
    raise ValueError("无法从图中推断类别数（未找到带有 y 向量的样本）。")

def load_labels_from_csv(label_csv: str, label_map: Dict[str,int],
                         id_col: str = "protein_id", label_prefix: str = "can_degrade_") -> Dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(label_csv)
    if id_col not in df.columns:
        raise ValueError(f"label_csv 缺少 id 列: {id_col}")
    def norm(x): return str(x).strip().lower().replace("-", "").replace("_", "")
    norm_cols = {norm(c): c for c in df.columns}
    class_cols = []
    for lab, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        full = lab if lab.startswith(label_prefix) else f"{label_prefix}{lab}"
        cand = norm(full)
        class_cols.append(norm_cols.get(cand, None))

    y_dict = {}
    for _, row in df.iterrows():
        sid = str(row[id_col])
        y = np.zeros(len(label_map), dtype=np.float32)
        for lab, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
            col = class_cols[idx]
            v = float(row[col]) if (col is not None and not pd.isna(row[col])) else 0.0
            y[idx] = 1.0 if v >= 0.5 else 0.0
        y_dict[sid] = y
    return y_dict

# --------------------------
# Dataset
# --------------------------
class GraphMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 graph_paths: List[str],
                 num_classes: int,
                 external_labels: Optional[Dict[str, np.ndarray]] = None):
        self.graph_paths = graph_paths
        self.num_classes = num_classes
        self.external = external_labels

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, i: int) -> Data:
        p = self.graph_paths[i]
        g: Data = load_graph_safe(p)
        y = None
        sid = stem(p)
        if self.external is not None and sid in self.external:
            y = torch.tensor(self.external[sid], dtype=torch.float32)
        else:
            if hasattr(g, "y") and g.y is not None and isinstance(g.y, torch.Tensor):
                y_raw = g.y.detach().cpu().view(-1)
                if y_raw.numel() == self.num_classes:
                    y = y_raw.to(torch.float32)
                elif y_raw.numel() == 1:
                    # 标量 -> one-hot
                    y = torch.zeros(self.num_classes, dtype=torch.float32)
                    idx = int(y_raw.item())
                    if 0 <= idx < self.num_classes:
                        y[idx] = 1.0
        if y is None:
            y = torch.zeros(self.num_classes, dtype=torch.float32)

        # 关键修复：存成 [1, C]，让 PyG batch 后得到 [B, C]（而不是 [B*C]）
        y = y.view(1, -1)

        g.y = y
        return g

# --------------------------
# Utils
# --------------------------
def split_paths(paths: List[str], ratio=0.8) -> Tuple[List[str], List[str]]:
    paths = paths[:]  # copy
    random.shuffle(paths)
    n = len(paths)
    k = int(round(n * ratio))
    return paths[:k], paths[k:]

def compute_pos_weight_and_stats(train_loader: DataLoader, num_classes: int, device: torch.device):
    """
    统计 per-class 正样本数并计算 pos_weight（带形状兜底，确保 batch.y -> [B, C]）
    """
    pos = torch.zeros(num_classes, dtype=torch.float64)
    total = 0
    for batch in train_loader:
        y = batch.y.to(dtype=torch.float64)  # 期望 [B, C]

        # --- 形状兜底：即便 y 意外是一维 [B*C]，也能恢复成 [B, C] ---
        if y.ndim == 1:
            # [B*C] -> [B, C]（能整除才 reshape）
            if y.numel() % num_classes == 0:
                y = y.view(-1, num_classes)
            else:
                y = y.view(1, -1)  # 避免崩
        elif y.ndim == 2 and y.shape[1] != num_classes and (y.numel() % num_classes == 0):
            y = y.view(-1, num_classes)
        # -------------------------------------------------------------------

        pos += y.sum(dim=0).cpu()
        total += y.shape[0]
    neg = total - pos
    pos_safe = torch.clamp(pos, min=1.0)
    pw = neg / pos_safe
    pw = torch.clamp(pw, min=1.0).to(dtype=torch.float32, device=device)
    return pw, pos.numpy().astype(np.int64), int(total)

def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)

def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    out = {}
    y_pred = (y_score >= thr).astype(int)
    try:
        out["micro_F1"] = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="binary", zero_division=0)
        out["macro_F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        out["micro_F1"] = np.nan
        out["macro_F1"] = np.nan

    if HAS_SK:
        aps, aucs = [], []
        for c in range(y_true.shape[1]):
            if np.sum(y_true[:, c]) > 0:
                try: aps.append(average_precision_score(y_true[:, c], y_score[:, c]))
                except Exception: pass
            if len(np.unique(y_true[:, c])) >= 2:
                try: aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
                except Exception: pass
        out["mAP_macro"] = float(np.mean(aps)) if len(aps) > 0 else float("nan")
        out["macro_AUROC"] = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")
    else:
        out["mAP_macro"] = np.nan
        out["macro_AUROC"] = np.nan
    return out

def per_class_ap(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    C = y_true.shape[1]
    ap = np.full(C, np.nan, dtype=float)
    if not HAS_SK:
        return ap
    for c in range(C):
        if np.sum(y_true[:, c]) > 0:
            try:
                ap[c] = average_precision_score(y_true[:, c], y_score[:, c])
            except Exception:
                ap[c] = np.nan
    return ap

# --------------------------
# 绘图（保存 PNG + 写入 TensorBoard）
# --------------------------
def plot_and_save_bar(values: np.ndarray, labels: List[str], title: str, ylabel: str, out_path: str,
                      ylim: Optional[Tuple[float,float]] = None) -> plt.Figure:
    fig = plt.figure(figsize=(max(8, len(labels) * 0.35), 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return fig

# --------------------------
# Train / Eval
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)                 # [B, C]
        y = batch.y.to(device=device, dtype=torch.float32)  # [B, C]
        # 兜底：万一是 [B*C]，恢复
        if y.ndim == 1 and y.numel() % logits.shape[1] == 0:
            y = y.view(-1, logits.shape[1])
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, logits_list = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)                 # [B, C]
        logits_list.append(logits.detach().cpu())
        yb = batch.y.detach().cpu()
        if yb.ndim == 1 and yb.numel() % logits.shape[1] == 0:
            yb = yb.view(-1, logits.shape[1])
        ys.append(yb.numpy())
    if len(logits_list) == 0:
        return {"micro_F1": np.nan, "macro_F1": np.nan, "mAP_macro": np.nan, "macro_AUROC": np.nan}, None, None
    logits = torch.cat(logits_list, dim=0)
    probs = sigmoid_probs(logits).numpy()
    y_true = np.concatenate(ys, axis=0)
    return multilabel_metrics(y_true, probs, thr=0.5), y_true, probs

# --------------------------
# Main
# --------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 1) 收集图清单
    all_paths = list_graph_files(args.dataset_dir)
    if not all_paths:
        raise FileNotFoundError(f"未在 {args.dataset_dir} 找到任何 *.pt 图文件。")

    # 2) label_map：读取或自动推断
    label2id = load_label_map(getattr(args, "label_map", None))
    if label2id is None:
        if getattr(args, "label_csv", None):
            label2id = infer_label_map_from_csv(args.label_csv, args.label_prefix)
            print(f"[INFO] label_map inferred from CSV: {len(label2id)} classes.")
        else:
            C = infer_num_classes_from_graphs(all_paths)
            label2id = {f"class_{i}": i for i in range(C)}
            print(f"[INFO] label_map inferred from graphs: {C} classes.")
        auto_map_path = os.path.join(args.save_dir, "label2id_auto.json")
        with open(auto_map_path, "w") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved inferred label_map to: {auto_map_path}")

    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    # 3) 可选：从 CSV 载入外部多标签
    external_labels = None
    if getattr(args, "label_csv", None):
        external_labels = load_labels_from_csv(
            args.label_csv, label2id,
            id_col=args.id_col, label_prefix=args.label_prefix
        )

    # 4) 划分/Loader
    tr_paths, va_paths = split_paths(all_paths, ratio=args.train_ratio)
    train_ds = GraphMultiLabelDataset(tr_paths, num_classes, external_labels)
    val_ds   = GraphMultiLabelDataset(va_paths,   num_classes, external_labels)

    pin_memory = torch.cuda.is_available() and bool(getattr(args, "pin_memory", False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    # 5) 模型
    model = DeepFRIModel(
        gnn_type=args.gnn_type,
        gnn_dims=args.gnn_dims,
        fc_dims=args.fc_dims,
        out_dim=num_classes,
        dropout=args.dropout,
        use_residue_level_output=False,
        in_dim=None,  # 懒构建
    ).to(device)

    # warm-up 一次（构建参数形状，便于权重统计）
    model.train()
    for batch in train_loader:
        _ = model(batch.to(device))
        break

    # 6) pos_weight + 训练集频次可视化
    pos_weight, pos_count, total_samples = compute_pos_weight_and_stats(train_loader, num_classes, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[INFO] pos_weight (first 10): {pos_weight[:min(10, num_classes)].tolist()}")

    # 训练集每类频次/占比图
    freq_out = os.path.join(args.save_dir, "train_label_frequency.png")
    class_names = [id2label[i] for i in range(num_classes)]
    frac = pos_count / max(total_samples, 1)
    fig_freq = plot_and_save_bar(frac, class_names, "Train label frequency (positive ratio per class)", "Positive ratio", freq_out, ylim=(0, 1))
    writer.add_figure("Stats/Train_Label_Positive_Ratio", fig_freq, global_step=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # 7) 训练
    best_key = -1e9
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, y_true_va, y_prob_va = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        key = (0.0 if np.isnan(metrics["mAP_macro"]) else metrics["mAP_macro"]) * 1000.0 \
              + (0.0 if np.isnan(metrics.get("micro_F1", np.nan)) else metrics["micro_F1"])
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | "
            f"mAP={metrics['mAP_macro']:.4f} | microF1={metrics.get('micro_F1', np.nan):.4f} | "
            f"macroF1={metrics.get('macro_F1', np.nan):.4f} | AUROC={metrics['macro_AUROC']:.4f} | lr={lr:.6f}"
        )

        # TensorBoard curves
        log_curve(writer, "Loss/train", [tr_loss], [epoch])
        log_curve(writer, "Val/mAP_macro", [metrics['mAP_macro']], [epoch])
        if "micro_F1" in metrics: log_curve(writer, "Val/micro_F1", [metrics['micro_F1']], [epoch])
        if "macro_F1" in metrics: log_curve(writer, "Val/macro_F1", [metrics['macro_F1']], [epoch])
        if not (np.isnan(metrics["macro_AUROC"]) or np.isinf(metrics["macro_AUROC"])):
            log_curve(writer, "Val/macro_AUROC", [metrics["macro_AUROC"]], [epoch])
        log_curve(writer, "LearningRate", [lr], [epoch])
        log_weights_histogram(writer, model, epoch)

        # ===== 可视化输出 =====
        if y_true_va is not None and y_prob_va is not None:
            # 1) 展平后的整体混淆矩阵（0/1）
            y_pred_va = (y_prob_va >= 0.5).astype(int)
            flat_true = y_true_va.reshape(-1).astype(int).tolist()
            flat_pred = y_pred_va.reshape(-1).astype(int).tolist()
            cm_path = os.path.join(args.save_dir, f"cm_epoch{epoch:02d}.png")
            log_confusion_matrix(flat_true, flat_pred, ["0", "1"], cm_path)
            # 同步到 TB
            try:
                import PIL.Image as Image
                img = Image.open(cm_path)
                writer.add_image("Val/ConfusionMatrix", np.asarray(img).transpose(2,0,1), global_step=epoch)
            except Exception:
                pass

            # 2) 按塑料类别的“阈值0.5”准确率
            per_label_acc = []
            for c in range(y_true_va.shape[1]):
                yt = y_true_va[:, c].astype(int)
                yp = y_pred_va[:, c].astype(int)
                if yt.size == 0:
                    per_label_acc.append(np.nan)
                else:
                    per_label_acc.append(float((yt == yp).mean()))
            per_label_acc = np.array(per_label_acc, dtype=float)
            acc_plot_out = os.path.join(args.save_dir, f"per_label_acc_epoch{epoch:02d}.png")
            fig_acc = plot_and_save_bar(per_label_acc, class_names, f"Per-label accuracy @0.5 (epoch {epoch:02d})", "Accuracy", acc_plot_out, ylim=(0, 1))
            writer.add_figure("Val/PerLabel_Accuracy", fig_acc, global_step=epoch)

            # 3) 每类 AP 条形图
            if HAS_SK:
                ap_c = per_class_ap(y_true_va, y_prob_va)
                ap_plot_out = os.path.join(args.save_dir, f"val_per_class_AP_epoch{epoch:02d}.png")
                fig_ap = plot_and_save_bar(ap_c, class_names, f"Per-class AP (epoch {epoch:02d})", "AP", ap_plot_out, ylim=(0, 1))
                writer.add_figure("Val/PerClass_AP", fig_ap, global_step=epoch)

        scheduler.step()

    # 8) 保存最佳
    if best_state is not None:
        out_pt = os.path.join(args.save_dir, "best_model_multilabel.pt")
        torch.save(best_state, out_pt)
        print(f"[OK] Saved best model to: {out_pt}")
    print(f"Done. Best key (mAP*1000 + microF1) = {best_key:.3f}")

# --------------------------
# CLI / Entrypoint
# --------------------------
if __name__ == "__main__":
    if USE_CONFIG:
        class _NS:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        args = _NS(CONFIG)
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)
    else:
        parser = argparse.ArgumentParser(description="Train DeepFRI graph encoder for multi-label plastic classification (no plastic embedding).")
        parser.add_argument("--dataset_dir", type=str, required=True)
        parser.add_argument("--label_map", type=str, default=None, help="path to label2id.json; optional")
        parser.add_argument("--label_csv", type=str, default=None, help="CSV with id + can_degrade_* (optional)")
        parser.add_argument("--id_col", type=str, default="protein_id")
        parser.add_argument("--label_prefix", type=str, default="can_degrade_")
        parser.add_argument("--save_dir", type=str, default="checkpoints_ml")
        parser.add_argument("--log_dir", type=str, default="logs_ml")
        parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"])
        parser.add_argument("--gnn_dims", nargs="+", type=int, default=[128, 128])
        parser.add_argument("--fc_dims", nargs="+", type=int, default=[128, 64])
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--lr_step", type=int, default=10)
        parser.add_argument("--lr_gamma", type=float, default=0.5)
        parser.add_argument("--train_ratio", type=float, default=0.8)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--pin_memory", action="store_true")
        args = parser.parse_args()
        os.makedirs(args.save_dir, exist_ok=True)
        main(args)