#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_models_from_matrices.py  (with random baselines)

标准评估：输入为
  - 预测矩阵 CSV：行=enzyme，列=plastic，值=模型分数/概率（越大越像正）
  - 真实矩阵：
      A) 推荐：单一测试集 CSV（含列：<enzyme_id_col>、<bucket_col>、以及若干 can_degrade_* 列）
      B) 兼容：label_csv（矩阵 1/0） + test_meta_csv（含 <enzyme_id_col> / <bucket_col>）
功能：
  1) 自动对齐（酶与塑料的交集）；对塑料名忽略大小写与 -/_，可去掉前缀对齐
  2) 排序指标（宏平均/按酶）：mAP、Recall@{1,3,5,10}、Hit@{1,3,5}、NDCG@{5,10}、R-Precision
  3) 全局曲线指标：AUROC、AUPRC（0 当 Unknown-负，保守下界）
  4) 阈值评估：precision/recall/F1/F2/coverage/utility(λ)，可额外报告“测试集 F2 最优阈值”（偏乐观，仅参考）
  5) 分层评估：Overall + bucket_by_nn_all 各难度
  6) 可视化：按难度分层的汇总条形图与逐酶 AP 分布箱线图
  7) 随机基线（random baseline）：对每个桶宏平均给出 AP / Recall@k / Hit@k / NDCG@k / R-Precision / AUROC / AUPRC 的理论期望，
     并在图中以空心点叠加显示

依赖：numpy, pandas, scikit-learn, matplotlib（seaborn 可选）
"""

# ==============================
# 配置区（可被命令行覆盖）
# ==============================
USE_CONFIG = True
CONFIG = {
    # ---- 基本路径 ----
    "pred_csv":        "/tmp/pycharm_project_27/checkpoints/binary_01_balanced_04/preds.csv",
    # 入口 A：单一测试集 CSV（推荐）
    "testset_csv":     "/tmp/pycharm_project_27/dataset/testset.csv",
    "label_prefix":    "can_degrade_",  # 测试集里真实标签列的前缀
    # 入口 B：兼容老接口（若提供 label_csv 则优先生效）
    "label_csv":       None,            # "/path/to/labels.csv"
    "test_meta_csv":   None,            # "/path/to/test_meta.csv"
    "out_dir":         "/tmp/pycharm_project_27/checkpoints/eval_outputs/binary_01_balanced_04",

    # ---- 索引/对齐 ----
    "enzyme_index_col": 0,          # 预测矩阵的“行索引列”（int 或 None/"none"）
    "normalize_names": True,        # 对塑料列名忽略大小写与 -/_ 差异
    "strip_label_prefix_for_align": True,  # 对真实列去掉 label_prefix 后参与对齐
    "enzyme_id_col":   "protein_id",   # 测试集 CSV 中酶 ID 列名
    "bucket_col":      "bucket_by_nn_all",  # 测试集 CSV 中分层列名

    # ---- 阈值与 Utility ----
    "thresholds":      [0.9, 0.8, 0.7, 0.5, 0.3],
    "report_f2_best":  True,        # 额外在测试集上报告 F2 最优阈值（偏乐观，仅参考）
    "utility_lambdas": [0.1, 0.3, 0.5],

    # ---- 排序指标参数 ----
    "ks_recall":       [1, 3, 5, 10],
    "ks_ndcg":         [5, 10],

    # ---- 绘图开关 ----
    "plot_enable":     True,
}

# ==============================
# 代码主体
# ==============================
import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from math import comb

from sklearn.metrics import (
    average_precision_score, roc_auc_score
)

# 可选 seaborn
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 名称规范化 ----------
def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace("-", "").replace("_", "")

def strip_prefix(col: str, prefix: str) -> str:
    if prefix and str(col).startswith(prefix):
        return str(col)[len(prefix):]
    return str(col)

# ---------- 从单一测试集 CSV 构造 label_df 与 meta ----------
def build_label_and_meta_from_testcsv(
    test_df: pd.DataFrame,
    enzyme_id_col: str,
    bucket_col: str,
    label_prefix: str,
    normalize_names_flag: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
      - label_df: 行=enzyme（enzyme_id_col），列=塑料名（去前缀/规范化对齐后统一的原始名），值∈{0,1}
      - meta_df:  两列 [enzyme_id_col, bucket_col]
    """
    if enzyme_id_col not in test_df.columns:
        raise ValueError(f"testset_csv 缺少酶 ID 列：{enzyme_id_col}")
    if bucket_col not in test_df.columns:
        raise ValueError(f"testset_csv 缺少分层列：{bucket_col}")

    # 提取标签列
    label_cols = [c for c in test_df.columns if str(c).startswith(label_prefix)]
    if not label_cols:
        raise ValueError(f"testset_csv 中未发现前缀为 '{label_prefix}' 的标签列。")

    # 生成塑料名（去前缀）
    raw_plastics = [strip_prefix(c, label_prefix) for c in label_cols]
    # 规范化去重
    if normalize_names_flag:
        seen = set()
        keep_idx = []
        for i, p in enumerate(raw_plastics):
            key = normalize_name(p)
            if key not in seen:
                seen.add(key)
                keep_idx.append(i)
        label_cols = [label_cols[i] for i in keep_idx]
        raw_plastics = [raw_plastics[i] for i in keep_idx]

    # label_df
    label_df = test_df[[enzyme_id_col] + label_cols].copy()
    label_df[enzyme_id_col] = label_df[enzyme_id_col].astype(str)
    label_df = label_df.set_index(enzyme_id_col)
    label_df.columns = raw_plastics
    label_df = label_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)

    # meta_df
    meta_df = test_df[[enzyme_id_col, bucket_col]].copy()
    meta_df[enzyme_id_col] = meta_df[enzyme_id_col].astype(str)

    return label_df, meta_df

# ---------- 对齐预测/真实 ----------
def align_matrices(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    enzyme_index_col: Optional[int],
    normalize_names_flag: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """对齐预测/真实矩阵（行=enzyme，列=plastic），只保留交集；塑料列可做名称规范化对齐。"""
    # 设置行索引（仅对 pred_df）
    if enzyme_index_col is not None:
        if isinstance(enzyme_index_col, int):
            pred_df = pred_df.set_index(pred_df.columns[enzyme_index_col])

    # 若没索引名，兜底：认为第一列是 enzyme
    if pred_df.index.name is None:
        pred_df.index = pred_df.iloc[:, 0].astype(str)
        pred_df = pred_df.iloc[:, 1:]
    pred_df.index = pred_df.index.astype(str)

    # 列名（塑料）对齐
    if normalize_names_flag:
        pmap = {normalize_name(c): c for c in pred_df.columns}
        lmap = {normalize_name(c): c for c in label_df.columns}
        common_norm = sorted(set(pmap) & set(lmap))
        if not common_norm:
            raise ValueError("列名对齐失败：规范化后预测与标注无塑料列交集。")
        pred_cols = [pmap[k] for k in common_norm]
        lab_cols  = [lmap[k] for k in common_norm]
        pred_df = pred_df[pred_cols].copy()
        label_df = label_df[lab_cols].copy()
        label_df.columns = pred_df.columns  # 对齐列名顺序
    else:
        common_cols = sorted(set(pred_df.columns) & set(label_df.columns))
        if not common_cols:
            raise ValueError("列名对齐失败：预测与标注无塑料列交集。")
        pred_df = pred_df[common_cols].copy()
        label_df = label_df[common_cols].copy()

    # 行（酶）交集
    common_enzyme = sorted(set(pred_df.index) & set(label_df.index))
    if not common_enzyme:
        raise ValueError("行名对齐失败：预测与标注无酶样本交集。")
    pred_df = pred_df.loc[common_enzyme]
    label_df = label_df.loc[common_enzyme]

    # 转数值
    pred_df = pred_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    label_df = label_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return pred_df, label_df, common_enzyme, list(pred_df.columns)

# ---------- 排序指标（单酶） ----------
def per_query_ranking_metrics(scores: np.ndarray, labels: np.ndarray,
                              ks_rec=(1,3,5,10), ks_ndcg=(5,10)) -> Dict[str, float]:
    P = len(scores); assert P == len(labels)
    out = {}
    num_pos = int(labels.sum())

    # AP
    out["AP"] = float(average_precision_score(labels, scores)) if num_pos > 0 else float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Recall@k / Hit@k
    for k in ks_rec:
        k_eff = min(k, P)
        rec_k = (int(sorted_labels[:k_eff].sum()) / max(num_pos, 1)) if num_pos > 0 else np.nan
        out[f"Recall@{k}"] = rec_k
    for k in [1,3,5]:
        k_eff = min(k, P)
        hit_k = 1.0 if int(sorted_labels[:k_eff].sum()) > 0 else 0.0
        out[f"Hit@{k}"] = hit_k if num_pos > 0 else np.nan

    # NDCG@k（binary gain）
    def dcg_at_k(lbl, k):
        k_eff = min(k, len(lbl))
        gains = (2**lbl[:k_eff] - 1)  # binary -> gains = lbl
        discounts = 1.0 / np.log2(np.arange(2, 2 + k_eff))
        return float(np.sum(gains * discounts))

    ideal_sorted = np.sort(labels)[::-1]
    for k in ks_ndcg:
        if num_pos == 0:
            out[f"NDCG@{k}"] = np.nan
        else:
            ndcg = dcg_at_k(sorted_labels, k)
            idcg = dcg_at_k(ideal_sorted, k)
            out[f"NDCG@{k}"] = (ndcg / idcg) if idcg > 0 else 0.0

    # R-Precision（k=正样本数）
    if num_pos > 0:
        k = num_pos
        k_eff = min(k, P)
        rprec = float(sorted_labels[:k_eff].sum()) / max(k, 1)
    else:
        rprec = np.nan
    out["R-Precision"] = rprec

    return out

def macro_aggregate_per_query(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for d in metrics_list for k in d.keys()})
    agg = {}
    for k in keys:
        vals = [d[k] for d in metrics_list if k in d and (not np.isnan(d[k]))]
        agg[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        # 不做加权，保持“按酶”宏平均
    return agg

# ---------- 随机基线（理论期望） ----------
def _dcg_denominator(k: int) -> float:
    i = np.arange(2, 2 + k)
    return float(np.sum(1.0 / np.log2(i)))

def _idcg_at_k_binary(r: int, k: int) -> float:
    kk = min(k, r)
    i = np.arange(2, 2 + kk)
    return float(np.sum(1.0 / np.log2(i)))

def _recall_or_hit_baseline(P: int, r: int, k: int) -> float:
    if P <= 0 or r <= 0 or k <= 0:
        return 0.0
    k = min(k, P)
    if P - r < k:
        return 1.0
    # 1 - C(P-r, k)/C(P, k)
    return 1.0 - (comb(P - r, k) / comb(P, k))

def compute_random_baselines_per_enzyme(labels_row: np.ndarray,
                                        ks_rec=(1,3,5,10),
                                        ks_ndcg=(5,10)) -> dict:
    """
    单酶随机基线（期望）：AP = r/P；R-Precision = r/P；
    Recall@k / Hit@k = 1 - C(P-r,k)/C(P,k)；NDCG@k = ((r/P)*sum(1/log2(i+1)))/IDCG@k
    """
    P = int(labels_row.size)
    r = int(labels_row.sum())
    out = {}
    out["AP_base"] = (r / P) if P > 0 else np.nan
    out["R-Precision_base"] = (r / P) if P > 0 else np.nan
    for k in ks_rec:
        out[f"Recall@{k}_base"] = _recall_or_hit_baseline(P, r, k)
    for k in [1,3,5]:
        out[f"Hit@{k}_base"] = _recall_or_hit_baseline(P, r, k)
    for k in ks_ndcg:
        if r == 0 or P == 0:
            out[f"NDCG@{k}_base"] = np.nan
        else:
            edcg = (r / P) * _dcg_denominator(k)
            idcg = _idcg_at_k_binary(r, k)
            out[f"NDCG@{k}_base"] = (edcg / idcg) if idcg > 0 else 0.0
    return out

def macro_aggregate_baselines(label_df: pd.DataFrame,
                              ks_rec=(1,3,5,10),
                              ks_ndcg=(5,10)) -> dict:
    """对一个子集（如某个 bucket）的所有酶做随机基线宏平均。"""
    rows = []
    for i in range(label_df.shape[0]):
        labels = label_df.iloc[i, :].values.astype(int)
        rows.append(compute_random_baselines_per_enzyme(labels, ks_rec, ks_ndcg))
    keys = sorted({k for d in rows for k in d.keys()})
    agg = {}
    for k in keys:
        vals = [d[k] for d in rows if k in d and (not np.isnan(d[k]))]
        agg[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return agg

def global_curve_metrics(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> Dict[str, float]:
    y_true = label_df.values.reshape(-1).astype(int)
    y_score = pred_df.values.reshape(-1).astype(float)
    uniq = np.unique(y_true)
    out = {}
    if len(uniq) < 2:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
        return out
    out["AUROC"] = float(roc_auc_score(y_true, y_score))
    out["AUPRC"] = float(average_precision_score(y_true, y_score))
    return out

def global_baselines_AUROC_AUPRC(label_df: pd.DataFrame) -> dict:
    """全局（铺平）基线：AUROC=0.5；AUPRC=正样本比例（若只有一类则 NaN）。"""
    y_true = label_df.values.reshape(-1).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"AUROC_base": np.nan, "AUPRC_base": np.nan}
    pos_rate = float(np.mean(y_true))
    return {"AUROC_base": 0.5, "AUPRC_base": pos_rate}

# ---------- 阈值评估（全局） ----------
def threshold_eval_global(pred_df: pd.DataFrame, label_df: pd.DataFrame,
                          thresholds: List[float], utility_lambdas: List[float],
                          also_report_f2_best: bool) -> pd.DataFrame:
    y_true = label_df.values.reshape(-1).astype(int)
    y_score = pred_df.values.reshape(-1).astype(float)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        f2   = (5*prec*rec)/(4*prec+rec) if (prec+rec) > 0 else 0.0
        coverage = (tp + fp) / max(len(y_true), 1)
        row = {
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "F2": f2,
            "coverage": coverage,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }
        for lam in utility_lambdas:
            row[f"utility_lambda_{lam}"] = tp - lam * fp
        rows.append(row)

    df = pd.DataFrame(rows)

    if also_report_f2_best and len(y_true) > 0:
        grid = np.linspace(0.01, 0.99, 99)
        best_f2, best_thr = -1.0, 0.5
        best_row = None
        for thr in grid:
            y_pred = (y_score >= thr).astype(int)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f2   = (5*prec*rec)/(4*prec+rec) if (prec+rec) > 0 else 0.0
            if f2 > best_f2:
                best_f2, best_thr = f2, thr
                best_row = {
                    "threshold": thr,
                    "precision": prec,
                    "recall": rec,
                    "F1": (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0,
                    "F2": f2,
                    "coverage": (tp + fp) / max(len(y_true), 1),
                    "TP": tp, "FP": fp, "FN": fn, "note": "best_F2_on_test(optimistic)"
                }
                for lam in utility_lambdas:
                    best_row[f"utility_lambda_{lam}"] = tp - lam * fp
        if best_row is not None:
            df = pd.concat([df, pd.DataFrame([best_row])], ignore_index=True)

    return df

# ---------- 分桶 ----------
def build_buckets_from_meta(enzyme_ids: List[str],
                            test_meta_df: Optional[pd.DataFrame],
                            enzyme_id_col: str,
                            bucket_col: str) -> Dict[str, List[str]]:
    """返回 {bucket_name: [enzyme_id,...]}，总包含 Overall；若提供元数据则根据 bucket_col 细分。"""
    buckets = {"Overall": enzyme_ids}
    if test_meta_df is None:
        return buckets

    df = test_meta_df.copy()
    if enzyme_id_col not in df.columns or bucket_col not in df.columns:
        return buckets

    df[enzyme_id_col] = df[enzyme_id_col].astype(str)
    present = set(enzyme_ids)
    df = df[df[enzyme_id_col].isin(present)]

    df["__bucket__"] = df[bucket_col].astype(str)
    for b, sub in df.groupby("__bucket__"):
        members = sub[enzyme_id_col].tolist()
        if members:
            buckets[b] = members
    return buckets

# ---------- 绘图（叠加基线空心点） ----------
def plot_metrics_by_bucket(summary_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["AP", "Recall@5", "NDCG@10", "AUROC", "AUPRC"]
    base_map = {"AP":"AP_base", "Recall@5":"Recall@5_base", "NDCG@10":"NDCG@10_base",
                "AUROC":"AUROC_base", "AUPRC":"AUPRC_base"}
    df = summary_df[summary_df["bucket"].notna()].copy()

    df_long = df.melt(id_vars=["bucket"], value_vars=metrics, var_name="metric", value_name="value")
    plt.figure(figsize=(11, 6))
    if HAS_SNS:
        ax = sns.barplot(data=df_long, x="bucket", y="value", hue="metric", errorbar=None)
    else:
        buckets = df["bucket"].tolist()
        x = np.arange(len(buckets))
        w = 0.15
        for i, m in enumerate(metrics):
            vals = df[m].values
            plt.bar(x + (i - len(metrics)/2)*w + w/2, vals, width=w, label=m)
        plt.xticks(x, buckets)
        ax = plt.gca()

    # 叠加基线（空心圆点）
    for m in metrics:
        base_col = base_map[m]
        if base_col not in df.columns:
            continue
        for i, b in enumerate(df["bucket"]):
            yb = df.loc[df["bucket"]==b, base_col].values[0]
            ax.scatter(i, yb, s=60, facecolors='white', edgecolors='black', marker='o', zorder=5)

    plt.title("Performance by bucket (hollow dots = random baseline)")
    plt.xlabel("Bucket")
    plt.ylabel("Score")
    if HAS_SNS: plt.legend(title="Metric")
    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_by_bucket.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"[OK] Saved plot: {path}")

def plot_per_enzyme_ap_box(pred_df: pd.DataFrame, label_df: pd.DataFrame,
                           buckets: Dict[str, List[str]], out_dir: str):
    """逐酶 AP 分布（仅对有正样本的酶），按桶画箱线图。"""
    rows = []
    for bucket, members in buckets.items():
        if bucket == "Overall":
            continue
        if not members:
            continue
        sub_pred = pred_df.loc[members]
        sub_lab  = label_df.loc[members]
        for i, enz in enumerate(sub_pred.index):
            labels = sub_lab.iloc[i, :].values.astype(int)
            if labels.sum() == 0:
                continue
            scores = sub_pred.iloc[i, :].values.astype(float)
            ap = average_precision_score(labels, scores)
            rows.append({"bucket": bucket, "enzyme": enz, "AP": ap})
    if not rows:
        print("[WARN] No per-enzyme AP to plot (no positives).")
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5))
    if HAS_SNS:
        sns.boxplot(data=df, x="bucket", y="AP")
        sns.stripplot(data=df, x="bucket", y="AP", alpha=0.3, color="black")
    else:
        groups = [df[df["bucket"] == b]["AP"].values for b in df["bucket"].unique()]
        plt.boxplot(groups, labels=df["bucket"].unique())
    plt.title("Per-enzyme AP distribution by bucket")
    plt.xlabel("Bucket")
    plt.ylabel("AP")
    plt.tight_layout()
    path = os.path.join(out_dir, "ap_box_by_bucket.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"[OK] Saved plot: {path}")

# ---------- 主流程 ----------
def run_eval(args_ns):
    os.makedirs(args_ns.out_dir, exist_ok=True)

    # 读取预测
    pred_df_raw = pd.read_csv(args_ns.pred_csv)

    # 决定使用哪个真实入口
    use_separate = bool(args_ns.label_csv)
    if use_separate:
        label_df_raw = pd.read_csv(args_ns.label_csv)
        test_meta_df = pd.read_csv(args_ns.test_meta_csv) if args_ns.test_meta_csv else None
    else:
        # 单文件：从 testset_csv 构造 label_df + meta
        test_df = pd.read_csv(args_ns.testset_csv)
        label_df_raw, meta_df = build_label_and_meta_from_testcsv(
            test_df,
            enzyme_id_col=args_ns.enzyme_id_col,
            bucket_col=args_ns.bucket_col,
            label_prefix=args_ns.label_prefix,
            normalize_names_flag=bool(args_ns.normalize_names),
        )
        test_meta_df = meta_df

    # 对齐
    pred_df, label_df, enzymes, plastics = align_matrices(
        pred_df_raw, label_df_raw,
        enzyme_index_col=(None if str(args_ns.enzyme_index_col).lower()=="none" else int(args_ns.enzyme_index_col)),
        normalize_names_flag=bool(args_ns.normalize_names)
    )

    # 构建桶（总体 + 元数据分桶）
    buckets = build_buckets_from_meta(
        enzymes, test_meta_df,
        enzyme_id_col=args_ns.enzyme_id_col,
        bucket_col=args_ns.bucket_col
    )

    # ---- 排序指标 + 曲线指标（按桶）+ 随机基线 ----
    rows = []
    for bucket, members in buckets.items():
        sub_pred = pred_df.loc[members]
        sub_lab  = label_df.loc[members]

        # per-query 指标（宏平均）
        per_list = []
        for i, _row_id in enumerate(sub_pred.index):
            scores = sub_pred.iloc[i, :].values.astype(float)
            labels = sub_lab.iloc[i, :].values.astype(int)
            m = per_query_ranking_metrics(scores, labels,
                                          ks_rec=tuple(args_ns.ks_recall),
                                          ks_ndcg=tuple(args_ns.ks_ndcg))
            per_list.append(m)
        agg = macro_aggregate_per_query(per_list)

        # 全局曲线（此桶）
        curves = global_curve_metrics(sub_pred, sub_lab)

        # 随机基线（此桶，宏平均）
        base = macro_aggregate_baselines(sub_lab,
                                         ks_rec=tuple(args_ns.ks_recall),
                                         ks_ndcg=tuple(args_ns.ks_ndcg))
        base_global = global_baselines_AUROC_AUPRC(sub_lab)

        row = {"bucket": bucket}
        row.update(agg)
        row.update(curves)
        # 主要基线放进 summary
        row["AP_base"]          = base.get("AP_base", np.nan)
        row["Recall@5_base"]    = base.get("Recall@5_base", np.nan)
        row["NDCG@10_base"]     = base.get("NDCG@10_base", np.nan)
        row["R-Precision_base"] = base.get("R-Precision_base", np.nan)
        row["AUROC_base"]       = base_global["AUROC_base"]
        row["AUPRC_base"]       = base_global["AUPRC_base"]

        row["#enzymes"]   = int(sub_pred.shape[0])
        row["#plastics"]  = int(sub_pred.shape[1])
        row["pos_rate"]   = float(sub_lab.values.mean())
        rows.append(row)

        # 每个桶阈值扫
        thr_df = threshold_eval_global(
            sub_pred, sub_lab,
            thresholds=list(sorted(set(args_ns.thresholds))),
            utility_lambdas=list(sorted(set(args_ns.utility_lambdas))),
            also_report_f2_best=bool(args_ns.report_f2_best)
        )
        thr_out = os.path.join(args_ns.out_dir, f"threshold_sweep_{bucket}.csv".replace("/", "_"))
        thr_df.to_csv(thr_out, index=False)

    summary_df = pd.DataFrame(rows)
    summary_out = os.path.join(args_ns.out_dir, "summary_by_bucket.csv")
    summary_df.to_csv(summary_out, index=False)

    # Overall 的阈值扫（总表）
    overall_thr = threshold_eval_global(
        pred_df, label_df,
        thresholds=list(sorted(set(args_ns.thresholds))),
        utility_lambdas=list(sorted(set(args_ns.utility_lambdas))),
        also_report_f2_best=bool(args_ns.report_f2_best)
    )
    overall_thr_out = os.path.join(args_ns.out_dir, "threshold_sweep_overall.csv")
    overall_thr.to_csv(overall_thr_out, index=False)

    # 可视化（叠加基线空心点）
    if bool(args_ns.plot_enable):
        plot_metrics_by_bucket(summary_df, args_ns.out_dir)
        plot_per_enzyme_ap_box(pred_df, label_df, buckets, args_ns.out_dir)

    # 提示
    print(f"[OK] Saved: {summary_out}")
    print(f"[OK] Saved: {overall_thr_out}")
    print(f"[OK] Saved: per-bucket threshold_sweep_*.csv")
    if bool(args_ns.plot_enable):
        print(f"[OK] Saved plots under: {args_ns.out_dir}")
    print("\n[NOTE] AUROC/AUPRC 把 0 当 Unknown-负计算，属于保守下界；排序指标更能反映“把正样本排前”的能力。")
    if args_ns.report_f2_best:
        print("[NOTE] best_F2_on_test 仅用于参考，正式评估建议用验证集选阈值后在测试集固定评估。")

# ==============================
# 入口：CONFIG 或 CLI
# ==============================
if __name__ == "__main__":
    if USE_CONFIG:
        class _NS:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        args = _NS(CONFIG)
        run_eval(args)
    else:
        ap = argparse.ArgumentParser(description="Evaluate enzyme×plastic prediction matrix with bucketed reports, plots, and random baselines.")
        # 路径
        ap.add_argument("--pred_csv", type=str, required=True, help="预测矩阵 CSV（行=enzyme，列=plastic，值=概率/分数）")
        # 入口 A（推荐）
        ap.add_argument("--testset_csv", type=str, default=None, help="单一测试集 CSV（含真实标签与分层元信息）")
        ap.add_argument("--label_prefix", type=str, default="can_degrade_", help="测试集真实列前缀")
        # 入口 B（兼容）
        ap.add_argument("--label_csv", type=str, default=None, help="真实矩阵 CSV（1/0）——若提供则优先生效")
        ap.add_argument("--test_meta_csv", type=str, default=None, help="测试集元信息（包含 enzyme_id 与 bucket）")
        ap.add_argument("--out_dir", type=str, default="eval_outputs", help="结果输出目录")
        # 对齐
        ap.add_argument("--enzyme_index_col", type=str, default="0", help="预测矩阵的酶行索引列（int 或 'none'）")
        ap.add_argument("--normalize_names", action="store_true", help="塑料列名忽略大小写与 -/_ 差异对齐")
        ap.add_argument("--strip_label_prefix_for_align", action="store_true", help="对真实列去 label_prefix 后再参与对齐（单测集入口会自动去）")
        # 元信息列名
        ap.add_argument("--enzyme_id_col", type=str, default="nn_all_identity", help="酶 ID 列名")
        ap.add_argument("--bucket_col", type=str, default="bucket_by_nn_all", help="分层列名")
        # 阈值与 Utility
        ap.add_argument("--thresholds", type=float, nargs="+", default=[0.9,0.8,0.7,0.5,0.3], help="评估阈值集合")
        ap.add_argument("--report_f2_best", action="store_true", help="是否额外报告 F2 最优阈值（偏乐观，仅参考）")
        ap.add_argument("--utility_lambdas", type=float, nargs="+", default=[0.1,0.3,0.5], help="Utility(λ)=TP-λ·FP 的 λ 列表")
        # 排序指标参数
        ap.add_argument("--ks_recall", type=int, nargs="+", default=[1,3,5,10], help="Recall@k 的 k 列表")
        ap.add_argument("--ks_ndcg", type=int, nargs="+", default=[5,10], help="NDCG@k 的 k 列表")
        # 绘图
        ap.add_argument("--plot_enable", action="store_true", help="输出按难度分层的统计图")

        args = ap.parse_args()

        # 处理 enzyme_index_col
        if isinstance(args.enzyme_index_col, str):
            if args.enzyme_index_col.lower() == "none":
                args.enzyme_index_col = None
            else:
                args.enzyme_index_col = int(args.enzyme_index_col)

        run_eval(args)