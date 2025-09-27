# -*- coding: utf-8 -*-
"""
bucket_metrics_from_preds.py

输入：
- pred_csv:   预测分数表，列 = enzyme_id + 多个 'XXX.sdf'
- gt_csv:     真实独热表，列 = protein_id + 多个 'XXX'
- bucket_csv: 难度映射表，两列 protein_id,bucket_by_nn_all

输出：
- 返回一个 DataFrame（列：你指定的12项指标 + bucket），并把同样内容写到 out_csv
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def _align_columns(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                   pred_id_col: str = "enzyme_id", gt_id_col: str = "protein_id") -> (pd.DataFrame, pd.DataFrame, List[str]):
    """对齐ID与标签列：去掉预测列名中的 .sdf，与真实列求交集，返回重排后的两个表和最终标签列表"""
    # 复制防止修改原数据
    pred = pred_df.copy()
    gt   = gt_df.copy()

    # 预测列去掉 .sdf 后缀
    def strip_sdf(c: str) -> str:
        return c[:-4] if c.lower().endswith(".sdf") else c

    pred_label_map = {c: strip_sdf(c) for c in pred.columns if c != pred_id_col}
    pred_renamed = pred.rename(columns=pred_label_map)

    pred_labels = set(pred_renamed.columns) - {pred_id_col}
    gt_labels   = set(gt.columns) - {gt_id_col}

    final_labels = sorted(list(pred_labels & gt_labels))
    if not final_labels:
        raise ValueError("预测表(.sdf去后)与真实表没有共同的标签列，请检查列名。")

    # 取ID交集
    common_ids = sorted(set(pred_renamed[pred_id_col]) & set(gt[gt_id_col]))
    pred_renamed = pred_renamed[pred_renamed[pred_id_col].isin(common_ids)].copy()
    gt = gt[gt[gt_id_col].isin(common_ids)].copy()

    # 统一 ID 列名为 protein_id 方便后续 merge
    pred_renamed = pred_renamed.rename(columns={pred_id_col: "protein_id"})
    gt = gt.rename(columns={gt_id_col: "protein_id"})

    # 只保留所需列 & 排序
    pred_final = pred_renamed[["protein_id"] + final_labels].reset_index(drop=True)
    gt_final   = gt[["protein_id"] + final_labels].reset_index(drop=True)

    # 缺失补0
    pred_final[final_labels] = pred_final[final_labels].fillna(0.0)
    gt_final[final_labels]   = gt_final[final_labels].fillna(0.0)

    return pred_final, gt_final, final_labels

def _ranking_metrics_per_row(scores: np.ndarray, labels: np.ndarray, ks=(1,3,5)) -> Dict[str, float]:
    """
    对单个样本：分数排序计算 Hit@k / Recall@k
    - scores: [L], labels: [L] in {0,1}
    """
    order = np.argsort(-scores)
    out = {}
    pos_sum = float(labels.sum())
    for k in ks:
        k = min(k, len(scores))
        topk = order[:k]
        hit = float((labels[topk] > 0.5).any())
        rec = float(labels[topk].sum() / max(pos_sum, 1.0))
        out[f"hit@{k}"] = hit
        out[f"recall@{k}"] = rec
    return out

def _aggregate_ranking(all_rows: List[Dict[str, float]], ks=(1,3,5)) -> Dict[str, float]:
    out = {}
    if not all_rows:
        for k in ks:
            out[f"hit@{k}"] = np.nan
            out[f"recall@{k}"] = np.nan
        return out
    for k in ks:
        out[f"hit@{k}"] = float(np.mean([r[f"hit@{k}"] for r in all_rows]))
        out[f"recall@{k}"] = float(np.mean([r[f"recall@{k}"] for r in all_rows]))
    return out

def _closed_metrics(y_true_bin: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    使用阈值 0 二值化，给出 micro/macro PRF。
    - y_true_bin: [N,L] in {0,1}
    - y_score:    [N,L] 实数分数
    """
    y_pred_bin = (y_score > 0).astype(int)

    # micro
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true_bin.ravel(), y_pred_bin.ravel(), average="micro", zero_division=0
    )
    # macro（对每标签平均；sklearn 的宏平均）
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    return {
        "micro_precision": float(micro_p),
        "micro_recall":    float(micro_r),
        "micro_f1":        float(micro_f),
        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "macro_f1":        float(macro_f),
    }

def evaluate_by_bucket(
    pred_csv: str,
    gt_csv: str,
    bucket_csv: str,
    out_csv: Optional[str] = None,
    bucket_col: str = "bucket_by_nn_all",
) -> pd.DataFrame:
    pred_raw = pd.read_csv(pred_csv)
    gt_raw   = pd.read_csv(gt_csv)
    bucket_df = pd.read_csv(bucket_csv)[["protein_id", bucket_col]]

    # 对齐列（统一成 protein_id + labels）
    pred_df, gt_df, labels = _align_columns(
        pred_raw, gt_raw, pred_id_col="enzyme_id", gt_id_col="protein_id"
    )

    # 合并，并保留预测/真实的后缀
    df = pred_df.merge(gt_df, on="protein_id", suffixes=("_pred", "_true"))
    df = df.merge(bucket_df, on="protein_id", how="left")
    df[bucket_col] = df[bucket_col].fillna("Unknown")

    # -------- 关键修复：明确带后缀的列名 --------
    score_cols = [f"{l}_pred" for l in labels]
    true_cols  = [f"{l}_true" for l in labels]

    # 保险检查：如果有缺列，给出更友好的报错
    missing_score = [c for c in score_cols if c not in df.columns]
    missing_true  = [c for c in true_cols  if c not in df.columns]
    if missing_score or missing_true:
        raise KeyError(
            f"列缺失：score({missing_score}) / true({missing_true}).\n"
            f"可用列示例：{df.columns[:20].tolist()} ..."
        )

    # 取矩阵
    y_score = df[score_cols].to_numpy(dtype=float)
    y_true  = df[true_cols].to_numpy(dtype=float)
    y_true_bin = (y_true > 0.5).astype(int)

    # 分桶计算
    results = []
    for bucket in ["Easy", "Medium", "Hard"]:
        mask = (df[bucket_col] == bucket).to_numpy()
        if not mask.any():
            continue
        y_score_b = y_score[mask]
        y_true_b  = y_true_bin[mask]

        rows_rank = [
            _ranking_metrics_per_row(s, t, ks=(1,3,5))
            for s, t in zip(y_score_b, y_true_b)
        ]
        rank = _aggregate_ranking(rows_rank, ks=(1,3,5))
        closed = _closed_metrics(y_true_b, y_score_b)

        results.append({**rank, **closed, "bucket": bucket})

    # overall
    rows_rank = [
        _ranking_metrics_per_row(s, t, ks=(1,3,5))
        for s, t in zip(y_score, y_true_bin)
    ]
    rank_all = _aggregate_ranking(rows_rank, ks=(1,3,5))
    closed_all = _closed_metrics(y_true_bin, y_score)
    results.append({**rank_all, **closed_all, "bucket": "all"})

    out_df = pd.DataFrame(results, columns=[
        "hit@1","recall@1","hit@3","recall@3","hit@5","recall@5",
        "micro_precision","micro_recall","micro_f1",
        "macro_precision","macro_recall","macro_f1","bucket"
    ])
    if out_csv:
        out_df.to_csv(out_csv, index=False)
    return out_df


# ===================== 示例调用 =====================
if __name__ == "__main__":
    # 这里替换成你的实际路径
    pred_csv = "/Users/shulei/PycharmProjects/Plaszyme/train_script/listwise_cos_gvp/test_score_matrix.csv"     # 预测分数表（含 enzyme_id 与 *.sdf 列）
    gt_csv   = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_testset.csv"   # 真实独热表（protein_id 与不带 .sdf 的列）
    bucket_csv = "/Users/shulei/PycharmProjects/Plaszyme/dataset/testset.csv"             # 难度表（protein_id,bucket_by_nn_all）
    out_csv = "/Users/shulei/PycharmProjects/Plaszyme/train_script/listwise_cos_gvp/listwise_cos_gvp.csv"

    df_metrics = evaluate_by_bucket(
        pred_csv=pred_csv,
        gt_csv=gt_csv,
        bucket_csv=bucket_csv,
        out_csv=out_csv,
        bucket_col="bucket_by_nn_all",
    )
    print(df_metrics)