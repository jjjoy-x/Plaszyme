#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将塑料投到训练好的嵌入空间并基于塑料×塑料相似度为“蛋白×塑料”主表的未知项生成软标签。
- 仅使用已训练 Siamese/对比学习模型的 encoder 部分，将塑料特征映射为 z_p 并做 L2 归一化。
- 计算所有塑料之间的余弦相似度，并映射到 [0,1]：  s_hat = (cos + 1) / 2
- 对于每个酶 e（行），用其正样本集合 P_e 对每个未知塑料 c 聚合得到 y_{e,c}：
    • max：          y_{e,c} = max_{p∈P_e} s_hat(c,p)
    • noisy_or：     p_i = sigmoid((s_hat(c,p_i) - theta) / tau),  y_{e,c} = 1 - ∏_i (1 - p_i)
    • softmin：      d_i = 1 - s_hat(c,p_i), d_soft = softmin_tau(d_i), y_{e,c} = exp(-d_soft)
- 仅填充主表中原本为 0 的位置；原本为 1 的位置保持为 1。
- 列名输出为塑料名称（去掉前缀，如 can_degrade_），并忽略大小写与 -/_ 的差异进行对齐。

使用示例：
python label_from_plastic_sim.py \
  --plastic_features_csv /path/to/plastic_features.csv \
  --protein_label_csv /path/to/protein_x_plastic.csv \
  --model_path run/run_from_sdf_7/siamese_model.pt \
  --outdir run/labels_plastic_sim_v1 \
  --label_prefix can_degrade_ \
  --agg noisy_or \
  --theta 0.5 \
  --tau 0.07
"""

import os
import argparse
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===================== 用户配置（可被命令行覆盖） =====================
DEFAULT_CONFIG = dict(
    # 路径
    plastic_features_csv="/tmp/pycharm_project_27/test/outputs/all_description_new.csv",                # 塑料原始特征：行=塑料，列=数值特征
    protein_label_csv="/tmp/pycharm_project_27/dataset/trainset.csv",          # 蛋白×塑料主表：行=蛋白ID，列=can_degrade_*
    model_path="/tmp/pycharm_project_27/notebook/outputs/run10/siamese_model.pt",   # 仅用 encoder
    outdir="/tmp/pycharm_project_27/dataset_pre/run2",

    # 索引列（若 CSV 无显式索引列，传 none）
    plastic_index_col=0,
    protein_index_col=0,

    # 主表列名前缀（如 can_degrade_）
    label_prefix="can_degrade_",

    # 聚合器：max / noisy_or / softmin
    agg="max",

    # noisy_or 的中心与温度；softmin 的温度
    theta=0.5,     # 相似度中心（基于 s_hat ∈ [0,1]）
    tau=0.07,      # 温度（平滑程度）

    # 数值与随机性
    clip_01=True,  # 是否将输出裁剪到 [0,1]
    seed=42,
)


# ===================== 模型定义（需与训练一致，仅用 encoder） =====================
class SiameseRegressor(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward_once(self, x):
        return self.encoder(x)
    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# ===================== 名称规范化工具（忽略大小写与 -/_） =====================
def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")

def strip_prefix(col: str, prefix: str) -> str:
    return col[len(prefix):] if prefix and str(col).startswith(prefix) else str(col)


# ===================== 聚合器 =====================
def agg_max(s_hat_vec: np.ndarray) -> float:
    # s_hat_vec: (P_pos,), ∈[0,1]
    return float(np.max(s_hat_vec)) if s_hat_vec.size > 0 else 0.0

def agg_noisy_or(s_hat_vec: np.ndarray, theta: float, tau: float) -> float:
    # p_i = sigmoid((s_hat - theta) / tau); y = 1 - prod(1 - p_i)
    if s_hat_vec.size == 0:
        return 0.0
    z = (s_hat_vec - theta) / max(tau, 1e-12)
    p = 1.0 / (1.0 + np.exp(-z))
    # 稳定计算 1 - ∏(1 - p_i)
    log1m_p = np.log(np.clip(1.0 - p, 1e-12, 1.0))
    log_prod = log1m_p.sum()
    y = 1.0 - float(np.exp(log_prod))
    return max(0.0, min(1.0, y))

def softmin(values: np.ndarray, tau: float) -> float:
    # softmin_tau(d_i) = -tau * log ∑ exp(-d_i / tau)
    # 等价于：tau * logsumexp(-d_i / tau) 的负号
    if values.size == 0:
        return float("inf")
    t = max(tau, 1e-12)
    x = -values / t
    x = x - np.max(x)  # 数值稳定
    lse = np.log(np.sum(np.exp(x))) + np.max(-values / t)
    return -t * lse

def agg_softmin(s_hat_vec: np.ndarray, tau: float) -> float:
    # d_i = 1 - s_hat_i ; y = exp(- softmin_tau(d))
    if s_hat_vec.size == 0:
        return 0.0
    d = 1.0 - np.clip(s_hat_vec, 0.0, 1.0)
    d_soft = softmin(d, tau=max(tau, 1e-12))
    y = float(np.exp(-d_soft))
    # 由于 softmin 的定义，这里 y ∈ (0,1]，数值上接近 max(s_hat) 但更平滑
    return max(0.0, min(1.0, y))


# ===================== CLI =====================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于塑料×塑料相似度为蛋白×塑料主表生成软标签")
    p.add_argument("--plastic_features_csv", type=str, default=DEFAULT_CONFIG["plastic_features_csv"])
    p.add_argument("--protein_label_csv", type=str, default=DEFAULT_CONFIG["protein_label_csv"])
    p.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"])
    p.add_argument("--outdir", type=str, default=DEFAULT_CONFIG["outdir"])
    p.add_argument("--plastic_index_col", type=lambda x: None if x.lower()=="none" else int(x),
                   default=DEFAULT_CONFIG["plastic_index_col"])
    p.add_argument("--protein_index_col", type=lambda x: None if x.lower()=="none" else int(x),
                   default=DEFAULT_CONFIG["protein_index_col"])
    p.add_argument("--label_prefix", type=str, default=DEFAULT_CONFIG["label_prefix"])
    p.add_argument("--agg", type=str, choices=["max", "noisy_or", "softmin"], default=DEFAULT_CONFIG["agg"])
    p.add_argument("--theta", type=float, default=DEFAULT_CONFIG["theta"])
    p.add_argument("--tau", type=float, default=DEFAULT_CONFIG["tau"])
    p.add_argument("--clip_01", type=lambda x: str(x).lower() in {"1","true","yes","y"},
                   default=DEFAULT_CONFIG["clip_01"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return p


# ===================== 主流程 =====================
def main():
    args = build_argparser().parse_args()

    # 随机性
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 载入塑料特征 ----------
    if args.plastic_index_col is None:
        plast_df = pd.read_csv(args.plastic_features_csv)
        plast_df.index = plast_df.index.astype(str)
    else:
        plast_df = pd.read_csv(args.plastic_features_csv, index_col=args.plastic_index_col)

    # 仅保留数值特征列
    num_cols = plast_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("塑料特征表中未检测到数值特征列。")
    X_plast = plast_df[num_cols].astype(float).values
    plast_names = plast_df.index.astype(str).tolist()
    norm_names = [normalize_name(n) for n in plast_names]
    D_in = X_plast.shape[1]

    # ---------- 加载模型并获取 encoder 输出 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseRegressor(input_dim=D_in, embed_dim=64).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(f"模型缺失权重键：{missing}")
    if unexpected:
        warnings.warn(f"模型存在未使用的权重键：{unexpected}")
    model.eval()

    with torch.no_grad():
        Z = model.encoder(torch.tensor(X_plast, dtype=torch.float32, device=device)).cpu().numpy()  # (P, d)

    # L2 归一化
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    # ---------- 计算塑料×塑料相似度并映射到 [0,1] ----------
    # cos = Z_norm @ Z_norm.T ∈ [-1,1]； s_hat = (cos + 1)/2 ∈ [0,1]
    cos_sim = np.clip(Z_norm @ Z_norm.T, -1.0, 1.0)
    s_hat = (cos_sim + 1.0) / 2.0  # (P, P)

    # 名称映射
    norm_to_idx: Dict[str, int] = {norm_names[i]: i for i in range(len(norm_names))}
    norm_to_orig: Dict[str, str] = {norm_names[i]: plast_names[i] for i in range(len(norm_names))}

    # ---------- 载入蛋白×塑料主表并对齐 ----------
    if args.protein_index_col is None:
        ydf = pd.read_csv(args.protein_label_csv)
        ydf.index = ydf.index.astype(str)
    else:
        ydf = pd.read_csv(args.protein_label_csv, index_col=args.protein_index_col)

    # 找出可对齐的列（去前缀+规范化）
    y_cols = []
    mapped_norm = []
    mapped_out = []
    for c in ydf.columns:
        name_raw = strip_prefix(str(c), args.label_prefix)
        key = normalize_name(name_raw)
        if key in norm_to_idx:
            y_cols.append(c)
            mapped_norm.append(key)
            mapped_out.append(norm_to_orig[key])

    if not y_cols:
        raise ValueError("主表中没有任何列能与塑料特征表对齐，请检查 label_prefix/名称规范化。")

    # 子矩阵索引
    plast_idx = np.array([norm_to_idx[k] for k in mapped_norm], dtype=int)  # (P_mapped,)
    # 从全体相似度中抽取：候选×候选之间的 s_hat
    s_hat_mapped = s_hat[np.ix_(plast_idx, plast_idx)]  # (P_mapped, P_mapped)

    # 读取主表与基本信息
    Y = ydf[y_cols].astype(float).values  # (N_e, P_mapped)
    enzymes = ydf.index.astype(str).tolist()
    Pm = len(mapped_norm)

    # 聚合器选择
    def aggregate_for_candidate(s_vec: np.ndarray) -> float:
        if args.agg == "max":
            return agg_max(s_vec)
        elif args.agg == "noisy_or":
            return agg_noisy_or(s_vec, theta=args.theta, tau=args.tau)
        elif args.agg == "softmin":
            return agg_softmin(s_vec, tau=args.tau)
        else:
            raise ValueError(f"不支持的聚合器：{args.agg}")

    # ---------- 逐酶生成软标签 ----------
    out = np.array(Y, dtype=float)  # 先拷贝，已知 1 保持
    for i in range(Y.shape[0]):
        row = Y[i]               # (P_mapped,)
        pos_idx = np.where(row == 1)[0]
        unk_idx = np.where(row == 0)[0]

        if len(pos_idx) == 0 or len(unk_idx) == 0:
            # 无正样本 或 无未知列：直接跳过（保持原值）
            continue

        # 对每个未知候选 c，取其与所有正样本 p 的 s_hat(c,p) 向量并做聚合
        for c in unk_idx:
            s_vec = s_hat_mapped[c, pos_idx]  # (|P_e|,)
            y = aggregate_for_candidate(s_vec)
            if args.clip_01:
                y = float(np.clip(y, 0.0, 1.0))
            out[i, c] = y  # 写回未知位置

    # ---------- 导出：行=酶（蛋白ID），列=塑料名称（去前缀，原始大小写） ----------
    out_df = pd.DataFrame(out, index=enzymes, columns=mapped_out)
    # 强制已知 1 仍为 1（避免数值漂移）
    out_df[ydf[y_cols] == 1] = 1.0

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "soft_labels_by_plastic_similarity.csv")
    out_df.to_csv(out_csv, float_format="%.6g")

    # 记录参数
    with open(os.path.join(args.outdir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"agg={args.agg}\n")
        f.write(f"theta={args.theta}\n")
        f.write(f"tau={args.tau}\n")
        f.write(f"clip_01={args.clip_01}\n")
        f.write(f"label_prefix={args.label_prefix}\n")
        f.write(f"P_mapped={Pm}\n")
        f.write(f"num_enzymes={len(enzymes)}\n")

    print(f"✅ 完成：{out_csv}")
    print(f"   聚合器: {args.agg} | theta={args.theta} | tau={args.tau} | 裁剪到[0,1]: {args.clip_01}")
    print("   仅填充原始 0（未知）；原始 1 保持为 1。名称对齐忽略大小写与 -/_ 差异，输出列名不含前缀。")


if __name__ == "__main__":
    main()