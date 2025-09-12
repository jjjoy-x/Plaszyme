#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pairs_from_plastic_similarity.py

Purpose:
    Use a trained "plastic-plastic distance/similarity" Siamese model to score
    candidate negative plastics for each enzyme and generate a labeled pairs table
    for downstream enzyme–plastic training (binary 0/1).

功能：
    利用你训练好的“塑料间距离/相似度”模型，为每个酶挑选并打分负样本塑料，
    生成包含 {enzyme, plastic, label, score, weight} 的配对表（CSV），用于后续二分类训练。

Inputs (configured below, no argparse):
    - FEATURES_CSV: RDKit 特征表（由 PlasticFeaturizer 导出 CSV），index 为塑料名
    - POSITIVE_MAP_CSV: 酶-正样本塑料映射，列名 {enzyme_col, plastic_col}
        例如：
            enzyme,plastic
            EnzA,PET
            EnzA,PLA
            EnzB,PCL
    - MODEL_PATH: 已训练好的孪生网络权重（与你的训练脚本里的 SiameseRegressor 结构一致）
    - 归一化方式需与训练时一致（zscore / minmax / none）

Output:
    - PAIRS_OUT_CSV: 含列
        enzyme,plastic,label,score,weight,sim_to_pos
      其中：
        label: 正样本=1（来自 POSITIVE_MAP_CSV），负样本=0（由本脚本选择）
        sim_to_pos: 候选塑料与该酶的“所有正样本塑料”的最大相似度（cosine）
        score: 负样本得分 = 1 - sim_to_pos（越大越“负”）
        weight: 训练时可用作样本权重（默认 = score）

Notes:
    - 如果你的模型是用 "contrastive" 训练的，本脚本仍用编码后的 cosine 相似度做打分；
      若你另有距离度量，可替换 compute_similarity_matrix。
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ===================== Direct configuration =====================
FEATURES_CSV      = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/all_description.csv"   # PlasticFeaturizer 保存的 CSV（index=plastic 名称）
POSITIVE_MAP_CSV  = "/Users/shulei/PycharmProjects/Plaszyme/dataset/train_v0.3.1.fasta"   # 列：enzyme, plastic
MODEL_PATH        = "/Users/shulei/PycharmProjects/Plaszyme/notebook/outputs/run9/siamese_model.pt"      # 你的孪生网络权重
PAIRS_OUT_CSV     = "/pairs.csv"      # 输出路径

# Normalization — must match training
NORMALIZE   = True
NORM_METHOD = "zscore"   # "zscore", "minmax", "none"

# Negative sampling policy
TOP_K_NEG_PER_ENZYME     = 50     # 每个酶最多挑选的负样本数量
MAX_SIM_FOR_NEG          = 0.20   # 仅选择最大相似度 <= 该阈值的候选作为负样本
WEIGHT_EXPONENT          = 1.0    # 权重 = (score) ** exponent
RANDOM_SEED              = 42     # 可复现随机性（当候选过多时随机裁剪）

# Column names in POSITIVE_MAP_CSV
ENZYME_COL  = "enzyme"
PLASTIC_COL = "plastic"


# ===================== Model definition (must match your training) =====================
class SiameseRegressor(nn.Module):
    """
    Should match the structure used during training (encoder only is used at inference).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):  # [B, D] -> [B, 64]
        return self.encoder(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# ===================== Utilities =====================
def load_features(csv_path: str, normalize: bool, method: str) -> pd.DataFrame:
    """
    Load plastic features (index=plastic name). Apply the same normalization as training.
    """
    df = pd.read_csv(csv_path, index_col=0)
    if normalize and method != "none":
        if method == "zscore":
            df[:] = StandardScaler().fit_transform(df)
        elif method == "minmax":
            df[:] = MinMaxScaler().fit_transform(df)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    return df


@torch.no_grad()
def compute_embeddings(model: SiameseRegressor, features_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Encode all plastic feature rows into embedding vectors.
    Returns: Tensor [N, H]
    """
    model.eval()
    X = torch.tensor(features_df.values, dtype=torch.float32, device=device)
    Z = model.encoder(X)  # [N, H]
    return Z


def cosine_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity matrix from embeddings. Returns [N, N] in numpy.
    """
    Z = embeddings
    Z = Z / (torch.norm(Z, dim=1, keepdim=True) + 1e-12)
    S = torch.mm(Z, Z.t())  # [N, N]
    return S.cpu().numpy()


def build_positive_map(pos_csv: str, enzyme_col: str, plastic_col: str) -> Dict[str, Set[str]]:
    """
    Build mapping: enzyme -> set(positive plastics).
    """
    df = pd.read_csv(pos_csv)
    if enzyme_col not in df.columns or plastic_col not in df.columns:
        raise KeyError(f"Columns not found. Available: {list(df.columns)}")
    pos_map: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        e = str(row[enzyme_col])
        p = str(row[plastic_col])
        pos_map.setdefault(e, set()).add(p)
    return pos_map


def sample_negatives_for_enzyme(
    enzyme: str,
    pos_plastics: Set[str],
    all_plastics: List[str],
    name_to_idx: Dict[str, int],
    sim_matrix: np.ndarray,
    top_k: int,
    max_sim: float,
    rng: random.Random
) -> List[Tuple[str, float]]:
    """
    For a given enzyme, compute each candidate plastic's max similarity to any positive plastic.
    Select negatives with max_sim_to_pos <= max_sim, then take up to top_k (lowest similarity first).
    Returns list of (plastic_name, sim_to_pos).
    """
    if not pos_plastics:
        return []

    # Pre-compute positive indices
    pos_idx = [name_to_idx[p] for p in pos_plastics if p in name_to_idx]
    if not pos_idx:
        return []

    candidates = []
    for p in all_plastics:
        if p in pos_plastics:
            continue
        if p not in name_to_idx:
            continue
        j = name_to_idx[p]
        # candidate's max similarity to positives
        sim_to_pos = float(sim_matrix[j, pos_idx].max())
        if sim_to_pos <= max_sim:
            candidates.append((p, sim_to_pos))

    # Sort by ascending similarity (lower similarity = stronger negative)
    candidates.sort(key=lambda x: x[1])

    # If too many, keep top_k (you can also randomize slight shuffle before truncation)
    if len(candidates) > top_k:
        # Optional small shuffle among the last ones of the truncation boundary
        # to avoid always picking exactly the same items at ties.
        cutoff = top_k
        head = candidates[:cutoff]
        rng.shuffle(head)
        candidates = head

    return candidates


# ===================== Main logic (no argparse) =====================
def main():
    rng = random.Random(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load features and model
    feats_df = load_features(FEATURES_CSV, NORMALIZE, NORM_METHOD)
    plastics: List[str] = feats_df.index.astype(str).tolist()
    in_dim = feats_df.shape[1]

    model = SiameseRegressor(input_dim=in_dim).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 2) Compute embeddings & cosine similarity matrix
    emb = compute_embeddings(model, feats_df, device)            # [N, H]
    sim = cosine_similarity_matrix(emb)                          # [N, N]
    name_to_idx = {name: i for i, name in enumerate(plastics)}

    # 3) Build positive map (enzyme -> set(positives))
    pos_map = build_positive_map(POSITIVE_MAP_CSV, ENZYME_COL, PLASTIC_COL)
    enzymes = sorted(pos_map.keys())

    # 4) Assemble labeled pairs
    rows = []

    # 4.1 Add all positive pairs (label=1)
    for e in enzymes:
        for p in sorted(pos_map[e]):
            rows.append({
                "enzyme": e,
                "plastic": p,
                "label": 1,
                "sim_to_pos": 1.0,   # by definition to itself; (or max sim among positives)
                "score": 1.0,        # optional for completeness
                "weight": 1.0
            })

    # 4.2 Add mined negatives (label=0) with scores/weights
    for e in enzymes:
        neg_candidates = sample_negatives_for_enzyme(
            enzyme=e,
            pos_plastics=pos_map[e],
            all_plastics=plastics,
            name_to_idx=name_to_idx,
            sim_matrix=sim,
            top_k=TOP_K_NEG_PER_ENZYME,
            max_sim=MAX_SIM_FOR_NEG,
            rng=rng
        )
        for p, sim_to_pos in neg_candidates:
            score = max(0.0, 1.0 - float(sim_to_pos))  # bigger = more negative
            weight = math.pow(score, WEIGHT_EXPONENT)  # you can tune exponent
            rows.append({
                "enzyme": e,
                "plastic": p,
                "label": 0,
                "sim_to_pos": float(sim_to_pos),
                "score": float(score),
                "weight": float(weight)
            })

    # 5) Save as CSV
    out_df = pd.DataFrame(rows, columns=["enzyme", "plastic", "label", "sim_to_pos", "score", "weight"])
    os.makedirs(os.path.dirname(PAIRS_OUT_CSV) or ".", exist_ok=True)
    out_df.to_csv(PAIRS_OUT_CSV, index=False)
    print(f"[INFO] Wrote pairs with scores: {PAIRS_OUT_CSV}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()