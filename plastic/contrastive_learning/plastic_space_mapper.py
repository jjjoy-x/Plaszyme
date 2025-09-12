#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_plastic_similarity_from_sdf.py

用途：
  读取一个 SDF 目录，利用已训练好的孪生网络（.pt）把所有塑料结构映射到嵌入空间，
  输出：余弦相似度矩阵、（可选）欧氏距离矩阵、嵌入向量、降维坐标和散点图。

注意：
  - 模型的第一层输入维度必须与这里提取的 RDKit 特征维度一致。
  - 训练时若做过特征归一化，这里最好用相同策略（zscore / minmax / none）。
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---- UMAP 可选依赖（只在选择 umap 时需要）
UMAP_AVAILABLE = True
try:
    import umap  # type: ignore
except Exception:
    UMAP_AVAILABLE = False

# --------- 你的 RDKit 特征提取器 ----------
# 需要你的项目里已有该模块
from plastic.mol_features.descriptors_rdkit import PlasticFeaturizer


# ===================== 用户配置 =====================
RUN_NAME   = "runs/infer_from_sdf_7"

SDF_DIR    = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"  # 包含多个 .sdf 文件的目录
CONFIG_YAML= "/path/to/plastic/mol_features/rdkit_features.yaml"  # RDKit 特征配置

MODEL_PT   = "/Users/shulei/PycharmProjects/Plaszyme/run/run_from_sdf_7/siamese_model.pt"  # 训练好的权重（state_dict 或完整模型）
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 与训练一致的特征归一化
NORMALIZE  = True
NORM_METHOD= "zscore"    # "zscore" | "minmax" | "none"

# 相似度/距离输出
SAVE_COSINE_SIMILARITY = True    # 余弦相似度（-1 ~ 1）
SAVE_EUCLIDEAN_DISTANCE= True    # 欧氏距离（>=0）

# 降维可视化
REDUCTION_METHOD = "tsne"        # "pca" | "umap" | "tsne"
RANDOM_STATE     = 42

# ===================== 自动路径管理 =====================
OUTDIR = RUN_NAME
os.makedirs(OUTDIR, exist_ok=True)
FEATURES_CSV        = os.path.join(OUTDIR, "features.csv")
EMBEDDINGS_CSV      = os.path.join(OUTDIR, "embeddings.csv")
EMBEDDINGS_NPY      = os.path.join(OUTDIR, "embeddings.npy")
SIM_CSV             = os.path.join(OUTDIR, "plastic_similarity__cosine.csv")
DIST_CSV            = os.path.join(OUTDIR, "plastic_distance__euclidean.csv")
REDUCED_CSV         = os.path.join(OUTDIR, f"reduced_{REDUCTION_METHOD}.csv")
SCATTER_PNG         = os.path.join(OUTDIR, f"scatter_{REDUCTION_METHOD}.png")
INFO_PATH           = os.path.join(OUTDIR, "run_info.json")


# ===================== 模型结构（需与训练一致） =====================
class SiameseRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward_once(self, x):
        return self.encoder(x)


# ===================== 工具函数 =====================
def log(msg: str):
    print(msg, flush=True)

def save_info(payload: dict):
    with open(INFO_PATH, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def load_or_featurize_sdf(sdf_dir: str, config_yaml: str) -> pd.DataFrame:
    """
    使用你项目里的 PlasticFeaturizer 从 SDF 目录提特征，并保存到 FEATURES_CSV。
    """
    log("🧪 开始从 SDF 提取 RDKit 特征 ...")
    featurizer = PlasticFeaturizer(config_yaml)
    feature_dict, stats = featurizer.featurize_folder(sdf_dir)
    # 自动保存（带表头）
    prefix = FEATURES_CSV.replace(".csv", "")
    featurizer.save_features(feature_dict, prefix)
    df = pd.read_csv(FEATURES_CSV, index_col=0)
    log(f"✅ 提取完成：{df.shape[0]} 个样本，{df.shape[1]} 个特征 → {FEATURES_CSV}")
    return df

def maybe_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if not NORMALIZE or NORM_METHOD == "none":
        return df
    log(f"🔧 特征归一化：{NORM_METHOD}")
    out = df.copy()
    if NORM_METHOD == "zscore":
        out[:] = StandardScaler().fit_transform(out)
    elif NORM_METHOD == "minmax":
        out[:] = MinMaxScaler().fit_transform(out)
    else:
        raise ValueError(f"Unknown NORM_METHOD: {NORM_METHOD}")
    return out

def find_first_linear_in_shape(state_dict: dict) -> int:
    """
    从 state_dict 猜第一层 Linear 的输入维度。
    常见 key: 'encoder.0.weight' 或 'module.encoder.0.weight'
    """
    # 优先找包含 'encoder.0.weight'
    for k, v in state_dict.items():
        if k.endswith("encoder.0.weight") or "encoder.0.weight" in k:
            return int(v.shape[1])
    # 退而求其次：找 shape 类似 [128, in_dim] 的第一个 weight
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] == 128:
            return int(v.shape[1])
    raise RuntimeError("无法从 state_dict 推断第一层输入维度（encoder.0.weight 未找到）。")

def load_model(model_pt: str, in_dim: int | None, device: str) -> SiameseRegressor:
    ckpt = torch.load(model_pt, map_location=device)
    # 可能是 state_dict 或完整模型
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model.to(device)
        model.eval()
        # 尝试从模型第一层读输入维度（可选）
        return model
    elif isinstance(ckpt, dict):
        # 有些保存为 {"state_dict": ...}
        sd = ckpt.get("state_dict", ckpt)
        expected_in = find_first_linear_in_shape(sd)
        if in_dim is not None and in_dim != expected_in:
            raise RuntimeError(f"❌ 特征维度({in_dim}) 与模型期望输入维度({expected_in})不一致。"
                               f"请确保 featurizer 配置与训练时一致。")
        model = SiameseRegressor(input_dim=expected_in).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    else:
        raise RuntimeError("无法识别的模型文件格式（既不是 nn.Module 也不是 state_dict）。")

@torch.no_grad()
def compute_embeddings(model: nn.Module, feats: pd.DataFrame, device: str) -> np.ndarray:
    X = torch.tensor(feats.values, dtype=torch.float32, device=device)
    Z = model.encoder(X) if hasattr(model, "encoder") else model(X)
    return Z.detach().cpu().numpy()

def cosine_similarity_matrix(Z: np.ndarray) -> pd.DataFrame:
    # 归一化后点乘
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Zhat = Z / norms
    S = Zhat @ Zhat.T
    return pd.DataFrame(S, index=names, columns=names)

def euclidean_distance_matrix(Z: np.ndarray) -> pd.DataFrame:
    # ||zi - zj||2
    # 高效计算：||A-B||^2 = ||A||^2 + ||B||^2 - 2 A·B
    G = Z @ Z.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2 * G
    D2[D2 < 0] = 0.0
    D = np.sqrt(D2)
    return pd.DataFrame(D, index=names, columns=names)

def reduce_2d(Z: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2).fit_transform(Z)
    if method == "tsne":
        return TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=max(5, min(30, Z.shape[0]-1))).fit_transform(Z)
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise RuntimeError("需要安装 umap-learn：pip install umap-learn")
        return umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
    raise ValueError(f"Unsupported REDUCTION_METHOD: {method}")

def plot_scatter(coords: np.ndarray, names: list[str], out_png: str, title: str):
    plt.figure(figsize=(7.2, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=18)
    for i, name in enumerate(names):
        plt.text(coords[i, 0], coords[i, 1], name, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ===================== 主流程 =====================
if __name__ == "__main__":
    log("=== Step 1/6: 提取或载入特征 ===")
    feats_df = load_or_featurize_sdf(SDF_DIR, CONFIG_YAML)
    names = feats_df.index.astype(str).tolist()
    in_dim = feats_df.shape[1]
    log(f"[INFO] 特征维度 = {in_dim}, 样本数 = {len(names)}")

    if NORMALIZE and NORM_METHOD != "none":
        feats_df = maybe_normalize(feats_df)

    log("=== Step 2/6: 加载模型 ===")
    model = load_model(MODEL_PT, in_dim, DEVICE)
    # 如果模型 state_dict 推断的维度与 df 不一致，上面的 load_model 已经抛错

    # 记录关键信息
    info = {
        "run_name": RUN_NAME,
        "sdf_dir": SDF_DIR,
        "config_yaml": CONFIG_YAML,
        "model_pt": MODEL_PT,
        "device": DEVICE,
        "normalize": NORMALIZE,
        "norm_method": NORM_METHOD,
        "reduction_method": REDUCTION_METHOD,
        "n_samples": len(names),
        "n_features": in_dim,
    }
    save_info(info)

    log("=== Step 3/6: 计算嵌入 ===")
    Z = compute_embeddings(model, feats_df, DEVICE)
    np.save(EMBEDDINGS_NPY, Z)
    pd.DataFrame(Z, index=names).to_csv(EMBEDDINGS_CSV)
    log(f"[OK] 嵌入保存：{EMBEDDINGS_CSV} | {EMBEDDINGS_NPY} | 形状={Z.shape}")

    log("=== Step 4/6: 计算相似度/距离矩阵 ===")
    if SAVE_COSINE_SIMILARITY:
        sim_df = cosine_similarity_matrix(Z)
        sim_df.to_csv(SIM_CSV)
        log(f"[OK] 余弦相似度矩阵保存：{SIM_CSV}  （范围约 [-1, 1]，对角=1.0）")

    if SAVE_EUCLIDEAN_DISTANCE:
        dist_df = euclidean_distance_matrix(Z)
        dist_df.to_csv(DIST_CSV)
        log(f"[OK] 欧氏距离矩阵保存：{DIST_CSV}  （非负，对角=0）")

    log("=== Step 5/6: 降维与可视化 ===")
    coords2d = reduce_2d(Z, REDUCTION_METHOD)
    pd.DataFrame(coords2d, index=names, columns=["x", "y"]).to_csv(REDUCED_CSV)
    plot_scatter(coords2d, names, SCATTER_PNG, f"{REDUCTION_METHOD.upper()} of embeddings (N={len(names)})")
    log(f"[OK] 降维坐标：{REDUCED_CSV}")
    log(f"[OK] 散点图：{SCATTER_PNG}")

    log("=== Step 6/6: 完成 ===")
    log(f"输出目录：{OUTDIR}")