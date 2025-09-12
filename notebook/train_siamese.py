#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siamese training on plastic features + export pairwise distances/similarities.

新增功能：
  1) 训练结束后，基于最终 encoder 对每个塑料得到 embedding
  2) 计算并导出：余弦相似度矩阵（similarity）与对应距离矩阵（distance = 1 - sim）
  3) 同时保存 embedding 向量表，便于后续复用

导出文件（均在 OUTDIR 下）：
  - plastic_embeddings.csv      每个塑料的最终 embedding（64维，列名 emb_0..emb_63）
  - plastic_similarity.csv      余弦相似度矩阵（行列均为塑料名）
  - plastic_distance.csv        距离矩阵（1 - 余弦相似度）
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import imageio
import cv2
from fpdf import FPDF
import numpy as np  # 新增

# ===================== 用户配置 =====================
RUN_NAME = "run16"
FEATURE_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/all_description_new_less.csv"
CO_MATRIX_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/plastic_co_matrix.csv"
LOSS_MODE = "contrastive"  # "mse" 或 "contrastive"
MARGIN = 3
SIM_THRESHOLD = 0.01
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
PLOT_INTERVAL = 5

# 新增归一化配置
NORMALIZE = True
NORM_METHOD = "zscore"  # "zscore", "minmax", or "none"

# ===================== 自动路径管理 =====================
OUTDIR = os.path.join(RUN_NAME)
PLOT_DIR = os.path.join(OUTDIR, "embedding_plots")
GIF_PATH = os.path.join(OUTDIR, "embedding_evolution.gif")
VIDEO_PATH = os.path.join(OUTDIR, "embedding_evolution.mp4")
PDF_PATH = os.path.join(OUTDIR, "embedding_snapshots.pdf")
LOSS_PATH = os.path.join(OUTDIR, "loss_curve.png")
MODEL_PATH = os.path.join(OUTDIR, "siamese_model.pt")
INFO_PATH = os.path.join(OUTDIR, "run_info.txt")

# 新增导出路径
EMBED_CSV_PATH = os.path.join(OUTDIR, "plastic_embeddings.csv")
SIM_CSV_PATH   = os.path.join(OUTDIR, "plastic_similarity.csv")
DIST_CSV_PATH  = os.path.join(OUTDIR, "plastic_distance.csv")

os.makedirs(PLOT_DIR, exist_ok=True)

# ===================== 记录实验配置 =====================
with open(INFO_PATH, "w") as f:
    f.write("🔧 Hyperparameters and paths:\n")
    f.write(f"RUN_NAME = {RUN_NAME}\n")
    f.write(f"FEATURE_CSV = {FEATURE_CSV}\n")
    f.write(f"CO_MATRIX_CSV = {CO_MATRIX_CSV}\n")
    f.write(f"LOSS_MODE = {LOSS_MODE}\n")
    f.write(f"MARGIN = {MARGIN}\n")
    f.write(f"SIM_THRESHOLD = {SIM_THRESHOLD}\n")
    f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
    f.write(f"EPOCHS = {EPOCHS}\n")
    f.write(f"LR = {LR}\n")
    f.write(f"NORMALIZE = {NORMALIZE}\n")
    f.write(f"NORM_METHOD = {NORM_METHOD}\n")

# ===================== 数据加载 =====================
features = pd.read_csv(FEATURE_CSV, index_col=0)
co_matrix = pd.read_csv(CO_MATRIX_CSV, index_col=0)
features = features.loc[features.index.intersection(co_matrix.index)]
co_matrix = co_matrix.loc[features.index, features.index]

# 归一化处理
if NORMALIZE and NORM_METHOD != "none":
    if NORM_METHOD == "zscore":
        features[:] = StandardScaler().fit_transform(features)
    elif NORM_METHOD == "minmax":
        features[:] = MinMaxScaler().fit_transform(features)
    else:
        raise ValueError(f"Unknown normalization method: {NORM_METHOD}")

feature_dim = features.shape[1]

data_pairs = []
for i, j in itertools.combinations(range(len(features)), 2):
    name_i = features.index[i]
    name_j = features.index[j]
    if name_i in co_matrix.index and name_j in co_matrix.columns:
        sim = co_matrix.loc[name_i, name_j]
        if pd.notna(sim):
            x1 = torch.tensor(features.loc[name_i].values, dtype=torch.float32)
            x2 = torch.tensor(features.loc[name_j].values, dtype=torch.float32)
            if LOSS_MODE == "mse":
                y = torch.tensor([float(sim)], dtype=torch.float32)
            else:
                y = torch.tensor([1.0 if sim >= SIM_THRESHOLD else 0.0], dtype=torch.float32)
            data_pairs.append((x1, x2, y))

if not data_pairs:
    raise ValueError("❌ No valid training pairs found.")

print(f"✅ Training pairs: {len(data_pairs)}")
print(f"🧪 Loss mode: {LOSS_MODE}")

# ===================== Dataset =====================
class PairwiseDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

# ===================== 模型结构 =====================
class SiameseRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward_once(self, x): return self.encoder(x)
    def forward(self, x1, x2): return self.forward_once(x1), self.forward_once(x2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin): super().__init__(); self.margin = margin
    def forward(self, out1, out2, label):
        dist = torch.norm(out1 - out2, dim=1)
        loss = label.squeeze() * torch.pow(dist, 2) + \
               (1 - label.squeeze()) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()

# ===================== 可视化 =====================
def plot_embeddings(model, features_df, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features_df.values, dtype=torch.float32)
        embeddings = model.encoder(X).numpy()
        reduced = PCA(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(6, 5))
        for i, name in enumerate(features_df.index):
            plt.scatter(reduced[i, 0], reduced[i, 1])
            plt.text(reduced[i, 0], reduced[i, 1], name, fontsize=7)
        plt.title(f"Epoch {epoch}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
        plt.close()

# ===================== 训练 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(PairwiseDataset(data_pairs), batch_size=BATCH_SIZE, shuffle=True)
model = SiameseRegressor(feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() if LOSS_MODE == "mse" else ContrastiveLoss(MARGIN)

losses = []
print("🚀 Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        h1, h2 = model(x1, x2)
        if LOSS_MODE == "mse":
            pred = torch.nn.functional.cosine_similarity(h1, h2).unsqueeze(1)
            loss = criterion(pred, y)
        else:
            loss = criterion(h1, h2, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x1.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    losses.append(avg_loss)
    if epoch % PLOT_INTERVAL == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.5f}")
        plot_embeddings(model, features, epoch, PLOT_DIR)

# ===================== 保存输出（原有） =====================
plt.figure()
plt.plot(range(EPOCHS), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Siamese Training Loss ({LOSS_MODE})")
plt.tight_layout()
plt.savefig(LOSS_PATH)
plt.close()

imgs = [imageio.v2.imread(os.path.join(PLOT_DIR, f)) for f in sorted(os.listdir(PLOT_DIR)) if f.endswith(".png")]
imageio.mimsave(GIF_PATH, imgs, duration=0.8)
print(f"🎞️ GIF saved to {GIF_PATH}")

frame = cv2.imread(os.path.join(PLOT_DIR, sorted(os.listdir(PLOT_DIR))[0]))
height, width, _ = frame.shape
video = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 1.25, (width, height))
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        img = cv2.imread(os.path.join(PLOT_DIR, f))
        video.write(img)
video.release()
print(f"🎬 MP4 saved to {VIDEO_PATH}")

pdf = FPDF()
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        pdf.add_page()
        pdf.image(os.path.join(PLOT_DIR, f), x=10, y=10, w=180)
pdf.output(PDF_PATH)
print(f"📄 PDF report saved to {PDF_PATH}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# ===================== 新增：导出 embedding / 相似度 / 距离矩阵 =====================
with torch.no_grad():
    model.eval()
    X_all = torch.tensor(features.values, dtype=torch.float32, device=device)
    Z = model.encoder(X_all)  # [N, 64]
    Z = Z.detach().cpu().numpy()

# 保存 embedding 表
emb_cols = [f"emb_{i}" for i in range(Z.shape[1])]
emb_df = pd.DataFrame(Z, index=features.index, columns=emb_cols)
emb_df.to_csv(EMBED_CSV_PATH)
print(f"🧬 Embeddings saved to {EMBED_CSV_PATH}")

# 计算余弦相似度矩阵（行归一化后点积）
Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
sim_mat = np.clip(Z_norm @ Z_norm.T, -1.0, 1.0)
dist_mat = 1.0 - sim_mat  # 定义距离 = 1 - 余弦相似度

# 数值与对角线修正
np.fill_diagonal(sim_mat, 1.0)
np.fill_diagonal(dist_mat, 0.0)

# 保存矩阵
sim_df = pd.DataFrame(sim_mat, index=features.index, columns=features.index)
dist_df = pd.DataFrame(dist_mat, index=features.index, columns=features.index)
sim_df.to_csv(SIM_CSV_PATH)
dist_df.to_csv(DIST_CSV_PATH)

print(f"🔗 Similarity matrix saved to {SIM_CSV_PATH}")
print(f"📏 Distance matrix  saved to {DIST_CSV_PATH}")