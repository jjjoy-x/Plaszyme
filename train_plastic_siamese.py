import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import imageio
import cv2
from fpdf import FPDF

from plastic.mol_features.descriptors_rdkit import PlasticFeaturizer

# ===================== 用户配置 =====================
RUN_NAME = "run/run_from_sdf_4"
SDF_DIR = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf"
CONFIG_PATH = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mol_features/rdkit_features.yaml"
CO_MATRIX_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/plastic_co_matrix.csv"

LOSS_MODE = "contrastive"
MARGIN = 1.5
SIM_THRESHOLD = 0.01
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-4
PLOT_INTERVAL = 5
REDUCTION_METHOD = "pca" # 降维方式可选: "pca", "umap", "tsne"

# ===================== 自动路径管理 =====================
OUTDIR = os.path.join(RUN_NAME)
PLOT_DIR = os.path.join(OUTDIR, "embedding_plots")
GIF_PATH = os.path.join(OUTDIR, "embedding_evolution.gif")
VIDEO_PATH = os.path.join(OUTDIR, "embedding_evolution.mp4")
PDF_PATH = os.path.join(OUTDIR, "embedding_snapshots.pdf")
LOSS_PATH = os.path.join(OUTDIR, "loss_curve.png")
MODEL_PATH = os.path.join(OUTDIR, "siamese_model.pt")
FEATURE_CSV = os.path.join(OUTDIR, "features.csv")
INFO_PATH = os.path.join(OUTDIR, "run_info.txt")

os.makedirs(PLOT_DIR, exist_ok=True)

# ===================== 特征提取 =====================
print("🧪 正在提取SDF特征...")
featurizer = PlasticFeaturizer(CONFIG_PATH)
feature_dict, stats = featurizer.featurize_folder(SDF_DIR)

# 自动保存包含列名的 CSV 和 .pt 文件
output_prefix = FEATURE_CSV.replace(".csv", "")
featurizer.save_features(feature_dict, output_prefix)

# 读取 CSV（带列名）
features_df = pd.read_csv(FEATURE_CSV, index_col=0)
print(f"✅ 特征保存至 {FEATURE_CSV}")
print(f"💡 特征维度: {features_df.shape[1]}，塑料种类: {features_df.shape[0]}")
print(f"🧬 特征名示例: {features_df.columns[:5].tolist()}")

# ===================== 加载共降解矩阵 =====================
co_matrix = pd.read_csv(CO_MATRIX_CSV, index_col=0)
features_df = features_df.loc[features_df.index.intersection(co_matrix.index)]
co_matrix = co_matrix.loc[features_df.index, features_df.index]
feature_dim = features_df.shape[1]

# ===================== 构建训练对 =====================
data_pairs = []
for i, j in itertools.combinations(range(len(features_df)), 2):
    name_i = features_df.index[i]
    name_j = features_df.index[j]
    if name_i in co_matrix.index and name_j in co_matrix.columns:
        sim = co_matrix.loc[name_i, name_j]
        if pd.notna(sim):
            x1 = torch.tensor(features_df.loc[name_i].values, dtype=torch.float32)
            x2 = torch.tensor(features_df.loc[name_j].values, dtype=torch.float32)
            y = torch.tensor([1.0 if sim >= SIM_THRESHOLD else 0.0], dtype=torch.float32)
            data_pairs.append((x1, x2, y))

if not data_pairs:
    raise ValueError("❌ 无有效训练对")

print(f"✅ 构建训练对数：{len(data_pairs)} 对")

# ===================== Dataset =====================
class PairwiseDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

# ===================== 模型定义 =====================
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

def reduce_embeddings(embeddings, method="pca", random_state=42):
    if method == "pca":
        return PCA(n_components=2).fit_transform(embeddings)
    elif method == "umap":
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(embeddings)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=random_state, perplexity=5).fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

# ===================== 可视化函数 =====================
def plot_embeddings(model, features_df, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features_df.values, dtype=torch.float32)
        embeddings = model.encoder(X).numpy()
        reduced = reduce_embeddings(embeddings, method=REDUCTION_METHOD)

        plt.figure(figsize=(6, 5))
        for i, name in enumerate(features_df.index):
            plt.scatter(reduced[i, 0], reduced[i, 1])
            plt.text(reduced[i, 0], reduced[i, 1], name, fontsize=7)
        plt.title(f"{REDUCTION_METHOD.upper()} - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
        plt.close()

# ===================== 训练流程 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(PairwiseDataset(data_pairs), batch_size=BATCH_SIZE, shuffle=True)
model = SiameseRegressor(feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = ContrastiveLoss(MARGIN)

print("🚀 开始训练...")
losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        h1, h2 = model(x1, x2)
        loss = criterion(h1, h2, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x1.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    losses.append(avg_loss)
    if epoch % PLOT_INTERVAL == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Loss = {avg_loss:.5f}")
        plot_embeddings(model, features_df, epoch, PLOT_DIR)

# ===================== 结果保存 =====================
plt.figure()
plt.plot(range(EPOCHS), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig(LOSS_PATH)

imgs = [imageio.v2.imread(os.path.join(PLOT_DIR, f)) for f in sorted(os.listdir(PLOT_DIR)) if f.endswith(".png")]
imageio.mimsave(GIF_PATH, imgs, duration=0.8)

frame = cv2.imread(os.path.join(PLOT_DIR, sorted(os.listdir(PLOT_DIR))[0]))
height, width, _ = frame.shape
video = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 1.25, (width, height))
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        img = cv2.imread(os.path.join(PLOT_DIR, f))
        video.write(img)
video.release()

pdf = FPDF()
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        pdf.add_page()
        pdf.image(os.path.join(PLOT_DIR, f), x=10, y=10, w=180)
pdf.output(PDF_PATH)

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 模型保存至: {MODEL_PATH}")

