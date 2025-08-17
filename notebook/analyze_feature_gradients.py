import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import itertools

# ========== 参数设置 ==========
FEATURE_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/t1.csv"
CO_MATRIX_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/plastic_co_matrix.csv"
MODEL_PATH = "/notebook/outputs/run9/siamese_model.pt"
OUTPUT_DIR = "/Users/shulei/PycharmProjects/Plaszyme/notebook/grad_analysis/run9"

LOSS_MODE = "contrastive"   # "mse" 或 "contrastive"
SIM_THRESHOLD = 0.01
BATCH_SIZE = 16
TOP_N = 10
MARGIN = 2.0

# 自动生成输出路径
os.makedirs(OUTPUT_DIR, exist_ok=True)
GRAD_PLOT_PATH = os.path.join(OUTPUT_DIR, "feature_gradient_contributions.png")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "feature_gradient_importance.csv")

# ========== 加载数据 ==========
features = pd.read_csv(FEATURE_CSV, index_col=0)
co_matrix = pd.read_csv(CO_MATRIX_CSV, index_col=0)
features = features.loc[features.index.intersection(co_matrix.index)]
co_matrix = co_matrix.loc[features.index, features.index]
feature_names = features.columns.tolist()
feature_dim = features.shape[1]

# ========== 构建配对数据 ==========
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
            elif LOSS_MODE == "contrastive":
                y = torch.tensor([1.0 if sim >= SIM_THRESHOLD else 0.0], dtype=torch.float32)
            data_pairs.append((x1, x2, y))

# ========== 数据集 ==========
class PairwiseDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

dataloader = DataLoader(PairwiseDataset(data_pairs), batch_size=BATCH_SIZE, shuffle=False)

# ========== 模型结构 ==========
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

# ========== 加载模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseRegressor(feature_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========== 梯度累积 ==========
grad_accum = torch.zeros(feature_dim)

for x1, x2, y in dataloader:
    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    x1.requires_grad_(True)
    h1, h2 = model(x1, x2)

    if LOSS_MODE == "mse":
        pred = torch.nn.functional.cosine_similarity(h1, h2).unsqueeze(1)
        loss = nn.MSELoss()(pred, y)
    else:
        loss = ContrastiveLoss(margin=MARGIN)(h1, h2, y)

    model.zero_grad()
    loss.backward()
    grad = x1.grad.abs().sum(dim=0).cpu()
    grad_accum += grad

# ========== 结果整理 ==========
grad_mean = grad_accum / len(dataloader)
grad_df = pd.DataFrame({"feature": feature_names, "importance": grad_mean.tolist()})
grad_df = grad_df.sort_values(by="importance", ascending=False)

# ========== 保存 CSV ==========
grad_df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"📄 梯度分析表已保存：{CSV_OUTPUT_PATH}")

# ========== 可视化 ==========
plot_df = grad_df.head(TOP_N) if TOP_N else grad_df
plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Gradient Importance (avg abs)")
plt.title(f"Top {TOP_N} Feature Importances by Gradient" if TOP_N else "All Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(GRAD_PLOT_PATH)
print(f"✅ 梯度贡献图已保存：{GRAD_PLOT_PATH}")