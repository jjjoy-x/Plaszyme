import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class _PairwiseDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]  # (x1, x2, y) | y: float scalar


class _ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        # out1, out2: [B, D] (已 L2 normalize)
        # label: [B] 1=similar, 0=dissimilar
        dist = torch.norm(out1 - out2, dim=1)
        pos = label * (dist ** 2)
        neg = (1 - label) * (torch.clamp(self.margin - dist, min=0.0) ** 2)
        return (pos + neg).mean()


class PolymerTower(nn.Module):
    """
    简单的塑料 MLP 塔
    - 输入: 描述符向量 (来自 PolymerFeaturizer) [B, in_dim]
    - 输出: L2 归一化后的嵌入向量 [B, out_dim]
    - 用途: 与酶塔输出做余弦相似度 (cosine similarity) 训练
    """

    def __init__(self, in_dim: int, hidden_dims=(512, 256), out_dim=128, dropout=0.1):
        super().__init__()
        layers, last_dim = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(last_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim] 塑料描述符向量
        Returns:
            z: [B, out_dim] 单位向量 (L2 normalized)
        """
        h = self.mlp(x.float())
        z = self.out_proj(h)
        return F.normalize(z, dim=-1)

    def pretrain_with_co_matrix(
        self,
        features_df: pd.DataFrame,  # index=塑料名
        co_matrix_df: pd.DataFrame, # index/columns 同塑料名
        *,
        sim_threshold: float = 0.01,
        loss_mode: str = "contrastive",  # "contrastive" 或 "mse"
        margin: float = 1.5,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-4,
        device: str | None = None,
        verbose: bool = True,
    ):
        def _normalize_name(name: str) -> str:
            name = str(name)
            if name.lower().endswith(".sdf"):
                name = name[:-4]  # 去掉 .sdf
            return name.strip()

        features_df = features_df.copy()
        features_df.index = [_normalize_name(n) for n in features_df.index]
        co_matrix_df = co_matrix_df.copy()
        co_matrix_df.index = [_normalize_name(n) for n in co_matrix_df.index]
        co_matrix_df.columns = [_normalize_name(n) for n in co_matrix_df.columns]

        """用共降解矩阵对 PolymerTower 进行预训练"""
        assert isinstance(features_df, pd.DataFrame) and isinstance(co_matrix_df, pd.DataFrame)

        print(f"[PRETRAIN] features_df 塑料={len(features_df.index)} 个: {list(features_df.index)[:10]}")
        print(f"[PRETRAIN] co_matrix_df 塑料={len(co_matrix_df.index)} 个: {list(co_matrix_df.index)[:10]}")

        # 对齐前，检查差集
        names_feat = set(map(str, features_df.index))
        names_co = set(map(str, co_matrix_df.index))
        inter = names_feat & names_co
        print(
            f"[PRETRAIN] 交集数量={len(inter)} | 仅在 features 不在 co_matrix 的示例: {list(names_feat - names_co)[:10]}")
        print(f"[PRETRAIN] 仅在 co_matrix 不在 features 的示例: {list(names_co - names_feat)[:10]}")

        # 1) 对齐索引
        # 以 features_df.index 为基准，只保留在 co_matrix 里出现过的
        names = [n for n in features_df.index if n in co_matrix_df.index]
        features_df = features_df.loc[names]
        co_matrix_df = co_matrix_df.loc[names, names]

        if len(names) < 2:
            print(f"[PRETRAIN] ⚠️ 可用塑料少于两个 (n={len(names)}), 跳过预训练。")
            return

        # 2) 构造训练对
        pairs = []
        X = features_df.values.astype("float32")
        for i, j in itertools.combinations(range(len(names)), 2):
            sim = co_matrix_df.iat[i, j]
            if pd.isna(sim):
                continue
            x1 = torch.from_numpy(X[i])
            x2 = torch.from_numpy(X[j])
            if loss_mode == "contrastive":
                y = torch.tensor(1.0 if float(sim) >= sim_threshold else 0.0, dtype=torch.float32)
            else:  # "mse"
                y = torch.tensor(float(sim), dtype=torch.float32)
            pairs.append((x1, x2, y))
        if not pairs:
            raise ValueError("没有构造出有效的塑料对")

        dl = DataLoader(_PairwiseDataset(pairs), batch_size=batch_size, shuffle=True)

        # 3) 设备、优化器、损失函数
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = _ContrastiveLoss(margin) if loss_mode == "contrastive" else nn.MSELoss()

        # 4) 训练循环
        self.train()
        for ep in range(1, epochs + 1):
            total_loss, n = 0.0, 0
            for x1, x2, y in dl:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                z1, z2 = self(x1), self(x2)

                if loss_mode == "contrastive":
                    loss = criterion(z1, z2, y)
                else:
                    cos = F.cosine_similarity(z1, z2, dim=-1)  # [-1,1]
                    cos_01 = (cos + 1.0) / 2.0                  # -> [0,1]
                    loss = criterion(cos_01, y)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total_loss += loss.item() * x1.size(0)
                n += x1.size(0)

            if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
                print(f"[PolymerTower pretrain] epoch={ep:03d} loss={total_loss / n:.6f}")

        self.eval()
        if verbose:
            print("[PolymerTower pretrain] ✅ 完成")