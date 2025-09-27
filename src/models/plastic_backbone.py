import torch
import torch.nn as nn
import torch.nn.functional as F


class PolymerTower(nn.Module):
    """
    简单的塑料 MLP 塔
    - 输入: 描述符向量 (来自 PolymerFeaturizer) [B, in_dim]
    - 输出: L2 归一化后的嵌入向量 [B, out_dim]
    - 用途: 与酶塔输出做余弦相似度 (cosine similarity) 训练
    """

    def __init__(self, in_dim: int, hidden_dims=(512, 256), out_dim=128, dropout=0.1):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
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
        z = F.normalize(z, dim=-1)   # 单位化，便于余弦相似度
        return z