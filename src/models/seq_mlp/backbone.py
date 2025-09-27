# src/models/seq_mlp/backbone.py
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

class SeqBackbone(nn.Module):
    """
    与 GNN/GVP 完全同接口的“非图”主干：
    - 仅使用残基级特征（优先 data.seq_x，其次 data.x）
    - 残基 -> 池化 -> 图级向量 -> MLP 映射到 out_dim
    - residue_logits=True 时返回残基级 [N, out_dim]

    Args:
        hidden_dims:  残基级编码 MLP 的层维度
        out_dim:      图级输出维度（与原脚本 EMB_DIM_ENZ 对齐）
        dropout:      Dropout 概率
        pool:         "mean" 或 "max"
        residue_logits: 是否输出残基级 logits（默认 False）
        in_dim:       若已知可指定；否则首个 batch 自动推断
        feature_priority: 从这些字段中按顺序取残基特征
    """
    def __init__(
        self,
        hidden_dims: List[int] = [256, 128],
        out_dim: int = 128,
        dropout: float = 0.1,
        pool: str = "mean",
        residue_logits: bool = False,
        *,
        in_dim: Optional[int] = None,
        feature_priority: Optional[List[str]] = None,  # e.g. ["seq_x", "x"]
    ):
        super().__init__()
        assert pool in ("mean", "max")
        self.hidden_dims = list(hidden_dims)
        self.out_dim = int(out_dim)
        self.dropout_p = float(dropout)
        self.pool = pool
        self.residue_logits = residue_logits
        self.in_dim_fixed = in_dim
        self.feature_priority = feature_priority or ["seq_x", "x"]

        # 懒构建
        self._built = False
        self._enc: Optional[nn.Sequential] = None
        self._node_head: Optional[nn.Linear] = None
        self._graph_head: Optional[nn.Sequential] = None

    # ---------- helpers ----------
    def _pick_feature(self, data: Batch) -> torch.Tensor:
        for k in self.feature_priority:
            v = getattr(data, k, None)
            if v is not None:
                return v.float()
        raise ValueError(
            f"[SeqBackbone] 未找到残基特征，已尝试字段 {self.feature_priority}。"
            "请在 builder 中把序列嵌入放到 data.seq_x（或 data.x）。"
        )

    def _get_batch(self, data: Batch, N: int, device: torch.device) -> torch.Tensor:
        b = getattr(data, "batch", None)
        if b is None:
            b = torch.zeros(N, dtype=torch.long, device=device)
        return b.to(device)

    def _lazy_build(self, in_dim: int, device: torch.device):
        # 残基级编码
        layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(self.dropout_p)]
            prev = h
        self._enc = nn.Sequential(*layers)

        # 残基级头（当 residue_logits=True 使用）
        self._node_head = nn.Linear(prev, self.out_dim)

        # 图级头
        self._graph_head = nn.Sequential(
            nn.Linear(prev, prev), nn.GELU(), nn.Dropout(self.dropout_p),
            nn.Linear(prev, self.out_dim)
        )

        self.to(device)
        self._built = True

    # ---------- forward ----------
    def forward(self, data: Batch) -> torch.Tensor:
        x = self._pick_feature(data)  # [N, Din]
        device = x.device
        N = x.size(0)
        batch = self._get_batch(data, N, device)

        if not self._built:
            in_dim = int(self.in_dim_fixed or x.size(-1))
            self._lazy_build(in_dim, device)

        h = self._enc(x)  # [N, H]

        if self.residue_logits:
            return self._node_head(h)  # [N, out_dim]

        if self.pool == "mean":
            hg = global_mean_pool(h, batch)  # [B, H]
        else:
            hg = global_max_pool(h, batch)   # [B, H]

        return self._graph_head(hg)          # [B, out_dim]