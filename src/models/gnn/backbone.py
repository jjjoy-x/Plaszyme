# src/models/gnn/backbone.py  —— 修改后的 GNNBackbone

from __future__ import annotations
import warnings
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv, GATConv, GATv2Conv, SAGEConv, GINConv, GINEConv, global_mean_pool
)
from torch_geometric.data import Batch


def _edge_weight_from_attr(edge_attr: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """Convert edge_attr ([E] or [E,D]) to GCN's edge_weight ([E]).
    将 edge_attr 转成 GCN 所需 edge_weight（尽量鲁棒，不假设具体语义）。

    mode:
      - "auto": 均值到 [E]，min-max 归一化后翻转（“更近→更大权重”）
      - "mean_inv": w = 1 / (mean + eps)
      - "first_inv": w = 1 / (col0 + eps)
    """
    eps = 1e-8
    if edge_attr.dim() == 1:
        w = edge_attr
    else:
        if edge_attr.size(-1) == 1:
            w = edge_attr.squeeze(-1)
        else:
            if mode == "first_inv":
                col0 = edge_attr[:, 0]
                w = 1.0 / (col0 + eps)
            elif mode == "mean_inv":
                mean = edge_attr.mean(dim=-1)
                w = 1.0 / (mean + eps)
            else:
                m = edge_attr.mean(dim=-1)  # [E]
                m_min, m_max = m.min(), m.max()
                if (m_max - m_min) > eps:
                    m_norm = (m - m_min) / (m_max - m_min + eps)
                    w = 1.0 - m_norm
                else:
                    w = torch.ones_like(m)
    return torch.clamp(w, min=0.0)


class GNNBackbone(nn.Module):
    """
    通用 GNN 主干（GCN/GAT/GATv2/SAGE/GIN/GINE），兼容“有/无边特征”。

    Args:
        conv_type: "gcn" | "gat" | "gatv2" | "sage" | "gin" | "gine"
        hidden_dims: List[int]    每层隐藏维度（节点通道数）
        out_dim: int              最终输出维度
        dropout: float            Dropout 概率
        residue_logits: bool      True→残基级 [N,out_dim]；False→图级 [B,out_dim]
        gcn_edge_mode: str        GCN 的 edge_attr→edge_weight 策略："auto"|"mean_inv"|"first_inv"
        gine_missing_edge_policy: 当 conv_type="gine" 且缺 edge_attr：
                                  - "error": 抛错
                                  - "zeros": 用全零占位（edge_dim=1）
    """

    def __init__(
        self,
        conv_type: str,
        hidden_dims: List[int] | None = None,
        out_dim: int = 1,
        dropout: float = 0.3,
        residue_logits: bool = False,
        *,
        gcn_edge_mode: str = "auto",
        gine_missing_edge_policy: str = "error",
        **legacy_kwargs,  # 兼容历史参数
    ):
        super().__init__()
        self.conv_type = conv_type.lower()

        # 兼容旧参数 dims=
        if hidden_dims is None and "dims" in legacy_kwargs:
            warnings.warn("[GNNBackbone] `dims=` 已弃用，请改用 `hidden_dims=`；本次沿用 `dims` 的值。", stacklevel=2)
            hidden_dims = legacy_kwargs.pop("dims")
        if hidden_dims is None:
            raise ValueError("`hidden_dims` 不能为空（示例：[64,64]）。")

        self.hidden_dims = list(hidden_dims)
        self.out_dim = out_dim
        self.dropout_p = dropout
        self.residue_logits = residue_logits
        self.gcn_edge_mode = gcn_edge_mode
        self.gine_missing_edge_policy = gine_missing_edge_policy

        # 懒构建：首个 batch 自动获知 in_dim / edge_dim
        self._built = False
        self._edge_dim: Optional[int] = None

        # 告警开关：避免每个 batch 重复打印
        self._warned_edge_ignored = False
        self._warned_edge_v_ignored = False

    # ---- layer builders ----
    def _make_conv(self, in_dim: int, out_dim: int, edge_dim: Optional[int]) -> nn.Module:
        """Construct one conv layer with the chosen operator."""
        ct = self.conv_type
        if ct == "gcn":
            return GCNConv(in_dim, out_dim, normalize=True, add_self_loops=True)
        if ct == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        if ct == "gatv2":
            return GATv2Conv(in_dim, out_dim, heads=1, concat=False)
        if ct == "sage":
            return SAGEConv(in_dim, out_dim)
        if ct == "gin":
            return GINConv(nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            ))
        if ct == "gine":
            if edge_dim is None:
                raise ValueError("GINEConv requires 'edge_dim' at layer build time.")
            # GINE 的 nn 是作用在节点上的 MLP（消息聚合后）；edge_attr 维度通过 edge_dim 传入构造
            nn_node = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn_node, train_eps=True, edge_dim=edge_dim)
        raise ValueError(f"Unsupported conv_type: {ct}")

    def _build_layers(self, in_dim: int, edge_dim: Optional[int]) -> None:
        layers = []
        prev = in_dim
        for d in self.hidden_dims:
            layers.append(self._make_conv(prev, d, edge_dim if self.conv_type == "gine" else None))
            prev = d
        self.convs = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        # 读出：将所有层输出 concat 再线性映射
        self.readout = nn.Linear(sum(self.hidden_dims), self.out_dim)
        self._edge_dim = edge_dim
        self._built = True

    # ---- input prep ----
    def _prepare_inputs(
        self, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """统一取出 x / edge_index / edge_attr / edge_weight / batch，按算子策略处理。"""
        x = data.x
        device = x.device

        edge_index = data.edge_index.to(device)
        edge_attr = getattr(data, "edge_attr", None)
        edge_weight = getattr(data, "edge_weight", None)
        edge_v = getattr(data, "edge_v", None)

        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # 仅提示一次：纯 GNN 不使用向量边
        if edge_v is not None and not self._warned_edge_v_ignored:
            warnings.warn("[GNNBackbone] 'edge_v' detected but ignored by non-GVP backbones.")
            self._warned_edge_v_ignored = True

        ct = self.conv_type

        # GCN：尝试把 edge_attr 转成 edge_weight
        if ct == "gcn":
            if edge_weight is None and edge_attr is not None:
                try:
                    ew = _edge_weight_from_attr(edge_attr, mode=self.gcn_edge_mode).to(device)
                    if not torch.is_floating_point(ew):
                        ew = ew.float()
                    edge_weight = ew
                except Exception:
                    if not self._warned_edge_ignored:
                        warnings.warn("[GNNBackbone][GCN] edge_attr→edge_weight 转换失败，退回二值邻接。")
                        self._warned_edge_ignored = True
                    edge_weight = None

        # 不支持边特征的算子：忽略 edge_attr
        if ct in {"gat", "gatv2", "sage", "gin"}:
            if edge_attr is not None and not self._warned_edge_ignored:
                warnings.warn(f"[GNNBackbone][{ct.upper()}] edge_attr 已提供但会被忽略。")
                self._warned_edge_ignored = True
            edge_attr = None

        # GINE：必须有 edge_attr（或 zeros 策略）
        if ct == "gine":
            if edge_attr is None and self.gine_missing_edge_policy != "zeros":
                raise ValueError("[GNNBackbone][GINE] 需要 edge_attr；或设置 gine_missing_edge_policy='zeros'。")

        # batch（没有则全 0）
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        else:
            batch = batch.to(device)

        return x, edge_index, edge_attr, edge_weight, batch

    # ---- forward ----
    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, edge_weight, batch = self._prepare_inputs(data)

        # 懒构建：自动推断输入维度 & GINE 的 edge_dim
        if not self._built:
            in_dim = x.size(-1)
            edge_dim = None
            if self.conv_type == "gine":
                if edge_attr is not None:
                    edge_dim = edge_attr.size(-1) if edge_attr.dim() == 2 else 1
                elif self.gine_missing_edge_policy == "zeros":
                    edge_dim = 1  # 用 1 维占位
                else:
                    raise RuntimeError("[GNNBackbone][GINE] 无法推断 edge_dim。")
            self._build_layers(in_dim, edge_dim)
            self.to(x.device)

        # 若 GINE 缺 edge_attr 且选择 zeros，占位构造 [E, edge_dim] 全零
        if self.conv_type == "gine" and edge_attr is None and self.gine_missing_edge_policy == "zeros":
            E = edge_index.size(1)
            edge_attr = torch.zeros(E, self._edge_dim or 1, device=x.device)

        # GNN 堆叠
        outs = []
        h = x
        for conv in self.convs:
            if isinstance(conv, GCNConv):
                h = conv(h, edge_index, edge_weight=edge_weight)
            elif isinstance(conv, GINEConv):
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
            outs.append(h)

        # 拼接所有层的输出（残基维度不变）
        h_cat = torch.cat(outs, dim=-1)  # [N, sum(hidden_dims)]

        # 残基级输出：不做 pooling
        if self.residue_logits:
            return self.readout(h_cat)  # [N, out_dim]

        # 图级输出：pooling 后线性
        g = global_mean_pool(h_cat, batch)  # [B, sum(hidden_dims)]
        return self.readout(g)              # [B, out_dim]