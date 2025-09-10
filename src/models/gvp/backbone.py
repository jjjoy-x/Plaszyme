# src/models/gvp_local/backbone_official.py
# -*- coding: utf-8 -*-
"""
GVPBackbone（官方 gvp_local 模块拼装版）
--------------------------------
依赖: drorlab/gvp_local-pytorch（pip install gvp_local-pytorch 或本地安装）
与项目现有 builder 对齐的数据契约：
  data.x_s: [N, S_in]        标量节点特征
  data.x_v: [N, V_in, 3]     向量节点特征
  data.edge_s: [E, Se]       标量边特征（可选）
  data.edge_v: [E, Ve, 3]    向量边特征（可选）
  data.edge_index: [2, E]
  data.batch: [N] （可选；无则视为单图或自动全 0）

API 对齐：
  - hidden_dims: Tuple[int,int]  每层节点隐藏维度 (S_hidden, V_hidden)
  - n_layers: int                叠多少层 GVPConvLayer（该层残差更新，节点维度固定）
  - out_dim: int                 输出维数（图级或残基级）
  - dropout: float               Dropout 概率
  - residue_logits: bool         True→残基级 [N,out_dim]；False→图级 [B,out_dim]
  - activations, vector_gate, n_message, n_feedforward: 直接透传给官方层
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.models import gvp_local


class GVPBackbone(nn.Module):
    """
    使用官方 gvp_local.GVPConvLayer 堆叠的主干。

    Args:
        hidden_dims: (S_hidden, V_hidden)，每层节点隐藏的标量/向量通道数
        n_layers: 堆叠多少个 GVPConvLayer（残差层，节点维不变）
        out_dim: 输出维度（图级或残基级）
        dropout: Dropout 概率
        residue_logits: True 则输出 [N,out_dim]；否则图级 [B,out_dim]
        activations: (scalar_act, vector_act)，默认 (ReLU, None) 与论文一致
        vector_gate: 是否启用 vector gating（论文中效果更好）
        n_message: 每层消息网络 GVP 个数（官方层超参），默认 3
        n_feedforward: 每层前馈网络 GVP 个数（官方层超参），默认 2
    """
    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (128, 16),
        n_layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
        residue_logits: bool = False,
        *,
        activations: Tuple = (F.relu, None),
        vector_gate: bool = True,
        n_message: int = 3,
        n_feedforward: int = 2,
    ) -> None:
        super().__init__()
        self.S_h, self.V_h = hidden_dims
        self.n_layers = int(n_layers)
        self.out_dim = int(out_dim)
        self.dropout_p = float(dropout)
        self.residue_logits = bool(residue_logits)

        self.activations = activations
        self.vector_gate = bool(vector_gate)
        self.n_message = int(n_message)
        self.n_feedforward = int(n_feedforward)

        # 懒构建：首个 batch 自动探测 (S_in,V_in) 和 (Se,Ve)
        self._built = False

        # 占位：实际在 _lazy_build 中创建
        self.input_proj: Optional[nn.Module] = None
        self.layers: Optional[nn.ModuleList] = None
        self.node_head: Optional[nn.Module] = None
        self.graph_head: Optional[nn.Module] = None

        self.dropout = nn.Dropout(self.dropout_p)

    # ---------- 内部：懒构建 ----------
    def _lazy_build(self, data: Batch) -> None:
        device = data.x_s.device

        # 自动推断输入与边维度
        S_in = int(data.x_s.size(-1))
        V_in = int(data.x_v.size(-2))
        Se = int(getattr(data, "edge_s", torch.zeros(data.edge_index.size(1), 0, device=device)).size(-1))
        if hasattr(data, "edge_v") and data.edge_v is not None:
            Ve = int(data.edge_v.size(-2))
        else:
            Ve = 0

        node_in_dims = (S_in, V_in)
        edge_in_dims = (Se, Ve)
        node_hidden_dims = (self.S_h, self.V_h)

        # 1) 可选的输入投影（把 (S_in,V_in) → (S_h,V_h)）
        if (S_in, V_in) != node_hidden_dims:
            self.input_proj = gvp_local.GVP(
                in_dims=node_in_dims,
                out_dims=node_hidden_dims,
                activations=self.activations,
                vector_gate=self.vector_gate,
            ).to(device)
        else:
            self.input_proj = None

        # 2) 堆叠 n_layers 个官方 GVPConvLayer（每层输入输出维一致）
        layers = []
        for _ in range(self.n_layers):
            layers.append(
                gvp_local.GVPConvLayer(
                    node_dims=node_hidden_dims,
                    edge_dims=edge_in_dims,
                    n_message=self.n_message,
                    n_feedforward=self.n_feedforward,
                    drop_rate=self.dropout_p,
                    activations=self.activations,
                    vector_gate=self.vector_gate,
                )
            )
        self.layers = nn.ModuleList(layers).to(device)

        # 3) 读出头：只用标量通道（旋转不变）
        if self.residue_logits:
            self.node_head = nn.Linear(self.S_h, self.out_dim).to(device)
            self.graph_head = None
        else:
            # 简单 MLP：S_h → S_h → out_dim
            self.node_head = None
            self.graph_head = nn.Sequential(
                nn.Linear(self.S_h, self.S_h),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.S_h, self.out_dim),
            ).to(device)

        self._built = True

    # ---------- 前向 ----------
    def forward(self, data: Batch) -> torch.Tensor:
        """
        期望输入（与 builder 对齐）:
            data.x_s: [N, S_in], data.x_v: [N, V_in, 3]
            data.edge_s: [E, Se] or None
            data.edge_v: [E, Ve, 3] or None
            data.edge_index: [2, E]
            data.batch: [N] (optional)

        Returns:
            - 残基级: [N, out_dim]
            - 图  级: [B, out_dim]
        """
        # 懒构建
        if not self._built:
            self._lazy_build(data)

        s = data.x_s
        v = data.x_v
        edge_index = data.edge_index.to(s.device)
        e_s = getattr(data, "edge_s", None)
        e_v = getattr(data, "edge_v", None)

        # 缺省边特征 → 0 维占位（与官方层兼容）
        if e_s is None:
            e_s = torch.zeros(edge_index.size(1), 0, dtype=s.dtype, device=s.device)
        if e_v is None:
            e_v = torch.zeros(edge_index.size(1), 0, 3, dtype=s.dtype, device=s.device)

        # 可选输入投影
        if self.input_proj is not None:
            s, v = self.input_proj((s, v))

        # 堆叠 GVPConvLayer
        h_s, h_v = s, v
        assert self.layers is not None
        for layer in self.layers:
            h_s, h_v = layer((h_s, h_v), edge_index, (e_s, e_v))

        # 残基级或图级读出（只用标量通道）
        if self.residue_logits:
            assert self.node_head is not None
            return self.node_head(h_s)

        # 图级：batch 不存在则默认全 0（单图）
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(h_s.device)
        else:
            batch = torch.zeros(h_s.size(0), dtype=torch.long, device=h_s.device)

        assert self.graph_head is not None
        g = global_mean_pool(h_s, batch)
        g = self.graph_head(g)
        return g