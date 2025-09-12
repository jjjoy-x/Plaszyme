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
        hidden_dims: Tuple[int,int] 或 List[Tuple[int,int]]，每层节点隐藏的标量/向量通道数
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
        hidden_dims=(128, 16),  # 可为 (S_h, V_h) 或 [(64,8),(96,12),(128,16)]
        n_layers: int = 3,      # 当 hidden_dims 是元组时生效；是列表时忽略
        out_dim: int = 128,
        dropout: float = 0.1,
        residue_logits: bool = False,
        *,
        activations: Tuple = (F.relu, None),
        vector_gate: bool = True,
        n_message: int = 3,
        n_feedforward: int = 2,
    ) -> None:
        super().__init__()
        self.dropout_p = float(dropout)
        self.residue_logits = bool(residue_logits)

        self.activations = activations
        self.vector_gate = bool(vector_gate)
        self.n_message = int(n_message)
        self.n_feedforward = int(n_feedforward)

        # 统一成 per-layer 维度列表 dims_list
        if isinstance(hidden_dims[0], int):
            # 单个元组 -> 重复 n_layers 次
            S_h, V_h = hidden_dims
            self.dims_list = [(S_h, V_h) for _ in range(int(n_layers))]
        else:
            # 列表[(S,V), (S,V), ...]
            self.dims_list = [tuple(map(int, t)) for t in hidden_dims]

        self.out_dim = int(out_dim)
        self._built = False
        self.input_proj = None
        self.blocks = None      # 交替 [opt_proj, conv, opt_proj, conv, ...]
        self.node_head = None
        self.graph_head = None
        self.dropout = nn.Dropout(self.dropout_p)

    # ---------- 内部：懒构建 ----------
    def _lazy_build(self, data: Batch) -> None:
        device = data.x_s.device

        # 自动推断输入与边维度
        S_in = int(data.x_s.size(-1))
        V_in = int(data.x_v.size(-2))
        Se = int(getattr(data, "edge_s", torch.zeros(data.edge_index.size(1), 0, device=device)).size(-1))
        Ve = int(getattr(data, "edge_v", torch.zeros(data.edge_index.size(1), 0, 3, device=device)).size(-2))

        # 输入投影到第一层维度
        S0, V0 = self.dims_list[0]
        if (S_in, V_in) != (S0, V0):
            self.input_proj = gvp_local.GVP(
                in_dims=(S_in, V_in),
                out_dims=(S0, V0),
                activations=self.activations,
                vector_gate=self.vector_gate,
            ).to(device)

        edge_dims = (Se, Ve)
        blocks = nn.ModuleList().to(device)
        cur = (S0, V0)

        for (S_h, V_h) in self.dims_list:
            # 若上一层输出与本层要求不一致，先插入过渡 GVP
            if cur != (S_h, V_h):
                blocks.append(
                    gvp_local.GVP(
                        in_dims=cur,
                        out_dims=(S_h, V_h),
                        activations=self.activations,
                        vector_gate=self.vector_gate,
                    ).to(device)
                )
                cur = (S_h, V_h)
            # 再放一层等维的 GVPConvLayer
            blocks.append(
                gvp_local.GVPConvLayer(
                    node_dims=(S_h, V_h),
                    edge_dims=edge_dims,
                    n_message=self.n_message,
                    n_feedforward=self.n_feedforward,
                    drop_rate=self.dropout_p,
                    activations=self.activations,
                    vector_gate=self.vector_gate,
                ).to(device)
            )
            cur = (S_h, V_h)

        self.blocks = blocks

        # 读出：最后一层的标量通道维度
        S_last, V_last = self.dims_list[-1]
        if self.residue_logits:
            self.node_head = nn.Linear(S_last, self.out_dim).to(device)
            self.graph_head = None
        else:
            self.node_head = None
            self.graph_head = nn.Sequential(
                nn.Linear(S_last, S_last),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(S_last, self.out_dim),
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

        # 依次通过 blocks
        for m in self.blocks:
            if isinstance(m, gvp_local.GVPConvLayer):
                s, v = m((s, v), data.edge_index, (e_s, e_v))
            else:  # 过渡 GVP
                s, v = m((s, v))

        if self.residue_logits:
            return self.node_head(s)

        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(s.size(0), dtype=torch.long, device=s.device)
        g = global_mean_pool(s, batch)
        return self.graph_head(g)