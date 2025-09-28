#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResidueRoleHead: 基于节点级图结构的残基角色类型分类头（role_type）。
- 依赖你的 GNNBackbone（强制 residue_logits=True）
- 前向输出 [N, C] 残基级 logits
- 提供 loss/predict/predict_proba 等便捷接口
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.plaszyme.models.gnn.backbone import GNNBackbone


class ResidueRoleHead(nn.Module):
    """
    残基级 role_type 分类头。

    Args:
        backbone_cfg: dict，用于构建 GNNBackbone，需包含：
            - conv_type: "gcn"|"gat"|"gatv2"|"sage"|"gin"|"gine"
            - hidden_dims: List[int]
            - dropout: float
            - 其他（如 gcn_edge_mode / gine_missing_edge_policy 等）可选
            ⚠️ 即使传入 residue_logits，本类也会忽略并强制设为 True
        num_classes: int，分类数（包含 "None" 类）
        freeze_backbone: 是否冻结骨干
        classifier_dims: 额外的 MLP 分类头隐藏层（可选），例如 [256]
    """

    def __init__(
        self,
        backbone_cfg: Dict,
        num_classes: int,
        freeze_backbone: bool = False,
        classifier_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        # 复制并剔除 residue_logits，避免“multiple values”冲突
        _bb_cfg = dict(backbone_cfg or {})
        if "residue_logits" in _bb_cfg:
            # 仅提示一次：以本头任务需求为准
            if _bb_cfg["residue_logits"] is False:
                # 如果外部显式给了 False，给出友好提示
                import warnings
                warnings.warn("[ResidueRoleHead] 忽略 backbone_cfg['residue_logits']=False，任务头已强制使用残基级输出。")
            _bb_cfg.pop("residue_logits", None)

        # 强制残基级 logits
        self.backbone = GNNBackbone(
            residue_logits=True,
            **_bb_cfg
        )

        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)

        # 分类头（延迟无需；直接用已知的 sum(hidden_dims)）
        self._feat_dim = sum(self.backbone.hidden_dims)
        self.classifier = self._make_classifier(self._feat_dim, self.num_classes, classifier_dims)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _make_classifier(in_dim: int, out_dim: int, hidden: Optional[List[int]] = None) -> nn.Module:
        """简单 MLP 分类头：可选若干隐藏层 + 最终线性到 num_classes。"""
        layers: List[nn.Module] = []
        prev = in_dim
        hidden = hidden or []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=0.2)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data: torch_geometric.data.Batch，
                  需包含 data.x, data.edge_index, (可选) data.edge_attr/edge_weight, (可选) data.batch
        Returns:
            logits: [N, C] 残基级 logits
        """
        feats = self.backbone(data)       # [N, feat]; residue_logits=True → 残基维度
        logits = self.classifier(feats)   # [N, C]
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        计算交叉熵损失。
        Args:
            logits: [N, C]
            target: [N] 取值范围 [0, C-1]；0 通常为 "None"
            class_weights: [C] 张量（可选）
            mask: [N] bool 张量，仅对 True 位置计算（可选）
        """
        if mask is not None:
            logits = logits[mask]
            target = target[mask]
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        return loss_fn(logits, target)

    @torch.no_grad()
    def predict(self, data: Batch) -> torch.Tensor:
        """返回类别索引 [N]"""
        logits = self.forward(data)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def predict_proba(self, data: Batch) -> torch.Tensor:
        """返回 softmax 概率 [N, C]"""
        logits = self.forward(data)
        return torch.softmax(logits, dim=-1)