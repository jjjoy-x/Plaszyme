# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidueActivityHead(nn.Module):
    """
    残基活性 4 分类任务头（reactant / interaction / spectator / none）。
    - 兼容: 仅推理 (返回 logits)；或 训练 (返回 logits 和 {loss, metrics...})
    - 内置: 类别权重、label smoothing、ignore_index、mask、none 类下采样、可选 focal loss、指标计算

    Args:
        in_dim:        输入特征维度（骨干输出的残基嵌入）
        num_classes:   类别数（默认 4）
        dropout:       Dropout 概率
        hidden:        None=线性头；int=1 层 MLP 的隐藏维度
        use_batchnorm: 隐藏层后是否接 BatchNorm1d
        # 损失相关
        class_weights: 传入固定类别权重（Tensor[num_classes]）；若为 None 且 dynamic_class_weights=True，则按当前 batch 统计
        label_smoothing: 交叉熵的 label smoothing 系数
        ignore_index:  用于忽略的标签 id（mask/下采样会把不参与的样本改成该值）
        loss_type:     "ce" | "focal"
        focal_gamma:   focal loss 的 gamma（>0 更聚焦难样本）
        focal_alpha:   focal 的 alpha（None 或 float / Tensor[num_classes]）
        # 训练便利项
        compute_metrics:   是否在 forward(y, ...) 时计算 acc / macro_f1 / per-class recall
        none_keep_ratio:   None 关闭；否则对标签==none_class_id 的样本按比例随机保留（只影响损失/指标的记账）
        none_class_id:     “none/负类”的类别 id（默认 3）
        dynamic_class_weights: 当 class_weights=None 时，是否用当前 batch 的频次反比来动态估算权重
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 4,
        dropout: float = 0.1,
        hidden: Optional[int] = None,
        use_batchnorm: bool = False,
        *,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float | torch.Tensor] = None,
        compute_metrics: bool = True,
        none_keep_ratio: Optional[float] = None,
        none_class_id: int = 3,
        dynamic_class_weights: bool = False,
    ) -> None:
        super().__init__()
        assert in_dim > 0 and num_classes > 0
        assert loss_type in {"ce", "focal"}

        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden) if hidden is not None else None
        self.use_batchnorm = bool(use_batchnorm)
        self.dropout_p = float(dropout)

        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = focal_alpha  # 可为 None/float/Tensor
        self.compute_metrics = bool(compute_metrics)

        self.none_keep_ratio = none_keep_ratio if none_keep_ratio is None else float(none_keep_ratio)
        self.none_class_id = int(none_class_id)
        self.dynamic_class_weights = bool(dynamic_class_weights)

        # 分类器
        if self.hidden is None:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(self.in_dim, self.num_classes),
            )
        else:
            layers = [nn.Linear(self.in_dim, self.hidden)]
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden))
            layers += [nn.ReLU(inplace=True), nn.Dropout(self.dropout_p),
                       nn.Linear(self.hidden, self.num_classes)]
            self.classifier = nn.Sequential(*layers)

        # 预置 CE 的静态参数
        self._ce_kwargs = {
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
        }
        # 类别权重缓存在 buffer 里，方便移到 device
        self.register_buffer("_class_weights",
                             class_weights if class_weights is not None else None,
                             persistent=False)

    # ------------ 公共 API ------------
    def forward(
        self,
        h: torch.Tensor,                   # [N, S_h]
        y: Optional[torch.Tensor] = None,  # [N] long
        mask: Optional[torch.Tensor] = None,   # [N] bool
    ):
        assert h.dim() == 2, f"ResidueActivityHead expects [N, S_h], got {tuple(h.shape)}"
        logits = self.classifier(h)

        # 纯推理：只要 logits
        if y is None:
            return logits

        # 训练/评估：准备标签与 mask
        y = y.long()
        eff_mask = self._make_effective_mask(y, mask)  # [N] bool 或 None

        # 准备权重
        weight = self._resolve_class_weights(y, eff_mask, logits.device)

        # 计算损失
        if self.loss_type == "ce":
            loss = self._cross_entropy_with_mask(logits, y, eff_mask, weight)
        else:
            loss = self._focal_loss_with_mask(logits, y, eff_mask, weight,
                                              gamma=self.focal_gamma, alpha=self.focal_alpha)

        # 指标（可关）
        out = {"loss": loss}
        if self.compute_metrics:
            metrics = self._compute_metrics(logits.detach(), y.detach(), eff_mask)
            out.update(metrics)

        return logits, out

    # ------------ 内部：mask 与下采样 ------------
    def _make_effective_mask(
        self, y: torch.Tensor, user_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        合成最终 mask：
        - 若提供 user_mask：以其为基础
        - 若配置 none_keep_ratio：对 none 类随机丢弃一部分（仅参与损失/指标的统计）
        - 返回 None 表示“全量参与”
        """
        if user_mask is not None:
            eff = user_mask.bool().clone()
        else:
            eff = torch.ones_like(y, dtype=torch.bool)

        if self.none_keep_ratio is not None:
            none_mask = (y == self.none_class_id) & eff
            idx = torch.nonzero(none_mask, as_tuple=False).flatten()
            if idx.numel() > 0:
                k = max(1, int(idx.numel() * float(self.none_keep_ratio)))
                perm = torch.randperm(idx.numel(), device=idx.device)[:k]
                keep_idx = idx[perm]
                # 将 none 类中未采样到的置为 False
                eff[none_mask] = False
                eff[keep_idx] = True

        if eff.all():
            return None
        return eff

    # ------------ 内部：损失 ------------
    def _cross_entropy_with_mask(
        self,
        logits: torch.Tensor, y: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return F.cross_entropy(logits, y, weight=weight, **self._ce_kwargs)
        # 把不参与的样本改成 ignore_index（与 CE 对齐）
        y_eff = y.clone()
        y_eff[~mask] = self.ignore_index
        return F.cross_entropy(logits, y_eff, weight=weight, **self._ce_kwargs)

    def _focal_loss_with_mask(
        self,
        logits: torch.Tensor, y: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        gamma: float = 2.0,
        alpha: Optional[float | torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        多分类 focal loss（对 CE 的一般化）。支持：
        - 类别 weight（与 CE 一致）
        - alpha（float/Tensor[num_classes]，进一步调正负/类别不均衡）
        - mask/ignore_index
        """
        # 过滤 ignore/mask
        if mask is None:
            valid = (y != self.ignore_index)
        else:
            valid = mask & (y != self.ignore_index)

        if valid.sum() == 0:
            return logits.new_tensor(0.0, requires_grad=True)

        logits = logits[valid]
        y = y[valid]

        logp = F.log_softmax(logits, dim=-1)            # [M, C]
        p = logp.exp()                                   # [M, C]
        # 取对应类别概率
        pt = p.gather(1, y.view(-1, 1)).squeeze(1)      # [M]
        logpt = logp.gather(1, y.view(-1, 1)).squeeze(1)

        # 基础 CE：-logpt
        ce = -logpt                                      # [M]

        # 类别权重（与 CE 相同语义）
        if weight is not None:
            w = weight.gather(0, y)                      # [M]
            ce = ce * w

        # alpha 修正
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                a = alpha.to(logits.device).gather(0, y) # [M]
            else:
                a = logits.new_full((y.numel(),), float(alpha))
            ce = a * ce

        # focal 调制
        loss = ((1 - pt) ** gamma) * ce
        return loss.mean()

    # ------------ 内部：类别权重 ------------
    def _resolve_class_weights(
        self, y: torch.Tensor, mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        优先级：
          1) 明确传入的 _class_weights
          2) dynamic_class_weights=True 时，用 batch 频次的反比作为权重
          3) 否则 None
        """
        if self._class_weights is not None:
            return self._class_weights.to(device)

        if not self.dynamic_class_weights:
            return None

        # 仅统计有效样本
        valid = (y != self.ignore_index)
        if mask is not None:
            valid = valid & mask
        if valid.sum() == 0:
            return None

        yv = y[valid]
        bins = torch.bincount(yv, minlength=self.num_classes).float() + 1e-6
        inv = 1.0 / bins
        w = inv / inv.sum() * self.num_classes  # 归一化到均值≈1
        return w.to(device)

    # ------------ 内部：指标 ------------
    @torch.no_grad()
    def _compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        if mask is None:
            valid = (y != self.ignore_index)
        else:
            valid = mask & (y != self.ignore_index)

        n = int(valid.sum().item())
        if n == 0:
            return {"acc": 0.0, "macro_f1": 0.0}

        logits = logits[valid]
        y = y[valid]
        pred = logits.argmax(dim=-1)

        acc = (pred == y).float().mean().item()

        # macro-F1（纯 torch 版，避免依赖 sklearn）
        f1 = self._macro_f1(pred, y, num_classes=self.num_classes)

        # 也给出 per-class recall（便于观察正类是否被学到）
        rec = self._per_class_recall(pred, y, num_classes=self.num_classes)
        out = {"acc": float(acc), "macro_f1": float(f1)}
        for c, r in enumerate(rec):
            out[f"recall_c{c}"] = float(r)
        return out

    @staticmethod
    def _macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
        f1s = []
        for c in range(num_classes):
            tp = ((pred == c) & (y == c)).sum().item()
            fp = ((pred == c) & (y != c)).sum().item()
            fn = ((pred != c) & (y == c)).sum().item()
            denom_p = tp + fp
            denom_r = tp + fn
            if denom_p == 0 or denom_r == 0:
                f1s.append(0.0)
            else:
                p = tp / denom_p
                r = tp / denom_r
                f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
        return float(sum(f1s) / len(f1s))

    @staticmethod
    def _per_class_recall(pred: torch.Tensor, y: torch.Tensor, num_classes: int):
        rec = []
        for c in range(num_classes):
            tp = ((pred == c) & (y == c)).sum().item()
            pos = (y == c).sum().item()
            rec.append(0.0 if pos == 0 else tp / pos)
        return rec