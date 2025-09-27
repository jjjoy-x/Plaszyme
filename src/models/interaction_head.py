# src/models/interaction_head.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionHead(nn.Module):
    """
    蛋白-塑料交互打分：
      - cos: 余弦相似
      - bilinear: s = z_e^T W z_p
      - factorized_bilinear: W≈U V^T（推荐）
      - hadamard_mlp: s = MLP([z_e ⊙ z_p, z_e, z_p])
      - gated: (σ(g_p(z_p))*z_e)·(σ(g_e(z_e))*z_p)
    输入：z_e:[1,D], z_p:[L,D]；输出：scores:[L]
    """
    def __init__(self, dim: int, mode: str = "factorized_bilinear",
                 rank: int = 64, mlp_hidden: int = 256):
        super().__init__()
        self.mode = mode
        d = dim
        if mode == "bilinear":
            self.W = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(self.W)
        elif mode == "factorized_bilinear":
            self.U = nn.Linear(d, rank, bias=False)
            self.V = nn.Linear(d, rank, bias=False)
            nn.init.xavier_uniform_(self.U.weight)
            nn.init.xavier_uniform_(self.V.weight)
        elif mode == "hadamard_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d * 3, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, 1)
            )
        elif mode == "gated":
            self.g_e = nn.Linear(d, d)
            self.g_p = nn.Linear(d, d)
            self.out = nn.Linear(d, 1, bias=False)
        elif mode == "cos":
            pass
        else:
            raise ValueError(f"Unknown interaction mode: {mode}")

    def orthogonal_regularizer(self) -> torch.Tensor:
        if self.mode != "factorized_bilinear":
            return next(self.parameters()).new_tensor(0.0)
        loss = 0.0
        for M in [self.U.weight, self.V.weight]:
            MMt = M @ M.t()
            I = torch.eye(MMt.size(0), device=MMt.device)
            loss = loss + (MMt - I).pow(2).mean()
        return loss

    def score(self, z_e: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        if self.mode == "cos":
            z1 = F.normalize(z_e, dim=-1)
            z2 = F.normalize(z_p, dim=-1)
            return (z1 @ z2.t()).squeeze(0)
        if self.mode == "bilinear":
            return (z_e @ self.W @ z_p.t()).squeeze(0)
        if self.mode == "factorized_bilinear":
            ue = self.U(z_e)          # [1,R]
            vp = self.V(z_p)          # [L,R]
            return (ue * vp).sum(dim=-1)
        if self.mode == "hadamard_mlp":
            L = z_p.size(0)
            ze_rep = z_e.expand(L, -1)
            feats = torch.cat([ze_rep * z_p, ze_rep, z_p], dim=-1)
            return self.mlp(feats).squeeze(-1)
        if self.mode == "gated":
            ge = torch.sigmoid(self.g_e(z_e))
            gp = torch.sigmoid(self.g_p(z_p))
            ze = ge * z_e
            zp = gp * z_p
            L = zp.size(0)
            ze_rep = ze.expand(L, -1)
            return self.out(ze_rep * zp).squeeze(-1)
        raise RuntimeError("unreachable")