#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.plastic_backbone import PolymerTower  # 塑料塔

# ===== 配置 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

PAIR_CSV = "/path/to/pairs.csv"     # 必含: pdb_id, chain, plastic_path, label
PDB_DIR  = "/root/autodl-tmp/M-CSA/pdbs"

# Backbone 直接做图级输出
EMB_DIM  = 256            # GNNBackbone 的 out_dim（图级）
RESIDUE_LOGITS = False    # !!! 图级
HIDDENS = [128, 128]
DROPOUT = 0.10

# 双塔公共维度
PROJ_OUT_DIM   = 128
PLASTIC_IN_DIM = 404      # 你的 featurizer 实际输出（示例）

# 训练
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 3e-4
WD         = 1e-4
NUM_WORKERS= 2
VAL_SPLIT  = 0.1
MAX_NORM   = 5.0
USE_AMP    = True

# 损失
ALPHA_POINT = 1.0
BETA_PAIR   = 1.0
PAIRWISE_LOSS = "margin"   # "margin" | "infoNCE"
MARGIN = 0.2
TAU    = 0.07

def set_seed(seed: int = 42):
    import numpy as np, random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# --- Losses ---
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)

def pointwise_loss(z_e, z_p, labels):
    sim = cosine_sim(z_e, z_p)
    pos = (labels == 1); neg = (labels == -1)
    loss, n = 0.0, 0
    if pos.any():
        loss = loss + (1.0 - sim[pos]).mean(); n += 1
    if neg.any():
        loss = loss + torch.clamp(sim[neg] + 0.1, min=0.0).mean(); n += 1
    return loss / max(n, 1)

def pairwise_margin_loss(z_e, z_p, labels, margin: float):
    B = z_e.size(0)
    sims = z_e @ z_p.t()  # [B,B]
    pos_idx = torch.where(labels == 1)[0]
    if pos_idx.numel() == 0:
        return sims.new_zeros(())
    loss, cnt = 0.0, 0
    for i in pos_idx.tolist():
        pos_score = sims[i, i]
        neg_scores = torch.cat([sims[i, :i], sims[i, i+1:]], dim=0)
        loss = loss + torch.clamp(margin + neg_scores - pos_score, min=0.0).mean()
        cnt += 1
    return loss / max(cnt, 1)

def info_nce_loss(z_e, z_p, tau: float):
    sims = (z_e @ z_p.t()) / tau
    labels = torch.arange(sims.size(0), device=sims.device)
    ce = nn.CrossEntropyLoss()
    return 0.5 * (ce(sims, labels) + ce(sims.t(), labels))

# --- 模型 ---
class EnzymeTower(nn.Module):
    """图级骨干 + 投影 + L2"""
    def __init__(self, out_dim: int = PROJ_OUT_DIM):
        super().__init__()
        self.backbone = GNNBackbone(
            conv_type="gine",                # 根据你的图边特征；无边特征可改 "gine"->"gine/none"
            hidden_dims=HIDDENS,
            out_dim=EMB_DIM,                 # 图级输出维
            dropout=DROPOUT,
            residue_logits=RESIDUE_LOGITS,   # 图级
            gine_missing_edge_policy="zeros" # 无 edge_attr 时占位
        )
        self.proj = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(EMB_DIM, out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        g = self.backbone(batch)        # [B, EMB_DIM] 图级
        z = self.proj(g)                # [B, PROJ_OUT_DIM]
        return l2norm(z)

class PlasticTower(nn.Module):
    """塑料 MLP 塔 + L2"""
    def __init__(self, in_dim: int, out_dim: int = PROJ_OUT_DIM):
        super().__init__()
        self.tower = PolymerTower(in_dim=in_dim, hidden_dims=(512, 256), out_dim=out_dim)
    def forward(self, x):
        return l2norm(self.tower(x))

# --- 一个 step ---
@torch.no_grad()
def _sim_stats(z_e, z_p, labels):
    s = cosine_sim(z_e, z_p)
    pos = s[labels == 1]
    neg = s[labels == -1]
    return (pos.mean().item() if pos.numel() else 0.0,
            neg.mean().item() if neg.numel() else 0.0)

def train_or_eval(loader, enz, pla, optimizer=None, scaler=None) -> Dict[str, float]:
    train = optimizer is not None
    enz.train(train); pla.train(train)

    tot = {"loss":0.0, "point":0.0, "pair":0.0, "sim_pos":0.0, "sim_neg":0.0, "n":0}
    for batch_graph, plast_x, labels in loader:
        batch_graph = batch_graph.to(DEVICE)
        plast_x = plast_x.to(DEVICE)
        labels  = labels.to(DEVICE)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (USE_AMP and torch.cuda.is_available()) else torch.cpu.amp.autocast(enabled=False)
        with amp_ctx:
            z_e = enz(batch_graph)
            z_p = pla(plast_x)

            l_point = pointwise_loss(z_e, z_p, labels) * ALPHA_POINT
            if PAIRWISE_LOSS == "margin":
                l_pair  = pairwise_margin_loss(z_e, z_p, labels, MARGIN) * BETA_PAIR
            else:
                l_pair  = info_nce_loss(z_e, z_p, TAU) * BETA_PAIR
            loss = l_point + l_pair

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(list(enz.parameters()) + list(pla.parameters()), MAX_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(list(enz.parameters()) + list(pla.parameters()), MAX_NORM)
                optimizer.step()

        sp, sn = _sim_stats(z_e.detach(), z_p.detach(), labels)
        bs = labels.size(0)
        tot["loss"]   += float(loss.item()) * bs
        tot["point"]  += float(l_point.item()) * bs
        tot["pair"]   += float(l_pair.item()) * bs
        tot["sim_pos"]+= sp * bs
        tot["sim_neg"]+= sn * bs
        tot["n"]      += bs

    n = max(tot["n"], 1)
    return {k: tot[k]/n for k in ["loss","point","pair","sim_pos","sim_neg"]} | {"n": n}

def main():
    set_seed(SEED)

    # 数据
    full = PairedPlaszymeDataset(
        csv_path=PAIR_CSV, pdb_dir=PDB_DIR,
        builder_type="gnn", radius=10.0,
        embedder_cfg={"name":"onehot"}
    )
    n = len(full)
    idx = list(range(n)); random.shuffle(idx)
    n_val = max(1, int(n * VAL_SPLIT))
    val_ids = set(idx[:n_val])
    train_ids = [i for i in idx if i not in val_ids]
    val_ids   = [i for i in idx if i in val_ids]

    train_loader = DataLoader(
        torch.utils.data.Subset(full, train_ids),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_pairs
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full, val_ids),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_pairs
    )

    # 模型
    enz = EnzymeTower(out_dim=PROJ_OUT_DIM).to(DEVICE)
    pla = PlasticTower(in_dim=PLASTIC_IN_DIM, out_dim=PROJ_OUT_DIM).to(DEVICE)

    optim = torch.optim.AdamW(list(enz.parameters()) + list(pla.parameters()), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and torch.cuda.is_available())

    print(f"[INFO] train={len(train_ids)} | val={len(val_ids)} | device={DEVICE} | pairwise={PAIRWISE_LOSS}")

    best = 1e9
    for ep in range(1, EPOCHS+1):
        tr = train_or_eval(train_loader, enz, pla, optim, scaler)
        vl = train_or_eval(val_loader, enz, pla, None, None)
        sched.step(vl["loss"])
        print(f"[Epoch {ep:02d}] "
              f"train loss={tr['loss']:.4f} (pt={tr['point']:.4f}, pw={tr['pair']:.4f}, s+={tr['sim_pos']:.3f}, s-={tr['sim_neg']:.3f}, n={tr['n']}) | "
              f"val loss={vl['loss']:.4f} (pt={vl['point']:.4f}, pw={vl['pair']:.4f}, s+={vl['sim_pos']:.3f}, s-={vl['sim_neg']:.3f}, n={vl['n']})")

        if vl["loss"] < best:
            best = vl["loss"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": ep,
                "enzyme_state": enz.state_dict(),
                "plastic_state": pla.state_dict(),
                "cfg": {
                    "EMB_DIM": EMB_DIM,
                    "PROJ_OUT_DIM": PROJ_OUT_DIM,
                    "PAIRWISE_LOSS": PAIRWISE_LOSS,
                    "MARGIN": MARGIN, "TAU": TAU
                }
            }, "checkpoints/dual_tower_best.pt")
            print(f"[INFO] saved best @ epoch {ep} (val_loss={best:.4f})")
    print("[DONE]")

if __name__ == "__main__":
    main()