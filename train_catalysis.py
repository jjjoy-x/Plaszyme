#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四分类残基催化训练（严格按链）
CSV 列：PDB ID, CHAIN ID, RESIDUE NUMBER, ROLE_TYPE（reactant/interaction/spectator）
未出现在 CSV 的残基 → 统一记为 none

修复点：
- 屏蔽 Bio.PDB 非标准残基告警，且 builder 仅纳入标准氨基酸
- 训练下采样 none；验证禁用下采样
- 打印人均 loss（不再上百万）
- AMP 使用 torch.amp 新接口
"""
from __future__ import annotations
# ========== 1) 屏蔽 Bio.PDB 告警(必须在任何 import 前) ==========
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore:Assuming residue .* is an unknown modified amino acid:UserWarning"
warnings.filterwarnings(
    "ignore",
    message=r"Assuming residue .* is an unknown modified amino acid",
    category=UserWarning,
    module=r"Bio\.PDB\.Polypeptide"
)


import sys, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ====== 路径修正 ======
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent
for p in [str(_REPO), str(_REPO / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ====== 项目内模块（按你的仓库结构） ======
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.builders.gnn_builder  import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.models.gvp.backbone   import GVPBackbone
from src.models.gnn.backbone   import GNNBackbone
from src.models.heads.residue_activity_head import ResidueActivityHead  # 你的头（支持 mask）

# =========================
# 配置
# =========================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE_TYPE   = "gnn"     # "gnn" | "gvp"

# 数据路径（替换为你的真实路径）
CSV_PATH       = "/tmp/pycharm_project_317/dataset/M-CSA/literature_pdb_residues_roles.csv"
PDB_DIR        = "/root/autodl-tmp/M-CSA/pdbs"
CACHE_DIR      = None  # 稳定后可指定目录加速

# 构图
RADIUS         = 10.0
EMBEDDER_CFG   = {"name": "onehot"}  # 先跑通再换 ESM
GVP_BUILDER_KW = dict(node_vec_mode="bb2", edge_scalar="rbf", edge_vec_dim=1, add_self_loop=False)
GNN_BUILDER_KW = dict()

# 模型
EMB_DIM        = 128
DROPOUT        = 0.10
GVP_HIDDEN     = (128, 16)
GVP_LAYERS     = 3
GNN_HIDDENS    = [128, 128]

# 训练
SEED           = 42
VAL_SPLIT      = 0.1
BATCH_SIZE     = 4
EPOCHS         = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 2
MAX_NORM       = 5.0
USE_AMP        = True

# 头 / 损失
NUM_CLASSES     = 4   # 0 reactant, 1 interaction, 2 spectator, 3 none
IGNORE_INDEX    = -100
LABEL_SMOOTHING = 0.02
CLASS_WEIGHTS   = torch.tensor([2.0, 2.0, 2.0, 0.5], dtype=torch.float32)  # 简单起点
NONE_CLASS_ID   = 3
NONE_KEEP_RATIO = 0.25   # 训练时对 none 的下采样比例；验证将自动关闭

# =========================
# 实用
# =========================
def set_seed(seed: int = 42):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def role_type_to_id(rt: str) -> int:
    if not isinstance(rt, str): return 3
    rt = rt.strip().lower()
    return {"reactant": 0, "interaction": 1, "spectator": 2, "none": 3}.get(rt, 3)

# 多行同一残基取“优先级最高”的 ROLE_TYPE（reactant>interaction>spectator）
ROLE_RANK = {"reactant": 3, "interaction": 2, "spectator": 1}

def pick_best_role(types: List[str]) -> str:
    best = None; br = -1
    for t in types:
        tt = (t or "").strip().lower()
        r = ROLE_RANK.get(tt, 0)
        if r > br:
            br, best = r, tt
    return best or "none"

# =========================
# Dataset：严格链匹配
# =========================
class CatalysisResidueDataset(Dataset):
    """
    一个 item = (pdb_id, chain)
    - builder 只为该链建图
    - 未出现在 CSV 的残基 → none
    - 同一残基若多行 → 取优先级最高的 ROLE_TYPE
    """
    def __init__(self, csv_path: str, pdb_dir: str, cache_dir: Optional[str],
                 backbone_type: str = "gnn"):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        self.backbone_type = backbone_type.lower()

        df = pd.read_csv(csv_path)
        # 仅保留必要列，并标准化
        need = ["PDB ID", "CHAIN ID", "RESIDUE NUMBER", "ROLE_TYPE"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"CSV 缺少必要列: {miss}")
        df = df[need].copy()
        df.rename(columns={"PDB ID":"pdb_id", "CHAIN ID":"chain", "RESIDUE NUMBER":"resnum",
                           "ROLE_TYPE":"role_type"}, inplace=True)
        df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
        df["chain"]  = df["chain"].astype(str)
        # 合并同一 (pdb, chain, resnum) 多行 → 取优先级最高角色
        grouped = (df
            .groupby(["pdb_id","chain","resnum"])["role_type"]
            .apply(lambda s: pick_best_role(list(s)))
            .reset_index()
        )
        self.labels_df = grouped  # 唯一键 + 单一 role_type
        # 键集合（每个样本为一条链）
        self.keys_df = self.labels_df[["pdb_id","chain"]].drop_duplicates().reset_index(drop=True)

        # 选择 builder
        if self.backbone_type == "gvp":
            cfg = GVPBuilderConfig(pdb_dir=pdb_dir, out_dir=cache_dir or os.path.join(pdb_dir, "_tmp_pt"),
                                   radius=RADIUS, embedder=EMBEDDER_CFG)
            os.makedirs(cfg.out_dir, exist_ok=True)
            self.builder = GVPProteinGraphBuilder(cfg, **GVP_BUILDER_KW)
        elif self.backbone_type == "gnn":
            cfg = GNNBuilderConfig(pdb_dir=pdb_dir, out_dir=cache_dir or os.path.join(pdb_dir, "_tmp_pt"),
                                   radius=RADIUS, embedder=EMBEDDER_CFG)
            os.makedirs(cfg.out_dir, exist_ok=True)
            self.builder = GNNProteinGraphBuilder(cfg, **GNN_BUILDER_KW)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
        self._cfg = cfg

    def __len__(self) -> int:
        return len(self.keys_df)

    def __getitem__(self, idx: int) -> Data:
        r = self.keys_df.iloc[idx]
        pdb_id = str(r["pdb_id"]); chain = str(r["chain"])

        # 严格文件名（如你的库是全小写/大写可自行调整；这里只尝试 1 次）
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        # 限定当前链
        self.builder.cfg.chain_id = chain

        # 缓存（单链 + 模式）
        tag = f"{pdb_id}_{chain}_{self.backbone_type}"
        if self.cache_dir:
            pt_path = os.path.join(self.builder.cfg.out_dir, f"{tag}.pt")
            if os.path.exists(pt_path):
                data = torch.load(pt_path, weights_only=False)
            else:
                data, _ = self.builder.build_one(pdb_path, name=tag)
                torch.save(data, pt_path)
        else:
            data, _ = self.builder.build_one(pdb_path, name=tag)

        # 适配 GNN 的字段名（若 builder 已处理可忽略）
        if not hasattr(data, "x") and hasattr(data, "x_s"):
            data.x = data.x_s.float()
        if not hasattr(data, "edge_attr") and hasattr(data, "edge_s"):
            ea = data.edge_s
            if isinstance(ea, torch.Tensor) and ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            data.edge_attr = ea

        # 给每个节点打标签：默认 none, 若在 CSV 中 → 覆盖 0/1/2
        if not hasattr(data, "res_ids"):
            # 若 builder 没生成 res_ids，则假设每个节点顺序就是残基序号 1..N
            N = data.x.size(0)
            data.res_ids = [f"{chain}:{i+1}" for i in range(N)]

        # 建立 resnum -> index 映射（只取整数部分）
        idx_map: Dict[int, int] = {}
        for n, tag in enumerate(data.res_ids):
            # 期待格式 "A:123"
            try:
                ch, num = tag.split(":", 1)
                if ch != chain:  # 只保留目标链
                    continue
                num = int("".join([c for c in num if (c.isdigit() or c == "-")]))
                idx_map[num] = n
            except Exception:
                pass

        y = torch.full((len(data.res_ids),), 3, dtype=torch.long)  # 全部 none
        rows = self.labels_df[(self.labels_df["pdb_id"]==pdb_id) & (self.labels_df["chain"]==chain)]
        for _, rr in rows.iterrows():
            resnum = int(rr["resnum"])
            i = idx_map.get(resnum, None)
            if i is None:  # CSV 残基不在图中（可能缺 CA/缺该残基），跳过
                continue
            y[i] = role_type_to_id(rr["role_type"])
        data.y_role = y
        return data

# =========================
# 模型与训练
# =========================
def build_backbone(backbone_type: str) -> nn.Module:
    if backbone_type == "gvp":
        return GVPBackbone(
            hidden_dims=GVP_HIDDEN, n_layers=GVP_LAYERS,
            out_dim=EMB_DIM, dropout=DROPOUT, residue_logits=True
        ).to(DEVICE)
    elif backbone_type == "gnn":
        return GNNBackbone(
            conv_type="gine", hidden_dims=GNN_HIDDENS,
            out_dim=EMB_DIM, dropout=DROPOUT,
            residue_logits=True, gine_missing_edge_policy="zeros",
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown BACKBONE_TYPE={backbone_type}")

@dataclass
class StepStats:
    loss_sum: float
    acc: float
    n_lab: int

def step(backbone: nn.Module, head: ResidueActivityHead, batch: Data,
         optimizer: Optional[torch.optim.Optimizer] = None,
         scaler: Optional[torch.amp.GradScaler] = None,
         none_keep_ratio: Optional[float] = None) -> StepStats:
    is_train = optimizer is not None
    batch = batch.to(DEVICE)

    # —— 评估阶段禁用 none 下采样 —— #
    effective_keep = none_keep_ratio if is_train else 1.0

    # 训练用 AMP
    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (USE_AMP and torch.cuda.is_available()) else nullcontext()

    with amp_ctx:
        h = backbone(batch)  # [N, EMB_DIM]
        y = batch.y_role
        # 构建 mask（训练时对 none 采样；验证全量）
        if effective_keep is None or effective_keep >= 1.0 or NONE_CLASS_ID is None:
            mask = None
        else:
            with torch.no_grad():
                is_none = (y == NONE_CLASS_ID)
                keep_none = torch.zeros_like(is_none, dtype=torch.bool)
                if is_none.any():
                    idx = torch.where(is_none)[0]
                    k = max(1, int(len(idx) * float(effective_keep)))
                    kept = idx[torch.randperm(len(idx), device=idx.device)[:k]]
                    keep_none[kept] = True
                mask = (~is_none) | keep_none

        logits, out = head(h, y=y, mask=mask)
        # out["ce"] 是平均损失；为了“人均”累积更稳，我们把它乘有效样本数得到“求和”
        if mask is None:
            n_lab = int(y.numel())
        else:
            n_lab = int(mask.sum().item())
        loss_sum = out["ce"] * n_lab

    if is_train:
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss_sum / max(n_lab, 1)).backward()  # 回传用人均
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), MAX_NORM)
            scaler.step(optimizer); scaler.update()
        else:
            (loss_sum / max(n_lab, 1)).backward()
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), MAX_NORM)
            optimizer.step()

    # 计算精度（仅在有效样本上）
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        if mask is None:
            acc = (pred == y).float().mean().item()
        else:
            acc = (pred[mask] == y[mask]).float().mean().item()

    return StepStats(loss_sum=float(loss_sum.item()), acc=float(acc), n_lab=n_lab)

def run_epoch(loader: DataLoader, backbone: nn.Module, head: ResidueActivityHead,
              optimizer: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None) -> Dict[str, float]:
    is_train = optimizer is not None
    backbone.train(is_train); head.train(is_train)

    tot_loss_sum, tot_acc, tot_lab = 0.0, 0.0, 0
    for batch in loader:
        s = step(backbone, head, batch, optimizer, scaler,
                 none_keep_ratio=NONE_KEEP_RATIO if is_train else None)
        tot_loss_sum += s.loss_sum
        tot_acc      += s.acc * s.n_lab
        tot_lab      += s.n_lab

    avg_loss = tot_loss_sum / max(tot_lab, 1)
    avg_acc  = tot_acc / max(tot_lab, 1)
    return {"loss": avg_loss, "acc": avg_acc, "n_lab": tot_lab}

# =========================
# 主程序
# =========================
def main():
    set_seed(SEED)

    # 数据
    full = CatalysisResidueDataset(
        csv_path=CSV_PATH, pdb_dir=PDB_DIR, cache_dir=CACHE_DIR,
        backbone_type=BACKBONE_TYPE
    )
    # 用链划分训练/验证
    n_total = len(full)
    idx = list(range(n_total)); random.shuffle(idx)
    n_val = max(1, int(n_total * VAL_SPLIT))
    val_set = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_set]
    val_idx   = [i for i in idx if i in val_set]

    train_loader = DataLoader(torch.utils.data.Subset(full, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(torch.utils.data.Subset(full, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # 模型
    backbone = build_backbone(BACKBONE_TYPE)
    head = ResidueActivityHead(
        in_dim=EMB_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT,
        hidden=EMB_DIM, use_batchnorm=False,
        class_weights=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING,
        ignore_index=IGNORE_INDEX
    ).to(DEVICE)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and torch.cuda.is_available())

    print(f"[INFO] backbone={BACKBONE_TYPE} | train={len(train_idx)} | val={len(val_idx)} | device={DEVICE}")

    # 训练
    best_val = 1e9
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(train_loader, backbone, head, optimizer, scaler)
        vl = run_epoch(val_loader, backbone, head, optimizer=None, scaler=None)
        scheduler.step(vl["loss"])

        print(f"[Epoch {epoch:02d}] "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} (n={tr['n_lab']}) | "
              f"val loss={vl['loss']:.4f} acc={vl['acc']:.4f} (n={vl['n_lab']})")

        if vl["loss"] < best_val:
            best_val = vl["loss"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "backbone_type": BACKBONE_TYPE,
                "backbone_state": backbone.state_dict(),
                "head_state": head.state_dict(),
                "cfg": {
                    "EMB_DIM": EMB_DIM,
                    "GVP_HIDDEN": GVP_HIDDEN, "GVP_LAYERS": GVP_LAYERS,
                    "GNN_HIDDENS": GNN_HIDDENS,
                    "LABEL_SMOOTHING": LABEL_SMOOTHING,
                    "NONE_KEEP_RATIO": NONE_KEEP_RATIO,
                }
            }, f"checkpoints/catalysis_4cls_{BACKBONE_TYPE}_best.pt")
            print(f"[INFO] saved best checkpoint @ epoch {epoch} (val_loss={best_val:.4f})")

    print("[DONE]")

if __name__ == "__main__":
    main()