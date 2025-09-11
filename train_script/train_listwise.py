# train_listwise.py
from __future__ import annotations

import os
import warnings
import math
from typing import Dict, List

import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")  # 全局静默 numpy 浮点告警

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.builders.base_builder import BuilderConfig
from src.builders.gnn_builder import GNNProteinGraphBuilder
from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM_ENZ = 128
EMB_DIM_PL  = 128
PROJ_DIM    = 128

BATCH_SIZE  = 2
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 100

TEMP        = 0.07     # InfoNCE 温度
ALPHA       = 0.4      # 正-正相似度阈值
LAMBDA_REP  = 0.1      # 去冗余权重

OUT_DIR     = "./runs_listwise"  # 日志与best模型的保存目录
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 损失 ============
def multi_positive_infonce(sim: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    sim: [L]，该蛋白与 list 中每个塑料的余弦相似度
    labels: [L]，0/1
    MP-InfoNCE：-log( sum_{pos} exp(s/tau) / sum_{all} exp(s/tau) )
    """
    assert sim.dim() == 1 and labels.dim() == 1
    mask_pos = (labels > 0.5)
    if mask_pos.sum() == 0:
        # 无正样本，返回0（等价跳过）
        return sim.new_tensor(0.0)
    logits = sim / tau
    num = torch.logsumexp(logits[mask_pos], dim=0)
    den = torch.logsumexp(logits, dim=0)
    return -(num - den)

def positive_repulsion(zp_pos: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
    """
    zp_pos: [P, D]，同一蛋白下所有正塑料的归一化向量
    目标：让正-正相似度尽量 <= alpha。损失 = mean(ReLU(cos - alpha))
    """
    P = zp_pos.size(0)
    if P <= 1:
        return zp_pos.new_tensor(0.0)
    S = zp_pos @ zp_pos.t()  # [P,P]（z 已归一化）
    mask = ~torch.eye(P, dtype=torch.bool, device=zp_pos.device)
    viol = F.relu(S[mask] - alpha)
    return viol.mean()

# ============ 模型 ============
class TwinModel(nn.Module):
    def __init__(self, in_dim_enz: int, in_dim_pl: int, proj_dim: int):
        super().__init__()
        self.proj_enz = nn.Linear(in_dim_enz, proj_dim, bias=False)
        self.proj_pl  = nn.Linear(in_dim_pl,  proj_dim, bias=False)

    def forward(self, enz_vec: torch.Tensor, pl_vec: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        z_e = F.normalize(self.proj_enz(enz_vec), dim=-1)
        z_p = F.normalize(self.proj_pl(pl_vec),  dim=-1)
        return z_e, z_p

# ============ 评估指标 ============
@torch.no_grad()
def listwise_metrics(sim: torch.Tensor, labels: torch.Tensor, ks=(1, 3)) -> Dict[str, float]:
    """
    sim: [L], labels: [L]
    返回: mean_pos_sim, mean_neg_sim, hit@k...
    """
    out = {}
    mask_pos = labels > 0.5
    mask_neg = ~mask_pos
    if mask_pos.any():
        out["pos_sim"] = float(sim[mask_pos].mean().item())
    else:
        out["pos_sim"] = float("nan")
    if mask_neg.any():
        out["neg_sim"] = float(sim[mask_neg].mean().item())
    else:
        out["neg_sim"] = float("nan")

    # 命中率
    L = sim.numel()
    if L > 0:
        order = torch.argsort(sim, descending=True)  # 大到小
        for k in ks:
            k = min(k, L)
            hit = (labels[order[:k]] > 0.5).any().float().item()
            out[f"hit@{k}"] = float(hit)
    else:
        for k in ks:
            out[f"hit@{k}"] = float("nan")
    return out

def reduce_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """把每条 item 指标取均值（忽略 NaN）"""
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if (k in m and not math.isnan(m[k]))]
        out[k] = float(sum(vals) / max(len(vals), 1)) if len(vals) else float("nan")
    return out

# ============ 一个 epoch 的训练 or 评估 ============
def run_one_epoch(
    loader: DataLoader,
    enzyme_backbone: nn.Module,
    plastic_tower: nn.Module,
    twin: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    train: bool,
    temp: float,
    alpha: float,
    lambda_rep: float,
) -> Dict[str, float]:
    if train:
        enzyme_backbone.train(); plastic_tower.train(); twin.train()
    else:
        enzyme_backbone.eval(); plastic_tower.eval(); twin.eval()

    total_loss, n_items = 0.0, 0
    metrics_items: List[Dict[str, float]] = []

    for batch in loader:
        g_batch = batch["enzyme_graph"].to(DEVICE)
        enz_vec = enzyme_backbone(g_batch)  # [B, D_e]

        loss_batch = 0.0
        items_in_batch = 0

        for b_idx, item in enumerate(batch["items"]):
            plast = item["plastic_list"]    # [L, Dp] or empty
            labels = item["relevance"]      # [L]
            if plast is None or plast.numel() == 0:
                continue

            plast = plast.to(DEVICE)
            labels = labels.to(DEVICE)

            z_e_b = enz_vec[b_idx:b_idx+1, :]                 # [1,De]
            z_p   = plastic_tower(plast)                      # [L,Dp]
            z_e_b, z_p = twin(z_e_b, z_p)                     # [1,D], [L,D]
            sim = (z_e_b @ z_p.t()).squeeze(0)                # [L]

            loss_ce  = multi_positive_infonce(sim, labels, tau=temp)
            loss_rep = positive_repulsion(z_p[labels > 0.5], alpha=alpha) if (labels > 0.5).sum() >= 2 else sim.new_tensor(0.0)
            loss_item = loss_ce + lambda_rep * loss_rep

            if train:
                loss_batch = loss_batch + loss_item

            total_loss += float(loss_item.item())
            n_items += 1
            items_in_batch += 1

            # 记录该 item 的指标
            metrics_items.append(listwise_metrics(sim, labels, ks=(1, 3)))

        if train and items_in_batch > 0:
            optimizer.zero_grad(set_to_none=True)
            (loss_batch / items_in_batch).backward()
            optimizer.step()

    reduced = reduce_metrics(metrics_items)
    reduced["loss"] = total_loss / max(n_items, 1)
    reduced["items"] = float(n_items)
    return reduced

# ============ 组装一切 ============
def main():
    # ----- 构建器 -----
    cfg = BuilderConfig(
        pdb_dir="/root/autodl-tmp/pdb/pdb",
        out_dir="/root/autodl-tmp/pdb/pt",
        radius=10.0,
        embedder=[{"name": "onehot"}],
    )
    enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    plastic_featurizer = PlasticFeaturizer(config_path=None)

    # ----- 数据集（train/val） -----
    spec = MatrixSpec(
        csv_path="/tmp/pycharm_project_317/dataset/predicted_xid/plastics_onehot_trainset.csv",
        pdb_root="/root/autodl-tmp/pdb/pdb",
        sdf_root="/tmp/pycharm_project_317/plastic/mols_for_unimol_10_sdf_new",
    )

    ds_train = PairedPlaszymeDataset(
        matrix=spec,
        mode="list",
        split="train",
        enzyme_builder=enzyme_builder,
        plastic_featurizer=plastic_featurizer,
        max_list_len=64,
    )
    ds_val = PairedPlaszymeDataset(
        matrix=spec,
        mode="list",
        split="val",
        enzyme_builder=enzyme_builder,
        plastic_featurizer=plastic_featurizer,
        max_list_len=60,
    )

    # ----- 两个塔 -----
    # 取一个样本的塑料维度以确定 polymer MLP 输入维数
    probe_feat = None
    for s in ds_train.cols:
        probe_feat = ds_train._get_plastic(s)
        if probe_feat is not None:
            break
    assert probe_feat is not None, "没有可用的塑料特征（请检查 SDF 路径与可读性）"
    in_dim_plastic = probe_feat.shape[-1]

    enzyme_backbone = GNNBackbone(
        conv_type="gine", hidden_dims=[128,128], out_dim=EMB_DIM_ENZ,
        dropout=0.1, residue_logits=False, gine_missing_edge_policy="zeros"
    ).to(DEVICE)
    plastic_tower = PolymerTower(
        in_dim=in_dim_plastic, hidden_dims=[256,128], out_dim=EMB_DIM_PL, dropout=0.1
    ).to(DEVICE)
    twin = TwinModel(EMB_DIM_ENZ, EMB_DIM_PL, PROJ_DIM).to(DEVICE)

    optim = torch.optim.AdamW(
        list(enzyme_backbone.parameters()) + list(plastic_tower.parameters()) + list(twin.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ----- DataLoader -----
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs, num_workers=0)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pairs, num_workers=0)

    print(f"[INFO] dataset(listwise) train={len(ds_train)} | val={len(ds_val)} | device={DEVICE}")

    # ----- 训练主循环 -----
    history = {
        "train_loss": [], "val_loss": [],
        "train_hit@1": [], "val_hit@1": [],
        "train_hit@3": [], "val_hit@3": [],
        "train_pos_sim": [], "val_pos_sim": [],
        "train_neg_sim": [], "val_neg_sim": [],
    }
    best_val_hit1 = -1.0
    best_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        stats_tr = run_one_epoch(
            loader_train, enzyme_backbone, plastic_tower, twin, optim,
            train=True, temp=TEMP, alpha=ALPHA, lambda_rep=LAMBDA_REP
        )
        stats_va = run_one_epoch(
            loader_val, enzyme_backbone, plastic_tower, twin, optimizer=None,
            train=False, temp=TEMP, alpha=ALPHA, lambda_rep=LAMBDA_REP
        )

        # 记录
        history["train_loss"].append(stats_tr["loss"])
        history["val_loss"].append(stats_va["loss"])
        history["train_hit@1"].append(stats_tr.get("hit@1", float("nan")))
        history["val_hit@1"].append(stats_va.get("hit@1", float("nan")))
        history["train_hit@3"].append(stats_tr.get("hit@3", float("nan")))
        history["val_hit@3"].append(stats_va.get("hit@3", float("nan")))
        history["train_pos_sim"].append(stats_tr.get("pos_sim", float("nan")))
        history["val_pos_sim"].append(stats_va.get("pos_sim", float("nan")))
        history["train_neg_sim"].append(stats_tr.get("neg_sim", float("nan")))
        history["val_neg_sim"].append(stats_va.get("neg_sim", float("nan")))

        # 打印整洁日志
        log = (
            f"[Epoch {epoch:02d}] "
            f"train: loss={stats_tr['loss']:.4f}, hit@1={stats_tr.get('hit@1', float('nan')):.3f}, "
            f"hit@3={stats_tr.get('hit@3', float('nan')):.3f}, "
            f"pos_sim={stats_tr.get('pos_sim', float('nan')):.3f}, "
            f"neg_sim={stats_tr.get('neg_sim', float('nan')):.3f} | "
            f"val: loss={stats_va['loss']:.4f}, hit@1={stats_va.get('hit@1', float('nan')):.3f}, "
            f"hit@3={stats_va.get('hit@3', float('nan')):.3f}, "
            f"pos_sim={stats_va.get('pos_sim', float('nan')):.3f}, "
            f"neg_sim={stats_va.get('neg_sim', float('nan')):.3f}"
        )
        print(log)

        # 保存 best（以 val hit@1 为准）
        cur = stats_va.get("hit@1", -1.0)
        if cur > best_val_hit1:
            best_val_hit1 = cur
            torch.save({
                "epoch": epoch,
                "cfg": {
                    "emb_dim_enz": EMB_DIM_ENZ, "emb_dim_pl": EMB_DIM_PL, "proj_dim": PROJ_DIM,
                    "temp": TEMP, "alpha": ALPHA, "lambda_rep": LAMBDA_REP
                },
                "enzyme_backbone": enzyme_backbone.state_dict(),
                "plastic_tower": plastic_tower.state_dict(),
                "twin": twin.state_dict(),
                "optimizer": optim.state_dict(),
                "history": history,
            }, best_path)
            print(f"[INFO] Saved best model to {best_path} (val hit@1={best_val_hit1:.3f})")

        # 每个 epoch 保存曲线
        plot_curves(history, OUT_DIR)

    # 训练结束再画一次
    plot_curves(history, OUT_DIR)
    print(f"[DONE] Best val hit@1={best_val_hit1:.3f} | best saved: {best_path}")

# ============ 可视化 ============
def plot_curves(history: Dict[str, List[float]], out_dir: str):
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # Hit@k
    plt.figure()
    plt.plot(history["train_hit@1"], label="train_hit@1")
    plt.plot(history["val_hit@1"],   label="val_hit@1")
    plt.plot(history["train_hit@3"], label="train_hit@3")
    plt.plot(history["val_hit@3"],   label="val_hit@3")
    plt.xlabel("epoch"); plt.ylabel("hit"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hit.png"))
    plt.close()

    # pos/neg sim
    plt.figure()
    plt.plot(history["train_pos_sim"], label="train_pos_sim")
    plt.plot(history["val_pos_sim"],   label="val_pos_sim")
    plt.plot(history["train_neg_sim"], label="train_neg_sim")
    plt.plot(history["val_neg_sim"],   label="val_neg_sim")
    plt.xlabel("epoch"); plt.ylabel("sim"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sim.png"))
    plt.close()

if __name__ == "__main__":
    # 限制 PyTorch 自身产生的无谓告警
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    main()