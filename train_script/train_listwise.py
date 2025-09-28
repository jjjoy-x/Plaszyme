#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plaszyme — Listwise Training
====================================================

This script implements listwise training for enzyme–plastic
interaction prediction, with support for multiple backbones
(GNN / GVP / MLP), plastic tower projection, and flexible
interaction heads (cosine, bilinear, factorized, MLP, gated).

Key Features
------------
- Structured config (`TrainConfig`) for hyperparameters & paths
- Multi-backbone enzyme encoders + polymer tower
- Listwise InfoNCE loss with optional list-mitigation
- Diagnostics of embedding space (saved to CSV)
- Logging, checkpoints, and training curve plots
- Automatic test evaluation after training

Outputs
-------
- Logs: `train.log`, `run_config.txt`, `run_config.json`
- Models: `best_<mode>.pt`, `last_<mode>.pt`
- Curves: `loss.png`, `hit.png`, `score.png`
- Evaluation: `test_metrics.csv`, optional score matrix

Usage
-----
# Default training
python train_listwise.py

# Override with JSON config
python train_listwise.py --config config.json
from __future__ import annotations

Author
-----
Shuleihe (School of Science, Xi’an Jiaotong-Liverpool University)
XJTLU_AI_China — iGEM 2025
"""

import os
import math
import csv
import json
import warnings
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# ====== 项目内部依赖 ======
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone
from src.models.seq_mlp.backbone import SeqBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.models.interaction_head import InteractionHead


# =============================================================================
# 配置（默认值保持与原脚本一致；不传参时行为完全相同）
# =============================================================================
@dataclass
class TrainConfig:
    # 设备与主干
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enz_backbone: Literal["GNN", "GVP", "MLP"] = "GNN"

    # 数据路径
    pdb_root: str = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb"
    pt_out_root: str = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt"
    sdf_root: str = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"
    train_csv: str = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_trainset.csv"
    test_csv: str = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_testset.csv"

    out_dir: str = "./train_results/gnn_gated"

    # 超参（与原脚本一致）
    emb_dim_enz: int = 128
    emb_dim_pl: int = 128
    proj_dim: int = 128
    batch_size: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    max_list_len: int = 10

    # InfoNCE 温度
    temp: float = 0.2

    # 正-正去同质化/正则
    alpha: float = 0.4
    lambda_rep: float = 0.1  # （保留字段）
    var_target: float = 1.0
    lambda_var: float = 1e-3
    lambda_center: float = 1e-3
    plastic_diversify: bool = True
    lambda_diversify: float = 0.05

    # 交互方式
    interaction: Literal["cos","bilinear","factorized_bilinear","hadamard_mlp","gated"] = "gated"
    bilinear_rank: int = 64
    lambda_w_reg: float = 1e-4
    ortho_reg: float = 0.0

    # 采样缓解（默认 True，与后面 run_one_epoch 默认一致）
    enable_list_mitigation: bool = False
    max_list_len_train: int = 10
    neg_per_item: int = 32
    hard_neg_cand: int = 32
    hard_neg_ratio: float = 0.5

    # 预训练（默认与原脚本一致：False）
    use_plastic_pretrain: bool = False
    co_matrix_csv: str = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastic_co_matrix.csv"
    pretrain_epochs: int = 10
    pretrain_loss_mode: Literal["contrastive","mse"] = "contrastive"

    # 随机种子
    seed: int = 42


# =============================================================================
# 日志与随机性
# =============================================================================
def setup_logging(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "train.log"
    fmt = "[%(asctime)s] %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    )
    logging.info("Logger initialized. Writing to %s", log_path)


def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 诊断模块（保持逻辑不变）
# =============================================================================
class EmbeddingDiagnostics:
    """将嵌入空间统计写入 CSV，便于排查坍缩/尺度失衡。"""
    def __init__(self, out_dir: str):
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        self.csv_path = p/"emb_diag.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch","split","where","space","N","D","norm_mean","std_mean","cos_mu","cos_std","energy_top1"]
                )
    @torch.no_grad()
    def _pairwise_cos_stats(self, Z: torch.Tensor) -> Tuple[float,float]:
        if Z.size(0) <= 1: return float("nan"), float("nan")
        Z = F.normalize(Z, dim=-1)
        S = Z @ Z.t()
        N = S.size(0); mask = ~torch.eye(N, dtype=torch.bool, device=Z.device)
        vals = S[mask]
        return float(vals.mean()), float(vals.std())
    @torch.no_grad()
    def _energy_top1_ratio(self, Z: torch.Tensor) -> float:
        if Z.size(0) <= 1: return float("nan")
        Zc = Z - Z.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(Zc)
        except RuntimeError:
            C = Zc.t() @ Zc
            s = torch.linalg.eigvalsh(C).clamp_min(0).sqrt()
        s2 = s.square()
        den = s2.sum().item()
        return float((s2.max()/den).item()) if den>1e-12 else float("nan")
    @torch.no_grad()
    def report(self, Z: torch.Tensor, *, epoch:int, split:str, where:str, space:str):
        if Z is None or Z.numel()==0: return
        Z = Z.detach()
        norm_mean = float(Z.norm(dim=-1).mean().item())
        std_mean  = float(Z.std(dim=0).mean().item())
        cos_mu, cos_std = self._pairwise_cos_stats(Z)
        enr = self._energy_top1_ratio(Z)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, split, where, space, Z.size(0), Z.size(1),
                f"{norm_mean:.6f}", f"{std_mean:.6f}",
                f"{cos_mu:.6f}" if not math.isnan(cos_mu) else "nan",
                f"{cos_std:.6f}" if not math.isnan(cos_std) else "nan",
                f"{enr:.6f}" if not math.isnan(enr) else "nan",
            ])


# =============================================================================
# 损失函数与正则（保持逻辑不变）
# =============================================================================
def mp_infonce(scores: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """多正样本 InfoNCE（与原脚本一致）"""
    assert scores.dim()==1 and labels.dim()==1
    pos = labels > 0.5
    if pos.sum()==0:
        return scores.new_tensor(0.0)
    logits = scores / tau
    num = torch.logsumexp(logits[pos], dim=0)
    den = torch.logsumexp(logits, dim=0)
    return -(num - den)

def positive_repulsion(z_pos: torch.Tensor, alpha: float=0.4) -> torch.Tensor:
    """同一 item 的正样本相互“区别开”（与原脚本一致）"""
    P = z_pos.size(0)
    if P <= 1: return z_pos.new_tensor(0.0)
    S = F.normalize(z_pos, dim=-1) @ F.normalize(z_pos, dim=-1).t()
    mask = ~torch.eye(P, dtype=torch.bool, device=z_pos.device)
    viol = F.relu(S[mask] - alpha)
    return viol.mean()

def center_var_reg(z: torch.Tensor, var_target: float=1.0) -> torch.Tensor:
    """简单“均值-方差”正则（与原脚本一致）"""
    if z.numel()==0: return z.new_tensor(0.0)
    mu  = z.mean(dim=0)
    var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    loss_center = (mu.square()).mean()
    loss_var    = ((var - var_target).square()).mean()
    return loss_center + loss_var


# =============================================================================
# 模型组件（保持逻辑不变）
# =============================================================================
class TwinProjector(nn.Module):
    """蛋白/塑料各自线性投影到同一空间（与原脚本一致）"""
    def __init__(self, in_e:int, in_p:int, out:int):
        super().__init__()
        self.proj_e = nn.Linear(in_e, out, bias=False)
        self.proj_p = nn.Linear(in_p, out, bias=False)
        nn.init.xavier_uniform_(self.proj_e.weight)
        nn.init.xavier_uniform_(self.proj_p.weight)
    def forward(self, e: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.proj_e(e), self.proj_p(p)


# =============================================================================
# 训练期指标（保持逻辑不变）
# =============================================================================
@torch.no_grad()
def listwise_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(1,3)) -> Dict[str,float]:
    out = {}
    mask_pos = labels>0.5; mask_neg = ~mask_pos
    out["pos_score"] = float(scores[mask_pos].mean().item()) if mask_pos.any() else float("nan")
    out["neg_score"] = float(scores[mask_neg].mean().item()) if mask_neg.any() else float("nan")
    L = scores.numel()
    if L>0:
        order = torch.argsort(scores, descending=True)
        for k in ks:
            k=min(k,L)
            hit = (labels[order[:k]]>0.5).any().float().item()
            out[f"hit@{k}"] = float(hit)
    else:
        for k in ks: out[f"hit@{k}"]=float("nan")
    return out

def reduce_metrics(metrics_list: List[Dict[str,float]]) -> Dict[str,float]:
    if not metrics_list: return {}
    keys = metrics_list[0].keys()
    out={}
    for k in keys:
        vals=[m[k] for m in metrics_list if (k in m and not math.isnan(m[k]))]
        out[k]=float(sum(vals)/max(len(vals),1)) if vals else float("nan")
    return out


# =============================================================================
# 单个 epoch（保持逻辑不变）
# =============================================================================
def run_one_epoch(
    loader: DataLoader,
    enzyme_backbone: nn.Module,
    plastic_tower: nn.Module,
    projector: TwinProjector,
    inter_head: InteractionHead,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    cfg: TrainConfig,
    train: bool,
    epoch: int,
) -> Dict[str, float]:
    """
    训练/验证一个 epoch；内部逻辑严格保持原脚本一致。
    """
    split = "train" if train else "val"
    enzyme_backbone.train() if train else enzyme_backbone.eval()
    plastic_tower.train()   if train else plastic_tower.eval()
    projector.train()       if train else projector.eval()
    inter_head.train()      if train else inter_head.eval()

    diag = EmbeddingDiagnostics(cfg.out_dir)
    max_diag_batches = 2
    diag_cnt = 0

    total_loss, n_items = 0.0, 0
    metrics_items: List[Dict[str, float]] = []
    sum_L_sub, cnt_L_sub = 0, 0

    for batch in loader:
        g_batch = batch["enzyme_graph"].to(cfg.device)
        enz_vec = enzyme_backbone(g_batch)      # [B, De]
        z_e_raw = projector.proj_e(enz_vec)     # [B, D]

        if diag_cnt < max_diag_batches:
            diag.report(z_e_raw, epoch=epoch, split=split, where="protein", space="raw")

        loss_batch = 0.0
        items_in_batch = 0

        for b_idx, item in enumerate(batch["items"]):
            plast  = item["plastic_list"]
            labels = item["relevance"]
            if plast is None or plast.numel() == 0:
                continue

            plast  = plast.to(cfg.device)       # [L, Din_p]
            labels = labels.to(cfg.device)      # [L]
            L_full = int(labels.numel())

            # 塑料塔 + 投影（完整列表一次算完）
            z_p_back = plastic_tower(plast)               # [L, Dp]
            z_p_raw  = projector.proj_p(z_p_back)         # [L, D]

            if diag_cnt < max_diag_batches and b_idx == 0:
                diag.report(z_p_raw, epoch=epoch, split=split, where="plastic", space="raw")

            z_e_b = z_e_raw[b_idx:b_idx + 1, :]           # [1, D]
            scores_full = inter_head.score(z_e_b, z_p_raw) # [L]

            if cfg.enable_list_mitigation:
                # ---- 全正 + 难负/随机负 子采样 ----
                idx_all = torch.arange(L_full, device=cfg.device)
                idx_pos = idx_all[labels > 0.5]
                idx_neg = idx_all[labels <= 0.5]

                keep_pos = idx_pos
                neg_target = max(0, min(cfg.neg_per_item, cfg.max_list_len_train - int(len(keep_pos))))
                neg_target = min(neg_target, int(len(idx_neg)))

                if neg_target > 0 and len(idx_neg) > 0:
                    if cfg.hard_neg_cand > 0:
                        cand = idx_neg[torch.randperm(len(idx_neg), device=cfg.device)[:min(cfg.hard_neg_cand, len(idx_neg))]]
                        cand_scores = scores_full[cand]
                        k_hard = int(round(cfg.hard_neg_ratio * neg_target))
                        k_hard = min(k_hard, int(len(cand)))
                        hard_idx = cand[torch.topk(cand_scores, k=k_hard, largest=True).indices] if k_hard > 0 else cand[:0]

                        remain = neg_target - int(len(hard_idx))
                        if remain > 0:
                            hard_set = set(hard_idx.tolist())
                            pool = [i for i in idx_neg.tolist() if i not in hard_set]
                            if len(pool) > 0:
                                pool_t = torch.tensor(pool, device=cfg.device, dtype=torch.long)
                                rand_idx = pool_t[torch.randperm(len(pool_t), device=cfg.device)[:remain]]
                            else:
                                rand_idx = idx_neg[:0]
                            keep_neg = torch.cat([hard_idx, rand_idx], dim=0)
                        else:
                            keep_neg = hard_idx
                    else:
                        keep_neg = idx_neg[torch.randperm(len(idx_neg), device=cfg.device)[:neg_target]]
                else:
                    keep_neg = idx_neg[:0]

                keep_idx = torch.cat([keep_pos, keep_neg], dim=0)
                if len(keep_idx) > 1:
                    keep_idx = keep_idx[torch.randperm(len(keep_idx), device=cfg.device)]

                labels_sub = labels[keep_idx]          # [L_sub]
                z_p_sub    = z_p_raw[keep_idx]         # [L_sub]
                scores_sub = scores_full[keep_idx]     # [L_sub]
                L_sub      = int(labels_sub.numel())

                tau_eff = cfg.temp / max(1.0, math.log(1.0 + float(L_sub)))
                loss_ce = mp_infonce(scores_sub, labels_sub, tau=tau_eff) / math.log(1.0 + float(L_sub))

                if cfg.plastic_diversify and (labels_sub > 0.5).sum() >= 2:
                    loss_div = positive_repulsion(z_p_sub[labels_sub > 0.5], alpha=cfg.alpha)
                else:
                    loss_div = scores_sub.new_tensor(0.0)

                loss_reg_e = center_var_reg(z_e_raw, cfg.var_target)
                loss_reg_p = center_var_reg(z_p_sub, cfg.var_target)

                w_reg = scores_sub.new_tensor(0.0)
                if cfg.interaction == "bilinear":
                    w_reg = w_reg + cfg.lambda_w_reg * (inter_head.W.square().mean())
                elif cfg.interaction == "factorized_bilinear":
                    w_reg = w_reg + cfg.lambda_w_reg * (
                        inter_head.U.weight.square().mean() + inter_head.V.weight.square().mean()
                    )
                    if cfg.ortho_reg > 0:
                        w_reg = w_reg + cfg.ortho_reg * inter_head.orthogonal_regularizer()

                loss_item = loss_ce + cfg.lambda_diversify * loss_div + cfg.lambda_center * (loss_reg_e + loss_reg_p) + w_reg
                metrics_items.append(listwise_metrics(scores_sub.detach(), labels_sub.detach(), ks=(1, 3)))
                sum_L_sub += L_sub
                cnt_L_sub += 1

            else:
                # ---- 不启用缓解：保持原逻辑 ----
                scores = scores_full
                loss_ce = mp_infonce(scores, labels, tau=cfg.temp)
                if cfg.plastic_diversify and (labels > 0.5).sum() >= 2:
                    loss_div = positive_repulsion(z_p_raw[labels > 0.5], alpha=cfg.alpha)
                else:
                    loss_div = scores.new_tensor(0.0)

                loss_reg_e = center_var_reg(z_e_raw, cfg.var_target)
                loss_reg_p = center_var_reg(z_p_raw, cfg.var_target)

                w_reg = scores.new_tensor(0.0)
                if cfg.interaction == "bilinear":
                    w_reg = w_reg + cfg.lambda_w_reg * (inter_head.W.square().mean())
                elif cfg.interaction == "factorized_bilinear":
                    w_reg = w_reg + cfg.lambda_w_reg * (
                        inter_head.U.weight.square().mean() + inter_head.V.weight.square().mean()
                    )
                    if cfg.ortho_reg > 0:
                        w_reg = w_reg + cfg.ortho_reg * inter_head.orthogonal_regularizer()

                loss_item = loss_ce + cfg.lambda_diversify * loss_div + cfg.lambda_center * (loss_reg_e + loss_reg_p) + w_reg
                metrics_items.append(listwise_metrics(scores.detach(), labels.detach(), ks=(1, 3)))

            if train:
                loss_batch = loss_batch + loss_item
            total_loss += float(loss_item.item())
            n_items += 1
            items_in_batch += 1

        if train and items_in_batch > 0:
            optimizer.zero_grad(set_to_none=True)
            (loss_batch / items_in_batch).backward()
            optimizer.step()

        diag_cnt += 1

    reduced = reduce_metrics(metrics_items)
    if cfg.enable_list_mitigation and cnt_L_sub > 0:
        reduced["avg_L_sub"] = float(sum_L_sub / cnt_L_sub)
    reduced["loss"]  = total_loss / max(n_items, 1)
    reduced["items"] = float(n_items)
    return reduced


# =============================================================================
# 评测工具（保持逻辑不变）
# =============================================================================
@torch.no_grad()
def compute_metrics(all_labels: List[np.ndarray], all_scores: List[np.ndarray], ks=(1,3,5)) -> Dict[str,float]:
    """listwise 排序指标 + micro/macro（二分类阈值=0，仅用于参考）"""
    results = {f"hit@{k}": [] for k in ks}
    results.update({f"recall@{k}": [] for k in ks})
    all_preds_bin, all_true_bin = [], []

    for labels, scores in zip(all_labels, all_scores):
        order = np.argsort(-scores)
        for k in ks:
            k = min(k, len(scores))
            topk = order[:k]
            hit = (labels[topk] > 0.5).any()
            recall = labels[topk].sum() / max(labels.sum(), 1)
            results[f"hit@{k}"].append(float(hit))
            results[f"recall@{k}"].append(float(recall))

        preds_bin = (scores > 0).astype(int)  # 仅用于参考
        all_preds_bin.extend(preds_bin.tolist())
        all_true_bin.extend(labels.tolist())

    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(results[f"hit@{k}"])) if results[f"hit@{k}"] else float("nan")
        out[f"recall@{k}"] = float(np.mean(results[f"recall@{k}"])) if results[f"recall@{k}"] else float("nan")

    from sklearn.metrics import precision_recall_fscore_support
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(all_true_bin, all_preds_bin, average="micro", zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(all_true_bin, all_preds_bin, average="macro", zero_division=0)
    out.update({
        "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f),
    })
    return out


# 与训练脚本一致的评测：保持原逻辑
@torch.no_grad()
def _eval_on_csv(csv_path: str,
                 pdb_root: str,
                 sdf_root: str,
                 pt_out_root: str,
                 enzyme_backbone: nn.Module,
                 plastic_tower: nn.Module,
                 projector: TwinProjector,
                 inter_head: InteractionHead,
                 out_csv: str,
                 max_list_len: int = 10,
                 score_matrix_csv: Optional[str] = None,
                 split_name: str = "val",
                 enz_backbone: Literal["GNN","GVP","MLP"] = "GNN"):
    # --- 构建 builder（逻辑保持一致） ---
    if enz_backbone == "GNN":
        cfg = GNNBuilderConfig(pdb_dir=pdb_root, out_dir=pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    elif enz_backbone == "GVP":
        cfg = GVPBuilderConfig(pdb_dir=pdb_root, out_dir=pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GVPProteinGraphBuilder(cfg)
    else:
        cfg = GNNBuilderConfig(pdb_dir=pdb_root, out_dir=pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")

    plastic_featurizer = PlasticFeaturizer(config_path=None)
    spec = MatrixSpec(csv_path=csv_path, pdb_root=pdb_root, sdf_root=sdf_root)
    ds = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="full", split_ratio=None,
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=max_list_len, return_names=True,
    )
    print(f"[EVAL] csv={csv_path} | split='{split_name}' | usable_samples={len(ds)} | plastics={len(ds.cols)}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pairs)

    enzyme_backbone.eval(); plastic_tower.eval(); projector.eval(); inter_head.eval()

    # -------- 指标（按样本子列表） --------
    all_labels, all_scores = [], []

    # -------- 分数矩阵（一次性塑料表示） --------
    want_matrix = bool(score_matrix_csv)
    if want_matrix:
        pl_feats, valid_mask = [], []
        for s in ds.cols:
            feat = ds._get_plastic(s)
            if feat is None:
                pl_feats.append(torch.zeros(1))
                valid_mask.append(False)
            else:
                t = torch.as_tensor(feat, dtype=torch.float32, device=enzyme_backbone.out_proj.weight.device if hasattr(enzyme_backbone,'out_proj') else 'cpu')
                pl_feats.append(t)
                valid_mask.append(True)

        in_dim_plastic = max(f.numel() for f in pl_feats if f.numel() > 1) if any(valid_mask) else 0
        P_list = []
        for ok, f in zip(valid_mask, pl_feats):
            if not ok:
                P_list.append(torch.zeros(in_dim_plastic, device=f.device))
            else:
                vec = f.flatten()
                if vec.numel() != in_dim_plastic:
                    v = torch.zeros(in_dim_plastic, device=f.device)
                    n = min(in_dim_plastic, vec.numel()); v[:n] = vec[:n]
                    vec = v
                P_list.append(vec)
        P = torch.stack(P_list, dim=0) if in_dim_plastic > 0 else torch.empty(0, 0, device=f.device)
        if P.numel() > 0:
            Zp_all = projector.proj_p(plastic_tower(P))
        else:
            Zp_all = torch.empty(0, projector.proj_e.out_features, device=P.device)
        score_rows = []
        pl_names = list(ds.cols)

    for idx, batch in enumerate(loader):
        g = batch["enzyme_graph"].to(next(enzyme_backbone.parameters()).device)
        enz_vec = enzyme_backbone(g)              # [1, De]
        z_e = projector.proj_e(enz_vec)           # [1, D]

        item   = batch["items"][0]
        plast  = item["plastic_list"].to(z_e.device)
        labels = item["relevance"].cpu().numpy()
        z_p    = projector.proj_p(plastic_tower(plast))
        scores = inter_head.score(z_e, z_p).detach().cpu().numpy()

        all_labels.append(labels)
        all_scores.append(scores)

        if want_matrix:
            enzyme_id = str(item.get("enzyme_id") or item.get("pdb_id") or item.get("id") or f"sample_{idx}")
            if Zp_all.numel() > 0:
                s_all = inter_head.score(z_e, Zp_all).detach().cpu().numpy()
                row = {"enzyme_id": enzyme_id}
                row.update({pl: float(s) for pl, s in zip(pl_names, s_all)})
            else:
                row = {"enzyme_id": enzyme_id}
            score_rows.append(row)

    metrics = compute_metrics(all_labels, all_scores, ks=(1,3,5))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"[TEST] saved metrics to: {out_csv}")
    print(pd.DataFrame([metrics]))

    if want_matrix:
        os.makedirs(os.path.dirname(score_matrix_csv) or ".", exist_ok=True)
        cols = ["enzyme_id"] + (pl_names if Zp_all.numel() > 0 else [])
        df_mat = pd.DataFrame(score_rows).reindex(columns=cols)
        df_mat.to_csv(score_matrix_csv, index=False)
        print(f"[TEST] saved score matrix to: {score_matrix_csv}  (rows={len(df_mat)}, cols={len(df_mat.columns)})")


# =============================================================================
# 辅助：保存本次配置（保持原逻辑并扩展为 JSON）
# =============================================================================
from datetime import datetime
def save_run_config_txt(out_dir: str, *, append: bool=False, extra: dict | None=None, cfg: TrainConfig | None=None):
    """保持原始 txt 行为；同时把 dataclass config 以 JSON 另存一份（不影响原有文件）。"""
    lines = []
    if not append:
        lines.append("# Plaszyme listwise training config")
        lines.append(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if cfg:
        # 与原来逐行写法一致
        lines += [
            f"device: {cfg.device}",
            f"enz_backbone: {cfg.enz_backbone}",
            "[paths]",
            f"  pdb_root: {cfg.pdb_root}",
            f"  pt_out_root: {cfg.pt_out_root}",
            f"  sdf_root: {cfg.sdf_root}",
            f"  train_csv: {cfg.train_csv}",
            f"  test_csv: {cfg.test_csv}",
            f"  out_dir: {cfg.out_dir}",
            "[hparams]",
            f"  emb_dim_enz: {cfg.emb_dim_enz}",
            f"  emb_dim_pl: {cfg.emb_dim_pl}",
            f"  proj_dim: {cfg.proj_dim}",
            f"  batch_size: {cfg.batch_size}",
            f"  lr: {cfg.lr}",
            f"  weight_decay: {cfg.weight_decay}",
            f"  epochs: {cfg.epochs}",
            f"  max_list_len: {cfg.max_list_len}",
            f"  temp: {cfg.temp}",
            f"  alpha: {cfg.alpha}",
            f"  lambda_rep: {cfg.lambda_rep}",
            f"  interaction: {cfg.interaction}",
            f"  bilinear_rank: {cfg.bilinear_rank}",
            f"  lambda_w_reg: {cfg.lambda_w_reg}",
            f"  ortho_reg: {cfg.ortho_reg}",
            f"  var_target: {cfg.var_target}",
            f"  lambda_var: {cfg.lambda_var}",
            f"  lambda_center: {cfg.lambda_center}",
            f"  plastic_diversify: {cfg.plastic_diversify}",
            f"  lambda_diversify: {cfg.lambda_diversify}",
            f"  seed: {cfg.seed}",
        ]
    if extra:
        for k, v in extra.items():
            if isinstance(v, dict):
                lines.append(f"[{k}]")
                for kk, vv in v.items():
                    lines.append(f"  {kk}: {vv}")
            else:
                lines.append(f"{k}: {v}")

    os.makedirs(out_dir, exist_ok=True)
    mode = "a" if append else "w"
    with open(os.path.join(out_dir, "run_config.txt"), mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 额外：JSON 版本（不影响原逻辑）
    if cfg and not append:
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as jf:
            json.dump(asdict(cfg), jf, indent=2, ensure_ascii=False)


# =============================================================================
# 构建 builder / dataset / model（保持默认逻辑）
# =============================================================================
def build_builders(cfg: TrainConfig):
    if cfg.enz_backbone == "GNN":
        bcfg = GNNBuilderConfig(pdb_dir=cfg.pdb_root, out_dir=cfg.pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GNNProteinGraphBuilder(bcfg, edge_mode="none")
    elif cfg.enz_backbone == "GVP":
        bcfg = GVPBuilderConfig(pdb_dir=cfg.pdb_root, out_dir=cfg.pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GVPProteinGraphBuilder(bcfg)
    else:
        bcfg = GNNBuilderConfig(pdb_dir=cfg.pdb_root, out_dir=cfg.pt_out_root, radius=10.0, embedder=[{"name":"esm"}])
        enzyme_builder = GNNProteinGraphBuilder(bcfg, edge_mode="none")
    plastic_featurizer = PlasticFeaturizer(config_path=None)
    return enzyme_builder, plastic_featurizer

def build_datasets(cfg: TrainConfig, enzyme_builder, plastic_featurizer):
    spec = MatrixSpec(csv_path=cfg.train_csv, pdb_root=cfg.pdb_root, sdf_root=cfg.sdf_root)
    ds_train = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="train",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=cfg.max_list_len,
    )
    ds_val = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="val",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=cfg.max_list_len,
    )
    return ds_train, ds_val

def build_models(cfg: TrainConfig, ds_train) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    # 探测塑料维度（保持与原逻辑一致）
    probe_feat = None
    for s in ds_train.cols:
        probe_feat = ds_train._get_plastic(s)
        if probe_feat is not None: break
    assert probe_feat is not None, "没有可用的塑料特征（检查 SDF/可读性）"
    in_dim_plastic = probe_feat.shape[-1]

    # enzyme backbone（保持与原逻辑相同）
    if cfg.enz_backbone == "GNN":
        enzyme_backbone = GNNBackbone(conv_type="gcn", hidden_dims=[128,128,128], out_dim=cfg.emb_dim_enz,
                                      dropout=0.1, residue_logits=False).to(cfg.device)
    elif cfg.enz_backbone == "GVP":
        enzyme_backbone = GVPBackbone(hidden_dims=[(128,16),(128,16),(128,16)], out_dim=cfg.emb_dim_enz,
                                      dropout=0.1, residue_logits=False).to(cfg.device)
    else:
        enzyme_backbone = SeqBackbone(hidden_dims=[256,256,256], out_dim=cfg.emb_dim_enz, dropout=0.1,
                                      pool="mean", residue_logits=False,
                                      feature_priority=["seq_x","x"]).to(cfg.device)

    plastic_tower = PolymerTower(in_dim=in_dim_plastic, hidden_dims=[256,128], out_dim=cfg.emb_dim_pl, dropout=0.1).to(cfg.device)
    projector  = TwinProjector(cfg.emb_dim_enz, cfg.emb_dim_pl, cfg.proj_dim).to(cfg.device)
    inter_head = InteractionHead(cfg.proj_dim, mode=cfg.interaction, rank=cfg.bilinear_rank).to(cfg.device)
    return enzyme_backbone, plastic_tower, projector, inter_head


# =============================================================================
# 可视化（保持逻辑不变）
# =============================================================================
def plot_curves(history: Dict[str, List[float]], out_dir: str):
    def _save(fig_name:str):
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fig_name)); plt.close()

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); _save("loss.png")

    plt.figure()
    plt.plot(history["train_hit@1"], label="train@1")
    plt.plot(history["val_hit@1"],   label="val@1")
    plt.plot(history["train_hit@3"], label="train@3")
    plt.plot(history["val_hit@3"],   label="val@3")
    plt.xlabel("epoch"); plt.ylabel("hit"); plt.legend(); _save("hit.png")

    plt.figure()
    plt.plot(history["train_pos_score"], label="train_pos")
    plt.plot(history["val_pos_score"],   label="val_pos")
    plt.plot(history["train_neg_score"], label="train_neg")
    plt.plot(history["val_neg_score"],   label="val_neg")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); _save("score.png")


# =============================================================================
# 主流程（默认行为不变）
# =============================================================================
def main():
    # argparse 仅作参数化入口；不传参=默认值=>行为与原脚本一致
    import argparse
    parser = argparse.ArgumentParser(description="Plaszyme listwise training (bilinear / factorized)")
    parser.add_argument("--config", type=str, default=None, help="可选 JSON 配置文件（覆盖默认）")
    args = parser.parse_args()

    # 构建配置
    cfg = TrainConfig()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        # 仅覆盖提供字段；不影响默认逻辑
        for k, v in user_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # 准备环境
    os.makedirs(cfg.out_dir, exist_ok=True)
    setup_logging(cfg.out_dir)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    set_seed(cfg.seed)

    # 构建器/数据
    enzyme_builder, plastic_featurizer = build_builders(cfg)
    ds_train, ds_val = build_datasets(cfg, enzyme_builder, plastic_featurizer)

    # 模型
    enzyme_backbone, plastic_tower, projector, inter_head = build_models(cfg, ds_train)

    # 可选：塑料塔预训练（与原逻辑一致，默认 False）
    if cfg.use_plastic_pretrain:
        logging.info("[PRETRAIN] loading co_matrix from %s", cfg.co_matrix_csv)
        co_matrix_df = pd.read_csv(cfg.co_matrix_csv, index_col=0)
        # 构造 features_df：直接从 ds_train 提取一次
        feature_dict = {}
        for pl_name in ds_train.cols:
            feat = ds_train._get_plastic(pl_name)
            if feat is not None:
                feat = torch.as_tensor(feat, dtype=torch.float32).view(-1).cpu().numpy()
                feature_dict[pl_name] = feat
        features_df = pd.DataFrame.from_dict(feature_dict, orient="index")
        logging.info("[PRETRAIN] features_df=%s, co_matrix_df=%s", features_df.shape, co_matrix_df.shape)

        plastic_tower.pretrain_with_co_matrix(
            features_df,
            co_matrix_df,
            loss_mode=cfg.pretrain_loss_mode,
            epochs=cfg.pretrain_epochs,
            batch_size=64,
            lr=1e-4,
            device=cfg.device,
        )
    torch.save(plastic_tower.state_dict(), os.path.join(cfg.out_dir, "plastic_pretrained.pt"))

    # 优化器（保持不变）
    optim = torch.optim.AdamW(
        list(enzyme_backbone.parameters())
        + list(plastic_tower.parameters())
        + list(projector.parameters())
        + list(inter_head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # DataLoader（保持不变）
    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_pairs, num_workers=0)
    loader_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_pairs, num_workers=0)

    print(f"[INFO] dataset(listwise) train={len(ds_train)} | val={len(ds_val)} | device={cfg.device}")
    save_run_config_txt(cfg.out_dir, cfg=cfg, extra={"dataset":{
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "num_plastics": len(ds_train.cols) if hasattr(ds_train, "cols") else None,
    }})

    # 训练循环（保持不变）
    history = {
        "train_loss":[], "val_loss":[],
        "train_hit@1":[], "val_hit@1":[],
        "train_hit@3":[], "val_hit@3":[],
        "train_pos_score":[], "val_pos_score":[],
        "train_neg_score":[], "val_neg_score":[],
    }
    best_val_hit1 = -1.0
    best_path = os.path.join(cfg.out_dir, f"best_{cfg.interaction}.pt")
    last_path = os.path.join(cfg.out_dir, f"last_{cfg.interaction}.pt")

    for epoch in range(1, cfg.epochs+1):
        tr = run_one_epoch(loader_train, enzyme_backbone, plastic_tower, projector, inter_head,
                           optim, cfg=cfg, train=True, epoch=epoch)
        va = run_one_epoch(loader_val,   enzyme_backbone, plastic_tower, projector, inter_head,
                           optimizer=None, cfg=cfg, train=False, epoch=epoch)

        # 记录
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_hit@1"].append(tr.get("hit@1", float("nan")))
        history["val_hit@1"].append(va.get("hit@1", float("nan")))
        history["train_hit@3"].append(tr.get("hit@3", float("nan")))
        history["val_hit@3"].append(va.get("hit@3", float("nan")))
        history["train_pos_score"].append(tr.get("pos_score", float("nan")))
        history["val_pos_score"].append(va.get("pos_score", float("nan")))
        history["train_neg_score"].append(tr.get("neg_score", float("nan")))
        history["val_neg_score"].append(va.get("neg_score", float("nan")))

        print(
            f"[Epoch {epoch:02d}] "
            f"train: loss={tr['loss']:.4f}, hit@1={tr.get('hit@1',float('nan')):.3f}, "
            f"hit@3={tr.get('hit@3',float('nan')):.3f}, pos={tr.get('pos_score',float('nan')):.3f}, "
            f"neg={tr.get('neg_score',float('nan')):.3f} | "
            f"val: loss={va['loss']:.4f}, hit@1={va.get('hit@1',float('nan')):.3f}, "
            f"hit@3={va.get('hit@3',float('nan')):.3f}, pos={va.get('pos_score',float('nan')):.3f}, "
            f"neg={va.get('neg_score',float('nan')):.3f}"
        )

        # —— 保存 last —— #
        torch.save({
            "epoch": epoch,
            "cfg": {
                "emb_dim_enz": cfg.emb_dim_enz, "emb_dim_pl": cfg.emb_dim_pl, "proj_dim": cfg.proj_dim,
                "temp": cfg.temp, "alpha": cfg.alpha, "lambda_rep": cfg.lambda_rep,
                "interaction": cfg.interaction, "bilinear_rank": cfg.bilinear_rank,
                "backbone": cfg.enz_backbone,
            },
            "enzyme_backbone": enzyme_backbone.state_dict(),
            "plastic_tower": plastic_tower.state_dict(),
            "projector": projector.state_dict(),
            "inter_head": inter_head.state_dict(),
            "optimizer": optim.state_dict(),
            "history": history,
        }, last_path)
        # logging.info("[CKPT] Saved last -> %s", last_path)

        cur = va.get("hit@1", -1.0)
        if cur > best_val_hit1:
            best_val_hit1 = cur
            torch.save({
                "epoch": epoch,
                "cfg": {
                    "emb_dim_enz":cfg.emb_dim_enz, "emb_dim_pl":cfg.emb_dim_pl, "proj_dim":cfg.proj_dim,
                    "temp":cfg.temp, "alpha":cfg.alpha, "lambda_rep":cfg.lambda_rep,
                    "interaction": cfg.interaction, "bilinear_rank": cfg.bilinear_rank,
                    "backbone": cfg.enz_backbone,
                },
                "enzyme_backbone": enzyme_backbone.state_dict(),
                "plastic_tower":   plastic_tower.state_dict(),
                "projector":       projector.state_dict(),
                "inter_head":      inter_head.state_dict(),
                "optimizer":       optim.state_dict(),
                "history":         history,
            }, best_path)
            print(f"[INFO] Saved best -> {best_path} (val hit@1={best_val_hit1:.3f})")

        plot_curves(history, cfg.out_dir)

    plot_curves(history, cfg.out_dir)
    print(f"[DONE] Best val hit@1={best_val_hit1:.3f} | best: {best_path}")

    # ========= 训练结束：自动用 TEST_CSV 评测（保持原逻辑） =========
    ckpt = torch.load(best_path, map_location=cfg.device)
    save_run_config_txt(
        cfg.out_dir, append=True, cfg=cfg,
        extra={"best": {"best_val_hit@1": best_val_hit1, "best_ckpt": best_path, "best_epoch": ckpt.get("epoch", None)}}
    )

    enzyme_backbone.load_state_dict(ckpt["enzyme_backbone"], strict=True)
    plastic_tower.load_state_dict(ckpt["plastic_tower"],   strict=True)
    projector.load_state_dict(ckpt["projector"],           strict=True)
    inter_head.load_state_dict(ckpt["inter_head"],         strict=True)

    test_out_csv = os.path.join(cfg.out_dir, "test_metrics.csv")
    test_matrix_csv = os.path.join(cfg.out_dir, "test_score_matrix.csv")
    _eval_on_csv(
        csv_path=cfg.test_csv,
        pdb_root=cfg.pdb_root,
        sdf_root=cfg.sdf_root,
        pt_out_root=cfg.pt_out_root,
        enzyme_backbone=enzyme_backbone,
        plastic_tower=plastic_tower,
        projector=projector,
        inter_head=inter_head,
        out_csv=test_out_csv,
        max_list_len=cfg.max_list_len,
        score_matrix_csv=test_matrix_csv,
        split_name="test",
        enz_backbone=cfg.enz_backbone,
    )


if __name__ == "__main__":
    main()