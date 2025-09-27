# train_listwise_bilinear.py
from __future__ import annotations

import os
import warnings
import math
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone
from src.models.seq_mlp.backbone import SeqBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.models.interaction_head import InteractionHead

# ================== 全局配置（路径集中在这里改） ==================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ENZ_BACKBONE  = "GVP"        # "GNN" | "GVP"

# 数据根路径
PDB_ROOT      = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb"
PT_OUT_ROOT   = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt"
SDF_ROOT      = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"
TRAIN_CSV     = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_trainset.csv"
TEST_CSV      = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_testset.csv"

OUT_DIR       = "./listwise_cos_gvp"
os.makedirs(OUT_DIR, exist_ok=True)

# 训练超参
EMB_DIM_ENZ   = 128
EMB_DIM_PL    = 128
PROJ_DIM      = 128
BATCH_SIZE    = 10
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
MAX_LIST_LEN  = 10

# InfoNCE 温度（用于可学习打分）
TEMP          = 0.2

# 正-正去同质化阈值与权重
ALPHA         = 0.4
LAMBDA_REP    = 0.1

# 交互方式（建议：'factorized_bilinear'）
INTERACTION: Literal["cos","bilinear","factorized_bilinear","hadamard_mlp","gated"] = "factorized_bilinear"

# 双线性/低秩超参
BILINEAR_RANK = 64         # 低秩 W≈U V^T 的秩
LAMBDA_W_REG  = 1e-4       # 对 W/U/V 的范数正则
ORTHO_REG     = 0.0        # >0 时对 U,V 做列正交（可选）

# 塔平衡与防塌缩正则
VAR_TARGET    = 1.0
LAMBDA_VAR    = 1e-3       # 保持各维方差接近目标
LAMBDA_CENTER = 1e-3       # 维度均值靠近 0
PLASTIC_DIVERSIFY = True   # 同一 item 正样本内的去同质化
LAMBDA_DIVERSIFY  = 0.05

# 塑料共降解预训练
USE_PLASTIC_PRETRAIN = False   # 是否先对塑料塔做共降解预训练
CO_MATRIX_CSV        = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastic_co_matrix.csv"
PRETRAIN_EPOCHS      = 10   # 预训练轮数
PRETRAIN_LOSS_MODE   = "contrastive"  # 或 "mse"



# =============== 稳定性：固定随机种子 ==================
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ================== 诊断模块（沿用/精简） ==================
import csv
from pathlib import Path
class EmbeddingDiagnostics:
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

# ================== 损失函数 ==================
def mp_infonce(scores: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """
    多正样本 InfoNCE:
    scores: [L] (可学习打分, 非归一化)
    labels: [L] ∈ {0,1}
    """
    assert scores.dim()==1 and labels.dim()==1
    pos = labels > 0.5
    if pos.sum()==0:
        return scores.new_tensor(0.0)
    logits = scores / tau
    num = torch.logsumexp(logits[pos], dim=0)
    den = torch.logsumexp(logits, dim=0)
    return -(num - den)

def positive_repulsion(z_pos: torch.Tensor, alpha: float=0.4) -> torch.Tensor:
    """ 同一 item 的正样本相互“区别开”，避免塑料原型主导。 """
    P = z_pos.size(0)
    if P <= 1: return z_pos.new_tensor(0.0)
    S = F.normalize(z_pos, dim=-1) @ F.normalize(z_pos, dim=-1).t()
    mask = ~torch.eye(P, dtype=torch.bool, device=z_pos.device)
    viol = F.relu(S[mask] - alpha)
    return viol.mean()

def center_var_reg(z: torch.Tensor, var_target: float=1.0) -> torch.Tensor:
    """
    简单的“均值-方差”正则：让维度均值靠近0, 方差靠近 var_target。
    """
    if z.numel()==0: return z.new_tensor(0.0)
    mu  = z.mean(dim=0)
    var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    loss_center = (mu.square()).mean()
    loss_var    = ((var - var_target).square()).mean()
    return loss_center + loss_var

# ================== 投影与模型 ==================
class TwinProjector(nn.Module):
    def __init__(self, in_e:int, in_p:int, out:int):
        super().__init__()
        self.proj_e = nn.Linear(in_e, out, bias=False)
        self.proj_p = nn.Linear(in_p, out, bias=False)
        nn.init.xavier_uniform_(self.proj_e.weight)
        nn.init.xavier_uniform_(self.proj_p.weight)

    def forward(self, e: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.proj_e(e), self.proj_p(p)

# ================== 训练期指标（小批内） ==================
@torch.no_grad()
def listwise_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(1,3)) -> Dict[str,float]:
    out = {}
    mask_pos = labels>0.5; mask_neg = ~mask_pos
    if mask_pos.any(): out["pos_score"] = float(scores[mask_pos].mean().item())
    else:              out["pos_score"] = float("nan")
    if mask_neg.any(): out["neg_score"] = float(scores[mask_neg].mean().item())
    else:              out["neg_score"] = float("nan")
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

# ================== 一个 epoch ==================
def run_one_epoch(
    loader: DataLoader,
    enzyme_backbone: nn.Module,
    plastic_tower: nn.Module,
    projector: TwinProjector,
    inter_head: InteractionHead,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    train: bool,
    epoch: int,
    # ---- 新增开关与可调参数 ----
    enable_list_mitigation: bool = True,
    max_list_len_train: int = 32,
    neg_per_item: int = 31,
    hard_neg_cand: int = 200,
    hard_neg_ratio: float = 0.7,
    temp_base: float = TEMP,
) -> Dict[str, float]:

    split = "train" if train else "val"
    enzyme_backbone.train() if train else enzyme_backbone.eval()
    plastic_tower.train()   if train else plastic_tower.eval()
    projector.train()       if train else projector.eval()
    inter_head.train()      if train else inter_head.eval()

    diag = EmbeddingDiagnostics(OUT_DIR)
    max_diag_batches = 2
    diag_cnt = 0

    total_loss, n_items = 0.0, 0
    metrics_items: List[Dict[str, float]] = []

    # 仅在启用缓解策略时统计子列表长度
    sum_L_sub, cnt_L_sub = 0, 0

    for batch in loader:
        g_batch = batch["enzyme_graph"].to(DEVICE)
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

            plast  = plast.to(DEVICE)          # [L, Din_p]
            labels = labels.to(DEVICE)         # [L]
            L_full = int(labels.numel())

            # 塑料塔 + 投影（完整列表一次算完）
            z_p_back = plastic_tower(plast)               # [L, Dp]
            z_p_raw  = projector.proj_p(z_p_back)         # [L, D]

            if diag_cnt < max_diag_batches and b_idx == 0:
                diag.report(z_p_raw, epoch=epoch, split=split, where="plastic", space="raw")

            z_e_b = z_e_raw[b_idx:b_idx + 1, :]           # [1, D]
            scores_full = inter_head.score(z_e_b, z_p_raw) # [L]

            if enable_list_mitigation:
                # ---- 全正 + 难负/随机负 子采样 ----
                idx_all = torch.arange(L_full, device=DEVICE)
                idx_pos = idx_all[labels > 0.5]
                idx_neg = idx_all[labels <= 0.5]

                keep_pos = idx_pos
                # 目标负样本数受 neg_per_item 与 max_list_len_train 双约束
                neg_target = max(0, min(neg_per_item, max_list_len_train - int(len(keep_pos))))
                neg_target = min(neg_target, int(len(idx_neg)))

                if neg_target > 0 and len(idx_neg) > 0:
                    if hard_neg_cand > 0:
                        cand = idx_neg[torch.randperm(len(idx_neg), device=DEVICE)[:min(hard_neg_cand, len(idx_neg))]]
                        cand_scores = scores_full[cand]
                        k_hard = int(round(hard_neg_ratio * neg_target))
                        k_hard = min(k_hard, int(len(cand)))
                        hard_idx = cand[torch.topk(cand_scores, k=k_hard, largest=True).indices] if k_hard > 0 else cand[:0]

                        remain = neg_target - int(len(hard_idx))
                        if remain > 0:
                            hard_set = set(hard_idx.tolist())
                            pool = [i for i in idx_neg.tolist() if i not in hard_set]
                            if len(pool) > 0:
                                pool_t = torch.tensor(pool, device=DEVICE, dtype=torch.long)
                                rand_idx = pool_t[torch.randperm(len(pool_t), device=DEVICE)[:remain]]
                            else:
                                rand_idx = idx_neg[:0]
                            keep_neg = torch.cat([hard_idx, rand_idx], dim=0)
                        else:
                            keep_neg = hard_idx
                    else:
                        keep_neg = idx_neg[torch.randperm(len(idx_neg), device=DEVICE)[:neg_target]]
                else:
                    keep_neg = idx_neg[:0]

                keep_idx = torch.cat([keep_pos, keep_neg], dim=0)
                if len(keep_idx) > 1:
                    keep_idx = keep_idx[torch.randperm(len(keep_idx), device=DEVICE)]

                labels_sub = labels[keep_idx]          # [L_sub]
                z_p_sub    = z_p_raw[keep_idx]         # [L_sub]
                scores_sub = scores_full[keep_idx]     # [L_sub]
                L_sub      = int(labels_sub.numel())

                # 自适应温度 + 长度归一
                tau_eff = temp_base / max(1.0, math.log(1.0 + float(L_sub)))
                loss_ce = mp_infonce(scores_sub, labels_sub, tau=tau_eff) / math.log(1.0 + float(L_sub))

                # 正样本多样性（对子列表）
                if PLASTIC_DIVERSIFY and (labels_sub > 0.5).sum() >= 2:
                    loss_div = positive_repulsion(z_p_sub[labels_sub > 0.5], alpha=ALPHA)
                else:
                    loss_div = scores_sub.new_tensor(0.0)

                loss_reg_e = center_var_reg(z_e_raw, VAR_TARGET)
                loss_reg_p = center_var_reg(z_p_sub, VAR_TARGET)

                w_reg = scores_sub.new_tensor(0.0)
                if INTERACTION == "bilinear":
                    w_reg = w_reg + LAMBDA_W_REG * (inter_head.W.square().mean())
                elif INTERACTION == "factorized_bilinear":
                    w_reg = w_reg + LAMBDA_W_REG * (
                        inter_head.U.weight.square().mean() + inter_head.V.weight.square().mean()
                    )
                    if ORTHO_REG > 0:
                        w_reg = w_reg + ORTHO_REG * inter_head.orthogonal_regularizer()

                loss_item = loss_ce + LAMBDA_DIVERSIFY * loss_div + LAMBDA_CENTER * (loss_reg_e + loss_reg_p) + w_reg

                # 训练期与指标口径使用子列表
                metrics_items.append(listwise_metrics(scores_sub.detach(), labels_sub.detach(), ks=(1, 3)))
                sum_L_sub += L_sub
                cnt_L_sub += 1

            else:
                # ---- 不启用缓解：保持原逻辑 ----
                scores = scores_full
                loss_ce = mp_infonce(scores, labels, tau=temp_base)

                if PLASTIC_DIVERSIFY and (labels > 0.5).sum() >= 2:
                    loss_div = positive_repulsion(z_p_raw[labels > 0.5], alpha=ALPHA)
                else:
                    loss_div = scores.new_tensor(0.0)

                loss_reg_e = center_var_reg(z_e_raw, VAR_TARGET)
                loss_reg_p = center_var_reg(z_p_raw, VAR_TARGET)

                w_reg = scores.new_tensor(0.0)
                if INTERACTION == "bilinear":
                    w_reg = w_reg + LAMBDA_W_REG * (inter_head.W.square().mean())
                elif INTERACTION == "factorized_bilinear":
                    w_reg = w_reg + LAMBDA_W_REG * (
                        inter_head.U.weight.square().mean() + inter_head.V.weight.square().mean()
                    )
                    if ORTHO_REG > 0:
                        w_reg = w_reg + ORTHO_REG * inter_head.orthogonal_regularizer()

                loss_item = loss_ce + LAMBDA_DIVERSIFY * loss_div + LAMBDA_CENTER * (loss_reg_e + loss_reg_p) + w_reg

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
    if enable_list_mitigation and cnt_L_sub > 0:
        reduced["avg_L_sub"] = float(sum_L_sub / cnt_L_sub)
    reduced["loss"]  = total_loss / max(n_items, 1)
    reduced["items"] = float(n_items)
    return reduced

# ==================（新增）评测工具：与预测脚本一致 ==================
@torch.no_grad()
def compute_metrics(all_labels: List[np.ndarray], all_scores: List[np.ndarray], ks=(1,3,5)) -> Dict[str,float]:
    """
    listwise 排序指标 + micro/macro（二分类阈值=0，仅用于参考）
    """
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


@torch.no_grad()
def _eval_on_csv(csv_path: str,
                 pdb_root: str,
                 sdf_root: str,
                 enzyme_backbone: nn.Module,
                 plastic_tower: nn.Module,
                 projector: TwinProjector,
                 inter_head: InteractionHead,
                 out_csv: str,
                 max_list_len: int = 10,
                 score_matrix_csv: Optional[str] = None,
                 split_name: str = "val"):
    """
    与训练管线一致的 listwise 评估：读取 csv -> 构建数据集 -> 逐样本打分 -> 计算指标 -> 写指标 CSV。
    另外（可选）导出“分数矩阵 CSV”：行=酶ID，列=塑料名称（ds.cols），值=模型对所有塑料的打分。
    """
    # --- 构建与训练一致的 builder ---
    if ENZ_BACKBONE == "GNN":
        cfg = GNNBuilderConfig(
            pdb_dir=pdb_root,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    elif ENZ_BACKBONE == "GVP":
        cfg = GVPBuilderConfig(
            pdb_dir=pdb_root,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GVPProteinGraphBuilder(cfg)
    elif ENZ_BACKBONE == "MLP":
        cfg = GNNBuilderConfig(
            pdb_dir=pdb_root,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name": "esm"}],
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")


    plastic_featurizer = PlasticFeaturizer(config_path=None)
    spec = MatrixSpec(csv_path=csv_path, pdb_root=pdb_root, sdf_root=sdf_root)
    ds = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="full",split_ratio=None,
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=max_list_len, return_names=True,
    )
    print(f"[EVAL] csv={csv_path} | split='{split_name}' | usable_samples={len(ds)} | plastics={len(ds.cols)}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pairs)

    enzyme_backbone.eval(); plastic_tower.eval(); projector.eval(); inter_head.eval()

    # -------- 指标（按样本的子列表计算） --------
    all_labels, all_scores = [], []

    # -------- 分数矩阵准备：一次性预计算“全部塑料”的表示 --------
    want_matrix = bool(score_matrix_csv)
    if want_matrix:
        # 收集所有塑料的原始特征（保持与 ds.cols 一一对应）
        pl_feats = []
        valid_mask = []
        for s in ds.cols:
            feat = ds._get_plastic(s)  # np.ndarray 或 torch.Tensor
            if feat is None:
                pl_feats.append(torch.zeros(1))  # 占位
                valid_mask.append(False)
            else:
                t = torch.as_tensor(feat, dtype=torch.float32, device=DEVICE)
                pl_feats.append(t)
                valid_mask.append(True)
        # 对齐成 [M, in_dim]
        in_dim_plastic = max(f.numel() for f in pl_feats if f.numel() > 1) if any(valid_mask) else 0
        P_list = []
        for ok, f in zip(valid_mask, pl_feats):
            if not ok:
                P_list.append(torch.zeros(in_dim_plastic, device=DEVICE))
            else:
                vec = f.flatten()
                if vec.numel() != in_dim_plastic:
                    # 维度不一致时填充/截断到 in_dim_plastic（防御，理论上不该发生）
                    v = torch.zeros(in_dim_plastic, device=DEVICE)
                    n = min(in_dim_plastic, vec.numel()); v[:n] = vec[:n]
                    vec = v
                P_list.append(vec)
        P = torch.stack(P_list, dim=0) if in_dim_plastic > 0 else torch.empty(0, 0, device=DEVICE)  # [M, in_dim]
        # 通过塑料塔与投影，得到全部塑料的投影空间表示
        if P.numel() > 0:
            Zp_all = projector.proj_p(plastic_tower(P))  # [M, D]
        else:
            Zp_all = torch.empty(0, projector.proj_e.out_features, device=DEVICE)  # [0, D]
        # 准备矩阵的 DataFrame 容器
        score_rows = []  # 每行一个 dict：{"enzyme_id": ..., pl1:score, pl2:score, ...}
        pl_names = list(ds.cols)

    # -------- 遍历样本 --------
    for idx, batch in enumerate(loader):
        g = batch["enzyme_graph"].to(DEVICE)
        enz_vec = enzyme_backbone(g)              # [1, De]
        z_e = projector.proj_e(enz_vec)           # [1, D]

        item   = batch["items"][0]
        plast  = item["plastic_list"].to(DEVICE)  # [L, in_dim]
        labels = item["relevance"].cpu().numpy()  # [L]
        z_p    = projector.proj_p(plastic_tower(plast))  # [L, D]
        scores = inter_head.score(z_e, z_p).detach().cpu().numpy()

        all_labels.append(labels)
        all_scores.append(scores)

        if want_matrix:
            # 识别该样本的 enzyme_id
            enzyme_id = str(
                item.get("enzyme_id")
                or item.get("pdb_id")
                or item.get("id")
                or f"sample_{idx}"
            )
            # 对 **全部**塑料打分
            if Zp_all.numel() > 0:
                s_all = inter_head.score(z_e, Zp_all).detach().cpu().numpy()  # [M]
                row = {"enzyme_id": enzyme_id}
                row.update({pl: float(s) for pl, s in zip(pl_names, s_all)})
            else:
                row = {"enzyme_id": enzyme_id}  # 没有塑料特征可用时只写 ID
            score_rows.append(row)

    # -------- 保存指标 CSV --------
    metrics = compute_metrics(all_labels, all_scores, ks=(1,3,5))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"[TEST] saved metrics to: {out_csv}")
    print(pd.DataFrame([metrics]))

    # -------- 保存分数矩阵 CSV --------
    if want_matrix:
        os.makedirs(os.path.dirname(score_matrix_csv) or ".", exist_ok=True)
        # 确保列顺序：enzyme_id + ds.cols
        cols = ["enzyme_id"] + (pl_names if Zp_all.numel() > 0 else [])
        df_mat = pd.DataFrame(score_rows)
        # 按列重排（缺失的列会自动补 NaN）
        df_mat = df_mat.reindex(columns=cols)
        df_mat.to_csv(score_matrix_csv, index=False)
        print(f"[TEST] saved score matrix to: {score_matrix_csv}  (rows={len(df_mat)}, cols={len(df_mat.columns)})")

# ---- 极简：把本次运行的所有关键参数写到 OUT_DIR/run_config.txt ----
from datetime import datetime

def save_run_config_txt(out_dir: str, *, append: bool=False, extra: dict | None=None):
    lines = []
    if not append:
        lines.append("# Plaszyme listwise training config")
        lines.append(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"device: {DEVICE}")
    lines.append(f"enz_backbone: {ENZ_BACKBONE}")
    lines.append("[paths]")
    lines.append(f"  pdb_root: {PDB_ROOT}")
    lines.append(f"  pt_out_root: {PT_OUT_ROOT}")
    lines.append(f"  sdf_root: {SDF_ROOT}")
    lines.append(f"  train_csv: {TRAIN_CSV}")
    lines.append(f"  test_csv: {TEST_CSV}")
    lines.append(f"  out_dir: {OUT_DIR}")
    lines.append("[hparams]")
    lines.append(f"  emb_dim_enz: {EMB_DIM_ENZ}")
    lines.append(f"  emb_dim_pl: {EMB_DIM_PL}")
    lines.append(f"  proj_dim: {PROJ_DIM}")
    lines.append(f"  batch_size: {BATCH_SIZE}")
    lines.append(f"  lr: {LR}")
    lines.append(f"  weight_decay: {WEIGHT_DECAY}")
    lines.append(f"  epochs: {EPOCHS}")
    lines.append(f"  max_list_len: {MAX_LIST_LEN}")
    lines.append(f"  temp: {TEMP}")
    lines.append(f"  alpha: {ALPHA}")
    lines.append(f"  lambda_rep: {LAMBDA_REP}")
    lines.append(f"  interaction: {INTERACTION}")
    lines.append(f"  bilinear_rank: {BILINEAR_RANK}")
    lines.append(f"  lambda_w_reg: {LAMBDA_W_REG}")
    lines.append(f"  ortho_reg: {ORTHO_REG}")
    lines.append(f"  var_target: {VAR_TARGET}")
    lines.append(f"  lambda_var: {LAMBDA_VAR}")
    lines.append(f"  lambda_center: {LAMBDA_CENTER}")
    lines.append(f"  plastic_diversify: {PLASTIC_DIVERSIFY}")
    lines.append(f"  lambda_diversify: {LAMBDA_DIVERSIFY}")
    lines.append(f"  seed: {42}")

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


# ================== 组装主程序 ==================
def main():
    # ----- 构建器 -----
    if ENZ_BACKBONE == "GNN":
        cfg = GNNBuilderConfig(
            pdb_dir=PDB_ROOT,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    elif ENZ_BACKBONE == "GVP":
        cfg = GVPBuilderConfig(
            pdb_dir=PDB_ROOT,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GVPProteinGraphBuilder(cfg)
    elif ENZ_BACKBONE == "MLP":
        cfg = GNNBuilderConfig(
            pdb_dir=PDB_ROOT,
            out_dir=PT_OUT_ROOT,
            radius=10.0,
            embedder=[{"name": "esm"}],
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    else:
        warnings.warn("Unknown backbone"); return

    plastic_featurizer = PlasticFeaturizer(config_path=None)

    # ----- 数据集 -----
    spec = MatrixSpec(
        csv_path=TRAIN_CSV,
        pdb_root=PDB_ROOT,
        sdf_root=SDF_ROOT,
    )
    ds_train = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="train",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=MAX_LIST_LEN,
    )
    ds_val = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="val",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=MAX_LIST_LEN,
    )

    # ----- 塔与维度 -----
    # 探测塑料维度
    probe_feat = None
    for s in ds_train.cols:
        probe_feat = ds_train._get_plastic(s)
        if probe_feat is not None: break
    assert probe_feat is not None, "没有可用的塑料特征（检查 SDF/可读性）"
    in_dim_plastic = probe_feat.shape[-1]

    if ENZ_BACKBONE == "GNN":
        enzyme_backbone = GNNBackbone(
            conv_type="gcn", hidden_dims=[128,128,128], out_dim=EMB_DIM_ENZ,
            dropout=0.1, residue_logits=False
        ).to(DEVICE)
    elif ENZ_BACKBONE == "GVP":
        enzyme_backbone = GVPBackbone(
            hidden_dims=[(128,16),(128,16),(128,16)], out_dim=EMB_DIM_ENZ,
            dropout=0.1, residue_logits=False
        ).to(DEVICE)

    elif ENZ_BACKBONE == "MLP":
        enzyme_backbone = SeqBackbone(
            hidden_dims=[256, 256, 256], out_dim=EMB_DIM_ENZ,
            dropout=0.1,
            pool="mean",  # 或 "max"
            residue_logits=False,
            # in_dim=None,           # 让它自动从首个 batch 推断
            feature_priority=["seq_x", "x"],  # builder 放到哪个字段都能吃
        ).to(DEVICE)

    plastic_tower = PolymerTower(
        in_dim=in_dim_plastic, hidden_dims=[256,128], out_dim=EMB_DIM_PL, dropout=0.1
    ).to(DEVICE)

    # ==== 塑料塔预训练 ====
    if USE_PLASTIC_PRETRAIN:
        print(f"[PRETRAIN] loading co_matrix from {CO_MATRIX_CSV}")
        co_matrix_df = pd.read_csv(CO_MATRIX_CSV, index_col=0)

        # 构造 features_df：直接从 ds_train 提取一次
        feature_dict = {}
        for pl_name in ds_train.cols:
            feat = ds_train._get_plastic(pl_name)
            if feat is not None:
                feat = torch.as_tensor(feat, dtype=torch.float32).view(-1).cpu().numpy()
                feature_dict[pl_name] = feat
        features_df = pd.DataFrame.from_dict(feature_dict, orient="index")
        print(f"[PRETRAIN] features_df={features_df.shape}, co_matrix_df={co_matrix_df.shape}")

        plastic_tower.pretrain_with_co_matrix(
            features_df,
            co_matrix_df,
            loss_mode=PRETRAIN_LOSS_MODE,
            epochs=PRETRAIN_EPOCHS,
            batch_size=64,
            lr=1e-4,
            device=DEVICE,
        )
    torch.save(plastic_tower.state_dict(), os.path.join(OUT_DIR, "plastic_pretrained.pt"))

    projector  = TwinProjector(EMB_DIM_ENZ, EMB_DIM_PL, PROJ_DIM).to(DEVICE)
    inter_head = InteractionHead(PROJ_DIM, mode=INTERACTION, rank=BILINEAR_RANK).to(DEVICE)

    # ----- 优化器 -----
    optim = torch.optim.AdamW(
        list(enzyme_backbone.parameters())
        + list(plastic_tower.parameters())
        + list(projector.parameters())
        + list(inter_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ----- DataLoader -----
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pairs, num_workers=0)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_pairs, num_workers=0)

    print(f"[INFO] dataset(listwise) train={len(ds_train)} | val={len(ds_val)} | device={DEVICE}")

    save_run_config_txt(
        OUT_DIR,
        extra={
            "dataset": {
                "train_size": len(ds_train),
                "val_size": len(ds_val),
                "num_plastics": len(ds_train.cols) if hasattr(ds_train, "cols") else None,
            }
        },
    )

    # ----- 训练循环 -----
    history = {
        "train_loss":[], "val_loss":[],
        "train_hit@1":[], "val_hit@1":[],
        "train_hit@3":[], "val_hit@3":[],
        "train_pos_score":[], "val_pos_score":[],
        "train_neg_score":[], "val_neg_score":[],
    }
    best_val_hit1 = -1.0
    best_path = os.path.join(OUT_DIR, f"best_{INTERACTION}.pt")

    for epoch in range(1, EPOCHS+1):
        tr = run_one_epoch(loader_train, enzyme_backbone, plastic_tower, projector, inter_head,
                           optim, train=True, epoch=epoch)
        va = run_one_epoch(loader_val,   enzyme_backbone, plastic_tower, projector, inter_head,
                           optimizer=None, train=False, epoch=epoch)

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

        cur = va.get("hit@1", -1.0)
        if cur > best_val_hit1:
            best_val_hit1 = cur
            torch.save({
                "epoch": epoch,
                "cfg": {
                    "emb_dim_enz":EMB_DIM_ENZ, "emb_dim_pl":EMB_DIM_PL, "proj_dim":PROJ_DIM,
                    "temp":TEMP, "alpha":ALPHA, "lambda_rep":LAMBDA_REP,
                    "interaction": INTERACTION, "bilinear_rank": BILINEAR_RANK
                },
                "enzyme_backbone": enzyme_backbone.state_dict(),
                "plastic_tower":   plastic_tower.state_dict(),
                "projector":       projector.state_dict(),
                "inter_head":      inter_head.state_dict(),
                "optimizer":       optim.state_dict(),
                "history":         history,
            }, best_path)
            print(f"[INFO] Saved best -> {best_path} (val hit@1={best_val_hit1:.3f})")

        plot_curves(history, OUT_DIR)

    plot_curves(history, OUT_DIR)
    print(f"[DONE] Best val hit@1={best_val_hit1:.3f} | best: {best_path}")

    # ========= 训练结束：自动用 TEST_CSV 评测 =========
    ckpt = torch.load(best_path, map_location=DEVICE)

    # 追加写入最佳结果信息
    save_run_config_txt(
        OUT_DIR,
        append=True,
        extra={
            "best": {
                "best_val_hit@1": best_val_hit1,
                "best_ckpt": best_path,
                "best_epoch": ckpt.get("epoch", None),
            }
        },
    )

    enzyme_backbone.load_state_dict(ckpt["enzyme_backbone"], strict=True)
    plastic_tower.load_state_dict(ckpt["plastic_tower"],   strict=True)
    projector.load_state_dict(ckpt["projector"],           strict=True)
    inter_head.load_state_dict(ckpt["inter_head"],         strict=True)

    # 生成输出路径
    test_out_csv = os.path.join(OUT_DIR, "test_metrics.csv")
    test_matrix_csv = os.path.join(OUT_DIR, "test_score_matrix.csv")  # 分数矩阵

    # 评测 + 导出矩阵
    _eval_on_csv(
        csv_path=TEST_CSV,
        pdb_root=PDB_ROOT,
        sdf_root=SDF_ROOT,
        enzyme_backbone=enzyme_backbone,
        plastic_tower=plastic_tower,
        projector=projector,
        inter_head=inter_head,
        out_csv=test_out_csv,
        max_list_len=MAX_LIST_LEN,
        score_matrix_csv=test_matrix_csv,
        split_name="test"
    )

# ================== 可视化 ==================
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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    main()