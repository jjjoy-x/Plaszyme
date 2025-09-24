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

# ===== 你的模块 =====
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer

# ================== 训练配置 ==================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ENZ_BACKBONE  = "GNN"        # "GNN" | "GVP"
EMB_DIM_ENZ   = 128
EMB_DIM_PL    = 128
PROJ_DIM      = 128

BATCH_SIZE    = 10
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50

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

# 塔平衡与防塌缩正则（建议保留，防止塑料塔“压着走”）
VAR_TARGET    = 1.0
LAMBDA_VAR    = 1e-3       # 保持各维方差接近目标
LAMBDA_CENTER = 1e-3       # 维度均值靠近 0
PLASTIC_DIVERSIFY = True   # 同一 item 正样本内的去同质化
LAMBDA_DIVERSIFY  = 0.05

OUT_DIR       = "./listwise_bilin_v1"
os.makedirs(OUT_DIR, exist_ok=True)

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

# ================== 交互头 ==================
class InteractionHead(nn.Module):
    """
    可切换的蛋白-塑料交互打分：
      - cos: 余弦相似（基线）
      - bilinear: s = z_e^T W z_p
      - factorized_bilinear: W≈U V^T, 低秩稳训练（推荐）
      - hadamard_mlp: s = MLP([z_e ⊙ z_p, z_e, z_p])
      - gated: s = (σ(g_p(z_p))*z_e)·(σ(g_e(z_e))*z_p)
    输入/期望：
      - 先经过 projector 投到 PROJ_DIM，再喂入本模块
    """
    def __init__(self, dim: int, mode: str="factorized_bilinear",
                 rank: int=64, mlp_hidden: int=256):
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
                nn.Linear(d*3, mlp_hidden),
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
        """ 对 factorized 的 U,V 施加列正交（可选）。 """
        if self.mode != "factorized_bilinear":
            return torch.tensor(0.0, device=self._device())
        loss = 0.0
        for M in [self.U.weight, self.V.weight]:
            # 让 M M^T 更接近对角
            MMt = M @ M.t()
            I = torch.eye(MMt.size(0), device=MMt.device)
            loss = loss + (MMt - I).pow(2).mean()
        return loss

    def _device(self):
        return next(self.parameters()).device

    def score(self, z_e: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """
        z_e: [1, D]  z_p: [L, D] -> scores: [L]
        """
        if self.mode == "cos":
            z1 = F.normalize(z_e, dim=-1)
            z2 = F.normalize(z_p, dim=-1)
            return (z1 @ z2.t()).squeeze(0)  # [L]

        if self.mode == "bilinear":
            # (1,D) W (D,L) = (1,L)
            return (z_e @ self.W @ z_p.t()).squeeze(0)

        if self.mode == "factorized_bilinear":
            # 低秩：⟨U z_e, V z_p⟩
            ue = self.U(z_e)         # [1, R]
            vp = self.V(z_p)         # [L, R]
            return (ue * vp).sum(dim=-1)  # [L]

        if self.mode == "hadamard_mlp":
            L = z_p.size(0)
            ze_rep = z_e.expand(L, -1)
            feats = torch.cat([ze_rep * z_p, ze_rep, z_p], dim=-1)  # [L, 3D]
            return self.mlp(feats).squeeze(-1)  # [L]

        if self.mode == "gated":
            # 双向门控后做线性打分
            ge = torch.sigmoid(self.g_e(z_e))    # [1,D]
            gp = torch.sigmoid(self.g_p(z_p))    # [L,D]
            ze = ge * z_e                        # [1,D]
            zp = gp * z_p                        # [L,D]
            L = zp.size(0)
            ze_rep = ze.expand(L, -1)
            return self.out(ze_rep * zp).squeeze(-1)  # [L]

        raise RuntimeError("unreachable")

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

# ================== 指标 ==================
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

# ================== 单个 epoch ==================
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
) -> Dict[str,float]:

    split = "train" if train else "val"
    mode  = "train" if train else "eval"
    enzyme_backbone.train() if train else enzyme_backbone.eval()
    plastic_tower.train()   if train else plastic_tower.eval()
    projector.train()       if train else projector.eval()
    inter_head.train()      if train else inter_head.eval()

    diag = EmbeddingDiagnostics(OUT_DIR)
    max_diag_batches=2; diag_cnt=0

    total_loss, n_items = 0.0, 0
    metrics_items: List[Dict[str,float]]=[]

    for batch in loader:
        g_batch = batch["enzyme_graph"].to(DEVICE)
        enz_vec = enzyme_backbone(g_batch)  # [B, De]

        # 先做一次 projector（避免重复）
        z_e_raw = projector.proj_e(enz_vec)         # [B, D]
        # 诊断
        if diag_cnt < max_diag_batches:
            diag.report(z_e_raw, epoch=epoch, split=split, where="protein", space="raw")

        loss_batch = 0.0
        items_in_batch = 0

        for b_idx, item in enumerate(batch["items"]):
            plast   = item["plastic_list"]
            labels  = item["relevance"]
            if plast is None or plast.numel()==0: continue

            plast  = plast.to(DEVICE)       # [L, Dp_in]
            labels = labels.to(DEVICE)      # [L]

            # 塑料塔 & projector
            z_p_back = plastic_tower(plast)           # [L, Dp]
            z_p_raw  = projector.proj_p(z_p_back)     # [L, D]

            # 诊断（仅首条 item 降噪日志）
            if diag_cnt < max_diag_batches and b_idx==0:
                diag.report(z_p_raw, epoch=epoch, split=split, where="plastic", space="raw")

            # 单个蛋白 b 的向量
            z_e_b = z_e_raw[b_idx:b_idx+1, :]         # [1, D]

            # ---- 可学习打分 ----
            scores = inter_head.score(z_e_b, z_p_raw) # [L]

            # ---- 多正样本 InfoNCE ----
            loss_ce = mp_infonce(scores, labels, tau=TEMP)

            # ---- 可选：正样本多样性（同一蛋白下正塑料互斥）----
            if PLASTIC_DIVERSIFY and (labels>0.5).sum() >= 2:
                loss_div = positive_repulsion(z_p_raw[labels>0.5], alpha=ALPHA)
            else:
                loss_div = scores.new_tensor(0.0)

            # ---- 塔平衡与防塌缩（对当前小批的蛋白、该 item 的塑料）----
            loss_reg_e  = center_var_reg(z_e_raw, VAR_TARGET)
            loss_reg_p  = center_var_reg(z_p_raw, VAR_TARGET)

            # ---- 双线性参数正则 ----
            w_reg = scores.new_tensor(0.0)
            if INTERACTION == "bilinear":
                w_reg = w_reg + LAMBDA_W_REG * (inter_head.W.square().mean())
            elif INTERACTION == "factorized_bilinear":
                w_reg = w_reg + LAMBDA_W_REG * (inter_head.U.weight.square().mean()
                                                + inter_head.V.weight.square().mean())
                if ORTHO_REG > 0:
                    w_reg = w_reg + ORTHO_REG * inter_head.orthogonal_regularizer()

            loss_item = loss_ce \
                        + LAMBDA_DIVERSIFY * loss_div \
                        + LAMBDA_CENTER * (loss_reg_e + loss_reg_p) \
                        + w_reg

            if train:
                loss_batch = loss_batch + loss_item

            total_loss += float(loss_item.item())
            n_items    += 1
            items_in_batch += 1

            # 指标
            metrics_items.append(listwise_metrics(scores, labels, ks=(1,3)))

        if train and items_in_batch>0:
            optimizer.zero_grad(set_to_none=True)
            (loss_batch / items_in_batch).backward()
            optimizer.step()

        diag_cnt += 1

    reduced = reduce_metrics(metrics_items)
    reduced["loss"]  = total_loss / max(n_items,1)
    reduced["items"] = float(n_items)
    return reduced

# ================== 组装主程序 ==================
def main():
    # ----- 构建器 -----
    if ENZ_BACKBONE == "GNN":
        cfg = GNNBuilderConfig(
            pdb_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb",
            out_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt",
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    elif ENZ_BACKBONE == "GVP":
        cfg = GVPBuilderConfig(
            pdb_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb",
            out_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt",
            radius=10.0,
            embedder=[{"name":"esm"}],
        )
        enzyme_builder = GVPProteinGraphBuilder(cfg)
    else:
        warnings.warn("Unknown backbone"); return

    plastic_featurizer = PlasticFeaturizer(config_path=None)

    # ----- 数据集 -----
    spec = MatrixSpec(
        csv_path="/Users/shulei/PycharmProjects/Plaszyme/dataset/plastics_onehot_trainset.csv",
        pdb_root="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb",
        sdf_root="/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new",
    )
    ds_train = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="train",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=10,
    )
    ds_val = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="val",
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=10,
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
    else:
        enzyme_backbone = GVPBackbone(
            hidden_dims=[(128,16),(128,16),(128,16)], out_dim=EMB_DIM_ENZ,
            dropout=0.1, residue_logits=False
        ).to(DEVICE)

    plastic_tower = PolymerTower(
        in_dim=in_dim_plastic, hidden_dims=[256,128], out_dim=EMB_DIM_PL, dropout=0.1
    ).to(DEVICE)

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