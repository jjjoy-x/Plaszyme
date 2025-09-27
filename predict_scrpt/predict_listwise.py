# predict_listwise_bilinear.py
import os
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# ==== 训练期模块 ====
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.models.gnn.backbone import GNNBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer
from train_listwise import TwinProjector, InteractionHead  # 复用训练定义

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- utils: metrics ----------------
@torch.no_grad()
def compute_metrics(all_labels: List[np.ndarray], all_scores: List[np.ndarray], ks=(1,3,5)) -> Dict[str,float]:
    results = {f"hit@{k}": [] for k in ks}
    results.update({f"recall@{k}": [] for k in ks})
    all_preds_bin, all_true_bin = [], []

    for labels, scores in zip(all_labels, all_scores):
        order = np.argsort(-scores)
        for k in ks:
            topk = order[:k]
            hit = (labels[topk] > 0.5).any()
            rec = float(labels[topk].sum()) / max(float(labels.sum()), 1.0)
            results[f"hit@{k}"].append(float(hit))
            results[f"recall@{k}"].append(rec)
        preds_bin = (scores > 0).astype(int)
        all_preds_bin.extend(preds_bin.tolist())
        all_true_bin.extend(labels.tolist())

    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(results[f"hit@{k}"])) if results[f"hit@{k}"] else float("nan")
        out[f"recall@{k}"] = float(np.mean(results[f"recall@{k}"])) if results[f"recall@{k}"] else float("nan")

    from sklearn.metrics import precision_recall_fscore_support
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        all_true_bin, all_preds_bin, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        all_true_bin, all_preds_bin, average="macro", zero_division=0
    )
    out.update({
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
    })
    return out

# ---------------- utils: loading safety ----------------
def infer_linear_in_dim_from_state_dict(sd: dict) -> int:
    """
    从塑料塔的 state_dict 里推断首层 Linear 的输入维度（第二维）。
    找到第一个 2D 权重 (out, in)，返回 in。
    """
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2:
            return v.shape[1]
    raise RuntimeError("Cannot infer plastic Tower in_dim from checkpoint state_dict.")

def strict_load(module: torch.nn.Module, sd: dict, name: str):
    """
    严格检查权重加载情况：
      - 打印 missing/unexpected
      - 如发现形状不匹配，直接报错（避免 silent fail）
    """
    res = module.load_state_dict(sd, strict=False)
    if res.missing_keys:
        print(f"[WARN] {name} missing keys: {res.missing_keys}")
    if res.unexpected_keys:
        print(f"[WARN] {name} unexpected keys: {res.unexpected_keys}")
    # 形状检查
    cur = module.state_dict()
    for k, w in sd.items():
        if k in cur:
            if tuple(cur[k].shape) != tuple(w.shape):
                raise RuntimeError(f"[FATAL] {name}.{k} shape mismatch: ckpt {tuple(w.shape)} vs model {tuple(cur[k].shape)}")

# ---------------- main ----------------
def main():
    # ====== 路径配置 ======
    ckpt_path = "/Users/shulei/PycharmProjects/Plaszyme/train_script/listwise_bilin_v1/best_factorized_bilinear.pt"
    out_csv   = "/Users/shulei/PycharmProjects/Plaszyme/result_data/pred_metrics.csv"

    # ====== 数据构造（和训练一致）======
    cfg = GNNBuilderConfig(
        pdb_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb",
        out_dir="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt",
        radius=10.0,
        embedder=[{"name":"esm"}],
    )
    enzyme_builder = GNNProteinGraphBuilder(cfg, edge_mode="none")
    plastic_featurizer = PlasticFeaturizer(config_path=None)

    spec = MatrixSpec(
        csv_path="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_testset.csv",
        pdb_root="/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb",
        sdf_root="/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new",
    )
    ds = PairedPlaszymeDataset(
        matrix=spec, mode="list", split="val",     # 这里保持和训练代码一致的 split 语义
        enzyme_builder=enzyme_builder, plastic_featurizer=plastic_featurizer,
        max_list_len=10,                           # 如担心截断正例，可临时调大
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pairs)

    # ====== 加载检查点 ======
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    EMB_DIM_ENZ = ckpt["cfg"]["emb_dim_enz"]
    EMB_DIM_PL  = ckpt["cfg"]["emb_dim_pl"]
    PROJ_DIM    = ckpt["cfg"]["proj_dim"]
    INTERACTION = ckpt["cfg"]["interaction"]
    RANK        = ckpt["cfg"].get("bilinear_rank", 64)

    # 优先从 ckpt cfg 读取塑料塔输入维度；没有就从 state_dict 反推
    if "plastic_in_dim" in ckpt["cfg"]:
        in_dim_plastic = ckpt["cfg"]["plastic_in_dim"]
    else:
        in_dim_plastic = infer_linear_in_dim_from_state_dict(ckpt["plastic_tower"])
        print(f"[INFO] inferred plastic in_dim from ckpt: {in_dim_plastic}")

    plastic_hidden = ckpt["cfg"].get("plastic_hidden", [256, 128])
    enzyme_hidden  = ckpt["cfg"].get("enzyme_hidden", [128, 128, 128])

    # ====== 构建模型（维度与训练一致）======
    enzyme_backbone = GNNBackbone(
        conv_type="gcn", hidden_dims=enzyme_hidden, out_dim=EMB_DIM_ENZ,
        dropout=0.1, residue_logits=False
    ).to(DEVICE)
    plastic_tower = PolymerTower(
        in_dim=in_dim_plastic, hidden_dims=plastic_hidden, out_dim=EMB_DIM_PL, dropout=0.1
    ).to(DEVICE)
    projector  = TwinProjector(EMB_DIM_ENZ, EMB_DIM_PL, PROJ_DIM).to(DEVICE)
    inter_head = InteractionHead(PROJ_DIM, mode=INTERACTION, rank=RANK).to(DEVICE)

    # ====== 严格加载并检查 ======
    strict_load(enzyme_backbone, ckpt["enzyme_backbone"], "enzyme_backbone")
    strict_load(plastic_tower,   ckpt["plastic_tower"],   "plastic_tower")
    strict_load(projector,       ckpt["projector"],       "projector")
    strict_load(inter_head,      ckpt["inter_head"],      "inter_head")

    enzyme_backbone.eval(); plastic_tower.eval(); projector.eval(); inter_head.eval()

    # ====== 预测 ======
    all_labels, all_scores = [], []
    pos_per_item = []  # sanity

    with torch.no_grad():
        for batch in loader:
            g = batch["enzyme_graph"].to(DEVICE)
            enz_vec = enzyme_backbone(g)             # [1, De]
            z_e = projector.proj_e(enz_vec)          # [1, D]

            item   = batch["items"][0]
            plast  = item["plastic_list"].to(DEVICE) # [L, Dp_in]
            labels = item["relevance"].cpu().numpy() # [L]
            z_p    = projector.proj_p(plastic_tower(plast))  # [L, D]

            scores = inter_head.score(z_e, z_p).cpu().numpy()

            all_labels.append(labels)
            all_scores.append(scores)
            pos_per_item.append(int(labels.sum()))

    # ====== Sanity checks ======
    print("[CHK] positives per item (first 20):", pos_per_item[:20])
    if len(all_scores) > 0:
        flat = np.concatenate(all_scores)
        print("[CHK] score stats: mean=%.4f std=%.4f min=%.4f max=%.4f" % (flat.mean(), flat.std(), flat.min(), flat.max()))
        for i in range(min(3, len(all_scores))):
            idx = np.argsort(-all_scores[i])[:5]
            print(f"[SAMPLE {i}] top5 labels:", all_labels[i][idx].tolist(), "scores:", np.round(all_scores[i][idx], 3).tolist())

    # ====== 计算指标并落盘 ======
    metrics = compute_metrics(all_labels, all_scores, ks=(1,3,5))
    df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Metrics saved to {out_csv}")
    print(df)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    main()