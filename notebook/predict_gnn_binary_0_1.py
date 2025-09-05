#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gnn_binary.py  (stable, strict load; warm-up build to avoid key mismatch)

- 直接导入你的 DeepFRIModel（from model.gcn_model import DeepFRIModel）
- 加载训练好的 FusionBinaryModel 权重（先“预跑”构建所有层，再 strict=True 加载）
- 对 graphs_dir 下所有 {enzyme}.pt 与 plastic_feat_pt 中所有塑料做两两打分
- 输出 行=酶 × 列=塑料 的概率矩阵 CSV（0~1）

命令行示例：
python predict_gnn_binary.py \
  --graphs_dir /path/to/test_graph \
  --plastic_feat_pt /path/to/plastics.pt \
  --checkpoint /path/to/best_model.pt \
  --out_csv /path/to/preds.csv \
  --gnn_type gat --gnn_dims 128 128 --fc_dims 128 64 \
  --gnn_embed_dim 128 --plastic_hidden 256 128 --fusion_hidden 128 64 --dropout 0.1
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from torch_geometric.data import Data, Batch as GeoBatch

# ==== 直接导入你的模型定义 ====
from model.gcn_model import DeepFRIModel   # 要与训练时同一模块

# =====================
# 用户配置（可被命令行覆盖）
# =====================
CONFIG = {
    # 路径
    "graphs_dir": "/tmp/pycharm_project_27/dataset/test_graph",  # 目录中应有 {enzyme}.pt
    "plastic_feat_pt": "/tmp/pycharm_project_27/test/outputs/all_description_new.pt",
    "checkpoint": "/tmp/pycharm_project_27/checkpoints/binary_01_balanced_04/best_model.pt",
    "out_csv": "/tmp/pycharm_project_27/checkpoints/binary_01_balanced_04/preds.csv",

    # 模型结构（需与训练时一致）
    "gnn_type": "gat",            # ["gcn", "gat"]
    "gnn_dims": [128, 128],
    "fc_dims": [128, 64],
    "gnn_embed_dim": 128,
    "plastic_hidden": [256, 128],
    "fusion_hidden": [128, 64],
    "dropout": 0.1,

    # 设备/批大小
    "device": "auto",             # "auto" | "cuda" | "cpu"
    "plastics_batch": 4096,       # 处理单个酶时，塑料拼接后的前向批大小
    "seed": 42,
}

# =====================
# 模型外壳（与训练一致）
# =====================
class PlasticEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # [N, Dp]
        return self.net(x) # [N, H]

class FusionBinaryModel(nn.Module):
    """
    用 DeepFRIModel 作为 enzyme encoder，MLP 作为 plastic encoder，拼接后接分类头（1 logit）
    """
    def __init__(
        self,
        gnn_type: str,
        gnn_dims: List[int],
        fc_dims: List[int],
        gnn_embed_dim: int,
        plastic_in_dim: int,
        plastic_hidden: List[int],
        fusion_hidden: List[int],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.enzyme_enc = DeepFRIModel(
            gnn_type=gnn_type,
            gnn_dims=list(gnn_dims),
            fc_dims=list(fc_dims),
            out_dim=gnn_embed_dim,
            dropout=dropout,
            use_residue_level_output=False,
            in_dim=None,   # 懒构建：第一次前向时用输入自动推断
        )
        self.plastic_enc = PlasticEncoder(plastic_in_dim, plastic_hidden, dropout=dropout)

        in_dim = int(gnn_embed_dim) + int(plastic_hidden[-1] if len(plastic_hidden)>0 else plastic_in_dim)
        dims = [in_dim] + list(fusion_hidden) + [1]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.fusion = nn.Sequential(*layers)

    def forward(self, batch_graph: GeoBatch, plastic_vecs: torch.Tensor):
        enz_embed = self.enzyme_enc(batch_graph)     # [B, Z]
        pla_embed = self.plastic_enc(plastic_vecs)   # [B, H]
        z = torch.cat([enz_embed, pla_embed], dim=1) # [B, Z+H]
        return self.fusion(z).squeeze(1)             # [B]

# =====================
# I/O & 工具函数
# =====================
def choose_device(flag: str):
    if flag == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_plastic_features(pt_path: str) -> Tuple[Dict[str, torch.Tensor], int, List[str]]:
    obj = torch.load(pt_path, map_location="cpu")
    feats: Dict[str, torch.Tensor] = obj["features"]
    if not feats:
        raise ValueError(f"No features found in {pt_path}")
    any_key = next(iter(feats.keys()))
    dim = int(feats[any_key].numel())
    names = list(feats.keys())
    print(f"[INFO] Loaded plastic features: {len(names)} plastics | dim={dim}")
    return feats, dim, names

def list_enzyme_graphs(graphs_dir: str) -> List[Tuple[str, str]]:
    """
    返回 [(enzyme_name, /path/to/enzyme.pt), ...]，按名字排序
    """
    add_safe_globals([Data])
    files = [f for f in os.listdir(graphs_dir) if f.endswith(".pt")]
    out = []
    for f in files:
        name = os.path.splitext(f)[0]
        out.append((name, os.path.join(graphs_dir, f)))
    out.sort(key=lambda x: x[0])
    if not out:
        raise FileNotFoundError(f"No *.pt graphs found under {graphs_dir}")
    print(f"[INFO] Found {len(out)} enzyme graphs.")
    return out

def load_graph(path: str) -> Data:
    add_safe_globals([Data])
    data: Data = torch.load(path, weights_only=False)
    return data

def ensure_out_csv(path_csv: str, checkpoint_path: str) -> str:
    if path_csv.lower().endswith(".csv"):
        return path_csv
    # 兜底：自动改名
    base = checkpoint_path
    if base.endswith(".pt"): base = base[:-3]
    return base + "_preds.csv"

# ============ 关键：预构建（warm-up）再加载权重 ============
def warm_build_then_load(
    model: FusionBinaryModel,
    checkpoint_state: dict,
    example_graph: Data,
    example_plastic_vec: torch.Tensor,
    device: torch.device
):
    """
    先用一个真实图 + 一个塑料特征向量跑一次，触发 DeepFRIModel 的懒构建；
    再 strict=True 加载 checkpoint，避免 Unexpected key(s)/missing key(s)。
    """
    model.eval()
    with torch.no_grad():
        # 保证 example_graph 的张量在同一 device
        for attr in ("x", "edge_index", "batch"):
            if hasattr(example_graph, attr) and getattr(example_graph, attr) is not None:
                setattr(example_graph, attr, getattr(example_graph, attr).to(device))
        example_plastic_vec = example_plastic_vec.to(device).unsqueeze(0)  # [1, Dp]

        # 触发 enzyme_enc 懒构建
        _ = model.enzyme_enc(example_graph)            # [1, Z]
        # 触发 plastic_enc + fusion 懒构建
        _ = model(example_graph, example_plastic_vec)  # [1]

    # 现在所有层都已存在，执行严格加载
    model_sd = model.state_dict()
    try:
        model.load_state_dict(checkpoint_state, strict=True)
        print("[INFO] Loaded checkpoint with strict=True (after warm-up build).")
    except RuntimeError as e:
        # 仍失败：给出清晰提示并回退到形状匹配加载
        print("[WARN] strict=True still failed after warm-up:\n", e)
        filtered = {k: v for k, v in checkpoint_state.items() if k in model_sd and model_sd[k].shape == v.shape}
        miss = [k for k in model_sd.keys() if k not in filtered]
        extra = [k for k in checkpoint_state.keys() if k not in filtered]
        print(f"[WARN] Partial load: matched={len(filtered)}/{len(model_sd)}; missing={len(miss)}; extra={len(extra)}")
        model.load_state_dict(filtered, strict=False)

# =====================
# 核心推理
# =====================
@torch.no_grad()
def infer_matrix(
    graphs_dir: str,
    plastic_feat_pt: str,
    checkpoint: str,
    out_csv: str,
    gnn_type: str,
    gnn_dims: List[int],
    fc_dims: List[int],
    gnn_embed_dim: int,
    plastic_hidden: List[int],
    fusion_hidden: List[int],
    dropout: float,
    device: torch.device,
    plastics_batch: int = 4096,
):
    # 载入塑料特征
    plastic_feats, plastic_dim, plastic_names = load_plastic_features(plastic_feat_pt)
    # 载入酶列表
    enzymes = list_enzyme_graphs(graphs_dir)

    # 构建模型
    model = FusionBinaryModel(
        gnn_type=gnn_type,
        gnn_dims=gnn_dims,
        fc_dims=fc_dims,
        gnn_embed_dim=gnn_embed_dim,
        plastic_in_dim=plastic_dim,
        plastic_hidden=plastic_hidden,
        fusion_hidden=fusion_hidden,
        dropout=dropout
    ).to(device)

    # 加载 checkpoint（先 warm-up 再 strict load）
    state = torch.load(checkpoint, map_location="cpu")
    # 选一个样本图与一个样本塑料向量进行 warm-up
    sample_graph = load_graph(enzymes[0][1])
    sample_plastic = plastic_feats[plastic_names[0]].float()
    warm_build_then_load(model, state, sample_graph, sample_plastic, device)

    model.eval()

    # 预编码全部塑料（一次性，避免重复算）
    plast_names_sorted = sorted(plastic_names)
    plast_tensor = torch.stack([plastic_feats[n].float() for n in plast_names_sorted], dim=0).to(device)  # [P, Dp]

    pla_emb_list = []
    for i in range(0, plast_tensor.size(0), plastics_batch):
        chunk = plast_tensor[i:i+plastics_batch]               # [b, Dp]
        emb = model.plastic_enc(chunk)                          # [b, H]
        pla_emb_list.append(emb)
    pla_emb_all = torch.cat(pla_emb_list, dim=0)                # [P, H]
    assert pla_emb_all.size(0) == len(plast_names_sorted)
    print(f"[INFO] Pre-encoded plastics: shape={tuple(pla_emb_all.shape)}")

    # 逐个酶推理（每个酶只算一次图嵌入）
    scores = []
    enzyme_names = []
    for enz, gpath in enzymes:
        data = load_graph(gpath)
        # 统一 to(device)
        for attr in ("x", "edge_index", "batch"):
            if hasattr(data, attr) and getattr(data, attr) is not None:
                setattr(data, attr, getattr(data, attr).to(device))

        # 一次性图级嵌入
        enz_emb = model.enzyme_enc(data)            # [1, Z]

        # 扩展与塑料拼接并过分类头（分块以控显存）
        row_scores = []
        for i in range(0, pla_emb_all.size(0), plastics_batch):
            pla_chunk = pla_emb_all[i:i+plastics_batch]        # [b, H]
            enz_rep = enz_emb.expand(pla_chunk.size(0), -1)    # [b, Z]
            z = torch.cat([enz_rep, pla_chunk], dim=1)         # [b, Z+H]
            logit = model.fusion(z).squeeze(1)                 # [b]
            prob = torch.sigmoid(logit)                        # [b]
            row_scores.append(prob)
        row = torch.cat(row_scores, dim=0).detach().cpu().numpy()  # [P]
        scores.append(row)
        enzyme_names.append(enz)

    # 组装为 DataFrame 并保存
    import pandas as pd
    df = pd.DataFrame(scores, index=enzyme_names, columns=plast_names_sorted)
    out_csv = ensure_out_csv(out_csv, checkpoint)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, float_format="%.6f")
    print(f"[OK] Saved prediction matrix to: {out_csv}")
    return out_csv, df

# =====================
# Entrypoint
# =====================
def parse_args_from_config():
    p = argparse.ArgumentParser(description="Predict enzyme×plastic probabilities using trained FusionBinaryModel (strict load with warm-up).")
    # 路径
    p.add_argument("--graphs_dir", type=str, default=CONFIG["graphs_dir"])
    p.add_argument("--plastic_feat_pt", type=str, default=CONFIG["plastic_feat_pt"])
    p.add_argument("--checkpoint", type=str, default=CONFIG["checkpoint"])
    p.add_argument("--out_csv", type=str, default=CONFIG["out_csv"])
    # 模型结构（与训练一致）
    p.add_argument("--gnn_type", type=str, default=CONFIG["gnn_type"], choices=["gcn", "gat"])
    p.add_argument("--gnn_dims", type=int, nargs="+", default=CONFIG["gnn_dims"])
    p.add_argument("--fc_dims", type=int, nargs="+", default=CONFIG["fc_dims"])
    p.add_argument("--gnn_embed_dim", type=int, default=CONFIG["gnn_embed_dim"])
    p.add_argument("--plastic_hidden", type=int, nargs="+", default=CONFIG["plastic_hidden"])
    p.add_argument("--fusion_hidden", type=int, nargs="+", default=CONFIG["fusion_hidden"])
    p.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    # 设备/批大小/随机种子
    p.add_argument("--device", type=str, default=CONFIG["device"], choices=["auto", "cuda", "cpu"])
    p.add_argument("--plastics_batch", type=int, default=CONFIG["plastics_batch"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    return p.parse_args()

def main():
    args = parse_args_from_config()
    device = choose_device(args.device)
    set_seed(int(args.seed))
    torch.set_grad_enabled(False)
    print(f"[INFO] Device: {device}")

    # 基本存在性检查
    if not os.path.isdir(args.graphs_dir):
        raise FileNotFoundError(f"graphs_dir not found: {args.graphs_dir}")
    if not os.path.isfile(args.plastic_feat_pt):
        raise FileNotFoundError(f"plastic_feat_pt not found: {args.plastic_feat_pt}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    infer_matrix(
        graphs_dir=args.graphs_dir,
        plastic_feat_pt=args.plastic_feat_pt,
        checkpoint=args.checkpoint,
        out_csv=args.out_csv,
        gnn_type=args.gnn_type,
        gnn_dims=list(args.gnn_dims),
        fc_dims=list(args.fc_dims),
        gnn_embed_dim=int(args.gnn_embed_dim),
        plastic_hidden=list(args.plastic_hidden),
        fusion_hidden=list(args.fusion_hidden),
        dropout=float(args.dropout),
        device=device,
        plastics_batch=int(args.plastics_batch),
    )

if __name__ == "__main__":
    main()