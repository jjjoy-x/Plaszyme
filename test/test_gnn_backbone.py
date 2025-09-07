#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gnn_with_builder.py

从 PDB 用 builder 构图，然后用 GNNBackbone 测试多种卷积算子在
- 开启边特征（RBF，edge_attr 存在）
- 关闭边特征（二值，edge_attr 不存在）
两种场景下的前向传播是否正常，以及输出形状是否符合预期。

依赖：
    - 你的 src.builders.*（base_builder / gnn_builder）
    - 你的 src.models.gnn.backbone.GNNBackbone
    - torch, torch_geometric

用法：
    直接改 MAIN 里的 PDB_FILE 路径后运行本脚本。
"""

from __future__ import annotations
import os
from typing import List, Tuple

import torch
from torch_geometric.data import Batch

# ===== 根据你的工程包名调整 import =====
from src.builders.base_builder import BuilderConfig
from src.builders.gnn_builder import GNNProteinGraphBuilder
from src.models.gnn.backbone import GNNBackbone


# -----------------------------
# Builder: 用 PDB 构图
# -----------------------------
def build_graph_from_pdb(
    pdb_file: str,
    edge_mode: str,          # "rbf" | "dist" | "inv_dist" | "none"
    radius: float = 10.0,
    embedder_cfg=None,
) -> Tuple[torch.Tensor, dict]:
    """
    用 GNN builder 从单个 PDB 构建 PyG Data（不落盘）。

    Args:
        pdb_file: PDB 文件路径
        edge_mode: 边特征模式（"rbf" 生成 edge_attr；"none" 不生成）
        radius: 构边半径
        embedder_cfg: 传给 BuilderConfig.embedder，可为单个或列表

    Returns:
        (data, misc)
    """
    pdb_dir = os.path.dirname(pdb_file)
    name = os.path.splitext(os.path.basename(pdb_file))[0]

    cfg = BuilderConfig(
        pdb_dir=pdb_dir,
        out_dir=os.path.join(pdb_dir, "tmp_test_out"),  # 这里不会真正写 .pt
        embedder=embedder_cfg or {"name": "onehot"},
        radius=radius,
    )
    builder = GNNProteinGraphBuilder(cfg, edge_mode=edge_mode)
    data, misc = builder.build_one(pdb_file, name=name)
    return data, misc


def make_batch(data, n_graphs: int = 2) -> Batch:
    """复制同一个图 n 次，构造一个 Batch。"""
    data_list = [data] * n_graphs
    return Batch.from_data_list(data_list)


# -----------------------------
# 核心测试逻辑
# -----------------------------
def run_one_case(
    pdb_file: str,
    conv_type: str,
    edge_mode: str,
    residue_logits: bool,
    *,
    gcn_edge_mode: str = "auto",
    gine_policy: str = "error",
    hidden_dims: List[int] = [64, 64],
    out_dim: int = 8,
):
    """
    构图 → 组 batch → 前向 → 断言形状。
    """
    # 1) 构图
    data, misc = build_graph_from_pdb(
        pdb_file=pdb_file,
        edge_mode=edge_mode,             # 决定是否生成 edge_attr
        radius=10.0,
        embedder_cfg={"name": "onehot"}, # 快速
    )
    batch = make_batch(data, n_graphs=2)  # B=2

    # 2) 建模（注意 GINE 的 edge_attr 要求）
    model = GNNBackbone(
        conv_type=conv_type,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        dropout=0.1,
        residue_logits=residue_logits,
        gcn_edge_mode=gcn_edge_mode,
        gine_missing_edge_policy=gine_policy,   # "error" or "zeros"
    )

    # 3) 前向
    out = model(batch)

    # 4) 形状断言
    if residue_logits:
        exp = (batch.num_nodes, out_dim)
    else:
        # 两个图
        exp = (2, out_dim)
    assert tuple(out.shape) == exp, (
        f"[{conv_type}|edge={edge_mode}|residue={residue_logits}] "
        f"shape mismatch: got {tuple(out.shape)}, expect {exp}"
    )

    # 5) 打印简单结果
    has_edge_attr = hasattr(batch, "edge_attr") and (getattr(batch, "edge_attr") is not None)
    print(
        f"[OK] conv={conv_type:<6} edge_mode={edge_mode:<4} "
        f"edge_attr={'Y' if has_edge_attr else 'N'} residue={str(residue_logits):<5} -> out={tuple(out.shape)}"
    )


def run_suite(pdb_file: str):
    """
    跑两组：edge on (RBF) / edge off (none)
    - 对应检查：GINE 在 edge off 下用 zeros 占位，其他算子忽略边特征或自动转 weight
    - 同时对每组分别跑：图级输出、残基级输出
    """
    convs = ["gcn", "gatv2", "sage", "gin", "gine"]

    # A) 边特征开启（RBF），edge_attr 存在
    for residue in (False, True):
        for ct in convs:
            # GINE: 需要 edge_attr；我们给 "error"（没有就报错）
            gine_policy = "error" if ct == "gine" else "error"
            run_one_case(
                pdb_file,
                conv_type=ct,
                edge_mode="rbf",
                residue_logits=residue,
                gcn_edge_mode="auto",
                gine_policy=gine_policy,
            )

    # B) 边特征关闭（none），edge_attr 不存在
    for residue in (False, True):
        for ct in convs:
            # GINE: 没有 edge_attr → 用 zeros 占位通过前向（只做健壮性测试）
            gine_policy = "zeros" if ct == "gine" else "error"
            run_one_case(
                pdb_file,
                conv_type=ct,
                edge_mode="none",
                residue_logits=residue,
                gcn_edge_mode="auto",
                gine_policy=gine_policy,
            )


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 改成你的 PDB 路径（确保是标准单链或由 chain_id 选择好）
    PDB_FILE = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0002.pdb"
    assert os.path.exists(PDB_FILE), f"PDB not found: {PDB_FILE}"

    torch.manual_seed(0)
    run_suite(PDB_FILE)