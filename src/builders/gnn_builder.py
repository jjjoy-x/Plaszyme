#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN graph builder (scalar-only nodes).

GNN 构建器：节点只有标量特征 x；
边为半径图 edge_index；edge_attr 可选：
- "none": 二值图（不返回 edge_attr）
- "dist": 单一距离标量
- "inv_dist": 1 / (dist + eps)
- "rbf": RBF 距离基展开（多维）
"""

from __future__ import annotations
import os
from typing import Dict, Literal, Optional

import numpy as np
import torch

from .base_builder import BaseProteinGraphBuilder, BuilderConfig


# -------------------------
# 边构建：半径图 + 多种 edge_attr 方案
# -------------------------
def _radius_graph(coords: np.ndarray, r: float, self_loop: bool = False) -> np.ndarray:
    """Undirected radius graph by KD-tree (numpy). 返回 edge_index [2, E]."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_ball_tree(tree, r=r)
    src, dst = [], []
    N = coords.shape[0]
    for i in range(N):
        for j in pairs[i]:
            if (i == j) and (not self_loop):
                continue
            src.append(i); dst.append(j)
            if i != j:  # 无向 → 存双向
                src.append(j); dst.append(i)
    return np.asarray([src, dst], dtype=np.int64)


def _pairwise_dist(coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    i, j = edge_index
    d = np.linalg.norm(coords[j] - coords[i], axis=-1, keepdims=True)
    return d.astype(np.float32)  # [E, 1]


def _rbf_expand(dist: np.ndarray,
                centers: np.ndarray,
                gamma: float) -> np.ndarray:
    """RBF: exp(-gamma * (d - c)^2).
    Args:
      dist: [E, 1]
      centers: [K] (float), e.g. np.linspace(2.0, 10.0, 16)
      gamma: float, e.g. 5.0 / (delta^2)
    Returns:
      [E, K]
    """
    d = dist  # [E,1]
    diff = d - centers.reshape(1, -1)  # [E, K]
    return np.exp(-gamma * (diff ** 2)).astype(np.float32)


class GNNProteinGraphBuilder(BaseProteinGraphBuilder):
    """Builder for conventional GNNs.

    面向常规 GCN/GAT/GraphSAGE：
      - 标量节点特征：来自 sequence embedder
      - 边：半径图；edge_attr 可选
    """

    def __init__(
        self,
        cfg: BuilderConfig,
        *,
        edge_mode: Literal["none", "dist", "inv_dist", "rbf"] = "none",
        rbf_centers: Optional[np.ndarray] = None,
        rbf_gamma: Optional[float] = None,
        add_self_loop: bool = False,
    ):
        """Init.

        Args:
          cfg: 运行配置（目录、半径、embedder 等来自 BaseBuilder）。
          edge_mode: 边特征方式：
            - "none": 不生成 edge_attr（纯二值邻接）；
            - "dist": edge_attr = 距离标量 d；
            - "inv_dist": edge_attr = 1/(d+eps)；
            - "rbf": 基于距离的 RBF 展开，edge_attr=[phi_1...phi_K]。
          rbf_centers: 当 edge_mode="rbf" 时的中心 array(K,)；若 None，默认 2.0~10.0 均匀 16 点。
          rbf_gamma: RBF 的 gamma；若 None，自动按中心间隔设置。
          add_self_loop: 是否保留自环（默认 False）。
        """
        super().__init__(cfg)
        self.edge_mode = edge_mode
        self.add_self_loop = bool(add_self_loop)

        if self.edge_mode == "rbf":
            if rbf_centers is None:
                rbf_centers = np.linspace(2.0, max(2.1, cfg.radius), num=16, dtype=np.float32)
            self.rbf_centers = np.asarray(rbf_centers, dtype=np.float32)
            if rbf_gamma is None:
                # 根据中心间隔给个温和的 gamma
                delta = float(self.rbf_centers[1] - self.rbf_centers[0]) if len(self.rbf_centers) > 1 else 1.0
                self.rbf_gamma = 1.0 / (2.0 * (delta ** 2))
            else:
                self.rbf_gamma = float(rbf_gamma)

    # ---- subclass hooks ----
    def _build_node_features(self, *, seq_full: str, kept_idx: np.ndarray, parsed: Dict[str, any]) -> Dict[str, torch.Tensor]:
        x_full = self.embedder.embed(seq_full)  # Tensor [L_full, D]
        x = x_full[torch.as_tensor(kept_idx, dtype=torch.long)].float()
        return {"x": x}

    def _build_edges(self, coords_ca_kept: np.ndarray, *, parsed: Dict[str, any], aln: Dict[str, any]) -> Dict[str, torch.Tensor]:
        edge_index_np = _radius_graph(coords_ca_kept, r=self.cfg.radius, self_loop=self.add_self_loop)
        out = {"edge_index": torch.as_tensor(edge_index_np, dtype=torch.long)}

        if self.edge_mode == "none":
            # 纯二值邻接，不返回 edge_attr
            return out

        dist = _pairwise_dist(coords_ca_kept, edge_index_np)  # [E,1]

        if self.edge_mode == "dist":
            out["edge_attr"] = torch.as_tensor(dist, dtype=torch.float32)  # [E,1]

        elif self.edge_mode == "inv_dist":
            inv = 1.0 / (dist + 1e-6)
            out["edge_attr"] = torch.as_tensor(inv, dtype=torch.float32)    # [E,1]

        elif self.edge_mode == "rbf":
            phi = _rbf_expand(dist, self.rbf_centers, self.rbf_gamma)       # [E,K]
            out["edge_attr"] = torch.as_tensor(phi, dtype=torch.float32)

        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        return out


# convenience
def build_gnn_folder(cfg: BuilderConfig, **kwargs) -> None:
    builder = GNNProteinGraphBuilder(cfg, **kwargs)
    builder.build_folder()


if __name__ == "__main__":
    # ==== 简单测试（直接在脚本里改路径）====
    import os
    pdb_file = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0002.pdb"
    out_dir = "/Users/shulei/PycharmProjects/Plaszyme/test/output_gnn"

    cfg = BuilderConfig(
        pdb_dir=os.path.dirname(pdb_file),
        out_dir=out_dir,
        embedder=[
            {"name": "physchem", "norm": "zscore", "concat_onehot": True},
            {"name": "onehot"},
            {"name": "esm", "model_name": "esm2_t12_35M_UR50D", "fp16": False},
        ],
    )

    # 1) 纯二值
    builder_bin = GNNProteinGraphBuilder(cfg, edge_mode="none")
    name = os.path.splitext(os.path.basename(pdb_file))[0]
    data_bin, misc = builder_bin.build_one(pdb_file, name=name)
    print(f"[BINARY] edge_attr? {'edge_attr' in data_bin}")