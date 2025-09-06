#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GVP graph builder (scalar + vector channels).

GVP 构建器：节点包含 x_s:[K,Ds] 与 x_v:[K,Vn,3]（默认 CA→N、CA→C 两个方向）；
边包含 edge_index、edge_s（可选：none/dist/inv_dist/rbf）与 edge_v（单位方向）。
"""

from __future__ import annotations
import os
from typing import Dict, Literal, Optional

import numpy as np
import torch

from .base_builder import BaseProteinGraphBuilder, BuilderConfig


# -------------------------
# 通用：半径图 & 距离派生特征
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
    return d.astype(np.float32)  # [E,1]


def _rbf_expand(dist: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    """RBF: exp(-gamma * (d - c)^2).
    Args:
      dist: [E, 1]
      centers: [K]
      gamma: float
    Returns:
      [E, K]
    """
    d = dist  # [E,1]
    diff = d - centers.reshape(1, -1)  # [E,K]
    return np.exp(-gamma * (diff ** 2)).astype(np.float32)


def _edge_unit_vectors(coords: np.ndarray, edge_index: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """CA→CA 单位方向向量，形状 [E,3]."""
    i, j = edge_index
    v = coords[j] - coords[i]
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return (v / n).astype(np.float32)


# -------------------------
# GVP 构建器
# -------------------------
class GVPProteinGraphBuilder(BaseProteinGraphBuilder):
    """Builder for GVP-GNN.

    节点：
      - x_s: 标量特征（来自 sequence embedder），形状 [K, Ds]
      - x_v: 向量特征（默认 CA→N、CA→C 两个单位向量），形状 [K, Vn=2, 3]；可关闭

    边：
      - edge_index: [2, E]
      - edge_s: 标量特征（'none' / 'dist' / 'inv_dist' / 'rbf'）
      - edge_v: 单位方向（CA→CA），形状 [E, Ve, 3]；默认 Ve=1，可增广
    """

    def __init__(
        self,
        cfg: BuilderConfig,
        *,
        node_vec_mode: Literal["bb2", "none"] = "bb2",
        edge_scalar: Literal["none", "dist", "inv_dist", "rbf"] = "none",
        rbf_centers: Optional[np.ndarray] = None,
        rbf_gamma: Optional[float] = None,
        edge_vec_dim: int = 1,
        add_self_loop: bool = False,
        vec_eps: float = 1e-8,
    ):
        """Init.

        Args:
          cfg: 运行配置（目录、半径、embedder 等来自 BaseBuilder）。
          node_vec_mode: 节点向量通道构造方式：
            - "bb2": 用主链 CA→N、CA→C 两个单位向量；
            - "none": 不输出节点向量通道（x_v 为空，形如 [K,0,3]）。
          edge_scalar: 边标量特征模式：
            - "none": 不生成 edge_s（只有向量 edge_v）；
            - "dist": edge_s = 距离标量 d；
            - "inv_dist": edge_s = 1/(d+eps)；
            - "rbf": 基于距离的 RBF 展开，edge_s=[phi_1...phi_K]。
          rbf_centers: 当 edge_scalar="rbf" 时的中心 array(K,)；None 则默认 2.0~radius 均匀 16 点。
          rbf_gamma: RBF 的 gamma；None 则按中心间隔给个合理默认。
          edge_vec_dim: 边向量通道数（通常 1 即可，将单位方向复制到 Ve 个通道）。
          add_self_loop: 边是否保留自环（默认 False）。
          vec_eps: 单位向量归一化的数值稳定项。
        """
        super().__init__(cfg)
        assert node_vec_mode in {"bb2", "none"}
        self.node_vec_mode = node_vec_mode
        self.edge_scalar = edge_scalar
        self.edge_vec_dim = int(edge_vec_dim)
        self.add_self_loop = bool(add_self_loop)
        self.vec_eps = float(vec_eps)

        if self.edge_scalar == "rbf":
            if rbf_centers is None:
                rbf_centers = np.linspace(2.0, max(2.1, cfg.radius), num=16, dtype=np.float32)
            self.rbf_centers = np.asarray(rbf_centers, dtype=np.float32)
            if rbf_gamma is None:
                delta = float(self.rbf_centers[1] - self.rbf_centers[0]) if len(self.rbf_centers) > 1 else 1.0
                self.rbf_gamma = 1.0 / (2.0 * (delta ** 2))
            else:
                self.rbf_gamma = float(rbf_gamma)

    # ---- subclass hooks ----
    def _build_node_features(
        self, *, seq_full: str, kept_idx: np.ndarray, parsed: Dict[str, any]
    ) -> Dict[str, torch.Tensor]:
        # 标量通道：对齐后索引
        x_s_full = self.embedder.embed(seq_full)  # [L_full, Ds]
        x_s = x_s_full[torch.as_tensor(kept_idx, dtype=torch.long)].float()

        # 向量通道
        if self.node_vec_mode == "none":
            x_v = torch.zeros((x_s.size(0), 0, 3), dtype=torch.float32)
        else:
            # 用主链几何：x_v = [CA→N, CA→C]
            N = parsed["coords_N"][kept_idx]    # [K,3]
            CA = parsed["coords_CA"][kept_idx]  # [K,3]
            C = parsed["coords_C"][kept_idx]    # [K,3]
            v1 = N - CA
            v2 = C - CA
            v1 /= (np.linalg.norm(v1, axis=-1, keepdims=True) + self.vec_eps)
            v2 /= (np.linalg.norm(v2, axis=-1, keepdims=True) + self.vec_eps)
            x_v = torch.as_tensor(np.stack([v1, v2], axis=1), dtype=torch.float32)  # [K,2,3]

        return {"x_s": x_s, "x_v": x_v}

    def _build_edges(
        self, coords_ca_kept: np.ndarray, *, parsed: Dict[str, any], aln: Dict[str, any]
    ) -> Dict[str, torch.Tensor]:
        edge_index_np = _radius_graph(coords_ca_kept, r=self.cfg.radius, self_loop=self.add_self_loop)
        out = {"edge_index": torch.as_tensor(edge_index_np, dtype=torch.long)}

        # 标量 edge_s
        if self.edge_scalar == "none":
            edge_s = None
        else:
            dist = _pairwise_dist(coords_ca_kept, edge_index_np)  # [E,1]
            if self.edge_scalar == "dist":
                edge_s = torch.as_tensor(dist, dtype=torch.float32)            # [E,1]
            elif self.edge_scalar == "inv_dist":
                edge_s = torch.as_tensor(1.0 / (dist + 1e-6), dtype=torch.float32)  # [E,1]
            elif self.edge_scalar == "rbf":
                phi = _rbf_expand(dist, self.rbf_centers, self.rbf_gamma)      # [E,K]
                edge_s = torch.as_tensor(phi, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown edge_scalar: {self.edge_scalar}")

        # 向量 edge_v（CA→CA 单位方向），复制到 Ve 个通道
        v = _edge_unit_vectors(coords_ca_kept, edge_index_np, eps=self.vec_eps)  # [E,3]
        if self.edge_vec_dim == 1:
            edge_v = torch.as_tensor(v, dtype=torch.float32).unsqueeze(1)       # [E,1,3]
        else:
            vv = np.repeat(v[:, None, :], self.edge_vec_dim, axis=1)
            edge_v = torch.as_tensor(vv, dtype=torch.float32)                   # [E,Ve,3]

        # 打包
        if edge_s is not None:
            out["edge_s"] = edge_s
        out["edge_v"] = edge_v
        return out


# convenience
def build_gvp_folder(cfg: BuilderConfig, **kwargs) -> None:
    builder = GVPProteinGraphBuilder(cfg, **kwargs)
    builder.build_folder()


# -------------------------
# 简单自测（直接改路径后运行：python -m src.builders.gvp_builder）
# -------------------------
if __name__ == "__main__":
    pdb_file = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0002.pdb"
    out_dir = "/Users/shulei/PycharmProjects/Plaszyme/test/output_gvp"

    cfg = BuilderConfig(
        pdb_dir=os.path.dirname(pdb_file),
        out_dir=out_dir,
        embedder={"name": "esm"},  # 可改为 "esm" / "physchem"
        radius=10.0,
    )
    builder = GVPProteinGraphBuilder(
        cfg,
        node_vec_mode="bb2",     # 节点向量：CA→N、CA→C
        edge_scalar="rbf",       # 边标量：RBF 展开
        edge_vec_dim=1,          # 边向量通道数
        add_self_loop=False,
    )

    name = os.path.splitext(os.path.basename(pdb_file))[0]
    data, misc = builder.build_one(pdb_file, name=name)

    print(f"[TEST] Built GVP graph for {name}")
    print(f" - Node x_s: {tuple(data.x_s.shape)}")   # [K, Ds]
    print(f" - Node x_v: {tuple(data.x_v.shape)}")   # [K, Vn, 3]
    print(f" - Edge index: {tuple(data.edge_index.shape)}")  # [2, E]")
    if hasattr(data, "edge_s"):
        print(f" - Edge s: {tuple(data.edge_s.shape)}")      # [E, 1 or K]
    print(f" - Edge v: {tuple(data.edge_v.shape)}")          # [E, Ve, 3]
    print(f" - Meta: {misc}")