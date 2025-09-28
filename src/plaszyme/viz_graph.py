#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_pymol_overlay.py

Build a protein graph (GNN or GVP) from a PDB and export a PyMOL script
to overlay edges (and optional GVP vectors) on the original structure.

从 PDB 自动构建（GNN/GVP）图，并导出 PyMOL 脚本，叠加显示：
- 边：distance 对象（半径图连边），支持按距离上色（近红远蓝）
- （可选）GVP 节点向量 x_v：CA→N / CA→C 箭头
- （可选）GVP 边向量 edge_v：边中点方向箭头

Usage（无需命令行，直接在 __main__ 的常量里改路径/参数）:
    python viz_pymol_overlay.py
"""

from __future__ import annotations
import os
import numpy as np
import torch

# === 根据你的包结构调整导入路径 ===
from src.plaszyme.builders import (
    BuilderConfig,
    _parse_pdb_single_chain,  # 用于兜底重建坐标
)
from src.plaszyme.builders.gnn_builder import GNNProteinGraphBuilder
from src.plaszyme.builders.gvp_builder import GVPProteinGraphBuilder


# -----------------------------
# 小工具：PyMOL 场景 & 颜色映射
# -----------------------------
def _pml_header(pdb_path: str) -> str:
    """Basic PyMOL scene."""
    return (
        "reinitialize\n"
        f"load {pdb_path}, prot\n"
        "hide everything, prot\n"
        "show cartoon, prot\n"
        "color gray70, prot\n"
        "set dash_gap, 0\n"
        "set dash_width, 2\n"
        "set ray_opaque_background, off\n"
        "bg_color white\n"
    )

def _resi_str_from_resid(res_id: str) -> tuple[str, str]:
    """Parse res_ids like 'A:123' or 'A:123A' -> (chain, resi_text)."""
    chain, resi = res_id.split(":")
    return chain, resi  # PyMOL 可以直接用 /obj//A/123A/CA

def _interp_red_to_blue(val: float, vmin: float, vmax: float) -> tuple[float, float, float]:
    """
    线性插值颜色：vmin -> 红(1,0,0)，vmax -> 蓝(0,0,1)。
    """
    if vmax <= vmin:
        t = 0.5
    else:
        t = (val - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))
    r = 1.0 - t
    g = 0.0
    b = t
    return r, g, b

def _pml_set_color_line(color_name: str, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"set_color {color_name}, [{r:.4f}, {g:.4f}, {b:.4f}]\n"


# ---------- 坐标获取的鲁棒兜底 ----------
def _ensure_positions(data, pdb_path: str, chain_id: str | None = None) -> np.ndarray:
    """返回 [N,3] 的 CA 坐标（与 data 节点顺序一致）。
    优先使用 data.pos；若无，则解析 PDB，用 data.res_ids 精确重排。
    """
    # 1) 直接用 data.pos（若存在）
    if hasattr(data, "pos") and (data.pos is not None):
        return data.pos.detach().cpu().numpy()

    # 2) 兜底：解析 PDB，按 res_ids 顺序取 CA 坐标
    parsed = _parse_pdb_single_chain(pdb_path, chain_id)
    id_to_ca = {rid: ca for rid, ca in zip(parsed["res_ids"], parsed["coords_CA"])}
    pos_list = []
    for rid in data.res_ids:
        if rid not in id_to_ca:
            raise KeyError(
                f"Residue id '{rid}' not found in parsed PDB. "
                f"Check chain_id or that PDB matches the graph."
            )
        pos_list.append(id_to_ca[rid])
    return np.asarray(pos_list, dtype=float)


# -----------------------------
# 边导出（支持按距离上色）
# -----------------------------
def _write_edges_as_distances(
    f,
    data,
    pdb_path: str,
    *,
    object_prefix="link",
    color_mode: str = "constant",   # "constant" | "distance"
    constant_color: str = "cyan",
    chain_id: str | None = None,
) -> None:
    """
    将连边写为 PyMOL distance 对象。
    color_mode="constant"：统一颜色；
    color_mode="distance"：按距离上色（近红远蓝）。

    说明：为避免双向重复，仅绘制 i<j 的一半边。
    """
    ei = data.edge_index.detach().cpu().numpy()
    res_ids = list(data.res_ids)  # 对应节点顺序

    # 选择要画的边（无向一半）
    mask = ei[0] < ei[1]
    src = ei[0][mask]
    dst = ei[1][mask]

    # 统一颜色模式：先清空同名组
    if color_mode == "constant":
        f.write(f"delete {object_prefix}*\n")
        f.write(f"set dash_color, {constant_color}\n")

    # 距离上色模式：先算距离并归一化
    if color_mode == "distance":
        # 用 CA 坐标直接算欧氏距离
        P = _ensure_positions(data, pdb_path, chain_id=chain_id)  # [N,3]
        dists = np.linalg.norm(P[dst] - P[src], axis=-1)  # [E_half]
        dmin = float(np.min(dists)) if len(dists) else 0.0
        dmax = float(np.max(dists)) if len(dists) else 1.0
        # 提示下范围（写到 pml 的注释中）
        f.write(f"# distance color mapping: min={dmin:.3f} Å (red) -> max={dmax:.3f} Å (blue)\n")

    # 逐条绘制
    for n, (i, j) in enumerate(zip(src, dst)):
        ch_i, r_i = _resi_str_from_resid(res_ids[int(i)])
        ch_j, r_j = _resi_str_from_resid(res_ids[int(j)])
        sel_i = f"/prot//{ch_i}/{r_i}/CA"
        sel_j = f"/prot//{ch_j}/{r_j}/CA"
        obj_name = f"{object_prefix}_{n}"

        # 画 distance 对象
        f.write(f"distance {obj_name}, {sel_i}, {sel_j}\n")

        # 着色
        if color_mode == "constant":
            # 全局 set dash_color 已经生效；也可以逐对象 color:
            # f.write(f"color {constant_color}, {obj_name}\n")
            pass
        elif color_mode == "distance":
            d = float(np.linalg.norm(P[int(j)] - P[int(i)]))
            rgb = _interp_red_to_blue(d, dmin, dmax)
            color_name = f"{object_prefix}_col_{n}"
            f.write(_pml_set_color_line(color_name, rgb))
            f.write(f"color {color_name}, {obj_name}\n")

    # 美化
    f.write("hide labels, {}*\n".format(object_prefix))
    f.write("zoom prot\n")


# -----------------------------
# CGO 箭头（节点/边向量）
# -----------------------------
def _write_cgo_header(f):
    """Write CGO import helper in PyMOL."""
    f.write("from pymol.cgo import *\n")
    f.write("from pymol import cmd\n")

def _write_cgo_cyl_arrow(f, name: str, start: np.ndarray, end: np.ndarray,
                         r_cyl=0.2, r_sph=0.35, color=(1.0, 0.5, 0.0)):
    """Emit a simple cylinder+two spheres 'arrow' CGO object."""
    x1, y1, z1 = map(float, start)
    x2, y2, z2 = map(float, end)
    r, g, b = color
    obj = (
        f"[ CYLINDER, {x1}, {y1}, {z1}, {x2}, {y2}, {z2}, {r_cyl}, "
        f"{r}, {g}, {b}, {r}, {g}, {b}, "
        f"SPHERE, {x1}, {y1}, {z1}, {r_sph}, "
        f"SPHERE, {x2}, {y2}, {z2}, {r_sph} ]"
    )
    f.write(f"{name} = {obj}\n")
    f.write(f"cmd.load_cgo({name}, '{name}')\n")

def _write_node_vectors(f, data, pdb_path: str, scale=2.5, color=(1.0, 0.2, 0.2), chain_id: str | None = None):
    """Draw GVP node vectors (x_v[K,Vn,3]) as CGO arrows from CA."""
    if not hasattr(data, "x_v") or data.x_v is None or data.x_v.size(1) == 0:
        return
    V = data.x_v.detach().cpu().numpy()         # [K, Vn, 3]
    P = _ensure_positions(data, pdb_path, chain_id=chain_id)  # [K, 3]
    name_prefix = "node_vec"
    _write_cgo_header(f)
    # 每个节点的每个向量画一个箭头
    for k in range(V.shape[0]):
        for vn in range(V.shape[1]):
            start = P[k]
            end = start + scale * V[k, vn]
            _write_cgo_cyl_arrow(f, f"{name_prefix}_{k}_{vn}", start, end, color=color)

def _write_edge_vectors(f, data, pdb_path: str, scale=2.0, color=(0.0, 0.4, 1.0), chain_id: str | None = None):
    """Draw GVP edge vectors (edge_v[E,Ve,3]) as arrows at edge midpoints."""
    if not hasattr(data, "edge_v") or data.edge_v is None:
        return
    V = data.edge_v[:, 0, :].detach().cpu().numpy()   # 取第一个通道 [E, 3]
    ei = data.edge_index.detach().cpu().numpy()
    P = _ensure_positions(data, pdb_path, chain_id=chain_id)  # [K, 3]
    name_prefix = "edge_vec"
    _write_cgo_header(f)
    # 画 i<j 的一半
    mask = ei[0] < ei[1]
    src = ei[0][mask]
    dst = ei[1][mask]
    vlist = V[mask]
    n = 0
    for i, j, v in zip(src, dst, vlist):
        mid = 0.5 * (P[int(i)] + P[int(j)])
        end = mid + scale * v
        _write_cgo_cyl_arrow(f, f"{name_prefix}_{n}", mid, end, color=color)
        n += 1


# -----------------------------
# 主流程
# -----------------------------
def build_graph_for_pdb(
    pdb_path: str,
    mode: str = "gvp_local",                 # "gnn" 或 "gvp_local"
    out_dir: str | None = None,
    radius: float = 10.0,
    embedder_cfg=None,                 # BuilderConfig.embedder；支持单个或列表（多嵌入拼接）
    gnn_edge_mode: str = "rbf",        # GNN 的 edge_attr 方案: "none"/"dist"/"inv_dist"/"rbf"
    gvp_edge_scalar: str = "rbf",      # GVP 的 edge_s 方案
    save_pt: bool = False,
    chain_id: str | None = None,       # 如需固定链
):
    """
    Build a graph for a single PDB and return Data.

    Args:
      pdb_path: 输入的 PDB 路径（标准、含完整序列者更佳）
      mode: "gnn"（标量）或 "gvp_local"（标量+向量）
      out_dir: 输出目录（保存 .pt 与 .pml）；None 则使用 PDB 同目录的 sibling 目录
      radius: 构边半径（Å）
      embedder_cfg: 传给 BuilderConfig.embedder；可单个 dict 或 list[dict]
      gnn_edge_mode: 参见 GNN 构建器
      gvp_edge_scalar: 参见 GVP 构建器
      save_pt: 是否把 PyG Data 另存 .pt
      chain_id: 指定链 ID（传给 builder）
    """
    pdb_dir = os.path.dirname(pdb_path)
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    if out_dir is None:
        out_dir = os.path.join(pdb_dir, f"{mode}_overlay_out")
    os.makedirs(out_dir, exist_ok=True)

    cfg = BuilderConfig(
        pdb_dir=pdb_dir,
        out_dir=out_dir,
        embedder=embedder_cfg or {"name": "onehot"},
        radius=radius,
        chain_id=chain_id,
    )

    if mode.lower() == "gnn":
        builder = GNNProteinGraphBuilder(cfg, edge_mode=gnn_edge_mode)
    elif mode.lower() == "gvp_local":
        builder = GVPProteinGraphBuilder(cfg, node_vec_mode="bb2", edge_scalar=gvp_edge_scalar)
    else:
        raise ValueError("mode must be 'gnn' or 'gvp_local'.")

    data, misc = builder.build_one(pdb_path, name=name)

    if save_pt:
        torch.save(data, os.path.join(out_dir, f"{name}.pt"))

    return data, misc, out_dir, name


def export_pymol_overlay(
    pdb_path: str,
    data,
    out_dir: str,
    name: str,
    draw_node_vectors: bool = True,
    draw_edge_vectors: bool = False,
    edge_color: str = "cyan",
    color_edges_by: str = "constant",  # "constant" | "distance"
    chain_id: str | None = None,
) -> str:
    """
    Export a PyMOL script (.pml) that overlays edges and optional vectors.
    导出 PyMOL 脚本，叠加显示边与（可选）向量箭头。

    Args:
      color_edges_by: "constant" 固定颜色；"distance" 距离映射（近红远蓝）
    Returns:
      pml_path: 生成的脚本路径
    """
    pml_path = os.path.join(out_dir, f"{name}_overlay.pml")
    with open(pml_path, "w") as f:
        f.write(_pml_header(pdb_path))
        _write_edges_as_distances(
            f, data, pdb_path,
            object_prefix="link",
            color_mode=color_edges_by,
            constant_color=edge_color,
            chain_id=chain_id,
        )

        # 可选：GVP 向量
        if draw_node_vectors and hasattr(data, "x_v") and data.x_v is not None and data.x_v.size(1) > 0:
            _write_node_vectors(f, data, pdb_path, scale=2.5, color=(1.0, 0.2, 0.2), chain_id=chain_id)   # 红色箭头

        if draw_edge_vectors and hasattr(data, "edge_v") and data.edge_v is not None:
            _write_edge_vectors(f, data, pdb_path, scale=2.0, color=(0.0, 0.4, 1.0), chain_id=chain_id)   # 蓝色箭头

    return pml_path


# -----------------------------
# 示例：直接在脚本里改参数跑
# -----------------------------
if __name__ == "__main__":
    # ====== 手动改这里做快速测试 ======
    PDB_FILE = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0002.pdb"
    MODE = "gnn"                 # "gnn" or "gvp_local"
    RADIUS = 10.0
    # 多嵌入拼接示例：物化+onehot+ESM
    EMBEDDERS = [
        {"name": "physchem", "norm": "zscore", "concat_onehot": True},
        {"name": "onehot"},
        {"name": "esm", "model_name": "esm2_t12_35M_UR50D"},
    ]

    data, misc, out_dir, name = build_graph_for_pdb(
        pdb_path=PDB_FILE,
        mode=MODE,
        radius=RADIUS,
        embedder_cfg=EMBEDDERS,
        gnn_edge_mode="rbf",
        gvp_edge_scalar="rbf",
        save_pt=True,
        chain_id=None,  # 如需固定链，填 'A' 等
    )

    # 导出 PyMOL 脚本（GVP 可选择画向量）
    pml = export_pymol_overlay(
        pdb_path=PDB_FILE,
        data=data,
        out_dir=out_dir,
        name=name,
        draw_node_vectors=(MODE == "gvp_local"),
        draw_edge_vectors=True,         # 如需展示 edge_v，把它改 True
        edge_color="cyan",              # color_edges_by="constant" 时生效
        color_edges_by="distance",      # ★ 按距离上色：近红远蓝
        chain_id=None,
    )
    print(f"[OK] Wrote PyMOL script: {pml}")
    print("Open in PyMOL:  @", pml)