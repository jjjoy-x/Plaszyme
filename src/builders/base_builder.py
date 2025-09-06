#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base protein graph builder (flat utilities inside).

基础蛋白质图构建器：内部自带 PDB 解析、序列对齐、嵌入器创建等通用能力；
GNN/GVP 子类只需实现节点/边特征的组织即可。

This module defines:
  - `BuilderConfig`: runtime config.
  - `BaseProteinGraphBuilder`: iterate PDBs → parse → align → delegate
    node/edge features → save PyG `Data`.
  - Inline helpers for PDB I/O and a simplified full-vs-atom alignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ---- embedder registry (load from your existing sequence_embedder.py) ----
# 与现有的 sequence_embedder.py 对接；若只使用某些类，也可按需精简导入。
from .sequence_embedder import ESMEmbedder, OneHotEmbedder, PhysChemEmbedder


# =============================================================================
# Config
# =============================================================================

@dataclass
class BuilderConfig:
    """Runtime configuration for builders.

    构建器运行配置。

    Attributes:
        pdb_dir (str): Directory containing standard PDB files (1 chain/file or choose a chain).
            包含标准 PDB 的目录（建议单链/文件或通过 chain_id 指定）。
        out_dir (str): Output directory for `.pt` graphs and `manifest.csv`.
            输出目录，保存 `.pt` 与 `manifest.csv`。
        chain_id (Optional[str]): Chain ID to parse; if None, use the first chain.
            指定链 ID；None 则取首链。
        radius (float): Distance cutoff (Å) for edge building (used by subclasses).
            构边半径阈值（Å），子类使用。
        embedder (Dict[str, Any]): Sequence embedder spec, e.g. {"name":"esm", ...}.
            序列嵌入器配置（示例：{"name":"esm", ...}）。
    """
    pdb_dir: str
    out_dir: str
    chain_id: Optional[str] = None
    radius: float = 10.0
    embedder: Dict[str, Any] = None


# =============================================================================
# Internal PDB I/O (single-chain)  内置 PDB 解析
# =============================================================================

def _parse_pdb_single_chain(pdb_path: str, chain_id: Optional[str]) -> Dict[str, Any]:
    """Parse a PDB file for one chain and extract backbone coords and sequences.

    解析单链 PDB：给出 SEQRES（若可得）、ATOM 序列，以及骨架 N/CA/C 坐标。

    Args:
        pdb_path: Path to PDB file.
        chain_id: Target chain ID; if None, pick the first chain.

    Returns:
        Dict[str, Any]: {
            "seq_full": Optional[str],   # SEQRES (or None)
            "seq_atom": str,             # from ATOM records after filtering
            "coords_N": np.ndarray[K,3],
            "coords_CA": np.ndarray[K,3],
            "coords_C": np.ndarray[K,3],
            "res_ids": List[str],        # e.g., "A:123" (with insertion code if present)
        }
    """
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())

    # pick chain
    ch = None
    for _ch in model.get_chains():
        if chain_id is None:
            ch = _ch
            break
        if _ch.id == chain_id:
            ch = _ch
            break
    if ch is None:
        raise ValueError(f"No chain found in {pdb_path} (requested={chain_id})")

    # SEQRES (best effort)
    seq_full = None
    try:
        ppb = PPBuilder()
        for poly in ppb.build_peptides(ch, aa_only=False):
            s = str(poly.get_sequence())
            seq_full = s if seq_full is None else (seq_full + s)
    except Exception:
        seq_full = None

    # ATOM sequence + backbone coords
    seq_atom, coords_N, coords_CA, coords_C, res_ids = [], [], [], [], []
    for res in ch.get_residues():
        # skip hetero/water
        if res.id[0] != " ":
            continue

        # 3-letter to 1-letter
        resname = res.get_resname()
        try:
            aa1 = seq1(
                resname,
                custom_map={
                    "MSE": "M",  # 硒代蛋氨酸常见于晶体结构
                    "SEC": "U",  # Selenocysteine（可按需要映射为 "C" 或 "X"）
                    "PYL": "O",  # Pyrrolysine（也可映射为 "K" 或 "X"）
                },
                undef_code="X",  # 未知或非常规残基 → 'X'
            )
        except Exception:
            aa1 = "X"

        def get_coord(atom_name: str) -> np.ndarray:
            if atom_name in res:
                return res[atom_name].get_coord().astype(float)
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        n = get_coord("N")
        ca = get_coord("CA")
        c = get_coord("C")

        # require CA present
        if not np.isfinite(ca).all():
            continue

        seq_atom.append(aa1)
        coords_N.append(n)
        coords_CA.append(ca)
        coords_C.append(c)

        resseq = res.id[1]
        icode = res.id[2].strip() if isinstance(res.id[2], str) else ""
        res_ids.append(f"{ch.id}:{resseq}{icode}")

    return {
        "seq_full": seq_full,
        "seq_atom": "".join(seq_atom),
        "coords_N": np.asarray(coords_N, dtype=float),
        "coords_CA": np.asarray(coords_CA, dtype=float),
        "coords_C": np.asarray(coords_C, dtype=float),
        "res_ids": res_ids,
    }


# =============================================================================
# Simplified alignment  简化序列对齐（前后缀裁剪）
# =============================================================================

def _align_seq_full_and_atom(seq_full: str, seq_atom: str) -> Dict[str, Any]:
    """Align full sequence (SEQRES) and observed ATOM sequence by prefix/suffix trimming.

    简化对齐策略：裁剪公共前缀与后缀，并对中段按较短序列对齐。适合 AlphaFold/多数标准 PDB。
    若差异较大，建议替换为 Needleman–Wunsch 全局对齐（仅需改此函数）。

    Args:
        seq_full: Full sequence (SEQRES or ATOM fallback).
        seq_atom: Observed sequence from ATOM.

    Returns:
        Dict[str, Any]: {
            "L_full": int,
            "kept_idx": np.ndarray[K],           # indices into full sequence
            "seq_to_atom_idx": np.ndarray[L_full],  # -1 if not observed
            "atom_kept_mask": np.ndarray[La],    # mask in atom seq kept
        }
    """
    Lf, La = len(seq_full), len(seq_atom)
    if seq_full == seq_atom:
        kept_idx = np.arange(Lf, dtype=int)
        return {
            "L_full": Lf,
            "kept_idx": kept_idx,
            "seq_to_atom_idx": np.arange(Lf, dtype=int),
            "atom_kept_mask": np.ones(La, dtype=bool),
        }

    # trim common prefix
    i0 = 0
    while i0 < min(Lf, La) and seq_full[i0] == seq_atom[i0]:
        i0 += 1
    # trim common suffix
    j0 = 0
    while j0 < min(Lf - i0, La - i0) and seq_full[Lf - 1 - j0] == seq_atom[La - 1 - j0]:
        j0 += 1

    mid = max(0, min(Lf - i0, La - i0) - j0)
    kept_idx = np.arange(i0, i0 + mid, dtype=int)

    seq_to_atom = np.full(Lf, -1, dtype=int)
    atom_kept_mask = np.zeros(La, dtype=bool)
    for k in range(mid):
        pos_f = i0 + k
        pos_a = i0 + k
        seq_to_atom[pos_f] = pos_a
        atom_kept_mask[pos_a] = True

    return {
        "L_full": Lf,
        "kept_idx": kept_idx,
        "seq_to_atom_idx": seq_to_atom,
        "atom_kept_mask": atom_kept_mask,
    }


# =============================================================================
# Simple embedder factory  内置嵌入器工厂
# =============================================================================

def _create_embedder(name: str = "esm", **kwargs) -> Any:
    """Create a residue-level embedder by name.

    根据名称创建残基级嵌入器。支持 "esm" / "onehot" / "physchem"。

    Args:
        name: Embedder name.
        **kwargs: Forwarded to the embedder constructor.

    Returns:
        Any: An object exposing `.embed(seq)->Tensor[L,D]`.
    """
    name = (name or "esm").lower()
    if name == "esm":
        return ESMEmbedder(**kwargs)
    if name == "onehot":
        return OneHotEmbedder(**kwargs)
    if name == "physchem":
        return PhysChemEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder name: {name}")


# =============================================================================
# Base builder  基类：流程 + 钩子
# =============================================================================

class BaseProteinGraphBuilder:
    """Base protein graph builder with flat utilities inside.

    通用流程（遍历→解析→对齐→委托节点/边→保存），对齐/解析等工具内嵌。
    子类只实现 `_build_node_features(...)` 与 `_build_edges(...)` 两个钩子。
    """

    def __init__(self, cfg: BuilderConfig):
        """Init.

        Args:
            cfg: Runtime configuration. 运行配置。
        """
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.embedder = _create_embedder(**(self.cfg.embedder or {}))

    # ---------- Public API ----------
    def build_folder(self) -> None:
        """Build graphs for all PDB files in the folder.

        遍历 PDB 目录构建图，并写出 manifest.csv。
        """
        rows: List[Dict[str, Any]] = []
        pdb_files = sorted(f for f in os.listdir(self.cfg.pdb_dir) if f.lower().endswith(".pdb"))

        for fname in tqdm(pdb_files, desc="Building graphs"):
            name = os.path.splitext(fname)[0]
            pdb_path = os.path.join(self.cfg.pdb_dir, fname)
            try:
                data, misc = self.build_one(pdb_path, name=name)
                out_pt = os.path.join(self.cfg.out_dir, f"{name}.pt")
                torch.save(data, out_pt)
                rows.append({**misc, "name": name, "status": "ok", "pt_path": out_pt})
            except Exception as e:
                rows.append({"name": name, "status": f"error:{e}", "pt_path": "", "pdb_path": pdb_path})

        pd.DataFrame(rows).to_csv(os.path.join(self.cfg.out_dir, "manifest.csv"), index=False)

    def build_one(self, pdb_path: str, name: Optional[str] = None) -> Tuple[Data, Dict[str, Any]]:
        """Build one PyG graph from a PDB.

        从单个 PDB 构建 PyG 图。

        Args:
            pdb_path: Path to PDB.
            name: Optional sample name (default: file stem).

        Returns:
            Tuple[Data, Dict[str, Any]]: (graph, misc for manifest).
        """
        if name is None:
            name = os.path.splitext(os.path.basename(pdb_path))[0]

        # 1) parse PDB
        parsed = _parse_pdb_single_chain(pdb_path, self.cfg.chain_id)

        # 2) choose full sequence (SEQRES preferred; fallback to ATOM)
        seq_full = parsed["seq_full"] if parsed["seq_full"] else parsed["seq_atom"]
        warn = "" if parsed["seq_full"] else "no_seqres_fallback_atom"

        # 3) simplified alignment
        aln = _align_seq_full_and_atom(seq_full, parsed["seq_atom"])

        # 4) delegate node features
        node_feats = self._build_node_features(
            seq_full=seq_full,
            kept_idx=aln["kept_idx"],
            parsed=parsed,     # includes coords and res_ids
        )

        # 5) delegate edges (use kept CA coords)
        coords_ca_kept = parsed["coords_CA"][aln["atom_kept_mask"]]
        edge_feats = self._build_edges(coords_ca_kept, parsed=parsed, aln=aln)

        # 6) assemble Data
        data = Data()
        data.name = name
        # alignment meta
        data.L_full = int(aln["L_full"])
        data.kept_idx = torch.as_tensor(aln["kept_idx"], dtype=torch.long)
        data.seq_to_atom_idx = torch.as_tensor(aln["seq_to_atom_idx"], dtype=torch.long)
        data.res_ids = parsed["res_ids"]  # list[str] for debugging/inspection

        # subclass fields
        for k, v in {**node_feats, **edge_feats}.items():
            data[k] = v

        misc = {
            "pdb_path": pdb_path,
            "seq_full_len": aln["L_full"],
            "seq_atom_len": len(parsed["seq_atom"]),
            "kept_len": len(aln["kept_idx"]),
            "warn": warn,
        }
        return data, misc

    # ---------- Hooks for subclass ----------
    def _build_node_features(self, *, seq_full: str, kept_idx: np.ndarray, parsed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build node features (override in subclasses).

        构建节点特征（子类实现）。
        GNN: return {"x": [K, D]}
        GVP: return {"x_s": [K, Ds], "x_v": [K, Dv, 3]}

        Args:
            seq_full: Full sequence used to embed.
            kept_idx: Indices in full sequence to keep (aligned).
            parsed: Parsed PDB dict (coords, seqs, res_ids).

        Returns:
            Dict[str, torch.Tensor]: Node feature dict.
        """
        raise NotImplementedError

    def _build_edges(self, coords_ca_kept: np.ndarray, *, parsed: Dict[str, Any], aln: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build edges (override in subclasses).

        构建边（子类实现）。
        GNN: at least {"edge_index"}
        GVP: typically {"edge_index", "edge_s", "edge_v"}

        Args:
            coords_ca_kept: Kept CA coordinates [K,3].
            parsed: Parsed PDB dict.
            aln: Alignment dict.

        Returns:
            Dict[str, torch.Tensor]: Edge feature dict.
        """
        raise NotImplementedError