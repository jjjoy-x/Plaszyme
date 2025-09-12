#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qc_pdb_format.py
----------------
对 PDB 进行“构图前”的格式与内容体检，输出：
  - CSV 报告：逐 (PDB, chain) 的体检结果
  - TXT 汇总：全局统计与建议

检查项（逐链）：
  - has_seqres, n_seqres_lines
  - n_models, multi_model
  - n_ca (可用 CA 个数), ca_lt2
  - has_nan, has_inf, coord_abs_max
  - has_altloc
  - has_hetatm
  - ppb_seq_empty, atom_seq_empty
  - res_ids_len_match
  - ok_for_builder（CA>=2 且无 NaN/Inf）
"""

from __future__ import annotations
import os, re, math
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# ========== 修改路径 ==========
PDB_DIR   = "/root/autodl-tmp/M-CSA/pdbs"      # 你的 PDB 目录
OUT_DIR   = "/root/autodl-tmp/M-CSA/qc_format" # 输出目录
# =============================

os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT   = os.path.join(OUT_DIR, "pdb_format_report.csv")
TXT_SUM   = os.path.join(OUT_DIR, "pdb_format_summary.txt")

# --------- 轻量 PDB 解析（依赖 Bio.PDB）---------
from Bio.PDB import PDBParser, PPBuilder

def fast_scan_seqres_and_models(pdb_path: str) -> Tuple[bool, int, int]:
    """快速扫描文本，统计 SEQRES 行数和模型数量。"""
    has_seqres, n_seqres = False, 0
    n_models = 0
    try:
        with open(pdb_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("SEQRES"):
                    has_seqres = True
                    n_seqres += 1
                elif line.startswith("MODEL "):
                    n_models += 1
    except Exception:
        pass
    return has_seqres, n_seqres, n_models

def analyze_pdb(pdb_path: str) -> List[Dict[str, Any]]:
    """返回逐链的体检结果记录列表。"""
    recs: List[Dict[str, Any]] = []
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

    has_seqres, n_seqres, n_models = fast_scan_seqres_and_models(pdb_path)
    multi_model = (n_models > 1)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", pdb_path)
    except Exception as e:
        recs.append({
            "pdb_id": pdb_id, "chain_id": "",
            "error": f"parse_fail:{e}", "ok_for_builder": False
        })
        return recs

    model = next(structure.get_models())  # 取第一个模型进行检查
    ppb = PPBuilder()

    for chain in model.get_chains():
        cid = chain.id

        # 统计 HETATM 是否存在（全链）
        has_hetatm = False
        for res in chain.get_residues():
            if res.id[0] != " ":
                has_hetatm = True
                break

        # 统计是否存在 altloc（任一原子 altloc 字段非空）
        has_altloc = False
        for atom in chain.get_atoms():
            altloc = getattr(atom, "get_altloc", None)
            alt = atom.get_altloc() if callable(altloc) else atom.get_altloc() if hasattr(atom, "get_altloc") else atom.altloc if hasattr(atom, "altloc") else ""
            if isinstance(alt, str) and alt not in ("", " "):
                has_altloc = True
                break

        # 收集骨架 CA 坐标 + 生成 res_ids
        ca_coords = []
        res_ids   = []
        atom_seq  = []
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            # CA
            if "CA" in res:
                xyz = res["CA"].get_coord().astype(float)
                ca_coords.append(xyz)
                resseq = res.id[1]
                icode  = res.id[2].strip() if isinstance(res.id[2], str) else ""
                res_ids.append(f"{cid}:{resseq}{icode}")
                # 简易 3→1，不追求完备
                atom_seq.append(res.get_resname())

        n_ca = len(ca_coords)
        ca_arr = np.asarray(ca_coords, dtype=float).reshape(n_ca, 3) if n_ca>0 else np.zeros((0,3), dtype=float)

        has_nan = bool(np.isnan(ca_arr).any())
        has_inf = bool(np.isinf(ca_arr).any())
        coord_abs_max = float(np.abs(ca_arr).max()) if n_ca>0 else 0.0

        # 与 PPBuilder 的序列（多肽）对比是否为空
        try:
            pp_seqs = [str(poly.get_sequence()) for poly in ppb.build_peptides(chain, aa_only=False)]
            ppb_seq_empty = (len(pp_seqs) == 0 or all(len(s)==0 for s in pp_seqs))
        except Exception:
            ppb_seq_empty = True

        atom_seq_empty = (len(atom_seq) == 0)

        # res_ids 与 CA 个数匹配
        res_ids_len_match = (len(res_ids) == n_ca)

        # 构图可用性：CA>=2 且无 NaN/Inf
        ok_build = (n_ca >= 2) and (not has_nan) and (not has_inf)

        # 建议
        advice = []
        if n_ca < 2:
            advice.append("CA<2: skip or merge chains")
        if has_nan or has_inf:
            advice.append("NaN/Inf coords: clean or skip")
        if multi_model:
            advice.append("multi-model: select MODEL 1")
        if has_altloc:
            advice.append("altloc present: keep 'A' or highest occ")
        if atom_seq_empty:
            advice.append("ATOM seq empty")
        if ppb_seq_empty and has_seqres:
            advice.append("PPBuilder empty but SEQRES present (ligand-only?)")
        if not res_ids_len_match:
            advice.append("res_ids length mismatch")

        recs.append({
            "pdb_id": pdb_id,
            "chain_id": cid,
            "has_seqres": has_seqres,
            "n_seqres_lines": n_seqres,
            "n_models": n_models,
            "multi_model": multi_model,
            "n_ca": n_ca,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "coord_abs_max": coord_abs_max,
            "has_altloc": has_altloc,
            "has_hetatm": has_hetatm,
            "ppb_seq_empty": ppb_seq_empty,
            "atom_seq_empty": atom_seq_empty,
            "res_ids_len_match": res_ids_len_match,
            "ok_for_builder": ok_build,
            "advice": "; ".join(advice),
            "error": "",
        })

    return recs


def main():
    pdb_files = sorted([f for f in os.listdir(PDB_DIR) if f.lower().endswith(".pdb")])
    all_rows: List[Dict[str, Any]] = []

    for i, fname in enumerate(pdb_files, 1):
        pdb_path = os.path.join(PDB_DIR, fname)
        rows = analyze_pdb(pdb_path)
        all_rows.extend(rows)
        if i % 100 == 0:
            print(f"  - scanned {i}/{len(pdb_files)} ...")

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] CSV written: {CSV_OUT} | rows={len(df)}")

    # 汇总
    lines = []
    push = lines.append
    n_pdb = len(pdb_files)
    n_rows = len(df)
    push(f"# Summary")
    push(f"- total PDB files: {n_pdb}")
    push(f"- total (PDB,chain) rows: {n_rows}")

    if "ok_for_builder" in df.columns and n_rows>0:
        ok_frac = 100.0 * df["ok_for_builder"].mean()
        push(f"- chains ok_for_builder: {df['ok_for_builder'].sum()} / {n_rows} ({ok_frac:.2f}%)")

    for col, title in [
        ("multi_model", "chains with multi-model PDB"),
        ("has_altloc", "chains with altloc"),
        ("has_hetatm", "chains with HETATM"),
        ("ppb_seq_empty", "chains with empty PPBuilder seq"),
        ("atom_seq_empty", "chains with empty ATOM seq"),
    ]:
        if col in df.columns:
            cnt = int(df[col].sum())
            push(f"- {title}: {cnt} / {n_rows}")

    # 常见失败原因 TopN
    if "advice" in df.columns:
        top_bad = (
            df.loc[df["advice"].astype(str)!="", "advice"]
              .value_counts()
              .head(20)
              .to_string()
        )
        push("\n# Top advice patterns")
        push(top_bad)

    with open(TXT_SUM, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] Summary written: {TXT_SUM}")

if __name__ == "__main__":
    main()