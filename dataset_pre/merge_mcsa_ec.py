#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge M-CSA roles (keep multiple role_type per residue) with curated EC labels,
and drop only EXACT duplicate rows.

- 不做“每残基一个角色”的合并；同一残基若有多个 role_type，一律保留
- 仅确保输出行不重复：按 (pdb_id, chain, resid_int, aa, role_type, ec_code) 去重
"""

from __future__ import annotations
from typing import List, Optional
import os, csv
import pandas as pd

# ========= 修改这里的路径 =========
ROLES_CSV   = "/tmp/pycharm_project_317/dataset/M-CSA/literature_pdb_residues_roles.csv"
CURATED_CSV = "/tmp/pycharm_project_317/dataset/M-CSA/curated_data.csv"
OUT_CSV     = "/tmp/pycharm_project_317/dataset/M-CSA/merged_mcsa_ec.csv"
STATS_TXT   = "/tmp/pycharm_project_317/dataset/M-CSA/stats_mcsa_ec.txt"
# =================================

# ---------- 读取 ----------
def safe_read_csv(path: str) -> pd.DataFrame:
    tries = [
        {},
        {"engine": "python"},
        {"engine": "python", "on_bad_lines": "skip"},
        {"engine": "python", "sep": ",", "quotechar": '"', "escapechar": "\\", "on_bad_lines": "skip"},
        {"engine": "python", "sep": ";", "on_bad_lines": "skip"},
        {"engine": "python", "sep": "\t", "on_bad_lines": "skip"},
    ]
    for kw in tries:
        try:
            return pd.read_csv(path, encoding="utf-8", **kw)
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(8192)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except Exception:
            delim = ","
    return pd.read_csv(path, encoding="utf-8", engine="python", sep=delim, on_bad_lines="skip")

# ---------- 工具 ----------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower() for c in df.columns})

def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for nm in candidates:
        key = nm.strip().lower()
        if key in m:
            return m[key]
    if required:
        raise ValueError(f"Required column not found. Tried: {candidates}\nAvailable: {list(df.columns)}")
    return None

def clean_pdb(x: str) -> str:
    return "" if pd.isna(x) else str(x).strip().upper()

def clean_chain(x: str) -> str:
    return "" if pd.isna(x) else str(x).strip()

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def split_ecs(s: str):
    if pd.isna(s):
        return []
    s = str(s).replace(";", ",").replace(" ", ",")
    return [t for t in s.split(",") if t]

def ec_top(ec: str) -> Optional[str]:
    if not ec:
        return None
    return ec.split(".")[0]

# ---------- schema 猜测 ----------
def roles_schema(df: pd.DataFrame):
    d = norm_cols(df)
    return {
        "pdb":      find_col(d, ["pdb id", "pdb_id", "pdb", "pdb code", "pdbcode"]),
        "chain":    find_col(d, ["chain id", "chain", "chain_name", "assembly_chain_name", "auth_chain_id"], required=False),
        "resid":    find_col(d, ["residue number", "resid", "auth_resid", "res_id", "resseq"]),
        "aa":       find_col(d, ["residue type", "residue", "residue_name", "code", "aa", "resname"]),
        "role":     find_col(d, ["role", "chemical function", "parent role"], required=False),
        "role_type":find_col(d, ["role_type", "function type", "function_type"]),
    }

def curated_schema(df: pd.DataFrame):
    d = norm_cols(df)
    return {
        "pdb": find_col(d, ["pdb_id", "pdb id", "reference pdb", "reference_pdb", "pdb", "pdb code", "reference pdb id", "reference_pdb_id"]),
        "ec":  find_col(d, ["ec_number", "ec", "ec code", "ecs", "ec_codes", "entries.reactions.ecs.codes", "reaction_ec"]),
        "uniprot": find_col(d, ["uniprot", "uniprot_id", "entries.proteins.sequences.uniprot_ids"], required=False),
    }

# ---------- 合并 ----------
def merge_keep_role_type(df_roles: pd.DataFrame, df_cur: pd.DataFrame) -> pd.DataFrame:
    rmap = roles_schema(df_roles)
    cmap = curated_schema(df_cur)

    r = norm_cols(df_roles.copy())
    keep_r = [rmap["pdb"], rmap["resid"], rmap["aa"], rmap["role_type"]]
    if rmap["chain"]: keep_r.append(rmap["chain"])
    if rmap["role"]:  keep_r.append(rmap["role"])  # 可选保留，便于检查
    r = r[keep_r].rename(columns={
        rmap["pdb"]: "pdb_id",
        rmap["resid"]: "resid",
        rmap["aa"]: "aa",
        rmap["role_type"]: "role_type",
        (rmap["chain"] or "chain"): "chain",
        (rmap["role"] or "role"): "role",
    })
    r["pdb_id"] = r["pdb_id"].map(clean_pdb)
    r["chain"]  = r.get("chain", "").map(clean_chain) if "chain" in r.columns else ""
    r["resid_int"] = r["resid"].map(to_int)
    r["is_catalytic"] = True

    c = norm_cols(df_cur.copy())
    keep_c = [cmap["pdb"], cmap["ec"]]
    if cmap["uniprot"]: keep_c.append(cmap["uniprot"])
    c = c[keep_c].rename(columns={
        cmap["pdb"]: "pdb_id",
        cmap["ec"]:  "ec",
        (cmap["uniprot"] or "uniprot"): "uniprot",
    })
    c["pdb_id"] = c["pdb_id"].map(clean_pdb)
    if "uniprot" not in c.columns:
        c["uniprot"] = ""
    c["ec_code"] = c["ec"].apply(split_ecs)
    c = c.explode("ec_code", ignore_index=True)
    c = c[c["ec_code"].notna() & (c["ec_code"] != "")]
    c["ec_top"] = c["ec_code"].map(ec_top)

    # 只保留 curated 中出现且有 EC 的 PDB
    valid_pdb = set(c["pdb_id"].unique())
    r = r[r["pdb_id"].isin(valid_pdb)]

    merged = r.merge(c[["pdb_id", "ec_code", "ec_top", "uniprot"]], on="pdb_id", how="inner")

    # 只去掉完全重复：键不含 'role'，避免不同细粒度 role 被误删
    dedup_keys = ["pdb_id", "chain", "resid_int", "aa", "role_type", "ec_code"]
    for k in dedup_keys:
        if k not in merged.columns:
            merged[k] = ""  # 安全兜底
    merged = merged.drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)

    ordered = ["pdb_id", "chain", "resid", "resid_int", "aa",
               "role_type", "is_catalytic", "ec_code", "ec_top", "uniprot"]
    if "role" in r.columns:
        ordered.insert(5, "role")  # 把 role 放在 role_type 前一列（可选）
    ordered = [c for c in ordered if c in merged.columns]
    return merged[ordered]

# ---------- 统计 ----------
def describe(merged: pd.DataFrame) -> str:
    lines = []
    push = lines.append
    push(f"# rows: {len(merged)}")
    push(f"# unique PDBs: {merged['pdb_id'].nunique()}")
    if "ec_code" in merged:
        push(f"# unique EC codes: {merged['ec_code'].nunique()}")
    if "role_type" in merged:
        push("\nrole_type distribution:\n" + merged["role_type"].value_counts().to_string())
    if "ec_top" in merged:
        push("\nEC top-level distribution:\n" + merged["ec_top"].value_counts().sort_index().to_string())
    return "\n".join(lines)

# ---------- 主函数 ----------
def main():
    assert os.path.exists(ROLES_CSV), f"roles not found: {ROLES_CSV}"
    assert os.path.exists(CURATED_CSV), f"curated not found: {CURATED_CSV}"

    roles = safe_read_csv(ROLES_CSV)
    cur   = safe_read_csv(CURATED_CSV)
    print(f"[INFO] roles rows={len(roles)}, curated rows={len(cur)}")

    merged = merge_keep_role_type(roles, cur)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"[OK] saved: {OUT_CSV} | rows={len(merged)}")

    rep = describe(merged)
    print(rep)
    with open(STATS_TXT, "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print(f"[OK] stats: {STATS_TXT}")

if __name__ == "__main__":
    main()