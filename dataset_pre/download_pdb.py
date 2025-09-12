#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载 PDB 文件 (ASCII .pdb 格式)
- 输入: merged_mcsa_ec.csv (需要包含 `pdb_id` 列)
- 输出: 每个 PDB 的 .pdb 文件保存到 OUT_DIR

依赖:
  pip install pandas requests tqdm
"""

import os
import requests
import pandas as pd
from tqdm import tqdm

# ===== 修改路径 =====
CSV_PATH = "/tmp/pycharm_project_317/dataset/M-CSA/literature_pdb_residues_roles.csv"
OUT_DIR  = "/root/autodl-tmp/M-CSA/pdbs"
# ===================

# RCSB 下载地址模板
URL_TEMPLATE = "https://files.rcsb.org/download/{}.pdb"

def download_pdb(pdb_id: str, out_dir: str) -> str:
    """下载单个 PDB 文件"""
    pdb_id = pdb_id.lower().strip()
    url = URL_TEMPLATE.format(pdb_id)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path  # 已存在

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
        else:
            print(f"[WARN] Failed {pdb_id}: HTTP {resp.status_code}")
            return ""
    except Exception as e:
        print(f"[ERROR] Failed {pdb_id}: {e}")
        return ""

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    if "PDB ID" not in df.columns:
        raise ValueError("CSV 缺少 'pdb_id' 列")

    pdb_ids = sorted(set(df["PDB ID"].dropna().astype(str)))
    print(f"[INFO] Found {len(pdb_ids)} unique PDB IDs.")

    downloaded, failed = 0, 0
    for pid in tqdm(pdb_ids, desc="Downloading PDBs"):
        path = download_pdb(pid, OUT_DIR)
        if path:
            downloaded += 1
        else:
            failed += 1

    print(f"[DONE] Downloaded={downloaded}, Failed={failed}, Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()