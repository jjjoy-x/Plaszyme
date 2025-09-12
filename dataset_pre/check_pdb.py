#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdb_sanity_check.py — 批量体检 PDB / mmCIF（不使用命令行）
用法：
    from pdb_sanity_check import check_pdbs
    rows, summary = check_pdbs("/path/to/pdbs", out_csv="/path/to/report.csv")

检查内容（全部针对 model 0）：
- n_models: 模型个数（>1 时经常需要只取 model 0）
- n_chains: 链条数
- std_aa_count: 标准氨基酸残基总数（包含无 CA 的）
- std_ca_count: 含 CA 的标准氨基酸数（==0 极可能触发 Bad coords_CA shape）
- std_no_ca_count: 没有 CA 的标准氨基酸数
- nonstd_count: 非标准氨基酸数（Bio.PDB:is_aa(standard=False) 但非 HET）
- het_count: HET 残基数（ligand/水等，hetflag != " "）
- insert_code_count: 残基插入码数（icode != " "）
- altloc_ca_count: CA 原子存在 altLoc 的残基数
- per_chain_std_ca: 每条链的标准AA+CA数（形如 "A:123;B:118"）
- error_category: 粗分类（OK / NO_STD_CA / PARSE_FAIL / UNKNOWN）
- note: 备注（multi_models / has_insert_codes / has_altloc_CA / has_nonstandard）

返回：
- rows: List[Dict]（逐文件结果）
- summary: Dict（总体统计与比率）
"""

from __future__ import annotations
import os
import csv
from typing import Dict, List, Tuple
from collections import defaultdict

# 可选：抑制 Bio.PDB 告警（注释掉则显示）
import warnings
warnings.filterwarnings("ignore")


def _load_structure(path: str):
    """根据扩展名选择 PDBParser 或 MMCIFParser，返回 Structure；失败则抛异常。"""
    from Bio.PDB import PDBParser
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
    except Exception:
        MMCIFParser = None

    ext = os.path.splitext(path)[1].lower()
    if ext in {".cif", ".mmcif"} and MMCIFParser is not None:
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure("x", path)
    else:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("x", path)
    return struct


def _analyze_one(path: str) -> Dict:
    """返回单个结构的统计信息字典（仅 model 0）。"""
    from Bio.PDB import is_aa

    result = {
        "pdb_id": os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "n_models": 0,
        "n_chains": 0,
        "std_aa_count": 0,
        "std_ca_count": 0,
        "std_no_ca_count": 0,
        "nonstd_count": 0,
        "het_count": 0,
        "insert_code_count": 0,
        "altloc_ca_count": 0,
        "per_chain_std_ca": "",   # 例如 "A:123;B:118"
        "error_category": "OK",
        "note": "",
    }

    try:
        struct = _load_structure(path)
    except Exception as e:
        result["error_category"] = "PARSE_FAIL"
        result["note"] = f"parse_error: {type(e).__name__}: {e}"
        return result

    # 统计 model 数
    models = list(struct.get_models())
    result["n_models"] = len(models)
    if len(models) == 0:
        result["error_category"] = "PARSE_FAIL"
        result["note"] = "no_models"
        return result

    # 仅 model 0
    model0 = models[0]

    per_chain_std_ca = defaultdict(int)
    per_chain_std_all = defaultdict(int)  # 包括无 CA 的标准残基

    for chain in model0:
        chain_id = str(chain.id)
        for res in chain:
            hetflag, resseq, icode = res.id  # (hetflag, int, icode)

            # HET 计数
            if hetflag != " ":
                result["het_count"] += 1
                continue

            # 插入码计数
            if icode != " ":
                result["insert_code_count"] += 1

            # 氨基酸识别
            if not is_aa(res, standard=False):
                # 既不是标准AA，也不是 HET（未知/异常）
                result["nonstd_count"] += 1
                continue

            is_standard = is_aa(res, standard=True)
            if is_standard:
                result["std_aa_count"] += 1
                per_chain_std_all[chain_id] += 1

                if "CA" in res:
                    ca = res["CA"]
                    try:
                        if ca.is_disordered():
                            result["altloc_ca_count"] += 1
                        _ = ca.get_coord()  # 确认可用
                        result["std_ca_count"] += 1
                        per_chain_std_ca[chain_id] += 1
                    except Exception:
                        result["std_no_ca_count"] += 1
                else:
                    result["std_no_ca_count"] += 1
            else:
                # 非标准氨基酸（修饰 AA）
                result["nonstd_count"] += 1

    result["n_chains"] = len(list(model0.get_chains()))

    # 风险判定：无任何“标准AA+可用CA”
    if result["std_ca_count"] == 0:
        result["error_category"] = "NO_STD_CA"

    # 每链标准AA+CA数
    if per_chain_std_ca:
        parts = [f"{cid}:{per_chain_std_ca[cid]}" for cid in sorted(per_chain_std_ca.keys())]
        result["per_chain_std_ca"] = ";".join(parts)

    # 备注
    if result["n_models"] > 1:
        result["note"] = (result["note"] + "; " if result["note"] else "") + "multi_models"
    if result["insert_code_count"] > 0:
        result["note"] = (result["note"] + "; " if result["note"] else "") + "has_insert_codes"
    if result["altloc_ca_count"] > 0:
        result["note"] = (result["note"] + "; " if result["note"] else "") + "has_altloc_CA"
    if result["nonstd_count"] > 0:
        result["note"] = (result["note"] + "; " if result["note"] else "") + "has_nonstandard"

    return result


def check_pdbs(pdb_dir: str,
               out_csv: str | None = None,
               exts: Tuple[str, ...] = (".pdb", ".ent", ".cif", ".mmcif")) -> Tuple[List[Dict], Dict]:
    """
    扫描目录并体检所有结构文件（仅 model 0），返回 (rows, summary)。
    - pdb_dir: 结构目录
    - out_csv: 若不为 None，则把 rows 写入该 CSV
    - exts: 允许的扩展名

    summary 字段示例：
      {
        "counts": {"OK": 123, "NO_STD_CA": 7, "PARSE_FAIL": 2, "UNKNOWN": 1},
        "total": 133,
        "ratios": {"OK": 0.925..., ...}
      }
    """
    files: List[str] = []
    allow = set(e.lower() for e in exts)
    for fn in os.listdir(pdb_dir):
        p = os.path.join(pdb_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in allow:
            files.append(p)
    files.sort()

    rows: List[Dict] = []
    stats = defaultdict(int)

    for path in files:
        try:
            r = _analyze_one(path)
        except Exception as e:
            r = {
                "pdb_id": os.path.splitext(os.path.basename(path))[0],
                "path": path,
                "n_models": 0, "n_chains": 0,
                "std_aa_count": 0, "std_ca_count": 0, "std_no_ca_count": 0,
                "nonstd_count": 0, "het_count": 0,
                "insert_code_count": 0, "altloc_ca_count": 0,
                "per_chain_std_ca": "",
                "error_category": "UNKNOWN",
                "note": f"uncaught: {type(e).__name__}: {e}",
            }
        rows.append(r)
        stats[r["error_category"]] += 1

    # 导出 CSV（可选）
    if out_csv:
        fieldnames = [
            "pdb_id", "path", "n_models", "n_chains",
            "std_aa_count", "std_ca_count", "std_no_ca_count",
            "nonstd_count", "het_count",
            "insert_code_count", "altloc_ca_count",
            "per_chain_std_ca",
            "error_category", "note",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    total = len(files)
    ratios = {k: (stats[k] / total if total else 0.0) for k in stats.keys()}
    summary = {"counts": dict(stats), "total": total, "ratios": ratios}
    return rows, summary


# 可选：直接在交互式里运行这段（不会自动执行）
if __name__ == "__main__":
    # 示例（请自行修改路径，或作为模块导入调用）
    folder = "/root/autodl-tmp/M-CSA/pdbs"
    csv_out = "/root/autodl-tmp/M-CSA/report.csv"  # 或者 "/path/to/report.csv"
    rows, summary = check_pdbs(folder, out_csv=csv_out)
    print(summary)