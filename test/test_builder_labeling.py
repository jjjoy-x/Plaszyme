#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_builder_labeling.py

目标：专测 builder 能否对指定目录的 PDB 成功构图，并且 kept 后的 (chain,resid) 能与标注 CSV 对齐。
策略：
  1) 读取标注 CSV（新版字段，含链 ID）→ {(pdb_id, chain, resid_int) -> role_type}
  2) 遍历 PDB：
     - 若该 pdb 在标注中出现过链 → 选“多数链”
     - 否则 / 或该链构图失败 → Bio.PDB 统计每条链的(标准AA+可用CA)数，选最多的链重试
  3) 从 misc（优先）或 data.res_ids（退路）得到 kept 后的 (chain,resid_int)，计算命中标注数量与覆盖率
  4) 输出 per-PDB 结果 + 总体统计；可选写 CSV

依赖：
  - biopython
  - torch, torch_geometric（因需调用 builder）
  - 你的项目内的 GNNProteinGraphBuilder 与 BuilderConfig
"""

from __future__ import annotations
import os
import csv
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional

import torch
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig

# ===================== 配置 =====================
CONFIG = {
    "pdb_dir": "/root/autodl-tmp/M-CSA/pdbs",
    "csv_path": "/tmp/pycharm_project_317/dataset/M-CSA/merged_mcsa_ec.csv",  # ← 用你新版（含 CHAIN ID）的 CSV 路径
    "out_report_csv": "/tmp/pycharm_project_317/dataset/M-CSA/pdb_builder_test_report.csv",  # 设为 None 则不写
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 构图器参数（与训练一致）
    "builder": {
        "radius": 10.0,
        "embedder": [
            {"name": "onehot"},  # 轻量嵌入，避免 ESM 影响测试速度/显存
        ],
        "edge_mode": "rbf",
        "rbf_centers": None,
        "rbf_gamma": None,
        "add_self_loop": False,
    },

    # 最多打印多少条失败样本细节
    "max_fail_log": 20,
}
# =================================================


# ---------- 读取新版标注 ----------
def load_annotations(csv_path: str) -> Dict[Tuple[str, str, int], str]:
    """
    读取新版 CSV（示例头）：
    PDB ID,CHAIN ID,RESIDUE NUMBER,ROLE_TYPE,ROLE,...
    仅取 (pdb_id, chain, resid_int) -> role_type（pdb_id 小写）
    """
    mapping: Dict[Tuple[str, str, int], str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 容错列名
        col_pid = None
        for c in reader.fieldnames or []:
            if c.strip().lower() in {"pdb id", "pdb_id"}:
                col_pid = c
                break
        col_chain = None
        for c in reader.fieldnames or []:
            if c.strip().lower() in {"chain id", "chain"}:
                col_chain = c
                break
        col_res = None
        for c in reader.fieldnames or []:
            if c.strip().lower() in {"residue number", "resid_int", "resseq", "resid"}:
                col_res = c
                break
        col_role_type = None
        for c in reader.fieldnames or []:
            if c.strip().lower() in {"role_type", "role type"}:
                col_role_type = c
                break

        if not (col_pid and col_chain and col_res and col_role_type):
            raise ValueError(f"CSV 列缺失：{reader.fieldnames}")

        for row in reader:
            pid = str(row[col_pid]).strip().lower()
            ch = str(row[col_chain]).strip()
            try:
                resid = int(str(row[col_res]).strip())
            except Exception:
                # 跳过无法转 int 的残基号
                continue
            role_type = str(row[col_role_type]).strip().lower()
            mapping[(pid, ch, resid)] = role_type

    return mapping


# ---------- 统计某 PDB 各链的“标准AA+有CA”个数 ----------
def per_chain_std_ca_counts(pdb_path: str) -> Dict[str, int]:
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure("x", pdb_path).get_models())
    counts = {}
    for ch in model.get_chains():
        n = 0
        for res in ch.get_residues():
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            if "CA" in res:
                try:
                    _ = res["CA"].get_coord()
                    n += 1
                except Exception:
                    pass
        counts[str(ch.id)] = n
    return counts


# ---------- 解析 misc/data 得到 kept 的 (chain,resid_int) ----------
def kept_chain_resid_from_misc_or_data(misc: dict, data) -> Tuple[List[str], List[int]]:
    # 优先 misc（builder 按我们建议应导出这两个）
    if "chain_kept" in misc and "resid_int_kept" in misc:
        chains = [str(x) for x in misc["chain_kept"]]
        resids = [int(x) for x in misc["resid_int_kept"]]
        return chains, resids

    # 退化用 data.res_ids（形如 "A:123A" 或 "A:123"）
    def _parse_res_id(s: str) -> Tuple[str, int]:
        # "A:123A" → ("A", 123)
        if ":" in s:
            ch, r = s.split(":", 1)
            # 去掉插码尾巴的非数字部分
            num = ""
            for ch_ in r:
                if ch_.isdigit():
                    num += ch_
                else:
                    break
            return ch, int(num) if num else -1
        return "A", -1

    if hasattr(data, "res_ids") and isinstance(data.res_ids, list):
        parsed = [_parse_res_id(s) for s in data.res_ids]
        chains = [c for c, _ in parsed]
        resids = [r for _, r in parsed]
        return chains, resids

    # 实在没有：按长度构造
    N = data.num_nodes
    return ["A"] * N, list(range(1, N + 1))


# ---------- 封装一次构图（支持临时链选择） ----------
def build_one_with_chain(builder: GNNProteinGraphBuilder, pdb_path: str, chain_id: Optional[str], name: Optional[str]):
    old = builder.cfg.chain_id
    builder.cfg.chain_id = chain_id
    try:
        return builder.build_one(pdb_path, name=name)
    finally:
        builder.cfg.chain_id = old


# ---------- 主测试逻辑 ----------
def main():
    cfg = CONFIG

    # 1) 读标注
    ann = load_annotations(cfg["csv_path"])

    # 2) 初始化 builder（先不指定链）
    bcfg = BuilderConfig(
        pdb_dir=cfg["pdb_dir"],
        out_dir=cfg["pdb_dir"],  # 不落盘，只是占位
        chain_id=None,
        radius=cfg["builder"]["radius"],
        embedder=cfg["builder"]["embedder"],
    )
    builder = GNNProteinGraphBuilder(
        bcfg,
        edge_mode=cfg["builder"]["edge_mode"],
        rbf_centers=cfg["builder"]["rbf_centers"],
        rbf_gamma=cfg["builder"]["rbf_gamma"],
        add_self_loop=cfg["builder"]["add_self_loop"],
    )

    # 3) 遍历 PDB
    pdb_files = sorted([f for f in os.listdir(cfg["pdb_dir"]) if f.lower().endswith((".pdb", ".ent", ".cif", ".mmcif"))])
    results = []
    n_ok = n_fail_build = n_mismatch = 0
    fail_logs = []

    for i, fn in enumerate(pdb_files, 1):
        path = os.path.join(cfg["pdb_dir"], fn)
        pdb_id = os.path.splitext(fn)[0].lower()

        # 标注里该 pdb 出现过的链
        chains_in_ann = [ch for (pid, ch, _), _role in ann.items() if pid == pdb_id]
        picked_by_ann = Counter(chains_in_ann).most_common(1)[0][0] if chains_in_ann else None

        # 先按标注链尝试
        tried = []
        success = False
        err_msg = ""
        data = misc = None

        for attempt in [picked_by_ann, None]:  # None 表示走“自动回退：选 stdCA 最多的链”
            if attempt in tried:
                continue
            tried.append(attempt)

            # 决定链
            if attempt is not None:
                chain_to_use = attempt
            else:
                # 自动回退：找 stdCA 最多的链
                counts = per_chain_std_ca_counts(path)
                chain_to_use = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None

            try:
                data, misc = build_one_with_chain(builder, path, chain_to_use, name=pdb_id)
                success = True
                break
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                continue

        if not success:
            n_fail_build += 1
            if len(fail_logs) < cfg["max_fail_log"]:
                fail_logs.append((pdb_id, err_msg))
            results.append({
                "pdb_id": pdb_id,
                "status": "BUILD_FAIL",
                "picked_chain": picked_by_ann or "",
                "final_chain": "",
                "nodes": 0,
                "align_ok": 0,
                "n_hit": 0,
                "n_nodes": 0,
                "hit_ratio": 0.0,
                "err": err_msg,
            })
            continue

        # 读取 kept 的 (chain,resid)
        chains_kept, resids_kept = kept_chain_resid_from_misc_or_data(misc, data)
        align_ok = (len(chains_kept) == data.num_nodes == len(resids_kept))
        if not align_ok:
            n_mismatch += 1

        # 命中标注数
        hits = 0
        for ch, resi in zip(chains_kept, resids_kept):
            if (pdb_id, str(ch), int(resi)) in ann:
                hits += 1

        hit_ratio = (hits / max(1, data.num_nodes))
        n_ok += 1

        results.append({
            "pdb_id": pdb_id,
            "status": "OK" if align_ok else "ALIGN_MISMATCH",
            "picked_chain": picked_by_ann or "",
            "final_chain": misc.get("picked_chain_id", ""),
            "nodes": int(data.num_nodes),
            "align_ok": int(align_ok),
            "n_hit": int(hits),
            "n_nodes": int(data.num_nodes),
            "hit_ratio": round(float(hit_ratio), 4),
            "err": "",
        })

    # 4) 汇总
    total = len(pdb_files)
    print("\n===== TEST SUMMARY =====")
    print(f"Total PDBs      : {total}")
    print(f"Build OK        : {n_ok}")
    print(f"Build Failed    : {n_fail_build}")
    print(f"Align Mismatch  : {n_mismatch}")
    if fail_logs:
        print("\n-- Some build failures (up to max_fail_log) --")
        for pid, em in fail_logs:
            print(f"{pid}: {em}")

    # 5) 导出报告（可选）
    out_csv = CONFIG.get("out_report_csv")
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f"\nReport saved to: {out_csv}")

    # 6) 返回结果（若你想在 REPL 中调用）
    return results


if __name__ == "__main__":
    main()