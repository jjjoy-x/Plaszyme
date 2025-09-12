#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_builder.py

Build graph dataset from FASTA sequences and PDB structures.
从 FASTA 与 PDB 构建图数据集（每个样本一个 .pt）。

Inputs:
    - FASTA: record.id must match the sample name
    - PDB folder: files named as {record.id}.pdb

Outputs:
    - {output_folder}/{name}.pt      (PyG Data with x, edge_index, name)
    - {output_folder}/manifest.csv   (status summary)

Dependencies:
    biopython, torch, torch-geometric, pandas, tqdm
"""

import os
from typing import Dict

import pandas as pd
import torch
from Bio import SeqIO
from torch_geometric.data import Data
from tqdm import tqdm

from sequence_embedder import ESMEmbedder
from structure_encoder import StructureEncoder


def build_dataset_from_fasta_pdb(
    fasta_file: str,
    pdb_folder: str,
    output_folder: str,
    threshold: float = 10.0,
    mode: str = 'CA'
):
    """
    Build graph dataset from FASTA and PDB.

    Args:
        fasta_file (str): Path to input FASTA file
                          输入的FASTA文件路径
        pdb_folder (str): Folder containing PDB files (named by sequence ID)
                          存储 PDB 结构的文件夹（文件名需与FASTA名称对应）
        output_folder (str): Output folder to store .pt files and manifest.csv
                             输出文件夹，保存 .pt 文件与 manifest.csv
        threshold (float): Distance threshold for contact map
                           接触图的距离阈值
        mode (str): Atom type used for distance calculation
                    用于计算距离的原子类型（'CA', 'CB' 等）
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load FASTA as dictionary: name → sequence
    # 将 FASTA 读为字典：name → 序列
    seq_dict: Dict[str, str] = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')}
    print(f"[INFO] Loaded {len(seq_dict)} sequences from FASTA.")

    # Initialize encoders
    # 初始化编码器
    seq_encoder = ESMEmbedder()
    struc_encoder = StructureEncoder(threshold=threshold, mode=mode)

    # Iterate and process each sample
    # 遍历并处理每个样本
    records = []
    for name, seq in tqdm(seq_dict.items(), total=len(seq_dict), desc="Building graphs"):
        pdb_path = os.path.join(pdb_folder, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"[WARNING] PDB file not found for: {name}")
            records.append({"name": name, "status": "missing_pdb", "pdb_path": pdb_path, "seq_len": len(seq)})
            continue

        try:
            x = seq_encoder(seq)                     # [L, D] node features
            edge_index = struc_encoder(pdb_path, seq)  # [2, E] adjacency edges

            # Construct PyG graph object
            # 构建 PyG 图对象
            data = Data(x=x, edge_index=edge_index)
            data.name = name  # keep sample name for downstream alignment / 下游对齐使用

            out_path = os.path.join(output_folder, f"{name}.pt")
            torch.save(data, out_path)

            records.append({
                "name": name,
                "status": "ok",
                "seq_len": len(seq),
                "pdb_path": pdb_path,
                "pt_path": out_path
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")
            records.append({
                "name": name,
                "status": f"error:{e}",
                "seq_len": len(seq),
                "pdb_path": pdb_path
            })

    # Write manifest for bookkeeping
    # 写出 manifest 记录处理状态
    manifest_path = os.path.join(output_folder, "manifest.csv")
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"[INFO] Saved graphs to {output_folder}")
    print(f"[INFO] Manifest written to {manifest_path}")


if __name__ == "__main__":
    build_dataset_from_fasta_pdb(
        fasta_file="/tmp/pycharm_project_27/dataset/test.fasta",
        pdb_folder="/tmp/pycharm_project_27/dataset/predicted_xid/pdb",
        output_folder="/tmp/pycharm_project_27/dataset/test_graph",
        threshold=10.0,
        mode='CA'
    )