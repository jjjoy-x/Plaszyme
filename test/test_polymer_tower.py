#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 PolymerTower:
- 输入: 一个 SDF 文件 (塑料分子)
- 流程:
    1. 使用 PolymerFeaturizer 提取描述符向量
    2. 输入 PolymerTower
    3. 打印输出嵌入 (归一化后的向量)

用法:
    python test_polymer_tower.py /path/to/polymer.sdf
"""

import sys
import torch
from pathlib import Path

# ====== 项目内模块 ======
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.models.plastic_backbone import PolymerTower   # 你之前写的塔


def main(sdf_path: str):
    # 1) 提取描述符
    featurizer = PlasticFeaturizer()  # 如需 yaml 配置可传 config_path
    feat = featurizer.featurize_file(sdf_path)  # 返回 torch.Tensor 或 None
    if feat is None:
        raise RuntimeError(f"无法从文件提取特征: {sdf_path}")

    x = feat.unsqueeze(0)  # [1, D]
    in_dim = x.shape[1]

    # 2) 构造塑料塔
    tower = PolymerTower(in_dim=in_dim, hidden_dims=(512, 256), out_dim=128)

    # 3) 前向
    with torch.no_grad():
        z = tower(x)  # [1, 128]

    print(f"Input file: {sdf_path}")
    print(f"Feature dim: {in_dim}")
    print(f"Embedding shape: {tuple(z.shape)}")
    print(f"Embedding (first 10 dims): {z[0, :10].cpu().numpy()}")
    print(f"L2 norm: {z.norm(dim=-1).item():.4f}")

if __name__ == "__main__":
    path='/tmp/pycharm_project_317/plastic/mols_for_unimol_10_sdf/LDPE.sdf'
    main(path)
