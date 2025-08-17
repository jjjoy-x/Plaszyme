# build_graph_from_mol.py
# 依据 YAML 配置文件，从 .mol 文件构建 PyTorch Geometric 图结构，支持自定义线性特征设置

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN
import yaml
import os
from typing import List

# === 完全接口 ===
ATOM_FEATURES = {
    "atomic_number": lambda atom: [atom.GetAtomicNum()],
    "is_aromatic": lambda atom: [int(atom.GetIsAromatic())],
    "is_in_ring": lambda atom: [int(atom.IsInRing())],
    "hybridization": lambda atom: [int(atom.GetHybridization())],
    "formal_charge": lambda atom: [atom.GetFormalCharge()],
    "num_explicit_hs": lambda atom: [atom.GetNumExplicitHs()],
    # 不同项可扩充
}

BOND_FEATURES = {
    "bond_type": lambda bond: [float(bond.GetBondTypeAsDouble())],
    "is_conjugated": lambda bond: [int(bond.GetIsConjugated())],
    "is_in_ring": lambda bond: [int(bond.IsInRing())],
    "stereo": lambda bond: [int(bond.GetStereo())],
}


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_main_chain_mask_with_ring_completion(mol, config) -> List[int]:
    n_atoms = mol.GetNumAtoms()
    mask = [0] * n_atoms

    # === Step 1: linear 主链识别 ===
    max_len = 100
    longest = []
    for L in range(max_len, 1, -1):
        paths = FindAllPathsOfLengthN(mol, L, useBonds=False)
        if paths:
            longest = max(paths, key=lambda p: len(p))
            break
    main_chain_atoms = set(longest)

    # === Step 2: 找出连接主链的环，并补全 ===
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    for ring in atom_rings:
        if any(a in main_chain_atoms for a in ring):
            main_chain_atoms.update(ring)  # 把整个环加进去

    # === Step 3: 构造掩码 ===
    for i in main_chain_atoms:
        mask[i] = 1

    return mask


def extract_node_features(mol, config):
    features = []
    for atom in mol.GetAtoms():
        feats = []
        for key in config['node_features']:
            if key == 'is_main_chain':
                continue  # 等待后面单独处理
            if key not in ATOM_FEATURES:
                raise ValueError(f"Unsupported atom feature: {key}")
            feats += ATOM_FEATURES[key](atom)
        features.append(feats)

    # 添加 is_main_chain 特征
    if "is_main_chain" in config['node_features']:
        mask = get_main_chain_mask(mol, config)
        features = [f + [m] for f, m in zip(features, mask)]

    return torch.tensor(features, dtype=torch.float)


def extract_edge_info(mol, config):
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = []
        for key in config['edge_features']:
            if key not in BOND_FEATURES:
                raise ValueError(f"Unsupported bond feature: {key}")
            feat += BOND_FEATURES[key](bond)

        # 双向边
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_attr, dtype=torch.float)
    )


def extract_pos(mol):
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return torch.tensor(coords, dtype=torch.float)


def build_graph_from_mol(mol_path, config_path):
    config = load_config(config_path)
    mol = Chem.MolFromMolFile(mol_path, removeHs=not config.get('add_hs', True))
    if mol is None:
        raise ValueError(f"Failed to load molecule from {mol_path}")

    if config.get("generate_conformers", False):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=config.get("num_conformers", 5))
        AllChem.UFFOptimizeMoleculeConfs(mol)

    x = extract_node_features(mol, config)
    edge_index, edge_attr = extract_edge_info(mol, config)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if config.get("pos", False):
        pos = extract_pos(mol)
        data.pos = pos

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mol", type=str, required=True, help="Path to .mol file")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--out", type=str, required=True, help="Output .pt file")
    args = parser.parse_args()

    data = build_graph_from_mol(args.mol, args.config)
    torch.save(data, args.out)
    print(f"[✔] Saved graph to {args.out}")