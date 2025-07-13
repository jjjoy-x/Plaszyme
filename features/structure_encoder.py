import os
import torch
import numpy as np
from Bio.PDB import PDBParser
from Bio.Data import IUPACData


class StructureEncoder:
    """
    StructureEncoder: Convert structure files (.pdb or .npz) into residue-level adjacency matrices.
    结构编码器：将结构文件转换为残基级邻接矩阵（Contact Map），仅生成原始 0/1 邻接矩阵，不加自环，不做归一化，输出为 PyTorch Geometric 所需的 edge_index 稀疏图格式

    Args:
        threshold (float): 距离阈值（Å），小于该值视为存在边
        mode (str): 'CA' | 'CB' | 'ALL'，用于指定原子类型
    """

    def __init__(self, threshold=10.0, mode='CA'):
        self.threshold = threshold
        self.mode = mode.upper()

    def load_pdb_with_alignment(self, pdb_path: str, sequence: str) -> torch.LongTensor:
        """
        Load contact map from a .pdb file and align to full sequence.
        从 PDB 文件加载邻接矩阵，并自动对齐至给定序列长度，对缺失残基建立线性边

        Args:
            pdb_path (str): PDB 文件路径
            sequence (str): 蛋白质主序列（FASTA 格式）

        Returns:
            torch.LongTensor: 稀疏图结构 edge_index [2, num_edges]
        """
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"[ERROR] File not found: {pdb_path}")
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"[ERROR] Must provide sequence string for alignment")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]

        coords = []
        valid_idx = []
        seq_len = len(sequence)

        for residue in model.get_residues():
            if not residue.has_id(self.mode):
                continue

            # 只处理标准氨基酸
            try:
                resname = residue.get_resname().strip().capitalize()
                res1 = IUPACData.protein_letters_3to1[resname]
            except KeyError:
                raise ValueError(f"[ERROR] Unknown residue name in PDB: {residue.get_resname()}")

            pdb_index = residue.get_id()[1]  # 实际的序列编号
            if pdb_index < 1 or pdb_index > seq_len:
                raise ValueError(f"[ERROR] Structure residue indices exceed sequence length: idx={pdb_index}, len={seq_len}")

            if sequence[pdb_index - 1] != res1:
                raise ValueError(f"[ERROR] Residue mismatch at position {pdb_index}: PDB={res1}, Seq={sequence[pdb_index - 1]}")

            atom = residue[self.mode]
            coords.append(atom.get_coord())
            valid_idx.append(pdb_index - 1)

        if len(valid_idx) == 0:
            raise ValueError(f"[ERROR] No valid residues with atom {self.mode} found")

        if len(valid_idx) / seq_len < 0.3:
            raise ValueError(f"[ERROR] Too few residues aligned: only {len(valid_idx)} of {seq_len}")

        coords = np.array(coords)
        edges = set()

        for i, idx_i in enumerate(valid_idx):
            for j, idx_j in enumerate(valid_idx):
                if idx_i == idx_j:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= self.threshold:
                    edges.add((idx_i, idx_j))

        # 添加顺序连接（链式残基）
        for i in range(seq_len - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

        print(f"[INFO] Loaded aligned contact graph from PDB: {pdb_path} (edges={edge_index.size(1)})")
        return edge_index

    def load_npz(self, npz_path: str) -> torch.LongTensor:
        """
        Load contact map from a .npz file (e.g., AlphaFold output)
        从 .npz 文件加载邻接矩阵（例如 AlphaFold 输出）

        Args:
            npz_path (str): .npz 文件路径

        Returns:
            torch.LongTensor: 稀疏图结构 edge_index [2, num_edges]
        """
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"[ERROR] File not found: {npz_path}")

        data = np.load(npz_path)
        if 'dist' in data:
            dist_mat = data['dist']
        elif 'distance' in data:
            dist_mat = data['distance']
        else:
            raise ValueError(f"[ERROR] No distance matrix in: {npz_path}")

        np.fill_diagonal(dist_mat, np.inf)
        edge_indices = np.where(dist_mat <= self.threshold)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)

        print(f"[INFO] Loaded contact graph from NPZ: {npz_path} (edges={edge_index.size(1)})")
        return edge_index

    def __call__(self, filepath: str, sequence: str = None) -> torch.LongTensor:
        """
        Unified entry point: auto-selects loader by file extension.
        统一接口：根据扩展名自动选择加载方式

        Args:
            filepath (str): 路径 (.pdb or .npz)
            sequence (str): 对于 .pdb 必须提供序列以对齐

        Returns:
            torch.LongTensor: 稀疏图结构 edge_index [2, num_edges]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] File not found: {filepath}")

        if filepath.endswith('.pdb'):
            return self.load_pdb_with_alignment(filepath, sequence)
        elif filepath.endswith('.npz'):
            return self.load_npz(filepath)
        else:
            raise ValueError(f"[ERROR] Unsupported file type: {filepath}")