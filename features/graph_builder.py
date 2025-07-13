import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from typing import Optional
from torch_geometric.data import Data

from features.sequence_embedder import ESMEmbedder
from features.structure_encoder import StructureEncoder

def build_dataset_from_csv(
    fasta_file: str,
    pdb_folder: str,
    csv_file: str,
    name_column: str,
    label_column: str,
    output_folder: str,
    threshold: float = 10.0,
    mode: str = 'CA'
):
    """
    Build graph dataset from fasta, pdb and label csv.
    从 fasta、pdb 和标签 CSV 构建图数据集

    Args:
        fasta_file (str): Path to input FASTA file
                          输入的FASTA文件路径
        pdb_folder (str): Folder containing PDB files (named by sequence ID)
                          存储 PDB 结构的文件夹（文件名需与FASTA名称对应）
        csv_file (str): CSV file containing sample names and labels
                        包含样本名和标签的 CSV 文件路径
        name_column (str): Column name in CSV for sample names
                           CSV中标识样本名的列名
        label_column (str): Column name in CSV for labels (str)
                            CSV中标识标签的列名（为字符串）
        output_folder (str): Output folder to store .pt and label2id.json
                             输出文件夹，保存.pt图和标签映射文件
        threshold (float): Distance threshold for contact map
                           接触图的距离阈值
        mode (str): Atom type used for distance calculation
                    用于计算距离的原子类型（'CA', 'CB' 等）
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load FASTA as dictionary: name → sequence
    from Bio import SeqIO
    seq_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')}

    # Load labels
    df = pd.read_csv(csv_file)
    name_list = df[name_column].astype(str).tolist()
    label_list = df[label_column].astype(str).tolist()

    # Build label2id
    unique_labels = sorted(set(label_list))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    with open(os.path.join(output_folder, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"[INFO] {len(label2id)} labels detected and saved to label2id.json")

    # Initialize encoders
    seq_encoder = ESMEmbedder()
    struc_encoder = StructureEncoder(threshold=threshold, mode=mode)

    # Iterate and process each sample
    for name, label in tqdm(zip(name_list, label_list), total=len(name_list)):
        if name not in seq_dict:
            print(f"[WARNING] Sequence not found for: {name}")
            continue

        seq = seq_dict[name]
        pdb_path = os.path.join(pdb_folder, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"[WARNING] PDB file not found for: {name}")
            continue

        try:
            x = seq_encoder(seq)            # [L, D] node features
            edge_index = struc_encoder(pdb_path, seq)  # [2, E] adjacency edges
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")
            continue

        y = torch.tensor([label2id[label]], dtype=torch.long)

        # Construct PyG graph object
        data = Data(x=x, edge_index=edge_index, y=y)
        torch.save(data, os.path.join(output_folder, f"{name}.pt"))

    print(f"[INFO] Dataset saved to {output_folder}")

if __name__ == "__main__":
    build_dataset_from_csv(
        fasta_file="/Users/shulei/PycharmProjects/Dataset/fri_dataset/fasta/combined.fasta",
        pdb_folder="/Users/shulei/PycharmProjects/Dataset/fri_dataset/pdb",
        csv_file="/Users/shulei/PycharmProjects/Dataset/dataset/PlaszymeDB_v0.2.3.csv",
        name_column="PLZ_ID",
        label_column="plastic",
        output_folder="/Users/shulei/PycharmProjects/Plaszyme/graph_dataset"
    )
