# test/test_loader_pair.py
import os
from torch.utils.data import DataLoader

from src.data.loader import MatrixSpec, PairedPlaszymeDataset, collate_pairs
from src.builders.gnn_builder import GNNProteinGraphBuilder
from src.builders.base_builder import BuilderConfig
from src.plastic.descriptors_rdkit import PlasticFeaturizer


def main():
    # ======== 路径配置（务必改成你的实际路径）========
    pdb_root = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb"     # 例如：/Users/xxx/Plaszyme/dataset/predicted_xid/pdb
    sdf_root = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"     # 例如：/Users/xxx/Plaszyme/plastic/mols_for_unimol_10_sdf_new
    csv_path = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/plastics_onehot_table.csv"  # 独热矩阵CSV（行=pdb名，列=sdf名），不要填 .py

    assert os.path.isdir(pdb_root), f"pdb_root 不存在: {pdb_root}"
    assert os.path.isdir(sdf_root), f"sdf_root 不存在: {sdf_root}"
    assert os.path.isfile(csv_path), f"csv_path 不存在: {csv_path}"

    # ======== 初始化构建器 ========
    # BuilderConfig 的字段以你的 BaseBuilder 实现为准；下面是常用/安全的最小配置
    cfg = BuilderConfig(
        pdb_dir=pdb_root,
        out_dir=os.path.join(os.path.dirname(csv_path), "_tmp_builder_out"),  # 仅占位；不写盘也没关系
        radius=8.0,                         # 半径图阈值
        embedder=[{"name": "onehot"}],      # 最简单的序列嵌入；和你的实现保持一致
    )

    enzyme_builder = GNNProteinGraphBuilder(
        cfg=cfg,
        edge_mode="none",   # 先用最简单的纯二值邻接；如需距离特征可改 "dist"/"inv_dist"/"rbf"
        add_self_loop=False
    )
    plastic_featurizer = PlasticFeaturizer(config_path=None)

    # ======== 数据集与 DataLoader ========
    spec = MatrixSpec(csv_path=csv_path, pdb_root=pdb_root, sdf_root=sdf_root)
    dataset = PairedPlaszymeDataset(
        matrix=spec,
        mode="pair",
        split="train",                    # 也可改 "val"/"test"
        enzyme_builder=enzyme_builder,
        plastic_featurizer=plastic_featurizer,
        # 如果某些 PDB 需要固定链，可传 chain_map={"xxx.pdb": "A"}
    )

    print(f"Dataset size (pair/train): {len(dataset)}")

    # —— 先看一个原始样本，核对名字是否对得上 ——
    sample = dataset[0]
    print("\n[Sample 0]")
    print(" mode:", sample["mode"])
    print(" pdb_name:", sample.get("pdb_name"))
    # PairedPlaszymeDataset 的 pair 模式样本里保存了 pos/neg 的特征张量，
    # 文件名不在字典里；为便于核对，这里从 dataset.samples 里取一下
    if hasattr(dataset, "samples"):
        pdb_name, sdf_pos, sdf_neg = dataset.samples[0]
        print(" sdf_pos:", sdf_pos)
        print(" sdf_neg:", sdf_neg)
    print(" enzyme_graph (PyG Data):", sample["enzyme_graph"])
    print(" plastic_pos shape:", sample["plastic_pos"].shape)
    print(" plastic_neg shape:", sample["plastic_neg"].shape)

    # —— 再走一遍 DataLoader + collate，看看 batch 格式 ——
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_pairs)
    batch = next(iter(loader))
    print("\n[Batch]")
    print(" keys:", batch.keys())
    print(" enzyme_graph (PyG Batch):", batch["enzyme_graph"])
    print(" plastic_pos shape:", batch["plastic_pos"].shape)
    print(" plastic_neg shape:", batch["plastic_neg"].shape)


if __name__ == "__main__":
    main()