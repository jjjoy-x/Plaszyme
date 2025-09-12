# test/test_gvp_backbone_official.py
import os, torch
from torch_geometric.data import Batch
from src.builders.base_builder import BuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder
from src.models.gvp.backbone import GVPBackbone

PDB = "/root/autodl-tmp/M-CSA/pdbs/13pk.pdb"

def build_batch(pdb, edge_scalar="rbf", node_vec_mode="bb2", copies=2):
    cfg = BuilderConfig(
        pdb_dir=os.path.dirname(pdb),
        out_dir=os.path.join(os.path.dirname(pdb), "tmp_gvp_test"),
        embedder={"name": "esm"},
        radius=10.0,
    )
    builder = GVPProteinGraphBuilder(cfg, node_vec_mode=node_vec_mode,
                                     edge_scalar=edge_scalar, edge_vec_dim=1)
    name = os.path.splitext(os.path.basename(pdb))[0]
    data, _ = builder.build_one(pdb, name=name)
    return Batch.from_data_list([data]*copies)

if __name__ == "__main__":
    batch = build_batch(PDB, edge_scalar="rbf", node_vec_mode="bb2", copies=2)

    # 图级
    model = GVPBackbone(hidden_dims=(128,16), n_layers=3, out_dim=8, dropout=0.1, residue_logits=False)
    with torch.no_grad():
        out = model(batch)
    print("graph out:", out.shape)  # 应为 (2, 8)

    # 残基级
    model2 = GVPBackbone(hidden_dims=(128,16), n_layers=3, out_dim=8, dropout=0.1, residue_logits=True)
    with torch.no_grad():
        out2 = model2(batch)
    print("residue out:", out2.shape)  # 应为 (batch.num_nodes, 8)