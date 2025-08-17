import os
from typing import List
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN

# === 手动指定路径 ===
INPUT_DIR  = "/plastic/mols_for_unimol_3"
OUTPUT_DIR = "outputs/mol_main_chains"

# === 手动指定 config ===
config = {
    "auto_main_chain": "linear_with_ring",  # 可选: linear, linear_with_ring, main_chain_smarts
}

def get_main_chain_mask(mol, config) -> List[int]:
    n_atoms = mol.GetNumAtoms()
    mask = [0] * n_atoms

    if "main_chain_smarts" in config:
        pattern = Chem.MolFromSmarts(config["main_chain_smarts"])
        matches = mol.GetSubstructMatches(pattern)
        main_chain_atoms = set(idx for match in matches for idx in match)
        for i in main_chain_atoms:
            mask[i] = 1
        return mask

    strategy = config.get("auto_main_chain", "linear")

    if strategy in ["linear", "linear_with_ring"]:
        max_len = 100
        longest = []
        for L in range(max_len, 1, -1):
            paths = FindAllPathsOfLengthN(mol, L, useBonds=False)
            if paths:
                longest = max(paths, key=lambda p: len(p))
                break
        main_chain_atoms = set(longest)

        if strategy == "linear_with_ring":
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            for ring in atom_rings:
                if any(a in main_chain_atoms for a in ring):
                    main_chain_atoms.update(ring)

        for i in main_chain_atoms:
            mask[i] = 1
        return mask

    elif strategy == "smart":
        raise NotImplementedError("'smart' strategy not implemented yet.")

    else:
        raise ValueError(f"Unknown auto_main_chain strategy: {strategy}")

def visualize_main_chain(mol_path, config, out_path):
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    if mol is None:
        print(f"[⚠️] 读取失败: {mol_path}")
        return

    mask = get_main_chain_mask(mol, config)
    highlight_atoms = [i for i, v in enumerate(mask) if v == 1]

    mol = Chem.RemoveHs(mol)
    Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(1500, 1200)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors={i: (1.0, 0.0, 0.0) for i in highlight_atoms}
    )
    drawer.FinishDrawing()

    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"[✔] 可视化保存: {out_path}")

def batch_process(input_dir, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mol"):
            mol_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_main_chain.png")
            visualize_main_chain(mol_path, config, out_path)

if __name__ == "__main__":
    batch_process(INPUT_DIR, OUTPUT_DIR, config)