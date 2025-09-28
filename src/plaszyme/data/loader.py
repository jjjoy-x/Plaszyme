# src/data/loader.py
"""Data loader module for Plaszyme dataset handling.

This module provides unified dataset classes for enzyme-plastic interaction prediction,
supporting multiple training modes (pointwise, pairwise, listwise) with flexible
data loading and preprocessing capabilities.
"""

from __future__ import annotations

import warnings
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData, Batch

Split = Literal["train", "val", "test"]
Mode = Literal["point", "pair", "list"]


# ========= 读取独热矩阵 =========
@dataclass
class MatrixSpec:
    """Specification for one-hot matrix location and corresponding root directories.

    Attributes:
        csv_path: Path to the CSV file containing the one-hot matrix.
        pdb_root: Root directory containing PDB files.
        sdf_root: Root directory containing SDF files.

    Note:
        Can be extended with chain column if certain PDBs need fixed chains,
        provide a {pdb:chain} mapping outside the one-hot matrix.
    """
    csv_path: str
    pdb_root: str
    sdf_root: str


def _read_onehot_matrix(spec: MatrixSpec) -> Tuple[List[str], List[str], Dict[Tuple[str, str], int]]:
    """Read one-hot matrix from CSV file.

    The matrix format:
    - Rows represent PDB filenames (with/without extension accepted, .pdb will be auto-appended)
    - Columns represent SDF filenames (with/without extension accepted, .sdf will be auto-appended)
    - Values are binary {0,1}

    Args:
        spec: Matrix specification containing file paths and directories.

    Returns:
        A tuple containing:
            - rows: List of PDB basenames
            - cols: List of SDF basenames
            - labels: Dictionary mapping (pdb_basename, sdf_basename) to 0/1

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If CSV has insufficient columns.
    """
    if not os.path.isfile(spec.csv_path):
        raise FileNotFoundError(spec.csv_path)

    with open(spec.csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("One-hot matrix CSV requires at least 1 plastic column")

        # First column is row name placeholder, subsequent columns are plastic names
        raw_cols = header[1:]
        cols = [c if os.path.splitext(c)[1].lower() == ".sdf" else f"{c}.sdf" for c in raw_cols]

        rows: List[str] = []
        labels: Dict[Tuple[str, str], int] = {}

        for line in reader:
            if not line:
                continue
            pdb_name = line[0]
            if os.path.splitext(pdb_name)[1].lower() != ".pdb":
                pdb_name = f"{pdb_name}.pdb"
            rows.append(pdb_name)

            # Each row contains subsequent 0/1 values
            for j, v in enumerate(line[1:]):
                try:
                    val = int(float(v))  # Compatible with "0","1","0.0","1.0"
                except Exception:
                    val = 0
                labels[(pdb_name, cols[j])] = 1 if val >= 1 else 0

    return rows, cols, labels


# ========= 核心数据集 =========
class PairedPlaszymeDataset(Dataset):
    """Unified dataset supporting point/pair/list output modes.

    This dataset only stores filenames and labels. The actual "PDB->graph / SDF->features"
    conversion is performed in __getitem__ by calling external builder/featurizer.
    Alternatively, this can be done in DataLoader's collate_fn for more efficient
    batch-level reuse/caching.

    Attributes:
        mode: Training mode - "point", "pair", or "list"
        matrix_spec: Matrix specification for data location
        neg_ratio: Ratio of negative samples per positive sample (for point mode)
        max_list_len: Maximum number of plastics per enzyme in list mode
        chain_map: Optional mapping from PDB basename to chain ID
        return_names: Whether to include names/IDs in returned samples
        no_truncate_list: Whether to avoid truncating negative samples in list mode
    """

    def __init__(
            self,
            matrix: MatrixSpec,
            *,
            mode: Mode = "list",
            split: Optional[Split] | Literal["full"] | None = "train",
            split_ratio: Optional[Tuple[float, float, float]] = (0.8, 0.2, 0),
            seed: int = 42,
            # Negative sampling strategy (common or partial use across point/pair/list)
            neg_ratio: float = 1.0,
            # point: how many negative samples per positive sample (random from current row's 0 columns)
            max_list_len: int = 16,  # list: max total plastic count per enzyme (including positive + negative)
            # Builders (pass instances!)
            enzyme_builder: Any = None,  # Must implement .build_one(pdb_path, name=...) -> (PyGData, misc)
            plastic_featurizer: Any = None,  # Must implement .featurize_file(sdf_path) -> Tensor[D] or None
            # Chain ID mapping (optional)
            chain_map: Optional[Dict[str, str]] = None,  # {pdb_basename: chain_id}
            # New: retain names/IDs for prediction/analysis
            return_names: bool = True,
            # New: allow not truncating negative samples in list mode (richer for prediction)
            no_truncate_list: bool = False,
    ):
        """Initialize the dataset.

        Args:
            matrix: Matrix specification containing data paths.
            mode: Output mode - "point", "pair", or "list".
            split: Data split to use - "train", "val", "test", "full", or None.
            split_ratio: Ratio for train/val/test split.
            seed: Random seed for reproducibility.
            neg_ratio: Negative sampling ratio for point mode.
            max_list_len: Maximum list length for list mode.
            enzyme_builder: Builder instance for protein graphs.
            plastic_featurizer: Featurizer instance for plastic features.
            chain_map: Optional chain ID mapping.
            return_names: Whether to return sample names/IDs.
            no_truncate_list: Whether to avoid truncating in list mode.

        Raises:
            ValueError: If builder instances are not provided or invalid split specified.
        """
        super().__init__()
        assert mode in ("point", "pair", "list")
        self.mode = mode
        self.matrix_spec = matrix

        # Require instance to avoid circular dependencies
        if enzyme_builder is None or plastic_featurizer is None:
            raise ValueError("Please provide enzyme_builder instance and plastic_featurizer instance.")
        self.enzyme_builder = enzyme_builder
        self.plastic_featurizer = plastic_featurizer

        self.neg_ratio = float(neg_ratio)
        self.max_list_len = int(max_list_len)
        self.chain_map = chain_map or {}
        self.return_names = bool(return_names)
        self.no_truncate_list = bool(no_truncate_list)

        # Read matrix
        rows, cols, labels = _read_onehot_matrix(matrix)
        self.rows = rows  # PDB names
        self.cols = cols  # SDF names
        self.labels = labels  # (pdb,sdf) -> 0/1

        # Split by rows (based on PDB)
        rng = random.Random(seed)
        row_ids = list(range(len(self.rows)))
        rng.shuffle(row_ids)

        # No split: split is None/"full" or split_ratio is None
        if (split is None) or (split == "full") or (split_ratio is None):
            keep = set(row_ids)
        else:
            # Normal three-way split
            assert isinstance(split_ratio, tuple) and len(split_ratio) == 3, "split_ratio must be (train,val,test)"
            n = len(row_ids)
            n_tr = int(n * split_ratio[0])
            n_va = int(n * split_ratio[1])
            id_tr = set(row_ids[:n_tr])
            id_va = set(row_ids[n_tr:n_tr + n_va])
            id_te = set(row_ids[n_tr + n_va:])

            if split == "train":
                keep = id_tr
            elif split == "val":
                keep = id_va
            elif split == "test":
                keep = id_te
            else:
                raise ValueError(f"Unknown split={split}")

        # Generate indices based on mode
        if mode == "point":
            # Pre-generate (pdb,sdf,label) triplets for pointwise with negative sampling
            self.samples: List[Tuple[str, str, int]] = []
            for i in range(len(self.rows)):
                if i not in keep:
                    continue
                pdb = self.rows[i]
                # Positive sample columns
                pos_cols = [s for s in self.cols if self.labels[(pdb, s)] == 1]
                neg_cols = [s for s in self.cols if self.labels[(pdb, s)] == 0]
                # Add all positive samples
                for s in pos_cols:
                    self.samples.append((pdb, s, 1))
                # Negative sampling
                k_neg = int(len(pos_cols) * self.neg_ratio) if self.neg_ratio > 0 else 0
                if k_neg > 0 and len(neg_cols) > 0:
                    rng.shuffle(neg_cols)
                    for s in neg_cols[:k_neg]:
                        self.samples.append((pdb, s, 0))

        elif mode == "pair":
            # Generate (pdb, pos_sdf, neg_sdf) triplets for pairwise (each positive paired with random negative)
            self.samples: List[Tuple[str, str, str]] = []
            for i in range(len(self.rows)):
                if i not in keep:
                    continue
                pdb = self.rows[i]
                pos_cols = [s for s in self.cols if self.labels[(pdb, s)] == 1]
                neg_cols = [s for s in self.cols if self.labels[(pdb, s)] == 0]
                if len(pos_cols) == 0 or len(neg_cols) == 0:
                    continue
                for s_pos in pos_cols:
                    s_neg = rng.choice(neg_cols)
                    self.samples.append((pdb, s_pos, s_neg))

        else:
            # Listwise: one sample per row, returns one enzyme + multiple plastics (pos/random neg) with 0/1 relevance
            self.row_indices = [i for i in range(len(self.rows)) if i in keep]

        # Simple memory cache (reuse across epochs)
        self._cache_graph: Dict[str, PyGData] = {}
        self._cache_plastic: Dict[str, torch.Tensor] = {}
        self._missing_sdf_warned: set[str] = set()
        self._failed_feat_warned: set[str] = set()
        self._missing_sdf_count: int = 0
        self._failed_feat_count: int = 0

    def __len__(self) -> int:
        """Return dataset length based on mode."""
        if self.mode in ("point", "pair"):
            return len(self.samples)
        return len(self.row_indices)

    # --- 内部工具 ---
    def _pdb_path(self, pdb_name: str) -> str:
        """Get full path for PDB file.

        Args:
            pdb_name: PDB filename.

        Returns:
            Full path to PDB file.
        """
        return os.path.join(self.matrix_spec.pdb_root, pdb_name)

    def _sdf_path(self, sdf_name: str) -> str:
        """Get full path for SDF file.

        Args:
            sdf_name: SDF filename.

        Returns:
            Full path to SDF file.
        """
        return os.path.join(self.matrix_spec.sdf_root, sdf_name)

    def _get_graph(self, pdb_name: str) -> PyGData:
        """Get protein graph with caching.

        Args:
            pdb_name: PDB filename.

        Returns:
            PyTorch Geometric Data object representing the protein graph.

        Raises:
            FileNotFoundError: If PDB file doesn't exist.
            TypeError: If builder doesn't return PyG Data object.
        """
        if pdb_name in self._cache_graph:
            return self._cache_graph[pdb_name]
        pdb_path = self._pdb_path(pdb_name)
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(pdb_path)
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        g, _ = self.enzyme_builder.build_one(pdb_path, name=name)
        if not isinstance(g, PyGData):
            raise TypeError("enzyme_builder.build_one must return (PyG Data, misc)")
        self._cache_graph[pdb_name] = g
        return g

    def _get_plastic(self, sdf_name: str) -> Optional[torch.Tensor]:
        """Get plastic features with caching and error handling.

        Args:
            sdf_name: SDF filename.

        Returns:
            Tensor of shape [1, D] containing plastic features, or None if failed.
        """
        if sdf_name in self._cache_plastic:
            return self._cache_plastic[sdf_name]

        sdf_path = self._sdf_path(sdf_name)
        if not os.path.isfile(sdf_path):
            # Warn only once per missing file and count
            if sdf_path not in self._missing_sdf_warned:
                warnings.warn(f"[PairedPlaszymeDataset] Missing SDF file: {sdf_path}, skipping sample")
                self._missing_sdf_warned.add(sdf_path)
            self._missing_sdf_count += 1
            return None

        f = self.plastic_featurizer.featurize_file(sdf_path)
        if f is None:
            # Warn only once per failed file and count
            if sdf_path not in self._failed_feat_warned:
                warnings.warn(f"[PairedPlaszymeDataset] Feature extraction failed: {sdf_path}, skipping sample")
                self._failed_feat_warned.add(sdf_path)
            self._failed_feat_count += 1
            return None

        f = f.view(1, -1)  # Ensure [1, D] shape
        self._cache_plastic[sdf_name] = f
        return f

    # --- 三种模式的 __getitem__ ---
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index with mode-specific behavior.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing sample data with mode-specific structure.
        """
        if self.mode == "point":
            pdb_name, sdf_name, y = self.samples[idx]
            g = self._get_graph(pdb_name)
            f = self._get_plastic(sdf_name)
            sample = {
                "mode": "point",
                "enzyme_graph": g,
                "plastic_feat": f,
                "label": torch.tensor([y], dtype=torch.float32),
            }
            # Retain names
            if self.return_names:
                sample["enzyme_id"] = os.path.splitext(os.path.basename(pdb_name))[0]
                sample["pdb_name"] = pdb_name
                sample["plastic_name"] = sdf_name
            return sample

        if self.mode == "pair":
            pdb_name, sdf_pos, sdf_neg = self.samples[idx]
            g = self._get_graph(pdb_name)
            f_pos = self._get_plastic(sdf_pos)
            f_neg = self._get_plastic(sdf_neg)
            sample = {
                "mode": "pair",
                "enzyme_graph": g,
                "plastic_pos": f_pos,
                "plastic_neg": f_neg,
            }
            if self.return_names:
                sample["enzyme_id"] = os.path.splitext(os.path.basename(pdb_name))[0]
                sample["pdb_name"] = pdb_name
                sample["plastic_pos_name"] = sdf_pos
                sample["plastic_neg_name"] = sdf_neg
            return sample

        # Listwise mode
        row_id = self.row_indices[idx]
        pdb_name = self.rows[row_id]
        g = self._get_graph(pdb_name)

        pos_cols = [s for s in self.cols if self.labels[(pdb_name, s)] == 1]
        neg_cols = [s for s in self.cols if self.labels[(pdb_name, s)] == 0]

        # Collect (name, feat_tensor, label_float) triplets, filter None later
        triples: List[Tuple[str, Optional[torch.Tensor], float]] = []

        # Add positive samples first (try to keep all)
        for s in pos_cols:
            f = self._get_plastic(s)
            triples.append((s, f, 1.0))

        # Add negative samples
        if self.no_truncate_list:
            # Add as many as possible (still only from this row's negative samples)
            random.shuffle(neg_cols)
            for s in neg_cols:
                f = self._get_plastic(s)
                if f is not None:
                    triples.append((s, f, 0.0))
        else:
            # Fill up to max_list_len
            k_neg = max(0, self.max_list_len - sum(1 for _, f, _ in triples if f is not None))
            if k_neg > 0 and len(neg_cols) > 0:
                random.shuffle(neg_cols)
                taken = 0
                for s in neg_cols:
                    if taken >= k_neg:
                        break
                    f = self._get_plastic(s)
                    if f is None:
                        continue
                    triples.append((s, f, 0.0))
                    taken += 1

        # Filter out None values
        triples = [(name, f, y) for (name, f, y) in triples if f is not None]

        if len(triples) == 0:
            plast_feat = torch.empty(0, 0)
            labels = torch.empty(0, dtype=torch.float32)
            names: List[str] = []
        else:
            names = [name for (name, _, _) in triples]
            plastics = [f for (_, f, _) in triples]  # list of [1, D]
            rels = [y for (_, _, y) in triples]  # list of float
            plast_feat = torch.cat(plastics, dim=0)  # [L, D]
            labels = torch.tensor(rels, dtype=torch.float32)  # [L]

        sample = {
            "mode": "list",
            "enzyme_graph": g,
            "plastic_list": plast_feat,
            "relevance": labels,
        }
        if self.return_names:
            sample["enzyme_id"] = os.path.splitext(os.path.basename(pdb_name))[0]
            sample["pdb_name"] = pdb_name
            sample["plastic_names"] = names  # One-to-one correspondence with plastic_list/relevance
        return sample


# ========= Collate 函数（把多个样本拼成批） =========
def collate_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Universal collate function for batching samples.

    Features:
    - Converts multiple PyG Data -> Batch
    - Stacks plastic features by mode
    - For point/pair modes: filters None (missing/failed) at collate stage
    - Retains name fields if present

    Args:
        batch: List of sample dictionaries from dataset.

    Returns:
        Batched dictionary with mode-specific structure.

    Raises:
        ValueError: If all samples in batch are invalid (contain None values).
    """
    assert len(batch) > 0
    mode: Mode = batch[0]["mode"]

    # Filter out entries containing None by mode
    if mode == "point":
        batch = [b for b in batch if b.get("plastic_feat", None) is not None]
    elif mode == "pair":
        batch = [b for b in batch if
                 (b.get("plastic_pos", None) is not None and b.get("plastic_neg", None) is not None)]
    else:
        # Listwise: each sample has already filtered None internally, no action needed here
        pass

    if len(batch) == 0:
        raise ValueError("[collate_pairs] Batch has 0 valid samples (all filtered due to missing/parsing failures).")

    # Enzyme graphs
    g_list = [b["enzyme_graph"] for b in batch]
    enzyme_batch = Batch.from_data_list(g_list)

    if mode == "point":
        feats = torch.cat([b["plastic_feat"] for b in batch], dim=0)  # [B, D]
        labels = torch.cat([b["label"] for b in batch], dim=0)  # [B]
        out = {"mode": "point", "enzyme_graph": enzyme_batch, "plastic_feat": feats, "label": labels}
        # Retain names
        if "enzyme_id" in batch[0]:
            out["enzyme_id"] = [b["enzyme_id"] for b in batch]
            out["pdb_name"] = [b["pdb_name"] for b in batch]
            out["plastic_name"] = [b["plastic_name"] for b in batch]
        return out

    if mode == "pair":
        pos = torch.cat([b["plastic_pos"] for b in batch], dim=0)  # [B, D]
        neg = torch.cat([b["plastic_neg"] for b in batch], dim=0)  # [B, D]
        out = {"mode": "pair", "enzyme_graph": enzyme_batch, "plastic_pos": pos, "plastic_neg": neg}
        if "enzyme_id" in batch[0]:
            out["enzyme_id"] = [b["enzyme_id"] for b in batch]
            out["pdb_name"] = [b["pdb_name"] for b in batch]
            out["plastic_pos_name"] = [b["plastic_pos_name"] for b in batch]
            out["plastic_neg_name"] = [b["plastic_neg_name"] for b in batch]
        return out

    # Listwise: each sample's list length may differ, training scripts usually process
    # item by item or with padding. For simplicity, we return list as-is (no padding),
    # training script handles internal sample processing in loops.
    return {"mode": "list", "enzyme_graph": enzyme_batch, "items": batch}