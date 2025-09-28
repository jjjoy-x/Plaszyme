"""Polymer tower module for plastic representation learning.

This module provides components for learning plastic polymer representations
through MLP networks and self-supervised pretraining using co-degradation matrices.
Includes support for contrastive learning and cosine similarity training.
"""

import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class _PairwiseDataset(Dataset):
    """Internal dataset for pairwise training data."""

    def __init__(self, pairs):
        """Initialize dataset with pairs.

        Args:
            pairs: List of (x1, x2, y) tuples where y is float scalar.
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (x1, x2, y) | y: float scalar


class _ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning."""

    def __init__(self, margin: float = 1.0):
        """Initialize contrastive loss.

        Args:
            margin: Margin for dissimilar pairs.
        """
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        """Compute contrastive loss.

        Args:
            out1: First embedding batch [B, D] (L2 normalized).
            out2: Second embedding batch [B, D] (L2 normalized).
            label: Binary labels [B] where 1=similar, 0=dissimilar.

        Returns:
            Contrastive loss scalar.
        """
        dist = torch.norm(out1 - out2, dim=1)
        pos = label * (dist ** 2)
        neg = (1 - label) * (torch.clamp(self.margin - dist, min=0.0) ** 2)
        return (pos + neg).mean()


class PolymerTower(nn.Module):
    """Simple plastic MLP tower for polymer representation learning.

    Transforms polymer descriptor vectors into L2-normalized embeddings
    suitable for cosine similarity training with enzyme representations.

    Architecture:
        Input: Descriptor vectors from PolymerFeaturizer [B, in_dim]
        Output: L2-normalized embedding vectors [B, out_dim]

    Purpose: Generate polymer embeddings for cosine similarity with enzyme embeddings.

    Attributes:
        mlp: MLP backbone with ReLU activations and dropout.
        out_proj: Final linear projection to output dimension.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_dim: int, hidden_dims=(512, 256), out_dim=128, dropout=0.1):
        """Initialize polymer tower.

        Args:
            in_dim: Input descriptor dimension.
            hidden_dims: Tuple of hidden layer dimensions.
            out_dim: Output embedding dimension.
            dropout: Dropout probability for hidden layers.
        """
        super().__init__()
        layers, last_dim = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(last_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through polymer tower.

        Args:
            x: Input descriptor vectors [B, in_dim].

        Returns:
            z: L2-normalized embedding vectors [B, out_dim].
        """
        h = self.mlp(x.float())
        z = self.out_proj(h)
        return F.normalize(z, dim=-1)

    def pretrain_with_co_matrix(
            self,
            features_df: pd.DataFrame,  # index=polymer names
            co_matrix_df: pd.DataFrame,  # index/columns same polymer names
            *,
            sim_threshold: float = 0.01,
            loss_mode: str = "contrastive",  # "contrastive" or "mse"
            margin: float = 1.5,
            epochs: int = 200,
            batch_size: int = 64,
            lr: float = 1e-4,
            device: str | None = None,
            verbose: bool = True,
    ):
        """Pretrain PolymerTower using co-degradation matrix.

        Performs self-supervised pretraining by learning to predict polymer
        similarity based on co-degradation relationships. Supports both
        contrastive learning and MSE regression training modes.

        Args:
            features_df: DataFrame with polymer features, indexed by polymer names.
            co_matrix_df: Co-degradation matrix with polymer names as index/columns.
            sim_threshold: Threshold for converting similarities to binary labels
                         (only used in contrastive mode).
            loss_mode: Training objective - "contrastive" for binary classification
                      or "mse" for regression on similarity values.
            margin: Margin for contrastive loss (only used in contrastive mode).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate for Adam optimizer.
            device: Training device (auto-detected if None).
            verbose: Whether to print training progress.

        Raises:
            ValueError: If no valid polymer pairs can be constructed.
        """

        def _normalize_name(name: str) -> str:
            """Normalize polymer names by removing .sdf extension."""
            name = str(name)
            if name.lower().endswith(".sdf"):
                name = name[:-4]  # Remove .sdf extension
            return name.strip()

        # Normalize polymer names in both dataframes
        features_df = features_df.copy()
        features_df.index = [_normalize_name(n) for n in features_df.index]
        co_matrix_df = co_matrix_df.copy()
        co_matrix_df.index = [_normalize_name(n) for n in co_matrix_df.index]
        co_matrix_df.columns = [_normalize_name(n) for n in co_matrix_df.columns]

        assert isinstance(features_df, pd.DataFrame) and isinstance(co_matrix_df, pd.DataFrame)

        print(f"[PRETRAIN] features_df polymers={len(features_df.index)} examples: {list(features_df.index)[:10]}")
        print(f"[PRETRAIN] co_matrix_df polymers={len(co_matrix_df.index)} examples: {list(co_matrix_df.index)[:10]}")

        # Check alignment before processing
        names_feat = set(map(str, features_df.index))
        names_co = set(map(str, co_matrix_df.index))
        inter = names_feat & names_co
        print(
            f"[PRETRAIN] intersection count={len(inter)} | "
            f"only in features not in co_matrix examples: {list(names_feat - names_co)[:10]}")
        print(f"[PRETRAIN] only in co_matrix not in features examples: {list(names_co - names_feat)[:10]}")

        # 1) Align indices
        # Use features_df.index as baseline, keep only those appearing in co_matrix
        names = [n for n in features_df.index if n in co_matrix_df.index]
        features_df = features_df.loc[names]
        co_matrix_df = co_matrix_df.loc[names, names]

        if len(names) < 2:
            print(f"[PRETRAIN] Warning: fewer than two usable polymers (n={len(names)}), skipping pretraining.")
            return

        # 2) Construct training pairs
        pairs = []
        X = features_df.values.astype("float32")
        for i, j in itertools.combinations(range(len(names)), 2):
            sim = co_matrix_df.iat[i, j]
            if pd.isna(sim):
                continue
            x1 = torch.from_numpy(X[i])
            x2 = torch.from_numpy(X[j])
            if loss_mode == "contrastive":
                y = torch.tensor(1.0 if float(sim) >= sim_threshold else 0.0, dtype=torch.float32)
            else:  # "mse"
                y = torch.tensor(float(sim), dtype=torch.float32)
            pairs.append((x1, x2, y))
        if not pairs:
            raise ValueError("No valid polymer pairs constructed")

        dl = DataLoader(_PairwiseDataset(pairs), batch_size=batch_size, shuffle=True)

        # 3) Setup device, optimizer, loss function
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = _ContrastiveLoss(margin) if loss_mode == "contrastive" else nn.MSELoss()

        # 4) Training loop
        self.train()
        for ep in range(1, epochs + 1):
            total_loss, n = 0.0, 0
            for x1, x2, y in dl:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                z1, z2 = self(x1), self(x2)

                if loss_mode == "contrastive":
                    loss = criterion(z1, z2, y)
                else:  # MSE mode
                    cos = F.cosine_similarity(z1, z2, dim=-1)  # [-1,1]
                    cos_01 = (cos + 1.0) / 2.0  # -> [0,1]
                    loss = criterion(cos_01, y)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total_loss += loss.item() * x1.size(0)
                n += x1.size(0)

            if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
                print(f"[PolymerTower pretrain] epoch={ep:03d} loss={total_loss / n:.6f}")

        self.eval()
        if verbose:
            print("[PolymerTower pretrain] Training completed")