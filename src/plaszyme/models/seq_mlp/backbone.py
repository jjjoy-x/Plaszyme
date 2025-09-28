# src/models/seq_mlp/backbone.py
"""Sequence-based MLP backbone module.

This module provides a non-graph backbone that processes residue-level features
through MLP layers, offering the same interface as GNN/GVP backbones but
operating purely on sequential data without graph connectivity.

The backbone prioritizes sequence embeddings (data.seq_x) over general node
features (data.x) and supports both residue-level and graph-level outputs
through configurable pooling strategies.
"""

from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool


class SeqBackbone(nn.Module):
    """Non-graph backbone with identical interface to GNN/GVP backbones.

    This backbone processes only residue-level features without using graph
    connectivity information. It applies MLP transformations to residue features,
    then optionally pools to graph-level representations.

    Processing Pipeline:
        residue features → MLP encoding → pooling → graph-level vector →
        MLP mapping to out_dim

    When residue_logits=True, returns residue-level output [N, out_dim].

    Attributes:
        hidden_dims: Layer dimensions for residue-level encoding MLP.
        out_dim: Graph-level output dimension (aligned with EMB_DIM_ENZ).
        dropout_p: Dropout probability applied throughout the network.
        pool: Pooling strategy - "mean" or "max".
        residue_logits: Whether to output residue-level logits.
        in_dim_fixed: Fixed input dimension if known beforehand.
        feature_priority: Ordered list of fields to try for residue features.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 128],
        out_dim: int = 128,
        dropout: float = 0.1,
        pool: str = "mean",
        residue_logits: bool = False,
        *,
        in_dim: Optional[int] = None,
        feature_priority: Optional[List[str]] = None,  # e.g. ["seq_x", "x"]
    ):
        """Initialize sequence MLP backbone.

        Args:
            hidden_dims: Layer dimensions for residue-level encoding MLP.
                Each dimension creates a Linear → GELU → Dropout block.
            out_dim: Final output dimension for both residue and graph levels.
            dropout: Dropout probability applied after each hidden layer.
            pool: Pooling strategy for graph-level output - "mean" or "max".
            residue_logits: If True, return residue-level output [N, out_dim].
                          If False, return graph-level output [B, out_dim].
            in_dim: Input feature dimension if known (enables eager construction).
                   If None, dimension is inferred from first batch (lazy construction).
            feature_priority: Ordered list of data fields to try for residue features.
                            Default tries ["seq_x", "x"] in order.

        Raises:
            AssertionError: If pool is not "mean" or "max".
        """
        super().__init__()
        assert pool in ("mean", "max")
        self.hidden_dims = list(hidden_dims)
        self.out_dim = int(out_dim)
        self.dropout_p = float(dropout)
        self.pool = pool
        self.residue_logits = residue_logits
        self.in_dim_fixed = in_dim
        self.feature_priority = feature_priority or ["seq_x", "x"]

        # Lazy construction components
        self._built = False
        self._enc: Optional[nn.Sequential] = None
        self._node_head: Optional[nn.Linear] = None
        self._graph_head: Optional[nn.Sequential] = None

    # ---------- helpers ----------
    def _pick_feature(self, data: Batch) -> torch.Tensor:
        """Extract residue features from data based on priority list.

        Tries each field in feature_priority order until finding valid data.

        Args:
            data: Input batch containing various feature fields.

        Returns:
            Residue feature tensor of shape [N, feature_dim].

        Raises:
            ValueError: If no valid residue features found in any priority field.
        """
        for k in self.feature_priority:
            v = getattr(data, k, None)
            if v is not None:
                return v.float()
        raise ValueError(
            f"[SeqBackbone] No residue features found, tried fields {self.feature_priority}. "
            "Please ensure sequence embeddings are stored in data.seq_x (or data.x) via builder."
        )

    def _get_batch(self, data: Batch, N: int, device: torch.device) -> torch.Tensor:
        """Get batch assignment vector, defaulting to single graph if not present.

        Args:
            data: Input batch data.
            N: Number of nodes.
            device: Target device for tensor.

        Returns:
            Batch assignment tensor of shape [N].
        """
        b = getattr(data, "batch", None)
        if b is None:
            b = torch.zeros(N, dtype=torch.long, device=device)
        return b.to(device)

    def _lazy_build(self, in_dim: int, device: torch.device) -> None:
        """Build network architecture after input dimension is known.

        Constructs the MLP encoder and output heads based on the inferred
        input dimension from the first batch.

        Args:
            in_dim: Input feature dimension.
            device: Target device for parameters.
        """
        # Residue-level encoder with layer normalization
        layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(self.dropout_p)]
            prev = h
        self._enc = nn.Sequential(*layers)

        # 残基级头（当 residue_logits=True 使用）
        self._node_head = nn.Linear(prev, self.out_dim)

        # Graph-level head (used when residue_logits=False)
        self._graph_head = nn.Sequential(
            nn.Linear(prev, prev), nn.GELU(), nn.Dropout(self.dropout_p),
            nn.Linear(prev, self.out_dim)
        )

        self.to(device)
        self._built = True

    # ---------- forward ----------
    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass through sequence MLP backbone.

        Processes residue-level features through MLP layers, then either
        returns residue-level outputs directly or pools to graph-level
        representations based on residue_logits setting.

        Args:
            data: Input batch containing residue features. Expected fields
                 (in priority order): seq_x, x. Also expects batch assignment
                 for graph-level pooling if multiple graphs present.

        Returns:
            Output tensor:
                - If residue_logits=True: [N, out_dim] (residue-level)
                - If residue_logits=False: [B, out_dim] (graph-level)

        Raises:
            ValueError: If no valid residue features found in input data.
        """
        # Extract residue features
        x = self._pick_feature(data)  # [N, D_in]
        device = x.device
        N = x.size(0)
        batch = self._get_batch(data, N, device)

        # Lazy construction on first forward pass
        if not self._built:
            in_dim = int(self.in_dim_fixed or x.size(-1))
            self._lazy_build(in_dim, device)

        # Encode residue features
        h = self._enc(x)  # [N, H]

        # Return residue-level output if requested
        if self.residue_logits:
            return self._node_head(h)  # [N, out_dim]

        # Pool to graph-level representation
        if self.pool == "mean":
            hg = global_mean_pool(h, batch)  # [B, H]
        else:  # self.pool == "max"
            hg = global_max_pool(h, batch)  # [B, H]

        # Apply graph-level head
        return self._graph_head(hg)  # [B, out_dim]