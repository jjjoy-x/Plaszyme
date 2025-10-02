# src/models/gvp_local/backbone_official.py
# -*- coding: utf-8 -*-
"""GVP (Geometric Vector Perceptron) backbone module.

This module provides a GVP backbone implementation that is compatible with the
drorlab/gvp-pytorch library and aligns with the project's existing builder
data contracts for handling both scalar and vector node/edge features.

Data Contract:
    Expected input data structure:
        data.x_s: [N, S_in] - Scalar node features
        data.x_v: [N, V_in, 3] - Vector node features
        data.edge_s: [E, Se] - Scalar edge features (optional)
        data.edge_v: [E, Ve, 3] - Vector edge features (optional)
        data.edge_index: [2, E] - Edge connectivity
        data.batch: [N] - Batch assignment (optional, defaults to single graph)

API Alignment:
    - hidden_dims: Per-layer hidden dimensions (S_hidden, V_hidden)
    - n_layers: Number of GVPConvLayer stacks (with residual updates, fixed node dims)
    - out_dim: Output dimension (graph-level or residue-level)
    - dropout: Dropout probability
    - residue_logits: True for residue-level [N,out_dim], False for graph-level [B,out_dim]
    - activations, vector_gate, n_message, n_feedforward: Passed directly to official layers
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from plaszyme.models.gvp import gvp_local


class GVPBackbone(nn.Module):
    """GVP backbone using official gvp_local.GVPConvLayer stacks.

    This backbone stacks multiple GVPConvLayer modules to process geometric
    graph data with both scalar and vector features. It supports both
    residue-level and graph-level outputs with flexible layer configurations.

    Attributes:
        dropout_p: Dropout probability applied throughout the network.
        residue_logits: Whether to output residue-level or graph-level features.
        activations: Tuple of (scalar_activation, vector_activation) functions.
        vector_gate: Whether to enable vector gating (improves performance per paper).
        n_message: Number of GVP modules in each layer's message network.
        n_feedforward: Number of GVP modules in each layer's feedforward network.
        dims_list: List of (S_hidden, V_hidden) tuples for each layer.
        out_dim: Final output dimension.
    """

    def __init__(
            self,
            hidden_dims=(128, 16),  # Can be (S_h, V_h) or [(64,8),(96,12),(128,16)]
            n_layers: int = 3,  # Used when hidden_dims is tuple; ignored when list
            out_dim: int = 128,
            dropout: float = 0.1,
            residue_logits: bool = False,
            *,
            activations: Tuple = (F.relu, None),
            vector_gate: bool = True,
            n_message: int = 3,
            n_feedforward: int = 2,
    ) -> None:
        """Initialize GVP backbone.

        Args:
            hidden_dims: Layer dimensions specification. Can be:
                - Single tuple (S_h, V_h): Repeated n_layers times
                - List of tuples [(S1,V1), (S2,V2), ...]: Per-layer dimensions
            n_layers: Number of layers (only used when hidden_dims is single tuple).
            out_dim: Output feature dimension for final readout.
            dropout: Dropout probability applied in layers and final head.
            residue_logits: If True, return [N, out_dim]; if False, return [B, out_dim].
            activations: Tuple of (scalar_activation, vector_activation) functions.
                Default (F.relu, None) follows original paper.
            vector_gate: Whether to enable vector gating mechanism (recommended).
            n_message: Number of GVP modules in message network per layer.
            n_feedforward: Number of GVP modules in feedforward network per layer.
        """
        super().__init__()
        self.dropout_p = float(dropout)
        self.residue_logits = bool(residue_logits)

        self.activations = activations
        self.vector_gate = bool(vector_gate)
        self.n_message = int(n_message)
        self.n_feedforward = int(n_feedforward)

        # Normalize to per-layer dimension list
        if isinstance(hidden_dims[0], int):
            # Single tuple -> repeat n_layers times
            S_h, V_h = hidden_dims
            self.dims_list = [(S_h, V_h) for _ in range(int(n_layers))]
        else:
            # List of tuples [(S,V), (S,V), ...]
            self.dims_list = [tuple(map(int, t)) for t in hidden_dims]

        self.out_dim = int(out_dim)

        # Lazy construction components
        self._built = False
        self.input_proj = None
        self.blocks = None  # Alternating [opt_proj, conv, opt_proj, conv, ...]
        self.node_head = None
        self.graph_head = None
        self.dropout = nn.Dropout(self.dropout_p)

    # ---------- 内部：懒构建 ----------
    def _lazy_build(self, data: Batch) -> None:
        """Build network architecture after inferring input dimensions.

        Performs lazy construction by examining the first batch to determine
        input feature dimensions for both scalar and vector features, as well
        as edge feature dimensions if present.

        Args:
            data: Input batch containing node and edge features.
        """
        device = data.x_s.device

        # Automatically infer input and edge dimensions
        S_in = int(data.x_s.size(-1))
        V_in = int(data.x_v.size(-2))
        Se = int(getattr(data, "edge_s", torch.zeros(data.edge_index.size(1), 0, device=device)).size(-1))
        Ve = int(getattr(data, "edge_v", torch.zeros(data.edge_index.size(1), 0, 3, device=device)).size(-2))

        # Input projection to first layer dimensions
        S0, V0 = self.dims_list[0]
        if (S_in, V_in) != (S0, V0):
            self.input_proj = gvp_local.GVP(
                in_dims=(S_in, V_in),
                out_dims=(S0, V0),
                activations=self.activations,
                vector_gate=self.vector_gate,
            ).to(device)

        edge_dims = (Se, Ve)
        blocks = nn.ModuleList().to(device)
        cur = (S0, V0)

        # Build layer sequence with dimension transitions
        for (S_h, V_h) in self.dims_list:
            # If previous layer output doesn't match current layer requirement,
            # insert transition GVP
            if cur != (S_h, V_h):
                blocks.append(
                    gvp_local.GVP(
                        in_dims=cur,
                        out_dims=(S_h, V_h),
                        activations=self.activations,
                        vector_gate=self.vector_gate,
                    ).to(device)
                )
                cur = (S_h, V_h)
            # Add GVPConvLayer with matching dimensions
            blocks.append(
                gvp_local.GVPConvLayer(
                    node_dims=(S_h, V_h),
                    edge_dims=edge_dims,
                    n_message=self.n_message,
                    n_feedforward=self.n_feedforward,
                    drop_rate=self.dropout_p,
                    activations=self.activations,
                    vector_gate=self.vector_gate,
                ).to(device)
            )
            cur = (S_h, V_h)

        self.blocks = blocks

        # Readout heads: use scalar channels from last layer
        S_last, V_last = self.dims_list[-1]
        if self.residue_logits:
            self.node_head = nn.Linear(S_last, self.out_dim).to(device)
            self.graph_head = None
        else:
            self.node_head = None
            self.graph_head = nn.Sequential(
                nn.Linear(S_last, S_last),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(S_last, self.out_dim),
            ).to(device)

        self._built = True

    # ---------- formard ----------
    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass through GVP backbone.

        Processes geometric graph data with scalar and vector features through
        a series of GVP convolution layers, returning either residue-level or
        graph-level representations.

        Args:
            data: Input batch with expected structure (aligned with builder):
                - data.x_s: [N, S_in] scalar node features
                - data.x_v: [N, V_in, 3] vector node features
                - data.edge_s: [E, Se] scalar edge features (optional)
                - data.edge_v: [E, Ve, 3] vector edge features (optional)
                - data.edge_index: [2, E] edge connectivity
                - data.batch: [N] batch assignment (optional)

        Returns:
            Output tensor:
                - If residue_logits=True: [N, out_dim] (residue-level)
                - If residue_logits=False: [B, out_dim] (graph-level)
        """
        # Lazy construction on first forward pass
        if not self._built:
            self._lazy_build(data)

        s = data.x_s
        v = data.x_v
        edge_index = data.edge_index.to(s.device)
        e_s = getattr(data, "edge_s", None)
        e_v = getattr(data, "edge_v", None)

        # Default edge features to zero-dimensional placeholders (compatible with official layers)
        if e_s is None:
            e_s = torch.zeros(edge_index.size(1), 0, dtype=s.dtype, device=s.device)
        if e_v is None:
            e_v = torch.zeros(edge_index.size(1), 0, 3, dtype=s.dtype, device=s.device)

        # Optional input projection
        if self.input_proj is not None:
            s, v = self.input_proj((s, v))

        # Process through layer blocks
        for m in self.blocks:
            if isinstance(m, gvp_local.GVPConvLayer):
                s, v = m((s, v), data.edge_index, (e_s, e_v))
            else:  # Transition GVP
                s, v = m((s, v))

        # Generate output based on mode
        if self.residue_logits:
            return self.node_head(s)

        # Graph-level output: pool then project
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(s.size(0), dtype=torch.long, device=s.device)
        g = global_mean_pool(s, batch)
        return self.graph_head(g)