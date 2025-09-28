# src/models/gnn/backbone.py
"""GNN backbone module for protein graph neural networks.

This module provides a universal GNN backbone supporting multiple graph convolution
operators (GCN/GAT/GATv2/SAGE/GIN/GINE) with flexible edge attribute handling
and both residue-level and graph-level output modes.
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv, GATConv, GATv2Conv, SAGEConv, GINConv, GINEConv, global_mean_pool
)
from torch_geometric.data import Batch


def _edge_weight_from_attr(edge_attr: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """Convert edge_attr to edge_weight for GCN compatibility.

    Converts edge_attr ([E] or [E,D]) to GCN's required edge_weight ([E]).
    Designed to be robust without assuming specific semantic meaning.

    Args:
        edge_attr: Edge attributes tensor of shape [E] or [E, D].
        mode: Conversion strategy:
            - "auto": Take mean across features, min-max normalize and invert
                     (closer distances → higher weights)
            - "mean_inv": w = 1 / (mean + eps)
            - "first_inv": w = 1 / (first_column + eps)

    Returns:
        Edge weights tensor of shape [E] with values >= 0.
    """
    eps = 1e-8
    if edge_attr.dim() == 1:
        w = edge_attr
    else:
        if edge_attr.size(-1) == 1:
            w = edge_attr.squeeze(-1)
        else:
            if mode == "first_inv":
                col0 = edge_attr[:, 0]
                w = 1.0 / (col0 + eps)
            elif mode == "mean_inv":
                mean = edge_attr.mean(dim=-1)
                w = 1.0 / (mean + eps)
            else:  # "auto"
                m = edge_attr.mean(dim=-1)  # [E]
                m_min, m_max = m.min(), m.max()
                if (m_max - m_min) > eps:
                    m_norm = (m - m_min) / (m_max - m_min + eps)
                    w = 1.0 - m_norm  # Invert: smaller distances → higher weights
                else:
                    w = torch.ones_like(m)
    return torch.clamp(w, min=0.0)


class GNNBackbone(nn.Module):
    """Universal GNN backbone supporting multiple convolution operators.

    Supports GCN/GAT/GATv2/SAGE/GIN/GINE with flexible edge attribute handling.
    Compatible with graphs that have or don't have edge features. Uses lazy
    construction to automatically infer input dimensions from the first batch.

    Attributes:
        conv_type: Type of graph convolution operator.
        hidden_dims: Hidden dimensions for each layer.
        out_dim: Final output dimension.
        dropout_p: Dropout probability.
        residue_logits: Whether to output residue-level or graph-level features.
        gcn_edge_mode: Strategy for converting edge_attr to edge_weight for GCN.
        gine_missing_edge_policy: Policy when GINE needs edge_attr but it's missing.
    """

    def __init__(
        self,
        conv_type: str,
        hidden_dims: List[int] | None = None,
        out_dim: int = 1,
        dropout: float = 0.3,
        residue_logits: bool = False,
        *,
        gcn_edge_mode: str = "auto",
        gine_missing_edge_policy: str = "error",
        **legacy_kwargs,
    ):
        """Initialize GNN backbone.

        Args:
            conv_type: Graph convolution type - "gcn", "gat", "gatv2", "sage", "gin", or "gine".
            hidden_dims: List of hidden dimensions for each layer (node feature channels).
            out_dim: Final output dimension.
            dropout: Dropout probability applied after each layer.
            residue_logits: If True, return residue-level output [N, out_dim].
                          If False, return graph-level output [B, out_dim].
            gcn_edge_mode: Strategy for GCN edge_attr→edge_weight conversion:
                         "auto", "mean_inv", or "first_inv".
            gine_missing_edge_policy: Policy when conv_type="gine" but edge_attr is missing:
                                    - "error": Raise error
                                    - "zeros": Use zero placeholders (edge_dim=1)
            **legacy_kwargs: Legacy parameters for backward compatibility.

        Raises:
            ValueError: If hidden_dims is not provided or conv_type is unsupported.
        """
        super().__init__()
        self.conv_type = conv_type.lower()

        # Handle legacy parameter compatibility
        if hidden_dims is None and "dims" in legacy_kwargs:
            warnings.warn(
                "[GNNBackbone] `dims=` is deprecated, please use `hidden_dims=`; "
                "using `dims` value for this run.",
                stacklevel=2
            )
            hidden_dims = legacy_kwargs.pop("dims")
        if hidden_dims is None:
            raise ValueError("`hidden_dims` cannot be empty (example: [64,64]).")

        self.hidden_dims = list(hidden_dims)
        self.out_dim = out_dim
        self.dropout_p = dropout
        self.residue_logits = residue_logits
        self.gcn_edge_mode = gcn_edge_mode
        self.gine_missing_edge_policy = gine_missing_edge_policy

        # Lazy construction: automatically infer in_dim/edge_dim from first batch
        self._built = False
        self._edge_dim: Optional[int] = None

        # Warning switches: avoid repeated prints for each batch
        self._warned_edge_ignored = False
        self._warned_edge_v_ignored = False

    # ---- layer builders ----
    def _make_conv(self, in_dim: int, out_dim: int, edge_dim: Optional[int]) -> nn.Module:
        """Construct one convolution layer with the chosen operator.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            edge_dim: Edge feature dimension (only used for GINE).

        Returns:
            Convolution layer module.

        Raises:
            ValueError: If conv_type is unsupported or GINE requires edge_dim but it's None.
        """
        ct = self.conv_type
        if ct == "gcn":
            return GCNConv(in_dim, out_dim, normalize=True, add_self_loops=True)
        if ct == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        if ct == "gatv2":
            return GATv2Conv(in_dim, out_dim, heads=1, concat=False)
        if ct == "sage":
            return SAGEConv(in_dim, out_dim)
        if ct == "gin":
            return GINConv(nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            ))
        if ct == "gine":
            if edge_dim is None:
                raise ValueError("GINEConv requires 'edge_dim' at layer build time.")
            # GINE's nn operates on nodes (after message aggregation); edge_attr dim passed via edge_dim
            nn_node = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn_node, train_eps=True, edge_dim=edge_dim)
        raise ValueError(f"Unsupported conv_type: {ct}")

    def _build_layers(self, in_dim: int, edge_dim: Optional[int]) -> None:
        """Build all network layers after input dimensions are known.

        Args:
            in_dim: Input node feature dimension.
            edge_dim: Edge feature dimension (for GINE only).
        """
        layers = []
        prev = in_dim
        for d in self.hidden_dims:
            layers.append(self._make_conv(prev, d, edge_dim if self.conv_type == "gine" else None))
            prev = d
        self.convs = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        # Readout: concatenate all layer outputs then linear projection
        self.readout = nn.Linear(sum(self.hidden_dims), self.out_dim)
        self._edge_dim = edge_dim
        self._built = True

    # ---- input prep ----
    def _prepare_inputs(
        self, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract and prepare inputs from batch data according to operator strategy.

        Uniformly extracts x/edge_index/edge_attr/edge_weight/batch and processes
        them according to the specific requirements of each convolution operator.

        Args:
            data: Input batch containing graph data.

        Returns:
            Tuple containing:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge attributes (may be None or modified)
                - edge_weight: Edge weights (may be None or computed from edge_attr)
                - batch: Batch assignment vector
        """
        x = data.x
        device = x.device

        edge_index = data.edge_index.to(device)
        edge_attr = getattr(data, "edge_attr", None)
        edge_weight = getattr(data, "edge_weight", None)
        edge_v = getattr(data, "edge_v", None)

        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Warn once: pure GNN doesn't use vector edges
        if edge_v is not None and not self._warned_edge_v_ignored:
            warnings.warn("[GNNBackbone] 'edge_v' detected but ignored by non-GVP backbones.")
            self._warned_edge_v_ignored = True

        ct = self.conv_type

        # GCN: Try to convert edge_attr to edge_weight
        if ct == "gcn":
            if edge_weight is None and edge_attr is not None:
                try:
                    ew = _edge_weight_from_attr(edge_attr, mode=self.gcn_edge_mode).to(device)
                    if not torch.is_floating_point(ew):
                        ew = ew.float()
                    edge_weight = ew
                except Exception:
                    if not self._warned_edge_ignored:
                        warnings.warn(
                            "[GNNBackbone][GCN] edge_attr→edge_weight conversion failed, falling back to binary adjacency."
                        )
                        self._warned_edge_ignored = True
                    edge_weight = None

        # Operators that don't support edge features: ignore edge_attr
        if ct in {"gat", "gatv2", "sage", "gin"}:
            if edge_attr is not None and not self._warned_edge_ignored:
                warnings.warn(f"[GNNBackbone][{ct.upper()}] edge_attr provided but will be ignored.")
                self._warned_edge_ignored = True
            edge_attr = None

        # GINE: Must have edge_attr (or zeros policy)
        if ct == "gine":
            if edge_attr is None and self.gine_missing_edge_policy != "zeros":
                raise ValueError(
                    "[GNNBackbone][GINE] edge_attr required; or set gine_missing_edge_policy='zeros'."
                )

        # Batch assignment (default to all zeros if not present)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        else:
            batch = batch.to(device)

        return x, edge_index, edge_attr, edge_weight, batch

    # ---- forward ----
    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass through the GNN backbone.

        Performs lazy construction on first call to automatically infer input dimensions.
        Then applies the specified graph convolution layers followed by readout.

        Args:
            data: Batch of graph data containing node features, edges, and optional
                 edge attributes.

        Returns:
            Output tensor:
                - If residue_logits=True: [N, out_dim] (residue-level)
                - If residue_logits=False: [B, out_dim] (graph-level)

        Raises:
            RuntimeError: If GINE cannot infer edge_dim during lazy construction.
        """
        x, edge_index, edge_attr, edge_weight, batch = self._prepare_inputs(data)

        # Lazy construction: automatically infer input dim & GINE's edge_dim
        if not self._built:
            in_dim = x.size(-1)
            edge_dim = None
            if self.conv_type == "gine":
                if edge_attr is not None:
                    edge_dim = edge_attr.size(-1) if edge_attr.dim() == 2 else 1
                elif self.gine_missing_edge_policy == "zeros":
                    edge_dim = 1  # Use 1D placeholder
                else:
                    raise RuntimeError("[GNNBackbone][GINE] Cannot infer edge_dim.")
            self._build_layers(in_dim, edge_dim)
            self.to(x.device)

        # If GINE lacks edge_attr and zeros policy selected, construct [E, edge_dim] zeros
        if self.conv_type == "gine" and edge_attr is None and self.gine_missing_edge_policy == "zeros":
            E = edge_index.size(1)
            edge_attr = torch.zeros(E, self._edge_dim or 1, device=x.device)

        # GNN layer stack
        outs = []
        h = x
        for conv in self.convs:
            if isinstance(conv, GCNConv):
                h = conv(h, edge_index, edge_weight=edge_weight)
            elif isinstance(conv, GINEConv):
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
            outs.append(h)

        # Concatenate all layer outputs (residue dimension unchanged)
        h_cat = torch.cat(outs, dim=-1)  # [N, sum(hidden_dims)]

        # Residue-level output: no pooling
        if self.residue_logits:
            return self.readout(h_cat)  # [N, out_dim]

        # Graph-level output: pooling then linear
        g = global_mean_pool(h_cat, batch)  # [B, sum(hidden_dims)]
        return self.readout(g)  # [B, out_dim]