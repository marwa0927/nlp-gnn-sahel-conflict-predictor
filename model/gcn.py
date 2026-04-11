
"""
model/gcn.py
============

2-layer Graph Convolutional Network for node-level conflict prediction.

Architecture:
  GCNConv(in, 64) → ReLU → Dropout
  GCNConv(64, 32) → ReLU → Dropout
  Linear(32, 2)   → Sigmoid

Two output heads per node:
  - p_conflict : probability of armed conflict event within 80km
  - p_unrest   : probability of civil unrest / protest within 80km

NOTE: GAT (Graph Attention Network) is the v2 upgrade target.
      GCN is the v1 baseline — simpler, faster, sufficient to
      validate that the graph structure adds signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ConflictGCN(nn.Module):
    """
    2-layer GCN for node-level binary conflict and unrest prediction.

    Args:
        in_features : input feature dimension per node (window * F = 7 * 28 = 196)
        hidden_dim  : hidden layer dimension (default: 64)
        dropout     : dropout probability (default: 0.3)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim:  int   = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Message-passing layers
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)

        # Output heads — separate linear layers for each task
        self.head_conflict = nn.Linear(hidden_dim // 2, 1)
        self.head_unrest   = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear output heads."""
        for head in [self.head_conflict, self.head_unrest]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : (N, in_features)  node feature matrix
            edge_index  : (2, E)             edge indices in COO format
            edge_weight : (E,)               optional edge weights

        Returns:
            out : (N, 2)  — columns are [p_conflict, p_unrest]
                            values in (0, 1) via sigmoid
        """
        # Layer 1: message passing + activation + dropout
        x = self.conv1(x, edge_index, edge_weight)     # (N, hidden_dim)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: message passing + activation + dropout
        x = self.conv2(x, edge_index, edge_weight)     # (N, hidden_dim // 2)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output heads — raw logits (sigmoid applied in loss / inference)
        logit_conflict = self.head_conflict(x)         # (N, 1)
        logit_unrest   = self.head_unrest(x)           # (N, 1)

        # Concatenate and apply sigmoid for inference
        out = torch.sigmoid(
            torch.cat([logit_conflict, logit_unrest], dim=1)
        )                                               # (N, 2)

        return out

    def forward_logits(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Same as forward() but returns raw logits (no sigmoid).
        Use this during training with BCEWithLogitsLoss for numerical stability.
        """
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        logit_conflict = self.head_conflict(x)
        logit_unrest   = self.head_unrest(x)

        return torch.cat([logit_conflict, logit_unrest], dim=1)   # (N, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        print(f"\n{'─'*50}")
        print(f"  ConflictGCN")
        print(f"{'─'*50}")
        print(f"  {self}")
        print(f"  Trainable parameters: {self.count_parameters():,}")
        print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    window:     int   = 7,
    n_features: int   = 28,
    hidden_dim: int   = 64,
    dropout:    float = 0.3,
) -> ConflictGCN:
    """
    Instantiate the GCN model with correct input dimension.

    in_features = window * n_features
    e.g., 7 days × 28 features = 196
    """
    in_features = window * n_features
    model = ConflictGCN(in_features=in_features, hidden_dim=hidden_dim, dropout=dropout)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model: ConflictGCN, path: Path, metadata: dict = None):
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata":   metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model: ConflictGCN, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Checkpoint loaded ← {path}")
    return checkpoint.get("metadata", {})


# ─────────────────────────────────────────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    N_NODES    = 25
    WINDOW     = 7
    N_FEATURES = 28
    IN_FEATURES = WINDOW * N_FEATURES   # 196

    print("Testing ConflictGCN forward pass...\n")

    model = build_model(window=WINDOW, n_features=N_FEATURES)
    model.describe()

    # Dummy inputs
    x           = torch.randn(N_NODES, IN_FEATURES)
    # Simple ring graph for testing
    src         = torch.arange(N_NODES)
    dst         = torch.roll(src, 1)
    edge_index  = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_weight = torch.ones(edge_index.shape[1])

    # Forward pass — inference mode
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)

    print(f"Forward pass output shape: {tuple(out.shape)}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]  (should be 0–1)")
    print(f"p_conflict sample: {out[:5, 0].tolist()}")
    print(f"p_unrest   sample: {out[:5, 1].tolist()}")

    # Assertions
    assert out.shape == (N_NODES, 2),       f"Expected ({N_NODES}, 2), got {out.shape}"
    assert out.min() >= 0.0,                "Output below 0 (sigmoid should prevent this)"
    assert out.max() <= 1.0,                "Output above 1 (sigmoid should prevent this)"
    assert not torch.any(torch.isnan(out)), "NaN in output"

    # Logit forward (for training)
    model.train()
    logits = model.forward_logits(x, edge_index, edge_weight)
    assert logits.shape == (N_NODES, 2), f"Logit shape wrong: {logits.shape}"
    print(f"\nLogit forward pass: shape={tuple(logits.shape)} ")

    print(f"\n ConflictGCN forward pass validated — model ready for training")

