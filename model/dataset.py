"""
model/dataset.py
"""

import importlib.util
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import Optional, Tuple, List


ROOT       = Path(__file__).parent.parent
NODES_PY   = ROOT / "data" / "nodes.py"
FEATURES_PATH = ROOT / "data" / "processed" / "dataset_daily.parquet"
GRAPH_DIR  = ROOT / "graph"

# Columns that are labels, not features
LABEL_COLS   = ["y_conflict", "y_unrest"]
# Columns to drop when building the feature vector
NON_FEATURE_COLS = {"node_id", "date", "y_conflict", "y_unrest",
                    "city", "location", "node", "id", "index",
                    "fatalities"}  # fatalities is a label, not a feature


# ─────────────────────────────────────────────────────────────────────────────
# Load nodes from nodes.py
# ─────────────────────────────────────────────────────────────────────────────
def load_nodes() -> list:
    spec = importlib.util.spec_from_file_location("nodes", NODES_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NODES


# ─────────────────────────────────────────────────────────────────────────────
# Build X (T, N, F) and Y (T, N, 2) from dataset_daily.parquet
# ─────────────────────────────────────────────────────────────────────────────
def build_matrices_from_parquet(
    nodes: list,
    parquet_path: Path = FEATURES_PATH,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Reads features_daily.parquet and pivots it into:
      X : (T, N, F)  float32 — feature matrix
      Y : (T, N, 2)  float32 — binary labels [y_conflict, y_unrest]

    Returns: X, Y, dates (list of YYYY-MM-DD strings), feature_names
    """
    df = pd.read_parquet(parquet_path)

    # ── Detect node column ───────────────────────────────────────────────────
    node_col = None
    for candidate in ["node_id", "city", "location", "node", "id"]:
        if candidate in df.columns:
            node_col = candidate
            break
    if node_col is None:
        raise ValueError(
            "Cannot find node identifier column in parquet. "
            "Expected one of: node_id, city, location, node, id. "
        )

    # ── Detect date column ───────────────────────────────────────────────────
    date_col = None
    for candidate in ["date", "event_date", "day", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Try index
        if isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index.strftime("%Y-%m-%d")
            date_col = "date"
        else:
            raise ValueError(
                "Cannot find date column in parquet. "
                "Expected one of: date, event_date, day, timestamp. "
            )

    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # ── Node ordering — must match nodes.py exactly ──────────────────────────
    node_order = [n["id"] for n in nodes]
    N = len(node_order)

    # ── Identify feature columns ─────────────────────────────────────────────
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != node_col and c != date_col
        and c not in LABEL_COLS
    ]
    F = len(feature_cols)
    print(f" Feature columns ({F}): {feature_cols}")

    # ── Check label columns ──────────────────────────────────────────────────
    has_labels = all(c in df.columns for c in LABEL_COLS)
    if has_labels:
        rate_c = df["y_conflict"].mean()
        rate_u = df["y_unrest"].mean()
        print(f"✅ Labels found — conflict rate={rate_c:.4f}  unrest rate={rate_u:.4f}")
    else:
        print(f"  Label columns {LABEL_COLS} not found in parquet.")
        print("   Y will be all zeros until Person A delivers ACLED labels.")

    # ── Pivot to (T, N, F) ───────────────────────────────────────────────────
    dates = sorted(df[date_col].unique())
    T = len(dates)
    date_idx  = {d: i for i, d in enumerate(dates)}
    node_idx  = {nid: i for i, nid in enumerate(node_order)}

    X = np.zeros((T, N, F), dtype=np.float32)
    Y = np.zeros((T, N, 2), dtype=np.float32)

    for _, row in df.iterrows():
        nid  = row[node_col]
        date = row[date_col]
        if nid not in node_idx or date not in date_idx:
            continue
        i = date_idx[date]
        j = node_idx[nid]
        X[i, j, :] = [row[c] for c in feature_cols]
        if has_labels:
            Y[i, j, 0] = float(row.get("y_conflict", 0))
            Y[i, j, 1] = float(row.get("y_unrest",   0))

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0).astype(np.float32)

    print(f"  Built X: {X.shape}  Y: {Y.shape}  dates: {dates[0]} → {dates[-1]}")
    print(f"   Conflict label rate: {Y[:,:,0].mean():.3f}")
    print(f"   Unrest   label rate: {Y[:,:,1].mean():.3f}")

    return X, Y, dates, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SahelConflictDataset(Dataset):
    """
    Sliding-window temporal dataset for node-level conflict prediction.

    Each sample:
      x          : (N, window * F)  — flattened window of node features
      edge_index : (2, E)           — graph structure
      edge_attr  : (E,)             — edge weights
      y          : (N, 2)           — binary labels [conflict, unrest]

    Args:
        window    : days of history used as input (default: 7)
        horizon   : days ahead to predict (default: 1)
        split     : 'train', 'val', 'test', or None (all)
        val_start : start of validation period
        test_start: start of test period
    """

    def __init__(
        self,
        window:     int  = 7,
        horizon:    int  = 1,
        split:      Optional[str] = None,
        val_start:  str  = "2023-01-01",
        test_start: str  = "2024-01-01",
    ):
        super().__init__()
        self.window  = window
        self.horizon = horizon
        self.split   = split

        # ── Load nodes ───────────────────────────────────────────────────────
        self.nodes = load_nodes()
        self.N     = len(self.nodes)

        # ── Load feature matrix from parquet ────────────────────────────────
        X, Y, dates, feature_cols = build_matrices_from_parquet(self.nodes)
        self.feature_cols = feature_cols
        self.F            = len(feature_cols)
        self.dates        = dates

        self.X = torch.tensor(X, dtype=torch.float32)   # (T, N, F)
        self.Y = torch.tensor(Y, dtype=torch.float32)   # (T, N, 2)

        # ── Load graph ───────────────────────────────────────────────────────
        self.edge_index = torch.tensor(
            np.load(GRAPH_DIR / "edge_index.npy"), dtype=torch.long
        )
        self.edge_attr = torch.tensor(
            np.load(GRAPH_DIR / "edge_weight.npy"), dtype=torch.float32
        )

        # ── Build index of valid samples ─────────────────────────────────────
        T = self.X.shape[0]
        all_indices = list(range(window, T - horizon + 1))

        if split is not None:
            all_indices = self._filter_by_split(
                all_indices, val_start, test_start, split
            )
        self.sample_indices = all_indices

        print(
            f"SahelConflictDataset ({split or 'all'}) — "
            f"{len(self.sample_indices)} samples | "
            f"N={self.N} nodes | F={self.F} features | "
            f"window={window}d horizon={horizon}d"
        )

    def _filter_by_split(self, indices, val_start, test_start, split):
        filtered = []
        for idx in indices:
            target_date = self.dates[idx + self.horizon - 1]
            if split == "train" and target_date < val_start:
                filtered.append(idx)
            elif split == "val" and val_start <= target_date < test_start:
                filtered.append(idx)
            elif split == "test" and target_date >= test_start:
                filtered.append(idx)
        return filtered

    def len(self) -> int:
        return len(self.sample_indices)

    def get(self, idx: int) -> Data:
        t = self.sample_indices[idx]

        x_window = self.X[t - self.window : t]                        # (window, N, F)
        x_flat   = x_window.permute(1, 0, 2).reshape(self.N, -1)     # (N, window*F)
        y        = self.Y[t + self.horizon - 1]                       # (N, 2)

        return Data(
            x          = x_flat,
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr,
            y          = y,
        )

    def get_input_dim(self) -> int:
        return self.window * self.F

    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y_all = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        n_pos_c = y_all[:, :, 0].sum().item()
        n_neg_c = y_all[:, :, 0].numel() - n_pos_c
        n_pos_u = y_all[:, :, 1].sum().item()
        n_neg_u = y_all[:, :, 1].numel() - n_pos_u
        w_conflict = torch.tensor([n_neg_c / (n_pos_c + 1e-6)], dtype=torch.float32)
        w_unrest   = torch.tensor([n_neg_u / (n_pos_u + 1e-6)], dtype=torch.float32)
        print(f"  pos_weight_conflict: {w_conflict.item():.2f}")
        print(f"  pos_weight_unrest:   {w_unrest.item():.2f}")
        return w_conflict, w_unrest

    def describe(self):
        print(f"\n{'─'*55}")
        print(f"  SahelConflictDataset ({self.split or 'all'})")
        print(f"{'─'*55}")
        print(f"  Samples:       {len(self)}")
        print(f"  Nodes (N):     {self.N}")
        print(f"  Features (F):  {self.F}  {self.feature_cols}")
        print(f"  Input dim:     {self.get_input_dim()} (window={self.window} × F={self.F})")
        print(f"  Edges:         {self.edge_index.shape[1]}")
        if len(self) > 0:
            s = self.get(0)
            print(f"  sample.x:      {tuple(s.x.shape)}")
            print(f"  sample.y:      {tuple(s.y.shape)}")
            first = self.dates[self.sample_indices[0]  + self.horizon - 1]
            last  = self.dates[self.sample_indices[-1] + self.horizon - 1]
            print(f"  Target dates:  {first} → {last}")
        print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SahelConflictDataset from real parquet...\n")
    dataset = SahelConflictDataset(split=None)
    dataset.describe()

    s = dataset.get(0)
    print(f"Sample 0:")
    print(f"  x shape:          {tuple(s.x.shape)}")
    print(f"  edge_index shape: {tuple(s.edge_index.shape)}")
    print(f"  y shape:          {tuple(s.y.shape)}")

    assert not torch.any(torch.isnan(s.x)), "NaN in x"
    assert not torch.any(torch.isnan(s.y)), "NaN in y"
    print("\nDataset ready")
