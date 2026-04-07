import numpy as np
import pandas as pd
from config import RADIUS_KM
from data.nodes import NODES

_NODE_IDS  = [n["id"] for n in NODES]
_NODE_LATS = np.array([n["lat"] for n in NODES])
_NODE_LONS = np.array([n["lon"] for n in NODES])


def assign_nodes_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorized node assignment — no row-by-row apply."""
    df = df.dropna(subset=["lat", "lon"]).copy()
    print(f"Assigning {len(df):,} rows to nodes...")

    R = 6371.0
    lat1 = np.radians(df["lat"].values)[:, None]   # (N, 1)
    lon1 = np.radians(df["lon"].values)[:, None]   # (N, 1)
    lat2 = np.radians(_NODE_LATS)[None, :]          # (1, 25)
    lon2 = np.radians(_NODE_LONS)[None, :]          # (1, 25)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist = R * 2 * np.arcsin(np.sqrt(a))            # (N, 25)

    nearest_idx  = np.argmin(dist, axis=1)           # (N,)
    nearest_dist = dist[np.arange(len(df)), nearest_idx]

    node_ids = np.array(_NODE_IDS)
    df["node_id"] = np.where(nearest_dist <= RADIUS_KM,
                             node_ids[nearest_idx], None)

    assigned = df["node_id"].notna().sum()
    print(f"Assigned {assigned:,} / {len(df):,} ({assigned/len(df):.1%})")
    return df.dropna(subset=["node_id"])


def assign_node(lat: float, lon: float) -> str | None:
    """Single-point lookup (kept for compatibility)."""
    if np.isnan(lat) or np.isnan(lon):
        return None
    R = 6371.0
    lat1, lon1 = np.radians(lat), np.radians(lon)
    dlat = np.radians(_NODE_LATS) - lat1
    dlon = np.radians(_NODE_LONS) - lon1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(np.radians(_NODE_LATS))*np.sin(dlon/2)**2
    dist = R * 2 * np.arcsin(np.sqrt(a))
    idx  = np.argmin(dist)
    return _NODE_IDS[idx] if dist[idx] <= RADIUS_KM else None