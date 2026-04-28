"""
Geo-filter ACLED events to 25 Sahel nodes and build daily binary labels.

ACLED columns used: event_date, event_type, latitude, longitude, fatalities
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.nodes import NODES
from config import RADIUS_KM

ACLED_RAW   = ROOT / "data" / "raw" / "acled_sahel_raw.csv"
LABELS_OUT  = ROOT / "data" / "processed" / "labels_daily.parquet"

CONFLICT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
]
UNREST_TYPES = [
    "Protests",
    "Riots",
    "Strategic developments",
]

_NODE_IDS  = [n["id"] for n in NODES]
_NODE_LATS = np.array([n["lat"] for n in NODES])
_NODE_LONS = np.array([n["lon"] for n in NODES])


def _assign_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised haversine node assignment (mirrors ingest/geo_filter.py logic)."""
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"Assigning {len(df):,} rows to nodes...")

    R = 6371.0
    lat1 = np.radians(df["latitude"].values)[:, None]   # (N, 1)
    lon1 = np.radians(df["longitude"].values)[:, None]  # (N, 1)
    lat2 = np.radians(_NODE_LATS)[None, :]               # (1, 25)
    lon2 = np.radians(_NODE_LONS)[None, :]               # (1, 25)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = R * 2 * np.arcsin(np.sqrt(a))                 # (N, 25)

    nearest_idx  = np.argmin(dist, axis=1)               # (N,)
    nearest_dist = dist[np.arange(len(df)), nearest_idx]

    node_ids = np.array(_NODE_IDS)
    df["node_id"] = np.where(nearest_dist <= RADIUS_KM, node_ids[nearest_idx], None)

    assigned = df["node_id"].notna().sum()
    print(f"Assigned {assigned:,} / {len(df):,} ({assigned / len(df):.1%})")
    return df.dropna(subset=["node_id"])


def build_labels() -> pd.DataFrame:
    print(f"Reading {ACLED_RAW} ...")
    raw = pd.read_csv(ACLED_RAW, parse_dates=["event_date"], low_memory=False)
    print(f"Raw rows: {len(raw):,}  |  columns: {list(raw.columns)}")

    df = _assign_nodes(raw)

    df["is_conflict"] = df["event_type"].isin(CONFLICT_TYPES).astype(np.int8)
    df["is_unrest"]   = df["event_type"].isin(UNREST_TYPES).astype(np.int8)

    agg = (
        df.groupby(["node_id", "event_date"])
        .agg(
            y_conflict=("is_conflict", "max"),
            y_unrest=("is_unrest",   "max"),
            fatalities=("fatalities", "sum"),
        )
        .reset_index()
        .rename(columns={"event_date": "date"})
    )

    # Full grid: 25 nodes × every calendar day 2018–2024
    all_dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    idx = pd.MultiIndex.from_product([_NODE_IDS, all_dates], names=["node_id", "date"])
    full = (
        agg.set_index(["node_id", "date"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )
    full["y_conflict"] = full["y_conflict"].astype(np.int8)
    full["y_unrest"]   = full["y_unrest"].astype(np.int8)
    full["fatalities"] = full["fatalities"].astype(np.int32)
    return full


def main() -> None:
    labels = build_labels()

    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS_OUT, index=False)

    print(f"\nSaved → {LABELS_OUT}")
    print(f"Shape:  {labels.shape}")

    pct_c = labels["y_conflict"].mean() * 100
    pct_u = labels["y_unrest"].mean() * 100
    print(f"Class balance — conflict=1: {pct_c:.2f}%  |  unrest=1: {pct_u:.2f}%")

    top5 = (
        labels.sort_values("fatalities", ascending=False)
        .head(5)[["node_id", "date", "fatalities"]]
    )
    print("\nTop 5 (node, date) by fatalities:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
