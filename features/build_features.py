"""
Aggregate geo-filtered GDELT events into a daily node-level feature matrix
and compute lag / rolling features.
"""

import pathlib

import numpy as np
import pandas as pd

from config import BASE_FEATURES, FEATURES_PATH
from data.nodes import NODES


# ── helpers ───────────────────────────────────────────────────────────────────

def _full_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Build a complete (node_id × date) MultiIndex for *df*'s date range."""
    all_nodes = [n["id"] for n in NODES]
    all_dates = pd.date_range(df.index.get_level_values("date").min(),
                              df.index.get_level_values("date").max(),
                              freq="D")
    return pd.MultiIndex.from_product([all_nodes, all_dates],
                                      names=["node_id", "date"])


# ── public API ────────────────────────────────────────────────────────────────

def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate *df* (output of :func:`~ingest.geo_filter.assign_nodes_to_df`)
    to one row per (node_id, date).

    Missing (node, date) pairs are filled with 0.0 — not NaN.
    """
    agg = (
        df.groupby(["node_id", "date"])
        .agg(
            avg_tone=("avg_tone", "mean"),
            goldstein=("goldstein", "mean"),
            n_articles=("n_articles", "sum"),
            conflict_events=("is_conflict", "sum"),
            unrest_events=("is_unrest", "sum"),
            n_actors=("actor1", "nunique"),
        )
    )

    # Reindex to full grid, fill missing with 0
    full_idx = _full_index(agg)
    full_grid = pd.DataFrame(full_idx.tolist(), columns=["node_id", "date"])
    agg = agg.reset_index()
    agg = full_grid.merge(agg, on=["node_id", "date"], how="left").fillna(0.0)

    return agg


def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    from config import BASE_FEATURES

    # Reset to clean integer index before any sorting
    daily = daily.copy().reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["node_id", "date"]).reset_index(drop=True)

    result = []
    for node_id, grp in daily.groupby("node_id", sort=False):
        grp = grp.copy().reset_index(drop=True)
        for col in BASE_FEATURES:
            grp[f"{col}_lag7"]   = grp[col].shift(7).fillna(0.0)
            grp[f"{col}_roll7"]  = grp[col].rolling(7,  min_periods=1).mean()
            grp[f"{col}_roll14"] = grp[col].rolling(14, min_periods=1).mean()
        result.append(grp)

    return pd.concat(result, ignore_index=True)

def save_features(df: pd.DataFrame) -> None:
    """Persist *df* to FEATURES_PATH as parquet."""
    out_path = pathlib.Path(FEATURES_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(out_path, index=False)
    print(f"Saved features → {out_path}  shape={df.shape}")


if __name__ == "__main__":
    import sys

    raw_path = pathlib.Path("data/raw/gdelt_sahel_raw.parquet")
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run ingest/gdelt_pull.py first.")
        sys.exit(1)

    from ingest.geo_filter import assign_nodes_to_df

    raw = pd.read_parquet(raw_path)
    raw = assign_nodes_to_df(raw)

    daily = build_daily_features(raw)
    daily = add_lag_features(daily)
    save_features(daily)

    print("\n── Feature matrix summary ──────────────────────────")
    print(f"  Shape : {daily.shape}")
    print(f"  Columns: {list(daily.columns)}")
    print(f"  Missing: {daily.isna().sum().sum()}")
