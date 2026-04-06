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
    agg = agg.reindex(full_idx, fill_value=0.0)

    return agg


def add_lag_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column in BASE_FEATURES add:
      - {col}_lag7   — 7-day lag
      - {col}_roll7  — 7-day rolling mean
      - {col}_roll14 — 14-day rolling mean

    Computed per node_id group; NaNs introduced at group boundaries are
    filled with 0.
    """
    df = daily_df.sort_index(level=["node_id", "date"]).copy()

    new_cols: dict[str, pd.Series] = {}

    for col in BASE_FEATURES:
        if col not in df.columns:
            continue

        grouped = df[col].groupby(level="node_id")

        new_cols[f"{col}_lag7"] = grouped.shift(7)
        new_cols[f"{col}_roll7"] = grouped.transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
        new_cols[f"{col}_roll14"] = grouped.transform(
            lambda s: s.rolling(14, min_periods=1).mean()
        )

    lag_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, lag_df], axis=1).fillna(0.0)

    return df


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
