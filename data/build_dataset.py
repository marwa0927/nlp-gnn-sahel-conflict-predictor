"""
Merge features_daily.parquet (2020–2024) and labels_daily.parquet (2018–2024)
into the unified training file dataset_daily.parquet (2020–2024).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES_PATH = ROOT / "data" / "processed" / "features_daily.parquet"
LABELS_PATH   = ROOT / "data" / "processed" / "labels_daily.parquet"
DATASET_PATH  = ROOT / "data" / "processed" / "dataset_daily.parquet"

LABEL_COLS = ["y_conflict", "y_unrest", "fatalities"]


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    labels   = pd.read_parquet(LABELS_PATH)

    # Drop leftover index column if present
    features = features.drop(columns=["index"], errors="ignore")

    # Cast all feature columns to float32
    feature_cols = [c for c in features.columns if c not in ["node_id", "date"]]
    features[feature_cols] = features[feature_cols].astype("float32")

    # Inner join on (node_id, date) — result is 2020–2024 intersection
    merged = features.merge(
        labels[["node_id", "date"] + LABEL_COLS],
        on=["node_id", "date"],
        how="inner",
    )

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(DATASET_PATH, index=False)

    print(f"Saved → {DATASET_PATH}")
    print(f"Shape:  {merged.shape}")
    print(f"Nulls:  {merged.isnull().sum().sum()}")
    print(f"Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")


if __name__ == "__main__":
    main()
