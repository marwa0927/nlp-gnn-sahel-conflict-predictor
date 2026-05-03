"""
Walk-forward cross-validation splits for the Sahel conflict predictor.

Use these exact splits for reproducibility. Custom splitting may lead to data leakage.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.nodes import NODES

SPLITS = [
    {
        "train_start": "2020-01-01",
        "train_end":   "2021-12-31",
        "val_start":   "2022-01-01",
        "val_end":     "2022-12-31",
    },
    {
        "train_start": "2020-01-01",
        "train_end":   "2022-12-31",
        "val_start":   "2023-01-01",
        "val_end":     "2023-12-31",
    },
    {
        "train_start": "2020-01-01",
        "train_end":   "2023-12-31",
        "val_start":   "2024-01-01",
        "val_end":     "2024-12-31",
    },
]


def get_split_data(
    df: pd.DataFrame, split: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_train, df_val) filtered to the split's date ranges."""
    date = pd.to_datetime(df["date"])
    train = df[(date >= split["train_start"]) & (date <= split["train_end"])]
    val   = df[(date >= split["val_start"])   & (date <= split["val_end"])]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def get_node_order() -> list[str]:
    """Canonical list of 25 node ids in NODES declaration order.

    Use this to align X matrix rows with adjacency matrix rows.
    """
    return [n["id"] for n in NODES]
