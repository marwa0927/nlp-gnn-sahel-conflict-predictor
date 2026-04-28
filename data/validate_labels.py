"""
Validation plots for ACLED labels: conflict timelines and per-country imbalance.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.nodes import NODES

LABELS_PATH = ROOT / "data" / "processed" / "labels_daily.parquet"
OUTPUTS_DIR = ROOT / "outputs"

COUNTRY_MAP = {n["id"]: n["country"] for n in NODES}
COUNTRY_NAMES = {"ML": "Mali", "NI": "Niger", "UV": "Burkina Faso", "CD": "Chad", "MR": "Mauritania"}


def _plot_conflict(labels: pd.DataFrame, node: str, outpath: Path) -> None:
    sub = labels[labels["node_id"] == node].sort_values("date")
    weekly = sub.set_index("date")["y_conflict"].resample("W").max()

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(weekly.index, weekly.values, alpha=0.65, color="crimson")
    ax.set_title(f"y_conflict — {node} (weekly max, 2018–2024)")
    ax.set_ylabel("conflict flag")
    ax.set_xlabel("date")
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"Saved {outpath}")


def main() -> None:
    labels = pd.read_parquet(LABELS_PATH)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    _plot_conflict(labels, "ndjamena",    OUTPUTS_DIR / "validation_ndjamena_conflict.png")
    _plot_conflict(labels, "ouagadougou", OUTPUTS_DIR / "validation_ouaga_conflict.png")

    labels["country"] = labels["node_id"].map(COUNTRY_MAP)
    labels["country_name"] = labels["country"].map(COUNTRY_NAMES)

    print("\nClass imbalance per country (% of node-days with event=1):")
    stats = (
        labels.groupby("country_name")[["y_conflict", "y_unrest"]]
        .mean()
        .mul(100)
        .rename(columns={"y_conflict": "conflict% days", "y_unrest": "unrest% days"})
        .sort_index()
    )
    print(stats.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
