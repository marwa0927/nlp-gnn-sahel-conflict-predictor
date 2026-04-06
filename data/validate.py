"""
Validation script for the daily feature matrix.

Usage:
    python data/validate.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

FEATURES_PATH = pathlib.Path("data/processed/features_daily.parquet")
OUTPUT_DIR = pathlib.Path("outputs")


def main() -> None:
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found. Run features/build_features.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)

    # ── 1. Shape & dtypes ─────────────────────────────────────────────────────
    print("── Shape ───────────────────────────────────────────────")
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n── dtypes ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    # ── 2. Missing value counts ───────────────────────────────────────────────
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\n── Missing values ──────────────────────────────────────")
    if missing.empty:
        print("  None — all cells filled.")
    else:
        print(missing.to_string())

    # ── 3. Gao avg_tone over time ─────────────────────────────────────────────
    gao = df[df["node_id"] == "gao"].copy()
    if gao.empty:
        print("\nWARN: no rows for node 'gao' — skipping plot.")
    else:
        gao["date"] = pd.to_datetime(gao["date"])
        gao = gao.sort_values("date")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(gao["date"], gao["avg_tone"], linewidth=0.8, color="steelblue",
                label="avg_tone")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

        # Annotate known conflict periods
        conflict_dates = {
            "Mali coup\nMay 2021": "2021-05-24",
            "BF coup\nJan 2022": "2022-01-24",
            "BF coup\nSep 2022": "2022-09-30",
        }
        for label, date_str in conflict_dates.items():
            ax.axvline(pd.Timestamp(date_str), color="red", linewidth=1,
                       linestyle=":", alpha=0.7)
            ax.text(pd.Timestamp(date_str), ax.get_ylim()[1] * 0.9, label,
                    fontsize=7, color="red", ha="center")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")
        ax.set_title("Gao — avg_tone (2020–2024)\n"
                     "Expected dips around 2021 Mali coup and 2022 Burkina coups")
        ax.set_xlabel("Date")
        ax.set_ylabel("avg_tone")
        ax.legend()
        plt.tight_layout()

        out_png = OUTPUT_DIR / "validation_gao_tone.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"\nPlot saved → {out_png}")

    # ── 4. Top-5 conflict events ──────────────────────────────────────────────
    print("\n── Top 5 (node, date) by conflict_events ───────────────")
    top5 = (
        df[["node_id", "date", "conflict_events"]]
        .sort_values("conflict_events", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
