"""
notebooks/plot_training.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def plot_training_curves():
    log_path = OUTPUTS_DIR / "training_log_best.csv"
    if not log_path.exists():
        print(f"❌ {log_path} not found. Run train_best.py first.")
        return

    df = pd.read_csv(log_path)
    folds = df["fold"].unique()
    n_folds = len(folds)

    fig = plt.figure(figsize=(16, 4 * n_folds))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(n_folds, 3, figure=fig, hspace=0.4, wspace=0.35)

    COLORS = {"train_loss": "#E63946", "val_loss": "#F4A261",
              "auroc_c": "#2A9D8F", "ap_c": "#457B9D"}

    for row, fold in enumerate(folds):
        fold_df = df[df["fold"] == fold].copy()

        # ── Loss curves ───────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.set_facecolor("#0d0d1a")
        ax1.plot(fold_df["epoch"], fold_df["train_loss"],
                 color=COLORS["train_loss"], label="Train loss", linewidth=2)
        ax1.plot(fold_df["epoch"], fold_df["val_loss"],
                 color=COLORS["val_loss"],  label="Val loss",   linewidth=2)
        ax1.set_title(f"{fold} — Loss", color="white", fontsize=10)
        ax1.set_xlabel("Epoch", color="#aaa", fontsize=8)
        ax1.tick_params(colors="#aaa")
        ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#333")

        # ── AUROC curve ───────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor("#0d0d1a")
        if "auroc_c" in fold_df.columns and fold_df["auroc_c"].notna().any():
            ax2.plot(fold_df["epoch"], fold_df["auroc_c"],
                     color=COLORS["auroc_c"], linewidth=2, label="AUROC (conflict)")
            ax2.axhline(y=0.65, color="white", linestyle="--",
                        alpha=0.4, label="Target (0.65)")
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, "No labels yet\n(AUROC = nan)",
                     ha="center", va="center", color="#aaa",
                     transform=ax2.transAxes, fontsize=9)
        ax2.set_title(f"{fold} — AUROC (conflict)", color="white", fontsize=10)
        ax2.set_xlabel("Epoch", color="#aaa", fontsize=8)
        ax2.tick_params(colors="#aaa")
        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")

        # ── AP curve ─────────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.set_facecolor("#0d0d1a")
        if "ap_c" in fold_df.columns and fold_df["ap_c"].notna().any():
            ax3.plot(fold_df["epoch"], fold_df["ap_c"],
                     color=COLORS["ap_c"], linewidth=2, label="AP (conflict)")
            ax3.set_ylim(0, 1)
        else:
            ax3.text(0.5, 0.5, "No labels yet\n(AP = nan)",
                     ha="center", va="center", color="#aaa",
                     transform=ax3.transAxes, fontsize=9)
        ax3.set_title(f"{fold} — Average Precision (conflict)", color="white", fontsize=10)
        ax3.set_xlabel("Epoch", color="#aaa", fontsize=8)
        ax3.tick_params(colors="#aaa")
        ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle("Sahel GNN — Training Curves (Best Config)",
                 color="white", fontsize=13, y=1.01)

    out_path = OUTPUTS_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"✅ Training curves saved → {out_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_training_curves()
