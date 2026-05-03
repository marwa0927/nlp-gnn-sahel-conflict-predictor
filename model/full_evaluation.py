"""
model/full_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import json
import importlib.util
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.dataset import SahelConflictDataset
from model.gcn import build_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2024-06-01"},
]


def load_best_config() -> dict:
    cfg_path = OUTPUTS_DIR / "best_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"window": 7, "hidden_dim": 64, "dropout": 0.3, "lr": 1e-3}


def load_nodes():
    nodes_py = ROOT / "data" / "nodes.py"
    spec = importlib.util.spec_from_file_location("nodes", nodes_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NODES


def safe(fn, y, p, **kw):
    try:
        return fn(y, p, **kw) if len(set(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_overall_metrics(probs_c, probs_u, labels_c, labels_u, threshold=0.5):
    preds_c = [1 if p > threshold else 0 for p in probs_c]
    preds_u = [1 if p > threshold else 0 for p in probs_u]
    return {
        "auroc_conflict": safe(roc_auc_score,          labels_c, probs_c),
        "ap_conflict":    safe(average_precision_score, labels_c, probs_c),
        "f1_conflict":    safe(f1_score, labels_c, preds_c, zero_division=0),
        "auroc_unrest":   safe(roc_auc_score,          labels_u, probs_u),
        "ap_unrest":      safe(average_precision_score, labels_u, probs_u),
        "f1_unrest":      safe(f1_score, labels_u, preds_u, zero_division=0),
        "pos_rate_c":     float(np.mean(labels_c)),
        "pos_rate_u":     float(np.mean(labels_u)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-node metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_per_node_metrics(
    all_probs_c, all_labels_c, nodes, N, n_samples
):
    """
    Reshape flat prediction lists back to (n_samples, N) and compute
    AUROC and AP per node.
    """
    probs_arr  = np.array(all_probs_c).reshape(n_samples, N)
    labels_arr = np.array(all_labels_c).reshape(n_samples, N)

    records = []
    for j, node in enumerate(nodes):
        p = probs_arr[:, j].tolist()
        l = labels_arr[:, j].tolist()
        records.append({
            "node_id":    node["id"],
            "country":    node["country"],
            "auroc":      safe(roc_auc_score,          l, p),
            "ap":         safe(average_precision_score, l, p),
            "pos_rate":   float(np.mean(l)),
            "n_positive": int(sum(l)),
        })

    df = pd.DataFrame(records).sort_values("auroc", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold analysis
# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold(labels_c, probs_c):
    """
    Find threshold that maximises F1 on the conflict head.
    Returns best threshold and its F1 score.
    """
    precision, recall, thresholds = precision_recall_curve(labels_c, probs_c)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0
    )
    best_idx  = np.argmax(f1_scores)
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1   = float(f1_scores[best_idx])
    return best_thr, best_f1, precision, recall, thresholds, f1_scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. Naive baseline
# ─────────────────────────────────────────────────────────────────────────────
def compute_baseline(labels_c, labels_u):
    """
    Majority-class baseline: always predict 0 (no conflict).
    Gives a lower bound — our model must beat this.
    """
    n = len(labels_c)
    preds_zero = [0] * n
    return {
        "baseline_f1_conflict": safe(f1_score, labels_c, preds_zero, zero_division=0),
        "baseline_f1_unrest":   safe(f1_score, labels_u, preds_zero, zero_division=0),
        "baseline_pos_rate_c":  float(np.mean(labels_c)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, title, ax):
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, color="white", fontsize=9)
    ax.set_xlabel("Predicted", color="#aaa", fontsize=8)
    ax.set_ylabel("Actual",    color="#aaa", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], color="#aaa")
    ax.set_yticklabels(["No", "Yes"], color="#aaa")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaa")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PR curve plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_curve(precision, recall, thresholds, f1_scores, best_thr, best_f1, ax):
    ax.plot(recall, precision, color="#E63946", linewidth=2, label="PR curve")
    best_idx = np.argmax(f1_scores)
    if best_idx < len(recall) - 1:
        ax.plot(recall[best_idx], precision[best_idx], "o",
                color="#F4A261", markersize=10,
                label=f"Best thr={best_thr:.2f} F1={best_f1:.3f}")
    ax.set_xlabel("Recall",    color="#aaa", fontsize=8)
    ax.set_ylabel("Precision", color="#aaa", fontsize=8)
    ax.set_title("Precision-Recall Curve (conflict)", color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="#aaa")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_full_evaluation():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg      = load_best_config()
    nodes    = load_nodes()
    N        = len(nodes)

    all_fold_results = []
    all_node_results = []

    for fold_cfg in SPLITS:
        fold_name  = fold_cfg["name"]
        ckpt_path  = CKPT_DIR / f"best_{fold_name}.pt"

        print(f"\n{'='*60}")
        print(f"  Evaluating: {fold_name}")
        print(f"{'='*60}")

        if not ckpt_path.exists():
            print(f"  ⚠️  No checkpoint — run train_best.py first.")
            continue

        test_ds = SahelConflictDataset(
            window=cfg.get("window", 7), horizon=1,
            split="test",
            val_start=fold_cfg["val_start"],
            test_start=fold_cfg["test_start"],
        )
        if len(test_ds) == 0:
            print("  ⚠️  No test samples for this fold.")
            continue

        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = build_model(
            window=cfg.get("window", 7),
            n_features=test_ds.F,
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.3),
        ).to(device)
        meta = load_checkpoint(model, ckpt_path)
        model.eval()

        # Collect predictions
        probs_c, probs_u, labels_c, labels_u = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch  = batch.to(device)
                logits = model.forward_logits(
                    batch.x, batch.edge_index, batch.edge_attr
                )
                p = torch.sigmoid(logits)
                probs_c.extend(p[:, 0].cpu().tolist())
                probs_u.extend(p[:, 1].cpu().tolist())
                labels_c.extend(batch.y[:, 0].cpu().tolist())
                labels_u.extend(batch.y[:, 1].cpu().tolist())

        n_samples = len(probs_c) // N

        # Overall metrics
        overall = compute_overall_metrics(probs_c, probs_u, labels_c, labels_u)
        overall["fold"] = fold_name
        all_fold_results.append(overall)

        print(f"\n  Overall metrics:")
        print(f"  {'Metric':30s} {'Conflict':>10} {'Unrest':>10}")
        print(f"  {'─'*55}")
        for metric in ["auroc", "ap", "f1"]:
            c = overall.get(f"{metric}_conflict", float("nan"))
            u = overall.get(f"{metric}_unrest",   float("nan"))
            fc = f"{c:.4f}" if not np.isnan(c) else "  nan "
            fu = f"{u:.4f}" if not np.isnan(u) else "  nan "
            print(f"  {metric.upper():30s} {fc:>10} {fu:>10}")

        # Baseline comparison
        baseline = compute_baseline(labels_c, labels_u)
        model_f1 = overall.get("f1_conflict", float("nan"))
        base_f1  = baseline["baseline_f1_conflict"]
        if not np.isnan(model_f1):
            margin = model_f1 - base_f1
            status = "✅" if margin > 0 else "❌"
            print(f"\n  {status} Model F1={model_f1:.4f}  vs  Baseline F1={base_f1:.4f}  "
                  f"(margin={margin:+.4f})")

        # Best threshold
        if len(set(labels_c)) > 1:
            best_thr, best_f1, precision, recall, thresholds, f1_scores = \
                find_best_threshold(labels_c, probs_c)
            print(f"\n  Best threshold: {best_thr:.3f}  "
                  f"→ F1={best_f1:.4f}  (default=0.5 gives F1={model_f1:.4f})")
        else:
            best_thr, best_f1 = 0.5, float("nan")
            precision, recall, thresholds, f1_scores = [], [], [], []

        # Per-node metrics
        if n_samples > 0:
            node_df = compute_per_node_metrics(
                probs_c, labels_c, nodes, N, n_samples
            )
            node_df["fold"] = fold_name
            all_node_results.append(node_df)

            print(f"\n  Per-node AUROC (top 10):")
            print(f"  {'Node':20s} {'Country':8s} {'AUROC':>8} {'AP':>8} "
                  f"{'PosRate':>8} {'N+':>6}")
            print(f"  {'─'*60}")
            for _, row in node_df.head(10).iterrows():
                auroc = f"{row['auroc']:.4f}" if not np.isnan(row["auroc"]) else "  nan "
                ap    = f"{row['ap']:.4f}"    if not np.isnan(row["ap"])    else "  nan "
                print(f"  {row['node_id']:20s} {row['country']:8s} "
                      f"{auroc:>8} {ap:>8} "
                      f"{row['pos_rate']:>8.4f} {row['n_positive']:>6}")

            # Flag weak nodes
            weak = node_df[
                (node_df["n_positive"] > 5) &
                (node_df["auroc"].notna()) &
                (node_df["auroc"] < 0.6)
            ]
            if len(weak) > 0:
                print(f"\n  ⚠️  Nodes with >5 events but AUROC < 0.6:")
                for _, row in weak.iterrows():
                    print(f"     {row['node_id']} (AUROC={row['auroc']:.4f})")

        # ── Plots ─────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 5))
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # Confusion matrix — conflict
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#0d0d1a")
        if len(set(labels_c)) > 1:
            preds_c = [1 if p > best_thr else 0 for p in probs_c]
            plot_confusion_matrix(labels_c, preds_c,
                                  f"{fold_name} — Conflict (thr={best_thr:.2f})", ax1)

        # Confusion matrix — unrest
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#0d0d1a")
        if len(set(labels_u)) > 1:
            preds_u = [1 if p > 0.5 else 0 for p in probs_u]
            plot_confusion_matrix(labels_u, preds_u,
                                  f"{fold_name} — Unrest (thr=0.5)", ax2)

        # PR curve
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor("#0d0d1a")
        if len(precision) > 0:
            plot_pr_curve(precision, recall, thresholds,
                          f1_scores, best_thr, best_f1, ax3)

        for ax in [ax1, ax2, ax3]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        plt.suptitle(f"Full Evaluation — {fold_name}", color="white", fontsize=12)
        plt.tight_layout()
        out_path = OUTPUTS_DIR / f"evaluation_{fold_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"\n  ✅ Plots saved → {out_path}")
        plt.close()

    # ── Cross-fold summary ────────────────────────────────────────────────────
    if all_fold_results:
        print(f"\n{'='*60}")
        print(f"  Cross-fold summary")
        print(f"{'='*60}")

        df_results = pd.DataFrame(all_fold_results)
        for metric in ["auroc_conflict", "ap_conflict", "f1_conflict"]:
            vals = df_results[metric].dropna().tolist()
            if vals:
                print(f"  {metric:30s}: "
                      f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

        # Save
        df_results.to_csv(OUTPUTS_DIR / "full_evaluation_report.csv", index=False)
        print(f"\n  ✅ Report → {OUTPUTS_DIR / 'full_evaluation_report.csv'}")

    # ── Node-level summary across folds ──────────────────────────────────────
    if all_node_results:
        df_nodes = pd.concat(all_node_results)
        avg_node = df_nodes.groupby("node_id")[["auroc", "ap"]].mean()
        avg_node = avg_node.sort_values("auroc", ascending=False)
        avg_node.to_csv(OUTPUTS_DIR / "per_node_metrics.csv")
        print(f"  ✅ Per-node metrics → {OUTPUTS_DIR / 'per_node_metrics.csv'}")

        print(f"\n  Average AUROC by node (across folds):")
        for nid, row in avg_node.iterrows():
            bar = "█" * int(row["auroc"] * 20) if not np.isnan(row["auroc"]) else ""
            print(f"    {nid:20s}  {row['auroc']:.4f}  {bar}")

    print(f"\n{'='*60}")
    mean_auroc = np.mean([r["auroc_conflict"] for r in all_fold_results
                          if not np.isnan(r.get("auroc_conflict", float("nan")))])
    if np.isnan(mean_auroc):
        print("  ⚠️  Could not compute mean AUROC.")
    elif mean_auroc >= 0.65:
        print(f"  ✅ Mean AUROC={mean_auroc:.4f} — model has signal.")
    elif mean_auroc >= 0.55:
        print(f"  ⚠️  Mean AUROC={mean_auroc:.4f} — borderline.")
        print("     Try: hidden_dim=128, check zero-coverage Niger nodes.")
    else:
        print(f"  ❌ Mean AUROC={mean_auroc:.4f} — below random.")
        print("     Check label quality and graph connectivity.")


if __name__ == "__main__":
    run_full_evaluation()
