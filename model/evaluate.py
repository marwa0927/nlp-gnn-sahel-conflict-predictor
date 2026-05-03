"""
model/evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, classification_report
)
from torch_geometric.loader import DataLoader

from model.dataset import SahelConflictDataset
from model.gcn import ConflictGCN, build_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2025-01-01"},
]



def evaluate_model(
    predict_fn,           # callable: (x, edge_index, edge_attr) → probs (N, 2)
    loader,               # DataLoader
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    """
    Evaluate any prediction function against ground truth labels.

    predict_fn must accept (x, edge_index, edge_attr) and return
    a tensor of shape (N, 2) with probabilities in [0, 1].

    This signature is flexible for wrapping different models.
    """
    all_probs_c,  all_probs_u  = [], []
    all_labels_c, all_labels_u = [], []

    for batch in loader:
        with torch.no_grad():
            probs = predict_fn(batch.x, batch.edge_index, batch.edge_attr)

        all_probs_c.extend(probs[:, 0].cpu().tolist())
        all_probs_u.extend(probs[:, 1].cpu().tolist())
        all_labels_c.extend(batch.y[:, 0].cpu().tolist())
        all_labels_u.extend(batch.y[:, 1].cpu().tolist())

    def safe(fn, labels, preds, **kwargs): # Add **kwargs here
        try:
            if len(set(labels)) < 2:
                return float("nan")
            return fn(labels, preds, **kwargs) # Pass **kwargs to the function
        except Exception:
            return float("nan")

    preds_c = [1 if p > threshold else 0 for p in all_probs_c]
    preds_u = [1 if p > threshold else 0 for p in all_probs_u]

    metrics = {
        "label":   label,
        "n_samples": len(all_labels_c),

        # Conflict head
        "auroc_conflict": safe(roc_auc_score,           all_labels_c, all_probs_c),
        "ap_conflict":    safe(average_precision_score,  all_labels_c, all_probs_c),
        "f1_conflict":    safe(f1_score, all_labels_c, preds_c, zero_division=0),

        # Unrest head
        "auroc_unrest":   safe(roc_auc_score,           all_labels_u, all_probs_u),
        "ap_unrest":      safe(average_precision_score,  all_labels_u, all_probs_u),
        "f1_unrest":      safe(f1_score, all_labels_u, preds_u, zero_division=0),

        # Label rates (sanity check)
        "pos_rate_conflict": float(np.mean(all_labels_c)),
        "pos_rate_unrest":   float(np.mean(all_labels_u)),
    }
    return metrics, all_probs_c, all_probs_u, all_labels_c, all_labels_u


# ─────────────────────────────────────────────────────────────────────────────
# Print metrics table
# ─────────────────────────────────────────────────────────────────────────────
def print_metrics(metrics: dict):
    def fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "  nan  "
        return f"{v:.4f}"

    print(f"\n  {'─'*50}")
    print(f"  Evaluation: {metrics['label']}  (n={metrics['n_samples']})")
    print(f"  {'─'*50}")
    print(f"  {'Metric':25s}  {'Conflict':>10}  {'Unrest':>10}")
    print(f"  {'─'*50}")
    print(f"  {'AUROC':25s}  {fmt(metrics['auroc_conflict']):>10}  {fmt(metrics['auroc_unrest']):>10}")
    print(f"  {'Avg Precision (AP)':25s}  {fmt(metrics['ap_conflict']):>10}  {fmt(metrics['ap_unrest']):>10}")
    print(f"  {'F1':25s}  {fmt(metrics['f1_conflict']):>10}  {fmt(metrics['f1_unrest']):>10}")
    print(f"  {'Positive label rate':25s}  {fmt(metrics['pos_rate_conflict']):>10}  {fmt(metrics['pos_rate_unrest']):>10}")
    print(f"  {'─'*50}")

    if np.isnan(metrics["auroc_conflict"]):
        print("\n  ⚠️  Metrics are nan — labels are all zero.")
        print("     AP and AUROC will be meaningful once ACLED labels arrive.")
    elif metrics["auroc_conflict"] < 0.55:
        print("\n  ⚠️  AUROC < 0.55 — model barely above random.")
        print("     Check: are labels correct? Is the graph connected?")
    elif metrics["auroc_conflict"] >= 0.65:
        print(f"\n  ✅ AUROC ≥ 0.65 — model has signal worth tuning.")


# ─────────────────────────────────────────────────────────────────────────────
# Save predictions CSV (input for risk map)
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions(
    dataset: SahelConflictDataset,
    probs_c: list,
    probs_u: list,
    fold_name: str,
):
    """
    Save per-node, per-date predictions to a CSV for visualization.
    """
    node_ids = [n["id"] for n in dataset.nodes]
    N = len(node_ids)
    records = []

    n_samples = len(probs_c) // N
    for s in range(n_samples):
        idx = dataset.sample_indices[s] if s < len(dataset.sample_indices) else s
        date = dataset.dates[min(idx + dataset.horizon - 1, len(dataset.dates) - 1)]
        for j, nid in enumerate(node_ids):
            flat_idx = s * N + j
            if flat_idx < len(probs_c):
                records.append({
                    "date":       date,
                    "node_id":    nid,
                    "p_conflict": round(probs_c[flat_idx], 4),
                    "p_unrest":   round(probs_u[flat_idx], 4),
                    "risk_level": (
                        "high"   if probs_c[flat_idx] > 0.6 else
                        "medium" if probs_c[flat_idx] > 0.3 else
                        "low"
                    ),
                })

    df = pd.DataFrame(records)
    out_path = OUTPUTS_DIR / f"predictions_{fold_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✅ Predictions saved → {out_path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Run one fold evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_fold_evaluation(fold_cfg: dict, device: torch.device) -> dict:
    fold_name  = fold_cfg["name"]
    val_start  = fold_cfg["val_start"]
    test_start = fold_cfg["test_start"]
    ckpt_path  = CKPT_DIR / f"best_{fold_name}.pt"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {fold_name}")
    print(f"{'='*60}")

    if not ckpt_path.exists():
        print(f"  ⚠️  No checkpoint found at {ckpt_path}")
        print(f"     Run python model/train.py first.")
        return {}

    # Load test dataset
    test_ds = SahelConflictDataset(
        window=7, horizon=1,
        split="test", val_start=val_start, test_start=test_start
    )

    if len(test_ds) == 0:
        print(f"  ⚠️  No test samples for {fold_name} — check date range.")
        return {}

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load model from checkpoint
    model = build_model(window=7, n_features=test_ds.F).to(device)
    meta  = load_checkpoint(model, ckpt_path)
    model.eval()
    print(f"  Loaded checkpoint (epoch={meta.get('epoch', '?')}, "
          f"val_loss={meta.get('val_loss', '?'):.4f})")

    # Build predict_fn
    def predict_fn(x, edge_index, edge_attr):
        x          = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)
        logits     = model.forward_logits(x, edge_index, edge_attr)
        return torch.sigmoid(logits)

    # Evaluate
    metrics, probs_c, probs_u, labels_c, labels_u = evaluate_model(
        predict_fn, test_loader, label=fold_name
    )
    print_metrics(metrics)

    # Save predictions CSV
    save_predictions(test_ds, probs_c, probs_u, fold_name)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold name to evaluate (fold_1, fold_2, fold_3). "
                             "Omit to evaluate all folds.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    folds_to_run = [f for f in SPLITS if args.fold is None or f["name"] == args.fold]
    if not folds_to_run:
        print(f"❌ Unknown fold '{args.fold}'. Choose from: "
              f"{[f['name'] for f in SPLITS]}")
        return

    all_metrics = []
    for fold_cfg in folds_to_run:
        m = run_fold_evaluation(fold_cfg, device)
        if m:
            all_metrics.append(m)

    # Summary across folds
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  Walk-forward summary ({len(all_metrics)} folds)")
        print(f"{'='*60}")
        for key in ["auroc_conflict", "ap_conflict", "f1_conflict"]:
            vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float("nan")))]
            if vals:
                print(f"  {key:30s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # Save summary JSON
    summary_path = OUTPUTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n✅ Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
