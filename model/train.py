"""
model/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import json
import csv
from datetime import datetime

from model.dataset import SahelConflictDataset
from model.gcn import ConflictGCN, build_model, save_checkpoint

ROOT         = Path(__file__).parent.parent
CKPT_DIR     = ROOT / "checkpoints"
OUTPUTS_DIR  = ROOT / "outputs"
CKPT_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    "window":     7,       # days of history as input
    "horizon":    1,       # days ahead to predict
    "hidden_dim": 64,      # GCN hidden dimension
    "dropout":    0.3,
    "lr":         1e-3,
    "weight_decay": 1e-4,
    "epochs":     50,
    "batch_size": 16,
    "threshold":  0.5,     # decision threshold for binary metrics
}

# Walk-forward splits — train on past, validate on next year
SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2025-01-01"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def build_loss(train_dataset: SahelConflictDataset):
    """
    Weighted BCEWithLogitsLoss to handle class imbalance.
    Weights are computed from actual label distribution in training set.
    Falls back to fixed weights if labels are all zero (pre-ACLED).
    """
    y_all = torch.stack(
        [train_dataset.get(i).y for i in range(min(len(train_dataset), 200))],
        dim=0
    )  # sample up to 200 to avoid slowness

    n_pos_c = y_all[:, :, 0].sum().item()
    n_pos_u = y_all[:, :, 1].sum().item()
    total   = y_all[:, :, 0].numel()

    if n_pos_c == 0 or n_pos_u == 0:
        print("⚠️  No positive labels found — using default weights (10x, 5x).")
        print("   This is expected when labels are not yet available.")
        w_conflict = torch.tensor([10.0])
        w_unrest   = torch.tensor([5.0])
    else:
        w_conflict = torch.tensor([(total - n_pos_c) / n_pos_c])
        w_unrest   = torch.tensor([(total - n_pos_u) / n_pos_u])

    print(f"  pos_weight_conflict: {w_conflict.item():.2f}")
    print(f"  pos_weight_unrest:   {w_unrest.item():.2f}")

    loss_conflict = nn.BCEWithLogitsLoss(pos_weight=w_conflict)
    loss_unrest   = nn.BCEWithLogitsLoss(pos_weight=w_unrest)
    return loss_conflict, loss_unrest


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_c, loss_u, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model.forward_logits(
            batch.x, batch.edge_index, batch.edge_attr
        )  # (batch_N, 2)

        lc = loss_c(logits[:, 0], batch.y[:, 0])
        lu = loss_u(logits[:, 1], batch.y[:, 1])
        loss = lc + lu

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, loss_c, loss_u, device, threshold=0.5):
    model.eval()
    total_loss  = 0.0
    all_probs_c, all_probs_u = [], []
    all_labels_c, all_labels_u = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model.forward_logits(
                batch.x, batch.edge_index, batch.edge_attr
            )
            lc = loss_c(logits[:, 0], batch.y[:, 0])
            lu = loss_u(logits[:, 1], batch.y[:, 1])
            total_loss += (lc + lu).item()

            probs = torch.sigmoid(logits)
            all_probs_c.extend(probs[:, 0].cpu().tolist())
            all_probs_u.extend(probs[:, 1].cpu().tolist())
            all_labels_c.extend(batch.y[:, 0].cpu().tolist())
            all_labels_u.extend(batch.y[:, 1].cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)

    # Only compute AUROC/AP if there are positive labels
    def safe_auroc(labels, probs):
        if len(set(labels)) < 2:
            return float("nan")
        return roc_auc_score(labels, probs)

    def safe_ap(labels, probs):
        if len(set(labels)) < 2:
            return float("nan")
        return average_precision_score(labels, probs)

    def safe_f1(labels, probs, thr):
        preds = [1 if p > thr else 0 for p in probs]
        if len(set(labels)) < 2:
            return float("nan")
        return f1_score(labels, preds, zero_division=0)

    metrics = {
        "loss":       avg_loss,
        "auroc_c":    safe_auroc(all_labels_c,  all_probs_c),
        "ap_c":       safe_ap(all_labels_c,      all_probs_c),
        "f1_c":       safe_f1(all_labels_c,      all_probs_c, threshold),
        "auroc_u":    safe_auroc(all_labels_u,   all_probs_u),
        "ap_u":       safe_ap(all_labels_u,       all_probs_u),
        "f1_u":       safe_f1(all_labels_u,       all_probs_u, threshold),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Print metrics row
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v):
    return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "  nan "

def print_metrics(epoch, train_loss, val_metrics):
    print(
        f"  Epoch {epoch:3d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={fmt(val_metrics['loss'])} | "
        f"AUROC_c={fmt(val_metrics['auroc_c'])} | "
        f"AP_c={fmt(val_metrics['ap_c'])} | "
        f"F1_c={fmt(val_metrics['f1_c'])}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# One full fold
# ─────────────────────────────────────────────────────────────────────────────
def run_fold(fold_cfg: dict, config: dict, device: torch.device, log_writer) -> dict:
    fold_name  = fold_cfg["name"]
    val_start  = fold_cfg["val_start"]
    test_start = fold_cfg["test_start"]

    print(f"\n{'='*60}")
    print(f"  Fold: {fold_name}  |  val_start={val_start}")
    print(f"{'='*60}")

    # Datasets
    train_ds = SahelConflictDataset(
        window=config["window"], horizon=config["horizon"],
        split="train", val_start=val_start, test_start=test_start
    )
    val_ds = SahelConflictDataset(
        window=config["window"], horizon=config["horizon"],
        split="val", val_start=val_start, test_start=test_start
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"⚠️  Skipping {fold_name} — not enough data for this split.")
        return {}

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)

    # Model
    model = build_model(
        window=config["window"],
        n_features=train_ds.F,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    loss_c, loss_u = build_loss(train_ds)
    loss_c = loss_c.to(device)
    loss_u = loss_u.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training loop
    best_val_loss = float("inf")
    best_ckpt     = CKPT_DIR / f"best_{fold_name}.pt"
    best_metrics  = {}

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'AUROC_c':>8}  {'AP_c':>8}")
    print(f"  {'─'*55}")

    for epoch in range(1, config["epochs"] + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, loss_c, loss_u, device)
        val_metrics = evaluate(model, val_loader, loss_c, loss_u, device, config["threshold"])
        scheduler.step()

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics  = val_metrics
            save_checkpoint(model, best_ckpt, metadata={
                "fold": fold_name, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "config": config,
            })

        # Log every epoch
        log_writer.writerow({
            "fold":       fold_name,
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            **{k: round(v, 6) if not np.isnan(v) else None
               for k, v in val_metrics.items()},
            "timestamp":  datetime.now().isoformat(),
        })

        # Print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            auroc = fmt(val_metrics["auroc_c"])
            ap    = fmt(val_metrics["ap_c"])
            print(f"  {epoch:5d}  {train_loss:10.4f}  "
                  f"{val_metrics['loss']:10.4f}  {auroc:>8}  {ap:>8}")

    print(f"\n  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint → {best_ckpt}")
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}\n")

    log_path = OUTPUTS_DIR / "training_log.csv"
    log_fields = [
        "fold", "epoch", "train_loss",
        "loss", "auroc_c", "ap_c", "f1_c",
        "auroc_u", "ap_u", "f1_u", "timestamp"
    ]

    all_fold_metrics = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for fold_cfg in SPLITS:
            metrics = run_fold(fold_cfg, CONFIG, device, writer)
            if metrics:
                all_fold_metrics.append(metrics)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Walk-forward summary ({len(all_fold_metrics)} folds)")
    print(f"{'='*60}")
    if all_fold_metrics:
        auroc_vals = [m["auroc_c"] for m in all_fold_metrics
                      if not np.isnan(m.get("auroc_c", float("nan")))]
        ap_vals    = [m["ap_c"]    for m in all_fold_metrics
                      if not np.isnan(m.get("ap_c",    float("nan")))]
        if auroc_vals:
            print(f"  Mean AUROC (conflict): {np.mean(auroc_vals):.4f}")
            print(f"  Mean AP    (conflict): {np.mean(ap_vals):.4f}")
        else:
            print("  Metrics are nan — labels are all zero (waiting for ACLED).")
            print("  Training loss behaviour is still valid to inspect.")

    print(f"\n✅ Training complete. Log → {log_path}")
    print(f"   Checkpoints → {CKPT_DIR}/")


if __name__ == "__main__":
    main()
