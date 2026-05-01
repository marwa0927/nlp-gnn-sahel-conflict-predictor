"""
model/tune.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import json
import csv
import itertools
from datetime import datetime

from model.dataset import SahelConflictDataset
from model.gcn import build_model, save_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
CKPT_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# Keep it small — goal is finding direction, not exhaustive search
# ─────────────────────────────────────────────────────────────────────────────
GRID = {
    "window":     [7, 14],
    "hidden_dim": [64, 128],
    "dropout":    [0.1, 0.3, 0.5],
    "lr":         [1e-3, 5e-4],
}

# Fixed for all runs
FIXED = {
    "horizon":      1,
    "weight_decay": 1e-4,
    "epochs":       30,        # shorter for sweep — full 50 epochs in final train
    "batch_size":   16,
    "threshold":    0.5,
    "val_start":    "2023-01-01",   # use fold_2 for tuning
    "test_start":   "2024-01-01",
}

LOG_FIELDS = [
    "run_id", "window", "hidden_dim", "dropout", "lr",
    "best_val_loss", "best_val_auroc_c", "best_val_ap_c",
    "best_val_auroc_u", "best_val_ap_u",
    "epochs_run", "timestamp",
]


# ─────────────────────────────────────────────────────────────────────────────
# Build weighted loss from actual label distribution
# ─────────────────────────────────────────────────────────────────────────────
def build_loss(train_ds):
    y_sample = torch.stack(
        [train_ds.get(i).y for i in range(min(300, len(train_ds)))], dim=0
    )
    n_pos_c = y_sample[:, :, 0].sum().item()
    n_pos_u = y_sample[:, :, 1].sum().item()
    total   = y_sample[:, :, 0].numel()

    if n_pos_c == 0:
        print("  ⚠️  y_conflict still all zero — check parquet labels.")
        w_c = torch.tensor([10.0])
    else:
        w_c = torch.tensor([(total - n_pos_c) / n_pos_c])

    if n_pos_u == 0:
        print("  ⚠️  y_unrest still all zero — check parquet labels.")
        w_u = torch.tensor([5.0])
    else:
        w_u = torch.tensor([(total - n_pos_u) / n_pos_u])

    print(f"  pos_weight_conflict={w_c.item():.2f}  pos_weight_unrest={w_u.item():.2f}")
    return nn.BCEWithLogitsLoss(pos_weight=w_c), nn.BCEWithLogitsLoss(pos_weight=w_u)


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate one config
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg: dict, device: torch.device) -> dict:
    train_ds = SahelConflictDataset(
        window=cfg["window"], horizon=cfg["horizon"],
        split="train",
        val_start=cfg["val_start"], test_start=cfg["test_start"],
    )
    val_ds = SahelConflictDataset(
        window=cfg["window"], horizon=cfg["horizon"],
        split="val",
        val_start=cfg["val_start"], test_start=cfg["test_start"],
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    model = build_model(
        window=cfg["window"],
        n_features=train_ds.F,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    loss_c, loss_u = build_loss(train_ds)
    loss_c = loss_c.to(device)
    loss_u = loss_u.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss  = float("inf")
    best_metrics   = {}

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(batch.x, batch.edge_index, batch.edge_attr)
            loss = (loss_c(logits[:, 0], batch.y[:, 0]) +
                    loss_u(logits[:, 1], batch.y[:, 1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        probs_c, probs_u, labels_c, labels_u = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                logits = model.forward_logits(
                    batch.x, batch.edge_index, batch.edge_attr
                )
                val_loss += (
                    loss_c(logits[:, 0], batch.y[:, 0]) +
                    loss_u(logits[:, 1], batch.y[:, 1])
                ).item()
                p = torch.sigmoid(logits)
                probs_c.extend(p[:, 0].cpu().tolist())
                probs_u.extend(p[:, 1].cpu().tolist())
                labels_c.extend(batch.y[:, 0].cpu().tolist())
                labels_u.extend(batch.y[:, 1].cpu().tolist())

        val_loss /= max(len(val_loader), 1)

        def safe_auroc(y, p):
            return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        def safe_ap(y, p):
            return average_precision_score(y, p) if len(set(y)) > 1 else float("nan")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "best_val_loss":    round(val_loss, 6),
                "best_val_auroc_c": safe_auroc(labels_c, probs_c),
                "best_val_ap_c":    safe_ap(labels_c,    probs_c),
                "best_val_auroc_u": safe_auroc(labels_u, probs_u),
                "best_val_ap_u":    safe_ap(labels_u,    probs_u),
                "epochs_run":       epoch,
            }

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate all combinations
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Total configurations to try: {len(combos)}\n")

    log_path    = OUTPUTS_DIR / "tuning_log.csv"
    all_results = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for run_id, combo in enumerate(combos, 1):
            cfg = {k: v for k, v in zip(keys, combo)}
            cfg.update(FIXED)

            print(f"── Run {run_id}/{len(combos)} ──────────────────────────────────")
            print(f"   window={cfg['window']}  hidden={cfg['hidden_dim']}  "
                  f"dropout={cfg['dropout']}  lr={cfg['lr']}")

            try:
                metrics = run_config(cfg, device)
            except Exception as e:
                print(f"   ❌ Run failed: {e}")
                metrics = None

            if metrics is None:
                print("   ⚠️  Skipped — no data for this split")
                continue

            row = {
                "run_id":     run_id,
                "window":     cfg["window"],
                "hidden_dim": cfg["hidden_dim"],
                "dropout":    cfg["dropout"],
                "lr":         cfg["lr"],
                "timestamp":  datetime.now().isoformat(),
                **{k: round(v, 6) if isinstance(v, float) and not np.isnan(v)
                   else v
                   for k, v in metrics.items()},
            }
            writer.writerow(row)
            f.flush()
            all_results.append(row)

            ap = metrics.get("best_val_ap_c", float("nan"))
            au = metrics.get("best_val_auroc_c", float("nan"))

            # Format the strings outside the f-string or move the logic inside the expression part
        au_str = f"{au:.4f}" if not np.isnan(au) else "nan"
        ap_str = f"{ap:.4f}" if not np.isnan(ap) else "nan"

        print(f"   val_loss={metrics['best_val_loss']:.4f}  "
            f"AUROC_c={au_str}  "
            f"AP_c={ap_str}")

    # ── Find best config ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Sweep complete — {len(all_results)} runs")
    print(f"{'='*60}")

    if not all_results:
        print("No results recorded.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUTS_DIR / "tuning_log.csv", index=False)

    # Primary metric: AP_c (Average Precision on conflict head)
    # Falls back to val_loss if all AP values are nan
    ap_col = "best_val_ap_c"
    valid  = df[df[ap_col].notna() & (df[ap_col] != float("nan"))]

    if len(valid) > 0:
        best_row = valid.loc[valid[ap_col].idxmax()]
        print(f"\n  Best config by AP_conflict:")
    else:
        print("\n  AP is nan (labels may still be sparse). Best by val_loss:")
        best_row = df.loc[df["best_val_loss"].idxmin()]

    best_cfg = {
        "window":     int(best_row["window"]),
        "hidden_dim": int(best_row["hidden_dim"]),
        "dropout":    float(best_row["dropout"]),
        "lr":         float(best_row["lr"]),
        "val_loss":   float(best_row["best_val_loss"]),
        "ap_c":       float(best_row.get(ap_col, float("nan"))),
        "auroc_c":    float(best_row.get("best_val_auroc_c", float("nan"))),
    }

    print(f"  window={best_cfg['window']}  hidden_dim={best_cfg['hidden_dim']}  "
          f"dropout={best_cfg['dropout']}  lr={best_cfg['lr']}")
    print(f"  val_loss={best_cfg['val_loss']:.4f}  "
          f"AP_c={best_cfg['ap_c']:.4f}  AUROC_c={best_cfg['auroc_c']:.4f}")

    best_cfg_path = OUTPUTS_DIR / "best_config.json"
    with open(best_cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\n  ✅ Best config saved → {best_cfg_path}")
    print(f"  ✅ Full log saved    → {log_path}")
    print(f"\n  Next step: run python model/train.py with this config")


if __name__ == "__main__":
    main()
