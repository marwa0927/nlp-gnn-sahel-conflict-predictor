"""
model/train_best.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import csv
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from datetime import datetime

from model.dataset import SahelConflictDataset
from model.gcn import build_model, save_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
CKPT_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2025-01-01"},
]


def load_best_config() -> dict:
    cfg_path = OUTPUTS_DIR / "best_config.json"
    if not cfg_path.exists():
        print("⚠️  best_config.json not found. Run tune.py first.")
        print("   Using default config instead.")
        return {
            "window": 7, "hidden_dim": 64,
            "dropout": 0.3, "lr": 1e-3,
        }
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"✅ Loaded best config from {cfg_path}")
    print(f"   window={cfg['window']}  hidden_dim={cfg['hidden_dim']}  "
          f"dropout={cfg['dropout']}  lr={cfg['lr']}")
    return cfg


def build_loss(train_ds, device):
    y_sample = torch.stack(
        [train_ds.get(i).y for i in range(min(300, len(train_ds)))], dim=0
    )
    total   = y_sample[:, :, 0].numel()
    n_pos_c = y_sample[:, :, 0].sum().item()
    n_pos_u = y_sample[:, :, 1].sum().item()

    w_c = torch.tensor([(total - n_pos_c) / (n_pos_c + 1e-6)])
    w_u = torch.tensor([(total - n_pos_u) / (n_pos_u + 1e-6)])
    print(f"  pos_weight_conflict={w_c.item():.2f}  "
          f"pos_weight_unrest={w_u.item():.2f}")

    return (nn.BCEWithLogitsLoss(pos_weight=w_c).to(device),
            nn.BCEWithLogitsLoss(pos_weight=w_u).to(device))


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  nan "


def train_fold(fold_cfg, best_cfg, device, log_writer):
    fold_name  = fold_cfg["name"]
    val_start  = fold_cfg["val_start"]
    test_start = fold_cfg["test_start"]

    print(f"\n{'='*60}")
    print(f"  Fold: {fold_name}  (val_start={val_start})")
    print(f"{'='*60}")

    train_ds = SahelConflictDataset(
        window=best_cfg["window"], horizon=1,
        split="train", val_start=val_start, test_start=test_start,
    )
    val_ds = SahelConflictDataset(
        window=best_cfg["window"], horizon=1,
        split="val", val_start=val_start, test_start=test_start,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  ⚠️  Skipping — not enough data.")
        return None

    # Label rate check
    y_all = torch.stack([train_ds.get(i).y for i in range(len(train_ds))], dim=0)
    rate_c = y_all[:, :, 0].mean().item()
    rate_u = y_all[:, :, 1].mean().item()
    print(f"  Conflict label rate: {rate_c:.4f}")
    print(f"  Unrest   label rate: {rate_u:.4f}")
    if rate_c == 0:
        print("  ⚠️  Conflict labels still zero — metrics will be nan.")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    model     = build_model(
        window=best_cfg["window"], n_features=train_ds.F,
        hidden_dim=best_cfg["hidden_dim"], dropout=best_cfg["dropout"],
    ).to(device)
    loss_c, loss_u = build_loss(train_ds, device)
    optimizer = AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    best_ap_c     = float("nan")
    best_auroc_c  = float("nan")
    ckpt_path     = CKPT_DIR / f"best_{fold_name}.pt"

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  "
          f"{'AUROC_c':>8}  {'AP_c':>8}  {'F1_c':>8}")
    print(f"  {'─'*55}")

    for epoch in range(1, 51):
        # Train
        model.train()
        total_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(batch.x, batch.edge_index, batch.edge_attr)
            loss = (loss_c(logits[:, 0], batch.y[:, 0]) +
                    loss_u(logits[:, 1], batch.y[:, 1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item()
        train_loss = total_train / max(len(train_loader), 1)
        scheduler.step()

        # Validate
        model.eval()
        total_val = 0.0
        pc, pu, lc, lu = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                logits = model.forward_logits(batch.x, batch.edge_index, batch.edge_attr)
                total_val += (
                    loss_c(logits[:, 0], batch.y[:, 0]) +
                    loss_u(logits[:, 1], batch.y[:, 1])
                ).item()
                p = torch.sigmoid(logits)
                pc.extend(p[:, 0].cpu().tolist())
                pu.extend(p[:, 1].cpu().tolist())
                lc.extend(batch.y[:, 0].cpu().tolist())
                lu.extend(batch.y[:, 1].cpu().tolist())

        val_loss = total_val / max(len(val_loader), 1)

        def safe(fn, y, p, **kw):
            try:
                return fn(y, p, **kw) if len(set(y)) > 1 else float("nan")
            except Exception:
                return float("nan")

        auroc_c = safe(roc_auc_score, lc, pc)
        ap_c    = safe(average_precision_score, lc, pc)
        f1_c    = safe(f1_score, lc,
                       [1 if x > 0.5 else 0 for x in pc],
                       zero_division=0)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ap_c    = ap_c
            best_auroc_c = auroc_c
            save_checkpoint(model, ckpt_path, metadata={
                "fold": fold_name, "epoch": epoch,
                "val_loss": val_loss, "ap_c": ap_c, "auroc_c": auroc_c,
                "config": best_cfg,
            })

        # Log
        log_writer.writerow({
            "fold": fold_name, "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "auroc_c":    round(auroc_c, 4) if not np.isnan(auroc_c) else None,
            "ap_c":       round(ap_c,    4) if not np.isnan(ap_c)    else None,
            "f1_c":       round(f1_c,    4) if not np.isnan(f1_c)    else None,
            "timestamp":  datetime.now().isoformat(),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                  f"{fmt(auroc_c):>8}  {fmt(ap_c):>8}  {fmt(f1_c):>8}")

    print(f"\n  Best → val_loss={best_val_loss:.4f}  "
          f"AUROC_c={fmt(best_auroc_c)}  AP_c={fmt(best_ap_c)}")
    print(f"  Checkpoint → {ckpt_path}")

    return {"fold": fold_name, "val_loss": best_val_loss,
            "ap_c": best_ap_c, "auroc_c": best_auroc_c}


def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_cfg   = load_best_config()
    log_path   = OUTPUTS_DIR / "training_log_best.csv"
    log_fields = ["fold", "epoch", "train_loss", "val_loss",
                  "auroc_c", "ap_c", "f1_c", "timestamp"]

    all_results = []
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for fold_cfg in SPLITS:
            result = train_fold(fold_cfg, best_cfg, device, writer)
            if result:
                all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Final results across {len(all_results)} folds")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['fold']}:  val_loss={r['val_loss']:.4f}  "
              f"AUROC_c={fmt(r['auroc_c'])}  AP_c={fmt(r['ap_c'])}")

    auroc_vals = [r["auroc_c"] for r in all_results
                  if not np.isnan(r.get("auroc_c", float("nan")))]
    ap_vals    = [r["ap_c"]    for r in all_results
                  if not np.isnan(r.get("ap_c",    float("nan")))]

    if auroc_vals:
        print(f"\n  Mean AUROC: {np.mean(auroc_vals):.4f}")
        print(f"  Mean AP:    {np.mean(ap_vals):.4f}")

        if np.mean(auroc_vals) >= 0.65:
            print("\n  ✅ AUROC ≥ 0.65 — model has real signal. Ready for Week 4.")
        else:
            print("\n  ⚠️  AUROC < 0.65 — see troubleshooting below:")
            print("     1. Check the 4 zero-coverage Niger nodes with Person A")
            print("     2. Check lag feature correlation with Person A")
            print("     3. Try hidden_dim=128 in best_config.json and re-run")

    print(f"\n  ✅ Log → {log_path}")


if __name__ == "__main__":
    main()
