"""
model/train_gat.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import json
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from model.dataset import SahelConflictDataset
from model.gat import ConflictGAT, build_gat_model, save_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
CKPT_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2024-06-01"},
]

# GAT config — tune heads and hidden_dim
GAT_CONFIG = {
    "window":     7,
    "hidden_dim": 64,
    "heads":      4,
    "dropout":    0.3,
    "lr":         5e-4,    # slightly lower lr for GAT stability
    "weight_decay": 1e-4,
    "epochs":     50,
    "batch_size": 16,
}


def load_gcn_results() -> pd.DataFrame:
    """Load existing GCN evaluation results for comparison."""
    path = OUTPUTS_DIR / "full_evaluation_report.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_loss(train_ds, device):
    y_sample = torch.stack(
        [train_ds.get(i).y for i in range(min(300, len(train_ds)))], dim=0
    )
    total   = y_sample[:, :, 0].numel()
    n_pos_c = y_sample[:, :, 0].sum().item()
    n_pos_u = y_sample[:, :, 1].sum().item()
    w_c = torch.tensor([(total - n_pos_c) / (n_pos_c + 1e-6)])
    w_u = torch.tensor([(total - n_pos_u) / (n_pos_u + 1e-6)])
    print(f"  pos_weight_conflict={w_c.item():.2f}  pos_weight_unrest={w_u.item():.2f}")
    return (nn.BCEWithLogitsLoss(pos_weight=w_c).to(device),
            nn.BCEWithLogitsLoss(pos_weight=w_u).to(device))


def safe(fn, y, p, **kw):
    try:
        return fn(y, p, **kw) if len(set(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  nan "


def train_fold(fold_cfg, cfg, device, log_writer) -> dict:
    fold_name  = fold_cfg["name"]
    val_start  = fold_cfg["val_start"]
    test_start = fold_cfg["test_start"]

    print(f"\n{'='*60}")
    print(f"  GAT Fold: {fold_name}  (val_start={val_start})")
    print(f"{'='*60}")

    train_ds = SahelConflictDataset(
        window=cfg["window"], horizon=1,
        split="train", val_start=val_start, test_start=test_start,
    )
    val_ds = SahelConflictDataset(
        window=cfg["window"], horizon=1,
        split="val", val_start=val_start, test_start=test_start,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  ⚠️  Skipping — not enough data.")
        return {}

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    model = build_gat_model(
        window=cfg["window"], n_features=train_ds.F,
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        dropout=cfg["dropout"],
    ).to(device)
    print(f"  GAT parameters: {model.count_parameters():,}")

    loss_c, loss_u = build_loss(train_ds, device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    best_metrics  = {}
    ckpt_path     = CKPT_DIR / f"gat_best_{fold_name}.pt"

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  "
          f"{'AUROC_c':>8}  {'AP_c':>8}  {'F1_c':>8}")
    print(f"  {'─'*55}")

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        total_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(batch.x, batch.edge_index)
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
                logits = model.forward_logits(batch.x, batch.edge_index)
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
        auroc_c  = safe(roc_auc_score, lc, pc)
        ap_c     = safe(average_precision_score, lc, pc)
        f1_c     = safe(f1_score, lc,
                        [1 if x > 0.5 else 0 for x in pc], zero_division=0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = {
                "fold": fold_name, "model": "GAT",
                "auroc_conflict": auroc_c, "ap_conflict": ap_c,
                "f1_conflict": f1_c, "val_loss": val_loss,
            }
            save_checkpoint(model, ckpt_path, metadata={
                "fold": fold_name, "epoch": epoch,
                "val_loss": val_loss, "auroc_c": auroc_c,
                "ap_c": ap_c, "config": cfg,
            })

        log_writer.writerow({
            "fold": fold_name, "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "auroc_c":    round(auroc_c, 4) if not np.isnan(auroc_c) else None,
            "ap_c":       round(ap_c,    4) if not np.isnan(ap_c)    else None,
            "f1_c":       round(f1_c,    4) if not np.isnan(f1_c)    else None,
            "timestamp":  datetime.now().isoformat(),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                  f"{fmt(auroc_c):>8}  {fmt(ap_c):>8}  {fmt(f1_c):>8}")

    print(f"\n  Best → val_loss={best_val_loss:.4f}  "
          f"AUROC_c={fmt(best_metrics.get('auroc_conflict', float('nan')))}  "
          f"AP_c={fmt(best_metrics.get('ap_conflict', float('nan')))}")
    return best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GAT Config: {json.dumps(GAT_CONFIG, indent=2)}\n")

    log_path   = OUTPUTS_DIR / "gat_training_log.csv"
    log_fields = ["fold", "epoch", "train_loss", "val_loss",
                  "auroc_c", "ap_c", "f1_c", "timestamp"]

    gat_results = []
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for fold_cfg in SPLITS:
            result = train_fold(fold_cfg, GAT_CONFIG, device, writer)
            if result:
                gat_results.append(result)

    # ── Compare GAT vs GCN ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAT vs GCN comparison")
    print(f"{'='*60}")

    gcn_df = load_gcn_results()
    rows   = []

    for r in gat_results:
        fold = r["fold"]
        row  = {
            "fold":          fold,
            "gat_auroc":     r.get("auroc_conflict", float("nan")),
            "gat_ap":        r.get("ap_conflict",    float("nan")),
            "gat_f1":        r.get("f1_conflict",    float("nan")),
        }

        if not gcn_df.empty:
            gcn_row = gcn_df[gcn_df["fold"] == fold]
            if not gcn_row.empty:
                row["gcn_auroc"] = float(gcn_row["auroc_conflict"].iloc[0])
                row["gcn_ap"]    = float(gcn_row["ap_conflict"].iloc[0])
                row["gcn_f1"]    = float(gcn_row["f1_conflict"].iloc[0])
                row["auroc_delta"] = row["gat_auroc"] - row["gcn_auroc"]
                row["ap_delta"]    = row["gat_ap"]    - row["gcn_ap"]

        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_path = OUTPUTS_DIR / "gat_vs_gcn_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    print(f"\n  {'Fold':10s} {'GCN AUROC':>12} {'GAT AUROC':>12} {'Delta':>8}")
    print(f"  {'─'*48}")
    for _, row in comp_df.iterrows():
        gcn_a = fmt(row.get("gcn_auroc", float("nan")))
        gat_a = fmt(row.get("gat_auroc", float("nan")))
        delta = row.get("auroc_delta", float("nan"))
        arrow = "⬆️ " if not np.isnan(delta) and delta > 0 else "⬇️ "
        delta_str = f"{arrow}{delta:+.4f}" if not np.isnan(delta) else "  n/a"
        print(f"  {row['fold']:10s} {gcn_a:>12} {gat_a:>12} {delta_str:>10}")

    # Summary
    gat_aurocs = [r.get("auroc_conflict", float("nan")) for r in gat_results
                  if not np.isnan(r.get("auroc_conflict", float("nan")))]
    if gat_aurocs:
        mean_gat = np.mean(gat_aurocs)
        print(f"\n  GAT mean AUROC: {mean_gat:.4f}")

        if not gcn_df.empty:
            gcn_aurocs = gcn_df["auroc_conflict"].dropna().tolist()
            mean_gcn = np.mean(gcn_aurocs)
            improvement = mean_gat - mean_gcn
            print(f"  GCN mean AUROC: {mean_gcn:.4f}")
            print(f"  Improvement:    {improvement:+.4f}")
            if improvement > 0.02:
                print(f"\n  ✅ GAT outperforms GCN by {improvement:.4f} — use GAT going forward.")
            elif improvement > 0:
                print(f"\n  ✅ GAT slightly better — marginal gain from attention.")
            else:
                print(f"\n  ⚠️  GCN matches or beats GAT.")
                print(f"     Try heads=8 or hidden_dim=128 in GAT_CONFIG.")

    print(f"\n  ✅ Comparison saved → {comp_path}")
    print(f"  ✅ GAT checkpoints → checkpoints/gat_best_fold_*.pt")


if __name__ == "__main__":
    main()
