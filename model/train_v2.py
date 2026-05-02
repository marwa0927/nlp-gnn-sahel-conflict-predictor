"""
model/train_v2.py
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
from model.gat import build_gat_model, save_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
GRAPH_DIR   = ROOT / "graph"
CKPT_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2024-06-01"},
]

GAT_CONFIG = {
    "window": 7, "hidden_dim": 64, "heads": 4,
    "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4,
    "epochs": 50, "batch_size": 16,
}


def load_v2_graph():
    """Load road-weighted v2 graph files."""
    ei_path = GRAPH_DIR / "edge_index_v2.npy"
    ew_path = GRAPH_DIR / "edge_weight_v2.npy"

    if not ei_path.exists():
        print("⚠️  v2 graph not found. Run: python graph/build_graph_v2.py first")
        print("   Falling back to v1 graph.")
        ei_path = GRAPH_DIR / "edge_index.npy"
        ew_path = GRAPH_DIR / "edge_weight.npy"

    edge_index = torch.tensor(np.load(ei_path), dtype=torch.long)
    edge_weight = torch.tensor(np.load(ew_path), dtype=torch.float32)
    print(f"✅ Loaded graph: {edge_index.shape[1]} edges")
    return edge_index, edge_weight


def build_loss(train_ds, device):
    y_sample = torch.stack(
        [train_ds.get(i).y for i in range(min(300, len(train_ds)))], dim=0
    )
    total   = y_sample[:, :, 0].numel()
    n_pos_c = y_sample[:, :, 0].sum().item()
    n_pos_u = y_sample[:, :, 1].sum().item()
    w_c = torch.tensor([(total - n_pos_c) / (n_pos_c + 1e-6)])
    w_u = torch.tensor([(total - n_pos_u) / (n_pos_u + 1e-6)])
    return (nn.BCEWithLogitsLoss(pos_weight=w_c).to(device),
            nn.BCEWithLogitsLoss(pos_weight=w_u).to(device))


def safe(fn, y, p, **kw):
    try:
        return fn(y, p, **kw) if len(set(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  nan "


def train_fold(fold_cfg, cfg, edge_index, edge_weight, device, log_writer):
    fold_name  = fold_cfg["name"]
    val_start  = fold_cfg["val_start"]
    test_start = fold_cfg["test_start"]

    print(f"\n{'='*60}")
    print(f"  v2 GAT Fold: {fold_name}")
    print(f"{'='*60}")

    train_ds = SahelConflictDataset(
        window=cfg["window"], horizon=1, split="train",
        val_start=val_start, test_start=test_start,
    )
    val_ds = SahelConflictDataset(
        window=cfg["window"], horizon=1, split="val",
        val_start=val_start, test_start=test_start,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  ⚠️  Skipping — not enough data.")
        return {}

    # Override graph with v2
    train_ds.edge_index = edge_index
    train_ds.edge_attr  = edge_weight
    val_ds.edge_index   = edge_index
    val_ds.edge_attr    = edge_weight

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    model = build_gat_model(
        window=cfg["window"], n_features=train_ds.F,
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        dropout=cfg["dropout"],
    ).to(device)

    loss_c, loss_u = build_loss(train_ds, device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    best_metrics  = {}
    ckpt_path     = CKPT_DIR / f"v2_gat_best_{fold_name}.pt"

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  "
          f"{'AUROC_c':>8}  {'AP_c':>8}")
    print(f"  {'─'*48}")

    for epoch in range(1, cfg["epochs"] + 1):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = {
                "fold": fold_name, "auroc_conflict": auroc_c,
                "ap_conflict": ap_c, "val_loss": val_loss,
            }
            save_checkpoint(model, ckpt_path, metadata={
                "fold": fold_name, "epoch": epoch,
                "val_loss": val_loss, "auroc_c": auroc_c,
            })

        log_writer.writerow({
            "fold": fold_name, "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "auroc_c": round(auroc_c, 4) if not np.isnan(auroc_c) else None,
            "ap_c":    round(ap_c,    4) if not np.isnan(ap_c)    else None,
            "timestamp": datetime.now().isoformat(),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                  f"{fmt(auroc_c):>8}  {fmt(ap_c):>8}")

    print(f"\n  Best → val_loss={best_val_loss:.4f}  "
          f"AUROC_c={fmt(best_metrics.get('auroc_conflict', float('nan')))}")
    return best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    edge_index, edge_weight = load_v2_graph()

    log_path   = OUTPUTS_DIR / "v2_training_log.csv"
    log_fields = ["fold", "epoch", "train_loss", "val_loss",
                  "auroc_c", "ap_c", "timestamp"]

    v2_results = []
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for fold_cfg in SPLITS:
            result = train_fold(fold_cfg, GAT_CONFIG, edge_index,
                                edge_weight, device, writer)
            if result:
                v2_results.append(result)

    # Compare v1 vs v2 graph impact
    print(f"\n{'='*60}")
    print(f"  v1 graph vs v2 road-weighted graph — GAT performance")
    print(f"{'='*60}")

    gat_path = OUTPUTS_DIR / "gat_evaluation_report.csv"
    if gat_path.exists():
        v1_df = pd.read_csv(gat_path)
        print(f"\n  {'Fold':10s} {'v1 AUROC':>12} {'v2 AUROC':>12} {'Delta':>8}")
        print(f"  {'─'*46}")
        for r in v2_results:
            fold = r["fold"]
            v2_a = r.get("auroc_conflict", float("nan"))
            v1_row = v1_df[v1_df["fold"] == fold]
            v1_a = float(v1_row["auroc_conflict"].iloc[0]) if not v1_row.empty else float("nan")
            delta = v2_a - v1_a if not (np.isnan(v2_a) or np.isnan(v1_a)) else float("nan")
            arrow = "⬆️ " if not np.isnan(delta) and delta > 0.005 else \
                    "⬇️ " if not np.isnan(delta) and delta < -0.005 else "➡️ "
            print(f"  {fold:10s} {fmt(v1_a):>12} {fmt(v2_a):>12} "
                  f"{arrow}{delta:+.4f}" if not np.isnan(delta) else f"  {fold:10s} {fmt(v1_a):>12} {fmt(v2_a):>12}  n/a")

        v2_aurocs = [r["auroc_conflict"] for r in v2_results
                     if not np.isnan(r.get("auroc_conflict", float("nan")))]
        v1_aurocs = v1_df["auroc_conflict"].dropna().tolist()

        if v2_aurocs and v1_aurocs:
            print(f"\n  v1 mean AUROC: {np.mean(v1_aurocs):.4f}")
            print(f"  v2 mean AUROC: {np.mean(v2_aurocs):.4f}")
            gain = np.mean(v2_aurocs) - np.mean(v1_aurocs)
            print(f"  Graph upgrade gain: {gain:+.4f}")
            if gain > 0.01:
                print("  ✅ Road-weighted graph improves performance — use v2.")
            else:
                print("  ➡️  Marginal difference — data quality remains the bottleneck.")

    # Save impact report
    impact_df = pd.DataFrame(v2_results)
    impact_df.to_csv(OUTPUTS_DIR / "v1_vs_v2_graph_impact.csv", index=False)
    print(f"\n✅ Training log → {log_path}")
    print(f"✅ Impact report → {OUTPUTS_DIR / 'v1_vs_v2_graph_impact.csv'}")


if __name__ == "__main__":
    main()
