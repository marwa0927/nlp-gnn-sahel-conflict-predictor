"""
model/evaluate_gat.py
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
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, precision_recall_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.dataset import SahelConflictDataset
from model.gat import build_gat_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SPLITS = [
    {"name": "fold_1", "val_start": "2022-01-01", "test_start": "2023-01-01"},
    {"name": "fold_2", "val_start": "2023-01-01", "test_start": "2024-01-01"},
    {"name": "fold_3", "val_start": "2024-01-01", "test_start": "2024-06-01"},
]

GAT_CONFIG = {
    "window": 7, "hidden_dim": 64, "heads": 4, "dropout": 0.3,
}


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


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  nan "


def evaluate_fold(fold_cfg, device, nodes):
    fold_name  = fold_cfg["name"]
    ckpt_path  = CKPT_DIR / f"gat_best_{fold_name}.pt"
    N          = len(nodes)

    print(f"\n{'='*60}")
    print(f"  GAT Evaluation: {fold_name}")
    print(f"{'='*60}")

    if not ckpt_path.exists():
        print(f"  ⚠️  No GAT checkpoint at {ckpt_path}")
        print(f"     Run: python model/train_gat.py first.")
        return {}

    test_ds = SahelConflictDataset(
        window=GAT_CONFIG["window"], horizon=1,
        split="test",
        val_start=fold_cfg["val_start"],
        test_start=fold_cfg["test_start"],
    )

    if len(test_ds) == 0:
        print("  ⚠️  No test samples.")
        return {}

    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_gat_model(
        window=GAT_CONFIG["window"], n_features=test_ds.F,
        hidden_dim=GAT_CONFIG["hidden_dim"], heads=GAT_CONFIG["heads"],
        dropout=GAT_CONFIG["dropout"],
    ).to(device)
    meta = load_checkpoint(model, ckpt_path)
    model.eval()

    pc, pu, lc, lu = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model.forward_logits(batch.x, batch.edge_index)
            p = torch.sigmoid(logits)
            pc.extend(p[:, 0].cpu().tolist())
            pu.extend(p[:, 1].cpu().tolist())
            lc.extend(batch.y[:, 0].cpu().tolist())
            lu.extend(batch.y[:, 1].cpu().tolist())

    auroc_c = safe(roc_auc_score,          lc, pc)
    ap_c    = safe(average_precision_score, lc, pc)
    f1_c    = safe(f1_score, lc, [1 if x > 0.5 else 0 for x in pc], zero_division=0)
    auroc_u = safe(roc_auc_score,          lu, pu)
    ap_u    = safe(average_precision_score, lu, pu)

    print(f"\n  {'Metric':25s} {'Conflict':>10} {'Unrest':>10}")
    print(f"  {'─'*48}")
    print(f"  {'AUROC':25s} {fmt(auroc_c):>10} {fmt(auroc_u):>10}")
    print(f"  {'AP':25s} {fmt(ap_c):>10} {fmt(ap_u):>10}")
    print(f"  {'F1':25s} {fmt(f1_c):>10}")

    # Per-node metrics
    n_samples  = len(pc) // N
    probs_arr  = np.array(pc).reshape(n_samples, N)
    labels_arr = np.array(lc).reshape(n_samples, N)

    node_records = []
    for j, node in enumerate(nodes):
        p_node = probs_arr[:, j].tolist()
        l_node = labels_arr[:, j].tolist()
        node_records.append({
            "node_id":    node["id"],
            "country":    node["country"],
            "auroc":      safe(roc_auc_score,          l_node, p_node),
            "ap":         safe(average_precision_score, l_node, p_node),
            "pos_rate":   float(np.mean(l_node)),
            "n_positive": int(sum(l_node)),
        })

    node_df = pd.DataFrame(node_records).sort_values("auroc", ascending=False)

    print(f"\n  Per-node AUROC (top 10):")
    print(f"  {'Node':20s} {'Country':8s} {'AUROC':>8} {'AP':>8} {'N+':>6}")
    print(f"  {'─'*52}")
    for _, row in node_df.head(10).iterrows():
        print(f"  {row['node_id']:20s} {row['country']:8s} "
              f"{fmt(row['auroc']):>8} {fmt(row['ap']):>8} "
              f"{row['n_positive']:>6}")

    return {
        "fold": fold_name, "model": "GAT",
        "auroc_conflict": auroc_c, "ap_conflict": ap_c,
        "f1_conflict": f1_c, "auroc_unrest": auroc_u, "ap_unrest": ap_u,
        "node_df": node_df,
    }


def print_final_comparison(gat_results):
    """Print GCN vs GAT side by side."""
    gcn_path = OUTPUTS_DIR / "full_evaluation_report.csv"
    gcn_df   = pd.read_csv(gcn_path) if gcn_path.exists() else pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  Final comparison: GCN vs GAT")
    print(f"{'='*60}")
    print(f"\n  {'Fold':10s} {'GCN AUROC':>12} {'GAT AUROC':>12} "
          f"{'GCN AP':>10} {'GAT AP':>10}")
    print(f"  {'─'*58}")

    for r in gat_results:
        fold    = r["fold"]
        gat_a   = r.get("auroc_conflict", float("nan"))
        gat_ap  = r.get("ap_conflict",    float("nan"))
        gcn_a   = float("nan")
        gcn_ap  = float("nan")

        if not gcn_df.empty:
            gcn_row = gcn_df[gcn_df["fold"] == fold]
            if not gcn_row.empty:
                gcn_a  = float(gcn_row["auroc_conflict"].iloc[0])
                gcn_ap = float(gcn_row["ap_conflict"].iloc[0])

        delta = gat_a - gcn_a if not (np.isnan(gat_a) or np.isnan(gcn_a)) else float("nan")
        arrow = ("⬆️ " if not np.isnan(delta) and delta > 0.005 else
                 "⬇️ " if not np.isnan(delta) and delta < -0.005 else "➡️ ")

        print(f"  {fold:10s} {fmt(gcn_a):>12} {fmt(gat_a):>12} "
              f"{fmt(gcn_ap):>10} {fmt(gat_ap):>10}  {arrow}")

    # Mean comparison
    gat_aurocs = [r["auroc_conflict"] for r in gat_results
                  if not np.isnan(r.get("auroc_conflict", float("nan")))]
    if gat_aurocs and not gcn_df.empty:
        mean_gat = np.mean(gat_aurocs)
        mean_gcn = gcn_df["auroc_conflict"].mean()
        print(f"\n  {'─'*58}")
        print(f"  {'Mean':10s} {fmt(mean_gcn):>12} {fmt(mean_gat):>12}")
        print(f"\n  Net improvement: {mean_gat - mean_gcn:+.4f} AUROC")

        if mean_gat > mean_gcn + 0.02:
            print("  ✅ GAT significantly outperforms GCN — adopt GAT as default.")
        elif mean_gat > mean_gcn:
            print("  ✅ GAT marginally better — worth keeping.")
        else:
            print("  ⚠️  GCN matches GAT — data quality is the bottleneck, not architecture.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes  = load_nodes()

    gat_results = []
    all_node_dfs = []

    for fold_cfg in SPLITS:
        result = evaluate_fold(fold_cfg, device, nodes)
        if result:
            node_df = result.pop("node_df", None)
            if node_df is not None:
                node_df["fold"] = result["fold"]
                all_node_dfs.append(node_df)
            gat_results.append(result)

    # Save GAT evaluation report
    if gat_results:
        df = pd.DataFrame(gat_results)
        out = OUTPUTS_DIR / "gat_evaluation_report.csv"
        df.to_csv(out, index=False)
        print(f"\n  ✅ GAT report → {out}")

    # Save per-node metrics
    if all_node_dfs:
        node_summary = (pd.concat(all_node_dfs)
                        .groupby("node_id")[["auroc", "ap"]]
                        .mean()
                        .sort_values("auroc", ascending=False))
        node_summary.to_csv(OUTPUTS_DIR / "gat_per_node_metrics.csv")

        print(f"\n  Average node AUROC (GAT):")
        for nid, row in node_summary.iterrows():
            bar = "█" * int(row["auroc"] * 20) if not np.isnan(row["auroc"]) else ""
            print(f"    {nid:20s}  {fmt(row['auroc'])}  {bar}")

    print_final_comparison(gat_results)


if __name__ == "__main__":
    main()
