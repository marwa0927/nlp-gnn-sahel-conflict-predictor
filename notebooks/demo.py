"""
notebooks/demo.py
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
from sklearn.metrics import roc_auc_score, average_precision_score

from model.dataset import SahelConflictDataset
from model.gcn import build_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"


def load_nodes():
    nodes_py = ROOT / "data" / "nodes.py"
    spec = importlib.util.spec_from_file_location("nodes", nodes_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NODES


def load_best_config():
    cfg_path = OUTPUTS_DIR / "best_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"window": 7, "hidden_dim": 64, "dropout": 0.3, "lr": 1e-3}


def print_banner(text):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes  = load_nodes()
    cfg    = load_best_config()
    N      = len(nodes)

    print_banner("Sahel Conflict GNN — Live Demo")
    print(f"\n  Model config: window={cfg.get('window',7)}d  "
          f"hidden_dim={cfg.get('hidden_dim',64)}  "
          f"dropout={cfg.get('dropout',0.3)}")
    print(f"  Nodes: {N}  |  Features: 24  |  Device: {device}")

    # ── Step 1: Load dataset ─────────────────────────────────────────────────
    print_banner("Step 1 — Loading data")

    ds_test = SahelConflictDataset(
        window=cfg.get("window", 7), horizon=1,
        split="test",
        val_start="2023-01-01", test_start="2024-01-01",
    )

    if len(ds_test) == 0:
        ds_test = SahelConflictDataset(
            window=cfg.get("window", 7), horizon=1,
            split="test",
            val_start="2023-01-01", test_start="2024-06-01",
        )

    print(f"\n  Test period: {ds_test.dates[ds_test.sample_indices[0]]} "
          f"→ {ds_test.dates[ds_test.sample_indices[-1]]}")
    print(f"  Test samples: {len(ds_test)}")

    # Label rates in test set
    y_sample = torch.stack(
        [ds_test.get(i).y for i in range(min(100, len(ds_test)))], dim=0
    )
    print(f"  Conflict rate (test): {y_sample[:,:,0].mean():.4f}")
    print(f"  Unrest   rate (test): {y_sample[:,:,1].mean():.4f}")

    # ── Step 2: Load model ───────────────────────────────────────────────────
    print_banner("Step 2 — Loading trained model")

    # Pick best available checkpoint
    ckpt_path = None
    for fold in ["fold_3", "fold_2", "fold_1"]:
        p = CKPT_DIR / f"best_{fold}.pt"
        if p.exists():
            ckpt_path = p
            fold_used = fold
            break

    if ckpt_path is None:
        print("❌ No checkpoint found. Run train_best.py first.")
        return

    model = build_model(
        window=cfg.get("window", 7),
        n_features=ds_test.F,
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    meta = load_checkpoint(model, ckpt_path)
    model.eval()
    print(f"\n  Using: {fold_used}  "
          f"(epoch={meta.get('epoch','?')}  "
          f"val_loss={meta.get('val_loss', 0):.4f})")
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Step 3: Run inference ────────────────────────────────────────────────
    print_banner("Step 3 — Running inference on test period")

    loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    all_probs_c, all_probs_u = [], []
    all_labels_c, all_labels_u = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model.forward_logits(
                batch.x, batch.edge_index, batch.edge_attr
            )
            p = torch.sigmoid(logits)
            all_probs_c.extend(p[:, 0].cpu().tolist())
            all_probs_u.extend(p[:, 1].cpu().tolist())
            all_labels_c.extend(batch.y[:, 0].cpu().tolist())
            all_labels_u.extend(batch.y[:, 1].cpu().tolist())

    # ── Step 4: Performance metrics ──────────────────────────────────────────
    print_banner("Step 4 — Model performance")

    def safe(fn, y, p):
        try:
            return fn(y, p) if len(set(y)) > 1 else float("nan")
        except Exception:
            return float("nan")

    auroc_c = safe(roc_auc_score,          all_labels_c, all_probs_c)
    ap_c    = safe(average_precision_score, all_labels_c, all_probs_c)
    auroc_u = safe(roc_auc_score,          all_labels_u, all_probs_u)
    ap_u    = safe(average_precision_score, all_labels_u, all_probs_u)

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "  nan "

    print(f"\n  {'Metric':25s} {'Conflict':>10} {'Unrest':>10}")
    print(f"  {'─'*48}")
    print(f"  {'AUROC':25s} {fmt(auroc_c):>10} {fmt(auroc_u):>10}")
    print(f"  {'Average Precision':25s} {fmt(ap_c):>10} {fmt(ap_u):>10}")

    # Interpretation
    if not np.isnan(auroc_c):
        if auroc_c >= 0.70:
            print(f"\n  ✅ AUROC={auroc_c:.4f} — strong signal.")
        elif auroc_c >= 0.65:
            print(f"\n  ✅ AUROC={auroc_c:.4f} — good signal, ready for production.")
        elif auroc_c >= 0.60:
            print(f"\n  ⚠️  AUROC={auroc_c:.4f} — moderate signal.")
        else:
            print(f"\n  ❌ AUROC={auroc_c:.4f} — weak signal.")

    # ── Step 5: Per-node risk scores (most recent 30 days) ───────────────────
    print_banner("Step 5 — Per-node conflict risk (last 30 days)")

    # Use last 30 samples
    recent_indices = ds_test.sample_indices[-min(30, len(ds_test.sample_indices)):]
    ds_recent = SahelConflictDataset(
        window=cfg.get("window", 7), horizon=1, split=None
    )
    ds_recent.sample_indices = recent_indices
    loader_recent = DataLoader(ds_recent, batch_size=32, shuffle=False)

    probs_c_recent, probs_u_recent = [], []
    with torch.no_grad():
        for batch in loader_recent:
            batch  = batch.to(device)
            logits = model.forward_logits(
                batch.x, batch.edge_index, batch.edge_attr
            )
            p = torch.sigmoid(logits)
            probs_c_recent.extend(p[:, 0].cpu().tolist())
            probs_u_recent.extend(p[:, 1].cpu().tolist())

    n_recent   = len(probs_c_recent) // N
    arr_c      = np.array(probs_c_recent).reshape(n_recent, N)
    arr_u      = np.array(probs_u_recent).reshape(n_recent, N)
    node_avg_c = arr_c.mean(axis=0)
    node_avg_u = arr_u.mean(axis=0)

    results = []
    for j, node in enumerate(nodes):
        p_c = node_avg_c[j]
        p_u = node_avg_u[j]
        results.append({
            "node_id":    node["id"],
            "country":    node["country"],
            "p_conflict": p_c,
            "p_unrest":   p_u,
            "risk_level": (
                "🔴 HIGH"     if p_c >= 0.6  else
                "🟠 MEDIUM"   if p_c >= 0.35 else
                "🟡 ELEVATED" if p_c >= 0.15 else
                "🟢 LOW"
            ),
        })

    df_results = pd.DataFrame(results).sort_values("p_conflict", ascending=False)

    print(f"\n  {'Node':20s} {'Country':8s} {'Conflict':>10} "
          f"{'Unrest':>8} {'Risk':>12}")
    print(f"  {'─'*65}")
    for _, row in df_results.iterrows():
        print(f"  {row['node_id']:20s} {row['country']:8s} "
              f"{row['p_conflict']:10.4f} {row['p_unrest']:8.4f} "
              f"{row['risk_level']:>12}")

    # ── Step 6: Top 5 alert ──────────────────────────────────────────────────
    print_banner("Step 6 — Top 5 highest risk cities")

    top5 = df_results.head(5)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n  #{rank}  {row['node_id'].upper()} ({row['country']})")
        print(f"      Conflict risk: {row['p_conflict']:.1%}  {row['risk_level']}")
        print(f"      Unrest risk:   {row['p_unrest']:.1%}")

    # Save predictions
    csv_path = OUTPUTS_DIR / "demo_predictions.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  ✅ Predictions saved → {csv_path}")

    # ── Step 7: Generate risk map ────────────────────────────────────────────
    print_banner("Step 7 — Generating risk map")

    try:
        import folium
        from notebooks.risk_map import build_risk_map

        # Add lat/lon to results
        node_coords = {n["id"]: {"lat": n["lat"], "lon": n["lon"]} for n in nodes}
        df_results["lat"] = df_results["node_id"].map(
            lambda x: node_coords[x]["lat"]
        )
        df_results["lon"] = df_results["node_id"].map(
            lambda x: node_coords[x]["lon"]
        )

        m = build_risk_map(df_results)
        map_path = OUTPUTS_DIR / "sahel_risk_map_demo.html"
        m.save(str(map_path))
        print(f"\n  ✅ Risk map saved → {map_path}")
        print("     Open in browser to view interactive map.")
    except ImportError:
        print("  ⚠️  folium not installed. Run: pip install folium")
    except Exception as e:
        print(f"  ⚠️  Map generation failed: {e}")
        print("     Run: python notebooks/risk_map.py separately.")

    # ── Final summary ────────────────────────────────────────────────────────
    print_banner("Demo complete")
    print(f"""
  Pipeline:      GDELT → Features → GCN → Risk Scores
  Nodes:         {N} Sahel cities
  Training data: 2020–2022 (fold 3)
  Test data:     2024

  Performance:
    AUROC (conflict): {fmt(auroc_c)}
    AP    (conflict): {fmt(ap_c)}

  Highest risk right now:
    {top5.iloc[0]['node_id'].upper()} — {top5.iloc[0]['p_conflict']:.1%} conflict probability

  Known limitations (v2 roadmap):
    - 4 Niger nodes have weak GDELT coverage
    - Edges are distance-based (v2: road-weighted via OSMnx)
    - GCN baseline (v2: GAT with attention)
    - No local RSS / Arabic NLP (v2: BERTopic layer)
    """)


if __name__ == "__main__":
    main()
