"""
model/predict.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import json
import argparse
import importlib.util
from torch_geometric.data import Data

from model.dataset import SahelConflictDataset, load_nodes, build_matrices_from_parquet
from model.gat import build_gat_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
GRAPH_DIR   = ROOT / "graph"


def load_config():
    cfg_path = OUTPUTS_DIR / "best_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"window": 7, "hidden_dim": 64, "heads": 4, "dropout": 0.3}


def load_graph():
    for prefix in ["edge_index_v2", "edge_index"]:
        p = GRAPH_DIR / f"{prefix}.npy"
        if p.exists():
            ei = torch.tensor(np.load(p), dtype=torch.long)
            ew = torch.tensor(
                np.load(str(p).replace("edge_index", "edge_weight")),
                dtype=torch.float32
            )
            return ei, ew
    raise FileNotFoundError("No edge_index.npy found in graph/")


def load_model(cfg, n_features, device):
    """Load best available GAT checkpoint."""
    model = build_gat_model(
        window=cfg.get("window", 7),
        n_features=n_features,
        hidden_dim=cfg.get("hidden_dim", 64),
        heads=cfg.get("heads", 4),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    for name in ["v2_gat_best_fold_3", "v2_gat_best_fold_2", "v2_gat_best_fold_1",
                 "gat_best_fold_3", "gat_best_fold_2", "best_fold_3"]:
        p = CKPT_DIR / f"{name}.pt"
        if p.exists():
            load_checkpoint(model, p)
            model.eval()
            print(f"  Model: {name}")
            return model

    raise FileNotFoundError("No checkpoint found. Run train_v2.py first.")


def predict_for_date(target_date: str, verbose: bool = True) -> list:
    """
    Run model inference for a specific date.

    Returns list of dicts:
        [{node_id, country, lat, lon, p_conflict, p_unrest, risk_level}, ...]
    sorted by p_conflict descending.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config()
    nodes  = load_nodes()
    N      = len(nodes)
    window = cfg.get("window", 7)

    if verbose:
        print(f"\nPredicting conflict risk for: {target_date}")
        print(f"  Window: {window} days  |  Nodes: {N}  |  Device: {device}")

    # Build full feature matrix
    X, Y, dates, feature_cols = build_matrices_from_parquet(nodes)
    F = len(feature_cols)

    if target_date not in dates:
        # Find closest available date
        available = [d for d in dates if d <= target_date]
        if not available:
            raise ValueError(f"No data available before {target_date}")
        target_date = available[-1]
        if verbose:
            print(f"  ⚠️  Exact date not found — using {target_date}")

    t = dates.index(target_date)

    if t < window:
        raise ValueError(
            f"Not enough history for window={window}. "
            f"Need at least {window} days before {target_date}."
        )

    # Build input window
    X_tensor = torch.tensor(X, dtype=torch.float32)
    x_window = X_tensor[t - window : t]                      # (window, N, F)
    x_flat   = x_window.permute(1, 0, 2).reshape(N, -1)     # (N, window*F)

    edge_index, edge_weight = load_graph()

    model = load_model(cfg, F, device)

    x_flat     = x_flat.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        logits = model.forward_logits(x_flat, edge_index)
        probs  = torch.sigmoid(logits).cpu().numpy()  # (N, 2)

    results = []
    for j, node in enumerate(nodes):
        p_c = float(probs[j, 0])
        p_u = float(probs[j, 1])
        results.append({
            "node_id":    node["id"],
            "country":    node["country"],
            "lat":        node["lat"],
            "lon":        node["lon"],
            "p_conflict": round(p_c, 4),
            "p_unrest":   round(p_u, 4),
            "risk_level": (
                "HIGH"     if p_c >= 0.6  else
                "MEDIUM"   if p_c >= 0.35 else
                "ELEVATED" if p_c >= 0.15 else
                "LOW"
            ),
        })

    results.sort(key=lambda x: x["p_conflict"], reverse=True)
    return results


def print_results(results: list, date: str):
    print(f"\n{'─'*65}")
    print(f"  Conflict risk scores — {date}")
    print(f"{'─'*65}")
    print(f"  {'City':20s} {'Country':8s} {'Conflict':>10} "
          f"{'Unrest':>8} {'Level':>10}")
    print(f"  {'─'*60}")

    for r in results:
        color_map = {"HIGH": "🔴", "MEDIUM": "🟠", "ELEVATED": "🟡", "LOW": "🟢"}
        icon = color_map.get(r["risk_level"], "⚪")
        print(f"  {r['node_id']:20s} {r['country']:8s} "
              f"{r['p_conflict']:10.1%} {r['p_unrest']:8.1%} "
              f"{icon} {r['risk_level']:>8}")

    high   = [r for r in results if r["risk_level"] == "HIGH"]
    medium = [r for r in results if r["risk_level"] == "MEDIUM"]
    print(f"\n  🔴 HIGH: {len(high)}  🟠 MEDIUM: {len(medium)}")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (optional — only if fastapi is installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Sahel Conflict Risk API",
        description="GNN-based conflict prediction for 25 Sahel cities",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "name": "Sahel Conflict Risk API",
            "version": "1.0.0",
            "endpoints": {
                "/predict/{date}": "Risk scores for all 25 nodes on given date (YYYY-MM-DD)",
                "/nodes": "List of all 25 city nodes",
                "/health": "Health check",
            }
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/nodes")
    def get_nodes():
        nodes = load_nodes()
        return {"nodes": nodes, "count": len(nodes)}

    @app.get("/predict/{date}")
    def predict(date: str):
        try:
            results = predict_for_date(date, verbose=False)
            return {
                "date":    date,
                "model":   "GAT-v2",
                "nodes":   results,
                "summary": {
                    "high":     sum(1 for r in results if r["risk_level"] == "HIGH"),
                    "medium":   sum(1 for r in results if r["risk_level"] == "MEDIUM"),
                    "elevated": sum(1 for r in results if r["risk_level"] == "ELEVATED"),
                    "low":      sum(1 for r in results if r["risk_level"] == "LOW"),
                }
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

except ImportError:
    app = None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sahel conflict risk predictor")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: latest available)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save predictions to outputs/predictions_{date}.csv"
    )
    args = parser.parse_args()

    # Use latest date if not specified
    if args.date is None:
        from model.dataset import build_matrices_from_parquet, load_nodes
        nodes = load_nodes()
        _, _, dates, _ = build_matrices_from_parquet(nodes)
        args.date = dates[-1]
        print(f"No date specified — using latest: {args.date}")

    results = predict_for_date(args.date)
    print_results(results, args.date)

    if args.save:
        df = pd.DataFrame(results)
        out = OUTPUTS_DIR / f"predictions_{args.date}.csv"
        df.to_csv(out, index=False)
        print(f"\n  ✅ Saved → {out}")
