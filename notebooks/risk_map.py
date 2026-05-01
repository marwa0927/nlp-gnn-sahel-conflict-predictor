"""
notebooks/risk_map.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import importlib.util
import json
import folium
from folium.plugins import HeatMap
from torch_geometric.loader import DataLoader

from model.dataset import SahelConflictDataset
from model.gcn import build_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


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
    return {"window": 7, "hidden_dim": 64, "dropout": 0.3}


def get_latest_predictions(nodes, cfg, device):
    """
    Run model on the most recent 30 days and return
    per-node average conflict probability.
    Uses fold_3 checkpoint (trained on most data).
    """
    ckpt_path = CKPT_DIR / "best_fold_3.pt"
    if not ckpt_path.exists():
        print("⚠️  best_fold_3.pt not found. Using fold_1.")
        ckpt_path = CKPT_DIR / "best_fold_1.pt"

    ds = SahelConflictDataset(
        window=cfg.get("window", 7), horizon=1,
        split="test",
        val_start="2024-01-01", test_start="2024-06-01",
    )

    if len(ds) == 0:
        print("⚠️  No test samples. Using full dataset last 30 samples.")
        ds = SahelConflictDataset(window=cfg.get("window", 7), split=None)

    # Use last 30 samples
    ds.sample_indices = ds.sample_indices[-min(30, len(ds.sample_indices)):]

    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_model(
        window=cfg.get("window", 7),
        n_features=ds.F,
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    N = len(nodes)
    all_probs_c = []
    all_probs_u = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model.forward_logits(
                batch.x, batch.edge_index, batch.edge_attr
            )
            p = torch.sigmoid(logits)
            all_probs_c.extend(p[:, 0].cpu().tolist())
            all_probs_u.extend(p[:, 1].cpu().tolist())

    # Average per node over the 30-day window
    probs_arr_c = np.array(all_probs_c).reshape(-1, N)
    probs_arr_u = np.array(all_probs_u).reshape(-1, N)

    node_probs = []
    for j, node in enumerate(nodes):
        node_probs.append({
            "node_id":    node["id"],
            "country":    node["country"],
            "lat":        node["lat"],
            "lon":        node["lon"],
            "p_conflict": float(probs_arr_c[:, j].mean()),
            "p_unrest":   float(probs_arr_u[:, j].mean()),
        })

    return pd.DataFrame(node_probs)


def build_risk_map(df_preds: pd.DataFrame) -> folium.Map:
    """Build a Folium map with colored circles per node."""

    m = folium.Map(
        location=[14.5, 2.0],
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    # Color scale: green → orange → red
    def risk_color(p):
        if p >= 0.6:
            return "#E63946"   # red
        elif p >= 0.35:
            return "#F4A261"   # orange
        elif p >= 0.15:
            return "#FFD166"   # yellow
        else:
            return "#2A9D8F"   # green

    def risk_label(p):
        if p >= 0.6:   return "HIGH"
        elif p >= 0.35: return "MEDIUM"
        elif p >= 0.15: return "ELEVATED"
        else:           return "LOW"

    # Add circles
    for _, row in df_preds.iterrows():
        p_c   = row["p_conflict"]
        p_u   = row["p_unrest"]
        color = risk_color(p_c)
        label = risk_label(p_c)

        popup_html = f"""
        <div style='font-family: monospace; font-size: 13px; min-width: 180px;'>
            <b>{row['node_id'].upper()}</b> ({row['country']})<br>
            <hr style='margin: 4px 0;'>
            <b>Conflict risk:</b>
            <span style='color:{color}; font-weight:bold;'>
                {p_c:.1%} — {label}
            </span><br>
            <b>Unrest risk:</b> {p_u:.1%}<br>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8 + p_c * 18,           # bigger circle = higher risk
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['node_id']} — {p_c:.1%} conflict risk",
        ).add_to(m)

    # Heatmap layer (conflict only)
    heat_data = [
        [row["lat"], row["lon"], row["p_conflict"]]
        for _, row in df_preds.iterrows()
        if row["p_conflict"] > 0.1
    ]
    if heat_data:
        HeatMap(
            heat_data,
            name="Conflict Heatmap",
            min_opacity=0.2,
            max_zoom=8,
            radius=40,
            blur=30,
            gradient={0.2: "blue", 0.5: "orange", 1.0: "red"},
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background-color: #1a1a2e; border: 2px solid #444;
        border-radius: 8px; padding: 12px 16px;
        font-family: monospace; color: white; font-size: 12px;
    '>
        <b>Conflict Risk</b><br><br>
        <span style='color:#E63946;'>⬤</span> HIGH    (&gt;60%)<br>
        <span style='color:#F4A261;'>⬤</span> MEDIUM  (35–60%)<br>
        <span style='color:#FFD166;'>⬤</span> ELEVATED(15–35%)<br>
        <span style='color:#2A9D8F;'>⬤</span> LOW     (&lt;15%)<br>
        <br><i>Circle size ∝ risk score</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes  = load_nodes()
    cfg    = load_best_config()

    print("Generating predictions for risk map...")
    df_preds = get_latest_predictions(nodes, cfg, device)

    print("\nNode risk scores:")
    print(f"  {'Node':20s} {'p_conflict':>12} {'p_unrest':>10} {'Risk':>10}")
    print(f"  {'─'*55}")
    for _, row in df_preds.sort_values("p_conflict", ascending=False).iterrows():
        label = (
            "HIGH"     if row["p_conflict"] >= 0.6  else
            "MEDIUM"   if row["p_conflict"] >= 0.35 else
            "ELEVATED" if row["p_conflict"] >= 0.15 else
            "LOW"
        )
        print(f"  {row['node_id']:20s} {row['p_conflict']:12.4f} "
              f"{row['p_unrest']:10.4f} {label:>10}")

    print("\nBuilding Folium map...")
    m = build_risk_map(df_preds)

    out_path = OUTPUTS_DIR / "sahel_risk_map.html"
    m.save(str(out_path))
    print(f"✅ Risk map saved → {out_path}")
    print("   Open this file in your browser to view the interactive map.")

    # Also save predictions CSV
    csv_path = OUTPUTS_DIR / "latest_predictions.csv"
    df_preds.to_csv(csv_path, index=False)
    print(f"✅ Predictions CSV → {csv_path}")


if __name__ == "__main__":
    main()
