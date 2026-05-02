"""
notebooks/dashboard.py
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
from model.gat import build_gat_model, load_checkpoint

ROOT        = Path(__file__).parent.parent
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
GRAPH_DIR   = ROOT / "graph"
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
    return {"window": 7, "hidden_dim": 64, "heads": 4, "dropout": 0.3}


def load_v2_graph():
    ei_path = GRAPH_DIR / "edge_index_v2.npy"
    ew_path = GRAPH_DIR / "edge_weight_v2.npy"
    if not ei_path.exists():
        ei_path = GRAPH_DIR / "edge_index.npy"
        ew_path = GRAPH_DIR / "edge_weight.npy"
    return (torch.tensor(np.load(ei_path), dtype=torch.long),
            torch.tensor(np.load(ew_path), dtype=torch.float32))


def get_predictions_over_time(nodes, cfg, device):
    """
    Run inference over the full test period.
    Returns DataFrame with columns: date, node_id, lat, lon,
    country, p_conflict, p_unrest, risk_level
    """
    ckpt_path = None
    for name in ["v2_gat_best_fold_3", "v2_gat_best_fold_2",
                 "gat_best_fold_3", "gat_best_fold_2", "best_fold_3"]:
        p = CKPT_DIR / f"{name}.pt"
        if p.exists():
            ckpt_path = p
            print(f"Using checkpoint: {name}")
            break

    if ckpt_path is None:
        print("❌ No checkpoint found.")
        return pd.DataFrame()

    ds = SahelConflictDataset(
        window=cfg.get("window", 7), horizon=1,
        split="test", val_start="2023-01-01", test_start="2024-01-01",
    )
    if len(ds) == 0:
        ds = SahelConflictDataset(window=cfg.get("window", 7), split=None)

    # Override with v2 graph
    edge_index, edge_weight = load_v2_graph()
    ds.edge_index = edge_index
    ds.edge_attr  = edge_weight

    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_gat_model(
        window=cfg.get("window", 7), n_features=ds.F,
        hidden_dim=cfg.get("hidden_dim", 64),
        heads=cfg.get("heads", 4),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    N = len(nodes)
    all_probs_c, all_probs_u, all_labels_c = [], [], []
    sample_dates = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model.forward_logits(batch.x, batch.edge_index)
            p = torch.sigmoid(logits)
            all_probs_c.extend(p[:, 0].cpu().tolist())
            all_probs_u.extend(p[:, 1].cpu().tolist())
            all_labels_c.extend(batch.y[:, 0].cpu().tolist())

    n_samples = len(all_probs_c) // N
    arr_c = np.array(all_probs_c[:n_samples * N]).reshape(n_samples, N)
    arr_u = np.array(all_probs_u[:n_samples * N]).reshape(n_samples, N)

    # Get dates for each sample
    for i in range(n_samples):
        idx = ds.sample_indices[i]
        sample_dates.append(ds.dates[idx + ds.horizon - 1])

    # Overall metrics
    labels_flat = all_labels_c[:n_samples * N]
    probs_flat  = all_probs_c[:n_samples * N]
    try:
        auroc = roc_auc_score(labels_flat, probs_flat) if len(set(labels_flat)) > 1 else float("nan")
        ap    = average_precision_score(labels_flat, probs_flat) if len(set(labels_flat)) > 1 else float("nan")
    except Exception:
        auroc, ap = float("nan"), float("nan")

    # Build DataFrame
    records = []
    for s, date in enumerate(sample_dates):
        for j, node in enumerate(nodes):
            p_c = float(arr_c[s, j])
            p_u = float(arr_u[s, j])
            records.append({
                "date":       date,
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

    df = pd.DataFrame(records)
    print(f"✅ Predictions: {n_samples} days × {N} nodes  "
          f"AUROC={auroc:.4f}  AP={ap:.4f}")
    return df, auroc, ap


def build_dashboard(df_all: pd.DataFrame, auroc: float, ap: float) -> str:
    """Build self-contained HTML dashboard."""

    nodes_data = df_all.groupby("node_id").first()[["lat", "lon", "country"]].reset_index()
    dates      = sorted(df_all["date"].unique())

    # Latest predictions (last available date)
    latest_date = dates[-1]
    df_latest   = df_all[df_all["date"] == latest_date].copy()

    # Top 5 highest risk nodes (averaged over all dates)
    avg_risk    = df_all.groupby("node_id")["p_conflict"].mean().sort_values(ascending=False)
    top5_nodes  = avg_risk.head(5).index.tolist()

    # Time series data for top 5
    ts_data = {}
    for node in top5_nodes:
        node_df = df_all[df_all["node_id"] == node].sort_values("date")
        ts_data[node] = {
            "dates":  node_df["date"].tolist(),
            "values": node_df["p_conflict"].tolist(),
        }

    # Map markers data
    markers = []
    for _, row in df_latest.iterrows():
        p_c = row["p_conflict"]
        color = ("#E63946" if p_c >= 0.6 else
                 "#F4A261" if p_c >= 0.35 else
                 "#FFD166" if p_c >= 0.15 else
                 "#2A9D8F")
        markers.append({
            "id":      row["node_id"],
            "country": row["country"],
            "lat":     row["lat"],
            "lon":     row["lon"],
            "p_c":     p_c,
            "p_u":     row["p_unrest"],
            "color":   color,
            "risk":    row["risk_level"],
        })

    # All predictions as JSON for time slider
    all_data = {}
    for date in dates[-90:]:   # last 90 days
        day_df = df_all[df_all["date"] == date]
        all_data[date] = {
            row["node_id"]: {
                "p_c": row["p_conflict"],
                "p_u": row["p_unrest"],
                "risk": row["risk_level"],
            }
            for _, row in day_df.iterrows()
        }

    # Table rows
    table_rows = ""
    for _, row in df_latest.sort_values("p_conflict", ascending=False).iterrows():
        color_map = {
            "HIGH": "#E63946", "MEDIUM": "#F4A261",
            "ELEVATED": "#FFD166", "LOW": "#2A9D8F"
        }
        color = color_map.get(row["risk_level"], "#888")
        table_rows += f"""
        <tr>
            <td>{row['node_id']}</td>
            <td>{row['country']}</td>
            <td><div class="risk-bar"><div class="risk-fill"
                style="width:{row['p_conflict']*100:.1f}%;background:{color}"></div></div>
                {row['p_conflict']:.1%}</td>
            <td>{row['p_unrest']:.1%}</td>
            <td><span class="badge" style="background:{color}20;color:{color};border:1px solid {color}">
                {row['risk_level']}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sahel Conflict Risk Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: system-ui, sans-serif; background:#0f0f1a; color:#e0e0e0; }}
  header {{ background:#1a1a2e; padding:16px 24px; border-bottom:1px solid #333;
            display:flex; align-items:center; gap:16px; }}
  header h1 {{ font-size:18px; font-weight:600; color:#fff; }}
  header .subtitle {{ font-size:13px; color:#888; }}
  .metrics {{ display:flex; gap:12px; margin-left:auto; }}
  .metric {{ background:#0f0f1a; border:1px solid #333; border-radius:8px;
             padding:8px 16px; text-align:center; }}
  .metric .val {{ font-size:22px; font-weight:700; color:#2A9D8F; }}
  .metric .lbl {{ font-size:11px; color:#666; margin-top:2px; }}
  .layout {{ display:grid; grid-template-columns:1fr 340px; height:calc(100vh - 65px); }}
  #map {{ width:100%; height:100%; }}
  .sidebar {{ background:#1a1a2e; border-left:1px solid #333;
              overflow-y:auto; display:flex; flex-direction:column; }}
  .panel {{ padding:16px; border-bottom:1px solid #222; }}
  .panel h2 {{ font-size:13px; font-weight:600; color:#aaa; text-transform:uppercase;
               letter-spacing:.05em; margin-bottom:12px; }}
  .slider-row {{ display:flex; align-items:center; gap:10px; }}
  .slider-row input {{ flex:1; accent-color:#2A9D8F; }}
  .slider-row span {{ font-size:12px; color:#888; min-width:90px; text-align:right; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ color:#666; font-weight:500; text-align:left; padding:6px 4px;
        border-bottom:1px solid #222; }}
  td {{ padding:5px 4px; border-bottom:1px solid #1a1a2e; vertical-align:middle; }}
  .risk-bar {{ background:#222; border-radius:3px; height:6px;
               width:80px; display:inline-block; vertical-align:middle; margin-right:6px; }}
  .risk-fill {{ height:100%; border-radius:3px; }}
  .badge {{ font-size:10px; padding:2px 6px; border-radius:4px; font-weight:500; }}
  canvas {{ max-height:180px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:8px; font-size:11px; }}
  .legend-item {{ display:flex; align-items:center; gap:4px; }}
  .legend-dot {{ width:10px; height:10px; border-radius:50%; }}
  .date-display {{ font-size:14px; font-weight:600; color:#2A9D8F; text-align:center;
                   padding:6px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Sahel Conflict Risk Monitor</h1>
    <div class="subtitle">GAT v2 · Road-weighted graph · 25 cities · 2024</div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="val">{auroc:.3f}</div><div class="lbl">AUROC</div></div>
    <div class="metric"><div class="val">{ap:.3f}</div><div class="lbl">Avg Precision</div></div>
    <div class="metric"><div class="val">25</div><div class="lbl">Cities</div></div>
    <div class="metric"><div class="val">{len(dates)}</div><div class="lbl">Days</div></div>
  </div>
</header>

<div class="layout">
  <div id="map"></div>

  <div class="sidebar">

    <div class="panel">
      <h2>Time slider</h2>
      <div class="date-display" id="current-date">{latest_date}</div>
      <div class="slider-row" style="margin-top:8px">
        <input type="range" id="date-slider" min="0" max="{len(list(all_data.keys()))-1}"
               value="{len(list(all_data.keys()))-1}" oninput="updateDate(this.value)">
      </div>
      <div class="legend" style="margin-top:12px">
        <div class="legend-item"><div class="legend-dot" style="background:#E63946"></div>HIGH ≥60%</div>
        <div class="legend-item"><div class="legend-dot" style="background:#F4A261"></div>MEDIUM 35–60%</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FFD166"></div>ELEVATED 15–35%</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2A9D8F"></div>LOW &lt;15%</div>
      </div>
    </div>

    <div class="panel">
      <h2>Top 5 risk cities — trend</h2>
      <canvas id="trend-chart"></canvas>
    </div>

    <div class="panel">
      <h2>All cities — latest</h2>
      <table>
        <thead><tr><th>City</th><th>Country</th><th>Conflict</th><th>Unrest</th><th>Level</th></tr></thead>
        <tbody id="risk-table">{table_rows}</tbody>
      </table>
    </div>

  </div>
</div>

<script>
const ALL_DATA   = {json.dumps(all_data)};
const DATE_KEYS  = Object.keys(ALL_DATA);
const MARKERS_META = {json.dumps({m['id']: m for m in markers})};
const TS_DATA    = {json.dumps(ts_data)};
const TOP5       = {json.dumps(top5_nodes)};

const COLOR_FN = p => p >= 0.6 ? '#E63946' : p >= 0.35 ? '#F4A261' : p >= 0.15 ? '#FFD166' : '#2A9D8F';
const RISK_FN  = p => p >= 0.6 ? 'HIGH' : p >= 0.35 ? 'MEDIUM' : p >= 0.15 ? 'ELEVATED' : 'LOW';

// ── Map ──────────────────────────────────────────────────────────────────────
const map = L.map('map', {{center:[14.5, 2.0], zoom:5}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:18
}}).addTo(map);

const circleLayer = {{}};
Object.values(MARKERS_META).forEach(m => {{
  const c = L.circleMarker([m.lat, m.lon], {{
    radius: 8 + m.p_c * 18,
    color: m.color, fillColor: m.color,
    fillOpacity: 0.75, weight: 2,
  }}).addTo(map);
  c.bindPopup(`<b>${{m.id.toUpperCase()}}</b> (${{m.country}})<br>
    Conflict: <b>${{(m.p_c*100).toFixed(1)}}%</b> — ${{m.risk}}<br>
    Unrest: ${{(m.p_u*100).toFixed(1)}}%`);
  circleLayer[m.id] = c;
}});

// ── Time slider ──────────────────────────────────────────────────────────────
function updateDate(idx) {{
  const date = DATE_KEYS[idx];
  document.getElementById('current-date').textContent = date;
  const dayData = ALL_DATA[date];
  if (!dayData) return;
  Object.entries(dayData).forEach(([nodeId, d]) => {{
    const c = circleLayer[nodeId];
    const m = MARKERS_META[nodeId];
    if (!c || !m) return;
    const color = COLOR_FN(d.p_c);
    c.setRadius(8 + d.p_c * 18);
    c.setStyle({{color, fillColor: color}});
    c.setPopupContent(`<b>${{nodeId.toUpperCase()}}</b> (${{m.country}})<br>
      ${{date}}<br>Conflict: <b>${{(d.p_c*100).toFixed(1)}}%</b> — ${{d.risk}}<br>
      Unrest: ${{(d.p_u*100).toFixed(1)}}%`);
  }});
}}

// ── Trend chart ──────────────────────────────────────────────────────────────
const COLORS = ['#E63946','#F4A261','#2A9D8F','#457B9D','#6A4C93'];
const chartCtx = document.getElementById('trend-chart').getContext('2d');
new Chart(chartCtx, {{
  type: 'line',
  data: {{
    labels: TS_DATA[TOP5[0]]?.dates.slice(-60) || [],
    datasets: TOP5.map((node, i) => ({{
      label: node,
      data: TS_DATA[node]?.values.slice(-60) || [],
      borderColor: COLORS[i],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    }})),
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#aaa', font: {{ size: 11 }} }} }},
    }},
    scales: {{
      x: {{ display: false }},
      y: {{
        min: 0, max: 1,
        ticks: {{ color: '#666', callback: v => (v*100).toFixed(0)+'%' }},
        grid: {{ color: '#222' }},
      }},
    }},
  }},
}});
</script>
</body>
</html>"""

    return html


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes  = load_nodes()
    cfg    = load_best_config()

    print("Running inference over test period...")
    result = get_predictions_over_time(nodes, cfg, device)

    if isinstance(result, tuple):
        df_all, auroc, ap = result
    else:
        print("❌ No predictions generated.")
        return

    if df_all.empty:
        print("❌ Empty predictions dataframe.")
        return

    print("Building dashboard...")
    html = build_dashboard(df_all, auroc, ap)

    out_path = OUTPUTS_DIR / "sahel_risk_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard saved → {out_path}")
    print("   Open in browser to view the interactive dashboard.")

    # Also save predictions CSV
    csv_path = OUTPUTS_DIR / "all_predictions.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"✅ All predictions → {csv_path}")


if __name__ == "__main__":
    main()
