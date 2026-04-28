# NLP-GNN Sahel Conflict Predictor

A research system that predicts armed conflict risk across 25 cities in the Sahel
region using media-derived signals processed through a three-layer deep learning
architecture.

---

## What it predicts

Given the last 14 days of geo-coded news event signals for each of 25 Sahel cities,
the system outputs a **conflict risk score** (0–1) per city for the next 7 days.
The model learns both the temporal dynamics at each location (via lag/rolling features)
and the spatial spread of conflict between neighbouring cities (via graph edges).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — NLP Semantic Sensor                              │
│  Source: GDELT BigQuery (gdelt-bq.gdeltv2.events)          │
│  • Filter Sahel events by country code and geo-bounding     │
│  • Extract tone, Goldstein scale, CAMEO event codes         │
│  • Assign events to nearest of 25 city nodes (≤ 80 km)     │
└────────────────────────┬────────────────────────────────────┘
                         │  data/processed/features_daily.parquet
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — Structural Graph                                  │
│  Source: data/nodes.py (25 nodes, lat/lon)                  │
│  • Build k-NN spatial graph between city nodes              │
│  • Edge weights derived from geographic distance and        │
│    historical conflict co-occurrence                        │
└────────────────────────┬────────────────────────────────────┘
                         │  graph/
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — GNN Predictive Core                              │
│  • Temporal Graph Convolutional Network (T-GCN or similar)  │
│  • Node-level binary classification: conflict in next 7d    │
│  • Trained on 2020-01-01 → 2023-12-31, validated on 2024   │
└─────────────────────────────────────────────────────────────┘
```

---

## The 25 nodes

| Country | Cities |
|---------|--------|
| Mali (ML) | Bamako, Mopti, Gao, Timbuktu, Kidal, Menaka, Ansongo |
| Niger (NI) | Niamey, Agadez, Zinder, Tahoua, Diffa, Tillaberi, Maradi |
| Burkina Faso (UV) | Ouagadougou, Dori, Djibo, Bobo-Dioulasso, Kaya |
| Chad (CD) | N'Djamena, Abéché, Mao, Mongo |
| Mauritania (MR) | Nouakchott, Néma |

Node definitions (id, lat, lon, country) live in `data/nodes.py` and serve as
the **shared source of truth** for both the data pipeline and the graph builder.

---

## Team split

| Person | Owns | Deliverables |
|--------|------|--------------|
| **Person A** (Data & Features) | `data/`, `ingest/`, `features/` | Raw GDELT pull, geo-filtering, daily feature matrix |
| **Person B** (Graph & GNN) | `graph/`, `notebooks/` | Graph construction, GNN model, training loop |

The handoff contract between Person A and Person B is documented in **`SCHEMA.md`**.

---

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/marwa0927/nlp-gnn-sahel-conflict-predictor.git
cd nlp-gnn-sahel-conflict-predictor
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — set GCP_PROJECT_ID to your Google Cloud project
gcloud auth application-default login
```

### 3. Reproduce data (Person A pipeline)

```bash
python -m ingest.gdelt_pull          # → data/raw/gdelt_sahel_raw.parquet
python -m features.build_features    # → data/processed/features_daily.parquet
python data/validate.py              # sanity checks + outputs/validation_gao_tone.png
```

See `data/README.md` for detailed instructions.

### 4. Build graph and train GNN (Person B pipeline)

```bash
python graph/build_graph.py          # reads data/nodes.py + features_daily.parquet
python graph/visualize_graph.py
# training notebook: notebooks/
```

---

## Project layout

```
.
├── config.py               # Shared constants (dates, codes, paths)
├── .env.example            # Environment variable template
├── SCHEMA.md               # Person A ↔ Person B data contract
├── requirements.txt        # Python dependencies (no torch — Person B installs separately)
├── data/
│   ├── nodes.py            # 25-city node definitions
│   ├── validate.py         # Feature matrix validation
│   └── README.md           # Data reproduction guide
├── ingest/
│   ├── gdelt_pull.py       # BigQuery GDELT pull
│   └── geo_filter.py       # Event-to-node assignment
├── features/
│   └── build_features.py   # Daily aggregation + lag features
├── graph/
│   ├── build_graph.py      # (Person B) Spatial graph builder
│   └── visualize_graph.py  # (Person B) Graph visualisation
├── outputs/                # Plots and validation artefacts
└── notebooks/              # (Person B) Training experiments
```
