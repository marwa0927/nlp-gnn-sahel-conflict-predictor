# Data Reproduction Guide

Follow these steps in order to reproduce all data files from scratch.

---

## Prerequisites

### 1. Copy and fill the environment file

```bash
cp .env.example .env
# Open .env and set your Google Cloud project ID:
#   GCP_PROJECT_ID=my-actual-project-id
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

Make sure the account used has **BigQuery Data Viewer** and **BigQuery Job User**
roles on `GCP_PROJECT_ID`.

---

## Step-by-step commands

### Step A — Pull raw GDELT data from BigQuery

```bash
python -m ingest.gdelt_pull
```

Output file: `data/raw/gdelt_sahel_raw.parquet`

This query is partition-filtered (`_PARTITIONTIME BETWEEN 2020-01-01 AND 2024-12-31`)
so you are only billed for ~5 years of Sahel events rather than the full GDELT table.
Estimated scan: ~50–80 GB depending on event volume.

### Step B — Build the daily feature matrix

```bash
python -m features.build_features
```

Output file: `data/processed/features_daily.parquet`

This script:
1. Loads `data/raw/gdelt_sahel_raw.parquet`
2. Geo-filters events, assigning each to the nearest of the 25 Sahel nodes (within 80 km)
3. Aggregates to (node_id, date) level
4. Adds 7-day lag and 7/14-day rolling features
5. Fills missing (node, date) pairs with 0.0

### Step C — Validate outputs

```bash
python data/validate.py
```

Prints shape, dtypes, missing counts, and top-5 conflict events.
Saves `outputs/validation_gao_tone.png` — a time-series plot of `avg_tone` for Gao
(expected visible dips around the 2021 Mali coup and 2022 Burkina Faso coups).

---

## ACLED supplementary data (optional)

ACLED provides geo-coded conflict event data that can complement GDELT.

1. Register for free access at: **https://acleddata.com/register/**
2. After approval, go to **Export** → set:
   - Region: **Africa**
   - Date range: **2018-01-01 → 2024-12-31**
   - Event types: **All**
3. Download as CSV and save to: `data/raw/acled_sahel_raw.csv`

ACLED integration is not yet scripted — planned for a future sprint.

---

## Directory layout after reproduction

```
data/
├── raw/
│   └── gdelt_sahel_raw.parquet     # ~50–200 MB
├── processed/
│   └── features_daily.parquet      # ~5–20 MB, 45,675 rows × 21 cols
├── nodes.py                        # 25-city node definitions (source of truth)
├── validate.py                     # validation / sanity-check script
└── README.md                       # this file
```
