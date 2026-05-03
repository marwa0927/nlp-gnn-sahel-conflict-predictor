# NLP-GNN Sahel Conflict Predictor

Predicts armed conflict risk across 25 cities in the Sahel region by combining
media-derived signals with a Graph Attention Network.

---

## How it works

1. **NLP layer** — pulls daily news events from GDELT, assigns them to 25 city nodes,
   and builds a feature vector per city per day (tone, event volume, conflict signal).
2. **Graph layer** — connects cities via a road-weighted geographic graph that captures
   how conflict spreads along Sahel highway corridors.
3. **GNN layer** — a Graph Attention Network learns to propagate risk signals across
   the graph and predict conflict probability for each city.

---

## Cities covered

Mali · Niger · Burkina Faso · Chad · Mauritania — 25 cities total.
Full node list in `data/nodes.py`.

---

## Quickstart

```bash
git clone https://github.com/marwa0927/nlp-gnn-sahel-conflict-predictor.git
cd nlp-gnn-sahel-conflict-predictor
pip install -r requirements.txt

cp .env.example .env          # add your GCP_PROJECT_ID
gcloud auth application-default login

# Run data pipeline
python -m ingest.gdelt_pull
python -m features.build_features
python -m data.build_dataset

# Train model
python model/train_v2.py

# Predict
python model/predict.py --date 2024-06-01
```

---

## Data sources

- [GDELT Project](https://www.gdeltproject.org/) — news event features (free, BigQuery)
- [ACLED](https://acleddata.com/) — ground truth conflict labels (free, research registration)

---

## License

MIT. ACLED data is subject to [ACLED's Terms of Use](https://acleddata.com/terms-of-use/).
