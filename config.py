"""Shared constants for the nlp-gnn-sahel-conflict-predictor project."""

DATE_START = "2020-01-01"
DATE_END = "2024-12-31"

COUNTRY_CODES = ["ML", "NI", "NE", "UV", "CD", "MR"]

CAMEO_CONFLICT = list(range(18, 21))   # 18, 19, 20
CAMEO_UNREST = list(range(14, 18))     # 14, 15, 16, 17

RADIUS_KM = 80

GDELT_TABLE = "gdelt-bq.gdeltv2.events"

RAW_GDELT_PATH = "data/raw/gdelt_sahel_raw.parquet"
FEATURES_PATH = "data/processed/features_daily.parquet"

BASE_FEATURES = [
    "avg_tone",
    "goldstein",
    "n_articles",
    "conflict_events",
    "unrest_events",
    "n_actors",
]
