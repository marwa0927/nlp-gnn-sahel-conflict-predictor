"""
Pull GDELT v2 events for the Sahel region from BigQuery.

Cost control: queries are bounded by _PARTITIONTIME so BigQuery only scans
partitions within DATE_START..DATE_END.
"""

import os

import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

from config import (
    DATE_START,
    DATE_END,
    COUNTRY_CODES,
    CAMEO_CONFLICT,
    CAMEO_UNREST,
    GDELT_TABLE,
    RAW_GDELT_PATH,
)

load_dotenv()
PROJECT_ID = os.environ["GCP_PROJECT_ID"]


_COUNTRY_LIST = ", ".join(f"'{c}'" for c in COUNTRY_CODES)

QUERY = f"""
SELECT
    SQLDATE                              AS date,
    ActionGeo_Lat                        AS lat,
    ActionGeo_Long                       AS lon,
    ActionGeo_CountryCode                AS country_code,
    CAST(EventRootCode AS INT64)         AS event_root_code,
    GoldsteinScale                       AS goldstein,
    AvgTone                              AS avg_tone,
    NumArticles                          AS n_articles,
    Actor1Name                           AS actor1,
    Actor2Name                           AS actor2
FROM `{GDELT_TABLE}`
WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('{DATE_START}') AND TIMESTAMP('{DATE_END}')
    AND ActionGeo_CountryCode IN ({_COUNTRY_LIST})
    AND ActionGeo_Lat  IS NOT NULL
    AND ActionGeo_Long IS NOT NULL
"""


def pull_gdelt() -> pd.DataFrame:
    """Query BigQuery and return a cleaned DataFrame."""
    client = bigquery.Client(project=PROJECT_ID)

    print(f"Querying {GDELT_TABLE} for {DATE_START} → {DATE_END} …")
    df = client.query(QUERY).to_dataframe()
    print(f"  Raw rows fetched: {len(df):,}")

    # Parse SQLDATE integer (YYYYMMDD) → datetime
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    # Boolean conflict / unrest flags from CAMEO root codes
    df["is_conflict"] = df["event_root_code"].isin(CAMEO_CONFLICT)
    df["is_unrest"] = df["event_root_code"].isin(CAMEO_UNREST)

    return df


def save_raw(df: pd.DataFrame) -> None:
    """Persist raw pull to RAW_GDELT_PATH as parquet."""
    import pathlib

    out_path = pathlib.Path(RAW_GDELT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    df = pull_gdelt()
    save_raw(df)

    print("\n── Shape ──────────────────────────────────────────")
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    print("\n── dtypes ─────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Conflict / unrest counts ────────────────────────")
    print(f"  is_conflict=True : {df['is_conflict'].sum():,}")
    print(f"  is_unrest=True   : {df['is_unrest'].sum():,}")
