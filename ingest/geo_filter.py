"""
Assign GDELT events to the nearest Sahel city node within RADIUS_KM.
"""

from typing import Optional

import pandas as pd
from geopy.distance import geodesic

from config import RADIUS_KM
from data.nodes import NODES


def assign_node(lat: float, lon: float) -> Optional[str]:
    """Return the id of the nearest node within RADIUS_KM, or None."""
    best_id: Optional[str] = None
    best_dist = float("inf")

    for node in NODES:
        dist = geodesic((lat, lon), (node["lat"], node["lon"])).km
        if dist < best_dist:
            best_dist = dist
            best_id = node["id"]

    return best_id if best_dist <= RADIUS_KM else None


def assign_nodes_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``node_id`` column to *df* by calling :func:`assign_node` on every
    row, then drop rows that have no node within RADIUS_KM.

    Prints an assignment-rate summary to stdout.
    """
    total = len(df)

    df = df.copy()
    df["node_id"] = df.apply(
        lambda row: assign_node(row["lat"], row["lon"]), axis=1
    )

    assigned = df["node_id"].notna().sum()
    rate = assigned / total * 100 if total else 0.0
    print(f"Assigned {assigned:,} / {total:,} events ({rate:.1f}%)")

    df = df[df["node_id"].notna()].reset_index(drop=True)
    return df
