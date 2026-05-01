"""
model/inspect_labels.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from model.dataset import SahelConflictDataset

ROOT = Path(__file__).parent.parent


def main():
    print("=" * 60)
    print("  Inspecting ACLED labels")
    print("=" * 60)

    ds = SahelConflictDataset(split=None)

    # Stack all labels
    print("\nLoading all labels (this takes ~30s)...")
    y_all = torch.stack([ds.get(i).y for i in range(len(ds))], dim=0)
    # y_all: (T, N, 2)

    T, N, _ = y_all.shape
    nodes    = ds.nodes

    # ── Overall rates ─────────────────────────────────────────────────────────
    rate_c = y_all[:, :, 0].mean().item()
    rate_u = y_all[:, :, 1].mean().item()
    print(f"\n  Overall conflict label rate: {rate_c:.4f}  "
          f"({y_all[:,:,0].sum().item():.0f} positive samples)")
    print(f"  Overall unrest   label rate: {rate_u:.4f}  "
          f"({y_all[:,:,1].sum().item():.0f} positive samples)")

    if rate_c == 0 and rate_u == 0:
        print("\n  ❌ LABELS ARE STILL ALL ZERO.")
        print("     The parquet does not contain y_conflict / y_unrest columns.")
        print("     ACLED labels needed joined in.")
        return

    if rate_c < 0.01:
        print("\n  ⚠️  Conflict label rate is very low (<1%).")
        print("     This is normal for conflict data but will need high pos_weight.")

    # ── Per-node label rates ──────────────────────────────────────────────────
    print(f"\n  Per-node conflict label rate:")
    print(f"  {'Node':20s}  {'Country':8s}  {'Rate':>8}  {'N events':>10}")
    print(f"  {'─'*55}")

    node_rates = []
    for j, node in enumerate(nodes):
        node_labels_c = y_all[:, j, 0]
        rate = node_labels_c.mean().item()
        n_events = node_labels_c.sum().item()
        node_rates.append((node["id"], node["country"], rate, n_events))

    node_rates.sort(key=lambda x: x[2], reverse=True)
    for nid, country, rate, n_events in node_rates:
        bar    = "█" * int(rate * 50)
        status = "⚠️ " if rate == 0 else "  "
        print(f"  {status}{nid:20s}  {country:8s}  {rate:8.4f}  {n_events:10.0f}  {bar}")

    zero_nodes = [r[0] for r in node_rates if r[2] == 0]
    if zero_nodes:
        print(f"\n  ⚠️  Nodes with ZERO conflict events: {zero_nodes}")
        print("     Check ACLED spatial join for these cities.")

    # ── Temporal distribution ─────────────────────────────────────────────────
    print(f"\n  Conflict events over time (yearly):")
    dates = ds.dates
    df_dates = pd.DataFrame({
        "date":       [dates[ds.sample_indices[i] + ds.horizon - 1]
                       for i in range(len(ds))],
        "y_conflict": y_all[:, :, 0].mean(dim=1).tolist(),
    })
    df_dates["year"] = df_dates["date"].str[:4]
    yearly = df_dates.groupby("year")["y_conflict"].mean()
    for year, rate in yearly.items():
        bar = "█" * int(rate * 100)
        print(f"    {year}: {rate:.4f}  {bar}")

    # ── Sanity check: known conflict years should have higher rates ───────────
    # Mali coup: 2021, Burkina coups: 2022, Niger coup: 2023
    print(f"\n  Sanity check — known high-conflict years:")
    for year in ["2021", "2022", "2023"]:
        if year in yearly:
            rate = yearly[year]
            flag = "✅" if rate > 0.05 else "⚠️  (lower than expected)"
            print(f"    {year}: {rate:.4f}  {flag}")

    print(f"\n{'─'*60}")
    if rate_c > 0:
        print("  ✅ Labels look ready")
    else:
        print("  ❌ Labels still zero")


if __name__ == "__main__":
    main()
