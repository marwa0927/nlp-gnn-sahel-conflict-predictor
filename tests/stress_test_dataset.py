"""
tests/stress_test_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from model.dataset import SahelConflictDataset


def run_stress_tests():
    print("=" * 60)
    print("  Dataset stress tests (Week 2)")
    print("=" * 60)

    results = {}

    # ── Test 1: Different window sizes ───────────────────────────────────────
    print("\n[1] Window size produces correct input dim...")
    for window in [3, 7, 14]:
        ds = SahelConflictDataset(window=window, split=None)
        expected_dim = window * ds.F
        actual_dim   = ds.get(0).x.shape[1]
        ok = actual_dim == expected_dim
        print(f"    window={window:2d} → expected_dim={expected_dim}  "
              f"actual_dim={actual_dim}  {'✅' if ok else '❌'}")
        results[f"window_{window}"] = ok

    # Reset to default window
    ds = SahelConflictDataset(window=7, split=None)

    # ── Test 2: Rolling features differ from base features ───────────────────
    print("\n[2] Rolling features are not identical to base features...")
    X = ds.X.numpy()  # (T, N, F)
    # avg_tone is col 0, avg_tone_lag7 is col 6 (after 6 base features)
    base_col    = 0
    lag_col     = ds.F // 4  # approximate position of lag features
    correlation = np.corrcoef(
        X[:, 0, base_col].flatten(),
        X[:, 0, lag_col].flatten()
    )[0, 1]
    # Should be correlated (same signal) but not identical (different time)
    ok = 0.3 < abs(correlation) < 1.0
    print(f"    Correlation between base and lag feature: {correlation:.4f}  "
          f"{'✅' if ok else '⚠️ (might be identical or unrelated)'}")
    results["rolling_differs"] = ok

    # ── Test 3: No node has all-zero features for entire range ───────────────
    print("\n[3] Checking node feature coverage...")
    zero_nodes = []
    for j in range(ds.N):
        node_features = X[:, j, :]  # (T, F)
        if np.all(node_features == 0):
            zero_nodes.append(ds.nodes[j]["id"])
    ok = len(zero_nodes) == 0
    if ok:
        print(f"    ✅ All {ds.N} nodes have at least some non-zero features")
    else:
        print(f"    ⚠️  {len(zero_nodes)} nodes are ALL ZERO: {zero_nodes}")
        print(f" no GDELT data for these cities.")
    results["node_coverage"] = ok

    # ── Test 4: Temporal ordering ─────────────────────────────────────────────
    print("\n[4] Checking temporal ordering of samples...")
    target_dates = [
        ds.dates[ds.sample_indices[i] + ds.horizon - 1]
        for i in range(min(20, len(ds)))
    ]
    ordered = all(target_dates[i] < target_dates[i+1]
                  for i in range(len(target_dates) - 1))
    ok = ordered
    print(f"    First 5 target dates: {target_dates[:5]}")
    print(f"    Dates are in order: {'✅' if ok else '❌'}")
    results["temporal_order"] = ok

    # ── Test 5: Feature value ranges ─────────────────────────────────────────
    print("\n[5] Checking feature value ranges...")
    X_nonzero = X[X != 0]
    if len(X_nonzero) > 0:
        p1,  p99 = np.percentile(X_nonzero, [1, 99])
        mean_val  = X_nonzero.mean()
        std_val   = X_nonzero.std()
        # Flag if values are extreme (likely un-normalized)
        ok = abs(mean_val) < 1000 and std_val < 10000
        print(f"    Non-zero values: mean={mean_val:.3f}  std={std_val:.3f}  "
              f"p1={p1:.3f}  p99={p99:.3f}")
        if not ok:
            print(f"    ⚠️  Values seem extreme — consider normalizing features.")
        else:
            print(f"    ✅ Value ranges look reasonable")
    else:
        print(f"    ⚠️  All features are zero — no real data loaded.")
        ok = False
    results["value_ranges"] = ok

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Stress test results:")
    all_pass = True
    for name, result in results.items():
        status = "✅" if result else "⚠️ "
        print(f"    {status}  {name}")
        if not result:
            all_pass = False
    print(f"{'─'*50}")
    if all_pass:
        print("  ✅ All stress tests passed")
    else:
        print("  ⚠️  Some tests failed — review output above")


if __name__ == "__main__":
    run_stress_tests()
