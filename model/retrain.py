"""
model/retrain.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

# Just call train_best.py — it reads best_config.json automatically
print("=" * 60)
print("  Week 5 — Retraining with fixed parquet")
print("  This overwrites checkpoints with better versions.")
print("=" * 60)

result = subprocess.run(
    [sys.executable, str(Path(__file__).parent / "train_best.py")],
    cwd=str(Path(__file__).parent.parent),
)

if result.returncode == 0:
    print("\n✅ Retrain complete.")
    print("   Next: python model/full_evaluation.py")
    print("   Then: python notebooks/demo.py")
else:
    print("\n❌ Retrain failed — check output above.")
