"""
Evaluation harness for the Sahel conflict/unrest predictor.

predict signature:
    predict(X_window: np.ndarray) -> np.ndarray
    X_window shape : (N, F)
    output shape   : (N, 2)  — columns [p_conflict, p_unrest]

Primary metric: Average Precision (area under PR curve).
"""

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.splits import SPLITS, get_split_data

DATASET_PATH = ROOT / "data" / "processed" / "dataset_daily.parquet"
OUTPUTS_DIR  = ROOT / "outputs"

PredictFn = Callable[[np.ndarray], np.ndarray]

_SKIP_COLS = {"node_id", "date", "y_conflict", "y_unrest", "fatalities"}


class DummyBaseline:
    """Always predicts 0 — the performance floor every model must beat."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 2), dtype=np.float32)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    ap  = float(average_precision_score(y_true, y_prob))
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    cm  = confusion_matrix(y_true, y_pred).tolist()
    return {"AP": round(ap, 4), "AUROC": round(auc, 4), "F1@0.5": round(f1, 4),
            "confusion_matrix": cm}


def evaluate(predict_fn: PredictFn, model_name: str = "model") -> dict:
    """Run walk-forward evaluation across all SPLITS and save results."""
    df = pd.read_parquet(DATASET_PATH)
    feature_cols = [c for c in df.columns if c not in _SKIP_COLS]

    results: dict = {"model": model_name, "splits": []}

    for i, split in enumerate(SPLITS, 1):
        _, val = get_split_data(df, split)
        X          = val[feature_cols].to_numpy(dtype=np.float32)
        y_conflict = val["y_conflict"].to_numpy()
        y_unrest   = val["y_unrest"].to_numpy()

        probs      = predict_fn(X)
        p_conflict, p_unrest = probs[:, 0], probs[:, 1]

        results["splits"].append({
            "split":     i,
            "val_range": f"{split['val_start']} → {split['val_end']}",
            "conflict":  _metrics(y_conflict, p_conflict),
            "unrest":    _metrics(y_unrest,   p_unrest),
        })

    # Mean across splits per target/metric
    agg: dict = {}
    for tgt in ("conflict", "unrest"):
        for metric in ("AP", "AUROC", "F1@0.5"):
            vals = [
                s[tgt][metric]
                for s in results["splits"]
                if not np.isnan(s[tgt][metric])
            ]
            agg.setdefault(tgt, {})[metric] = (
                round(float(np.mean(vals)), 4) if vals else float("nan")
            )
    results["aggregated"] = agg

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "evaluation_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved → {out_path}")

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"Model: {results['model']}")
    print(f"{'=' * width}")
    print(f"{'Split':<8}{'Target':<12}{'AP':>8}{'AUROC':>8}{'F1@0.5':>8}")
    print(f"{'-' * width}")
    for s in results["splits"]:
        for tgt in ("conflict", "unrest"):
            m = s[tgt]
            print(
                f"{s['split']:<8}{tgt:<12}"
                f"{m['AP']:>8.4f}{m['AUROC']:>8.4f}{m['F1@0.5']:>8.4f}"
            )
    print(f"{'-' * width}")
    agg = results.get("aggregated", {})
    for tgt in ("conflict", "unrest"):
        m = agg.get(tgt, {})
        print(
            f"{'mean':<8}{tgt:<12}"
            f"{m.get('AP', float('nan')):>8.4f}"
            f"{m.get('AUROC', float('nan')):>8.4f}"
            f"{m.get('F1@0.5', float('nan')):>8.4f}"
        )
    print(f"{'=' * width}\n")


if __name__ == "__main__":
    baseline = DummyBaseline()
    evaluate(baseline.predict, model_name="DummyBaseline")
