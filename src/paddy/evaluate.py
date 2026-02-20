from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from paddy.config import RANDOM_SEED, TARGET_COLUMN
from paddy.data import load_data, split_features_target, train_test_split_regression
from paddy.explain import save_json


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_evaluation(
    data_path: str | Path,
    model_path: str | Path,
    outdir: str | Path,
    target_column: str = TARGET_COLUMN,
) -> dict:
    outdir = Path(outdir)
    (outdir / "metrics").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y = split_features_target(df, target_column=target_column)
    _, X_test, _, y_test = train_test_split_regression(X, y, random_state=RANDOM_SEED)

    model = joblib.load(model_path)
    preds = model.predict(X_test)
    m = regression_metrics(y_test, preds)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    lo = min(float(y_test.min()), float(preds.min()))
    hi = max(float(y_test.max()), float(preds.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Predicted vs Actual Yield")
    plt.tight_layout()
    fig_path = outdir / "figures" / "pred_vs_actual.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    payload = {
        "target": target_column,
        "model_path": str(model_path),
        "evaluation": m,
        "figure": str(fig_path),
    }
    save_json(payload, outdir / "metrics" / "evaluation_metrics.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained paddy yield model")
    parser.add_argument("--data", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--model_path", required=True, help="Path to trained model artifact")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_evaluation(args.data, args.model_path, args.outdir, target_column=args.target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
