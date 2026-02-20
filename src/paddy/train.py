from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from paddy.config import CV_FOLDS, RANDOM_SEED, TARGET_COLUMN
from paddy.data import load_data, split_features_target, train_test_split_regression
from paddy.explain import extract_feature_importance, plot_feature_importance, plot_model_comparison, save_json
from paddy.features import build_regression_pipeline
from paddy.models import get_regression_models


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_training(data_path: str | Path, outdir: str | Path, target_column: str = TARGET_COLUMN) -> dict:
    outdir = Path(outdir)
    (outdir / "metrics").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y = split_features_target(df, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split_regression(X, y, random_state=RANDOM_SEED)

    rows = []
    trained_models = {}

    for spec in get_regression_models(random_state=RANDOM_SEED):
        pipe = build_regression_pipeline(X_train, spec.estimator, feature_selection=True)

        if spec.name == "hist_gradient_boosting_benchmark":
            search = GridSearchCV(
                pipe,
                param_grid={
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_iter": [200, 300],
                },
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            trained = search.best_estimator_
            tuning = {"best_params": search.best_params_, "best_cv_score": float(search.best_score_)}
        else:
            trained = pipe.fit(X_train, y_train)
            tuning = {}

        preds = trained.predict(X_test)
        m = regression_metrics(y_test, preds)
        row = {"model": spec.name, **m, **tuning}
        rows.append(row)
        trained_models[spec.name] = trained

    metrics_df = pd.DataFrame(rows).sort_values(by="rmse", ascending=True)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    model_path = outdir / "model.joblib"
    joblib.dump(best_model, model_path)

    fi_source_name = "random_forest_interpretable"
    fi_model = trained_models.get(fi_source_name, best_model)
    fi_df = extract_feature_importance(fi_model, X_test, y_test)
    fi_path = outdir / "metrics" / "feature_importance.json"
    save_json({"top_features": fi_df.head(20).to_dict(orient="records")}, fi_path)

    plot_feature_importance(fi_df, outdir / "figures" / "feature_importance.png")
    plot_model_comparison(metrics_df, outdir / "figures" / "model_rmse_comparison.png")

    payload = {
        "target": target_column,
        "best_model": best_model_name,
        "models": metrics_df.to_dict(orient="records"),
        "model_path": str(model_path),
    }
    save_json(payload, outdir / "metrics" / "supervised_metrics.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised paddy yield models")
    parser.add_argument("--data", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_training(args.data, args.outdir, target_column=args.target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
