from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier

from paddy.config import RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from paddy.data import load_data, split_features_target
from paddy.explain import save_json
from paddy.features import build_classification_pipeline
from paddy.models import get_semi_supervised_base_classifier


def make_quantile_labels(y_train: pd.Series, y_all: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    q1 = float(y_train.quantile(0.33))
    q2 = float(y_train.quantile(0.66))

    def encode(v: float) -> int:
        if v <= q1:
            return 0
        if v <= q2:
            return 1
        return 2

    labels = y_all.apply(encode)
    thresholds = {"low_max": q1, "medium_max": q2}
    return labels, thresholds


def run_semisupervised(
    data_path: str | Path,
    outdir: str | Path,
    target_column: str = TARGET_COLUMN,
    unlabeled_fraction: float = 0.5,
) -> dict:
    outdir = Path(outdir)
    (outdir / "metrics").mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y = split_features_target(df, target_column=target_column)

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    y_train_cls, thresholds = make_quantile_labels(y_train_reg, y_train_reg)
    y_test_cls, _ = make_quantile_labels(y_train_reg, y_test_reg)

    rng = np.random.default_rng(RANDOM_SEED)
    mask_unlabeled = rng.random(len(y_train_cls)) < unlabeled_fraction

    y_train_partial = y_train_cls.copy().to_numpy()
    y_train_partial[mask_unlabeled] = -1

    labeled_idx = ~mask_unlabeled
    X_labeled = X_train.loc[labeled_idx]
    y_labeled = y_train_cls.loc[labeled_idx]

    base_clf = get_semi_supervised_base_classifier(random_state=RANDOM_SEED)
    supervised_pipe = build_classification_pipeline(X_train, base_clf, feature_selection=True)
    supervised_pipe.fit(X_labeled, y_labeled)
    sup_preds = supervised_pipe.predict(X_test)

    semi_base = get_semi_supervised_base_classifier(random_state=RANDOM_SEED)
    semi_pipe = build_classification_pipeline(X_train, semi_base, feature_selection=True)
    semi_model = SelfTrainingClassifier(estimator=semi_pipe, threshold=0.75, max_iter=10)
    semi_model.fit(X_train, y_train_partial)
    semi_preds = semi_model.predict(X_test)

    payload = {
        "target": target_column,
        "binned_target": "Low(0)/Medium(1)/High(2) based on train quantiles",
        "quantile_thresholds": thresholds,
        "unlabeled_fraction": unlabeled_fraction,
        "supervised_baseline": {
            "accuracy": float(accuracy_score(y_test_cls, sup_preds)),
            "macro_f1": float(f1_score(y_test_cls, sup_preds, average="macro")),
        },
        "semi_supervised_self_training": {
            "accuracy": float(accuracy_score(y_test_cls, semi_preds)),
            "macro_f1": float(f1_score(y_test_cls, semi_preds, average="macro")),
        },
    }

    save_json(payload, outdir / "metrics" / "semisupervised_metrics.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semi-supervised classification experiment")
    parser.add_argument("--data", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    parser.add_argument("--unlabeled_fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_semisupervised(
        data_path=args.data,
        outdir=args.outdir,
        target_column=args.target,
        unlabeled_fraction=args.unlabeled_fraction,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
