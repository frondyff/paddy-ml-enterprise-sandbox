from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_feature_importance(trained_pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model = trained_pipeline.named_steps["model"]
    preprocess = trained_pipeline.named_steps["preprocess"]

    # Build feature names from preprocess and align with optional feature selector.
    feature_names = np.array(preprocess.get_feature_names_out())
    if "select" in trained_pipeline.named_steps:
        mask = trained_pipeline.named_steps["select"].get_support()
        feature_names = feature_names[mask]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Permutation importance fallback for models without direct importances.
        pi = permutation_importance(trained_pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = pi.importances_mean
        # permutation importance on raw X maps to original columns
        feature_names = np.array(X.columns)

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = fi_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["model"], metrics_df["rmse"])
    plt.ylabel("RMSE (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.title("Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
