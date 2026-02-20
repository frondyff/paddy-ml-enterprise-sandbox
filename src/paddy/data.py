from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from paddy.config import RANDOM_SEED, TARGET_COLUMN, TEST_SIZE


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load CSV and strip accidental column whitespace at ends."""
    df = pd.read_csv(data_path)
    df.columns = [c.strip() if c != "Hectares " else c for c in df.columns]
    return df


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols
