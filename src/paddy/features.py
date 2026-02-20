from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paddy.data import infer_feature_types


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols, categorical_cols = infer_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def build_regression_pipeline(X: pd.DataFrame, estimator, feature_selection: bool = True) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X)
    steps = [("preprocess", preprocessor)]
    if feature_selection:
        # Feature selection is fit strictly on train folds when pipeline is trained.
        steps.append(("select", SelectPercentile(score_func=f_regression, percentile=80)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def build_classification_pipeline(X: pd.DataFrame, estimator, feature_selection: bool = True) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X)
    steps = [("preprocess", preprocessor)]
    if feature_selection:
        steps.append(("select", SelectPercentile(score_func=f_classif, percentile=80)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def get_feature_names(preprocess: ColumnTransformer) -> np.ndarray:
    return preprocess.get_feature_names_out()
