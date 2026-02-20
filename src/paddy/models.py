from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def get_regression_models(random_state: int = 42) -> list[ModelSpec]:
    baseline = ModelSpec(name="ridge_baseline", estimator=Ridge())
    interpretable = ModelSpec(
        name="random_forest_interpretable",
        estimator=RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    )
    benchmark = ModelSpec(
        name="hist_gradient_boosting_benchmark",
        estimator=HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.08,
            max_depth=8,
            max_iter=300,
        ),
    )
    return [baseline, interpretable, benchmark]


def get_semi_supervised_base_classifier(random_state: int = 42):
    return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
