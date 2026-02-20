from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from paddy.data import load_data, split_features_target
from paddy.features import build_regression_pipeline


def test_pipeline_contains_preprocess_and_selector():
    df = load_data(Path("data/raw/paddydataset.csv"))
    X, _ = split_features_target(df)
    pipe = build_regression_pipeline(X, RandomForestRegressor(n_estimators=10, random_state=42))

    assert "preprocess" in pipe.named_steps
    assert "select" in pipe.named_steps
    assert "model" in pipe.named_steps
