from pathlib import Path

from paddy.train import run_training


def test_training_outputs_metrics_and_model(tmp_path: Path):
    payload = run_training("data/raw/paddydataset.csv", tmp_path)

    assert "models" in payload
    assert payload["best_model"]

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics" / "supervised_metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    first_model = payload["models"][0]
    for k in ["rmse", "mae", "r2"]:
        assert k in first_model
