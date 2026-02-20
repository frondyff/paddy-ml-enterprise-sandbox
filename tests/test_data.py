from pathlib import Path

from paddy.config import TARGET_COLUMN
from paddy.data import load_data


def test_load_data_has_target():
    df = load_data(Path("data/raw/paddydataset.csv"))
    assert TARGET_COLUMN in df.columns
    assert len(df) > 0
