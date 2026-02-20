from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET_COLUMN = "Paddy yield(in Kg)"
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

KEY_APP_FEATURES = [
    "Hectares ",
    "Agriblock",
    "Variety",
    "Soil Types",
    "Seedrate(in Kg)",
    "Urea_40Days",
    "30DRain( in mm)",
    "Relative Humidity_D1_D30",
]


@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    outdir: Path
    target_column: str = TARGET_COLUMN
    random_seed: int = RANDOM_SEED
    test_size: float = TEST_SIZE
