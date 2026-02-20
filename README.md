# Paddy Yield Decision Support Using Enterprise ML

## Project Goal
Build an enterprise-ready decision support pipeline that predicts paddy yield and supports actionable, leakage-safe analysis for agronomic planning.

## Hypotheses
1. Feature selection inside leakage-safe pipelines improves generalization versus using all transformed features.
2. Interpretable models (Random Forest) can provide competitive performance with clearer operational drivers than linear baselines.
3. Actionability improves when predictive outputs are paired with transparent top-driver guidance (non-causal recommendations).

## Hypotheses to GitHub Board Tasks
- H1 Feature Selection: `src/paddy/features.py`, pipeline selector and tests.
- H2 Model Comparison: `src/paddy/train.py`, baseline/interpretable/benchmark model training and metrics.
- H3 Actionability: `app/streamlit_app.py` + `src/paddy/explain.py` for model drivers and guidance text.

## Repository Structure
```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── paddydataset.csv
│   └── processed/
├── notebooks/
│   └── 01_initial_eda.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   └── paddy/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── explain.py
│       ├── features.py
│       ├── models.py
│       ├── semisupervised.py
│       └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_pipeline.py
│   └── test_train.py
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Setup (WSL/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
```bash
python -m paddy.train --data data/raw/paddydataset.csv --outdir results
python -m paddy.evaluate --data data/raw/paddydataset.csv --model_path results/model.joblib --outdir results
python -m paddy.semisupervised --data data/raw/paddydataset.csv --outdir results
streamlit run app/streamlit_app.py
pytest -q
```

## Leakage Prevention
- Train/test split is performed before preprocessing and model fitting.
- All imputation/encoding/scaling and feature selection are inside `Pipeline` + `ColumnTransformer`.
- Feature selection is fit only on train folds.
- Hyperparameter tuning uses `GridSearchCV` over training folds only.
- Test data is used exclusively for final evaluation.

## Semi-Supervised Demonstration
The semi-supervised module converts yield into Low/Medium/High classes using train quantile thresholds, masks a configurable fraction of training labels (default 50%), and compares:
- Supervised baseline classifier trained on labeled subset only.
- Self-training classifier that leverages unlabeled samples.

This reflects enterprise conditions where labels are partially available.

## Outputs
- Supervised: `results/metrics/supervised_metrics.json`
- Evaluation: `results/metrics/evaluation_metrics.json`
- Semi-supervised: `results/metrics/semisupervised_metrics.json`
- Figures: `results/figures/feature_importance.png`, `results/figures/model_rmse_comparison.png`, `results/figures/pred_vs_actual.png`

## Current vs Next
- Current: End-to-end code-first pipeline, supervised benchmarking, semi-supervised demonstration, tests, and Streamlit demo.
- Next: Add LightGBM/XGBoost benchmark when environment allows, deeper error analysis, and CI/CD + deployment hardening.
