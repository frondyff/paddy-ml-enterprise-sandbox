from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Ensure `paddy` package can be imported when running from app/ in src layout.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paddy.config import KEY_APP_FEATURES, PROJECT_ROOT
from paddy.data import load_data

st.set_page_config(page_title="Paddy Yield Decision Support", layout="centered")
st.title("Paddy Yield Decision Support")
st.caption("Enterprise ML demo: regression prediction + top drivers (non-causal guidance)")

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "paddydataset.csv"
DEFAULT_MODEL = PROJECT_ROOT / "results" / "model.joblib"
FI_PATH = PROJECT_ROOT / "results" / "metrics" / "feature_importance.json"

if not DEFAULT_DATA.exists():
    st.error(f"Missing dataset: {DEFAULT_DATA}")
    st.stop()

df = load_data(DEFAULT_DATA)
if not DEFAULT_MODEL.exists():
    st.warning("Model artifact not found at results/model.joblib. Run training first.")
    st.stop()

model = joblib.load(DEFAULT_MODEL)

st.subheader("Input Features")
input_data = {}

for col in KEY_APP_FEATURES:
    if col not in df.columns:
        continue
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        default = float(series.median())
        min_v = float(series.min())
        max_v = float(series.max())
        input_data[col] = st.number_input(col, min_value=min_v, max_value=max_v, value=default)
    else:
        options = sorted(series.dropna().astype(str).unique().tolist())
        default_idx = 0
        input_data[col] = st.selectbox(col, options=options, index=default_idx)

if st.button("Predict Yield"):
    row = {c: df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else str(df[c].mode().iloc[0]) for c in df.columns if c != "Paddy yield(in Kg)"}
    row.update(input_data)

    X_pred = pd.DataFrame([row])
    pred = float(model.predict(X_pred)[0])

    st.success(f"Predicted Paddy Yield: {pred:,.2f} Kg")

    st.subheader("Top Drivers (Global Importance)")
    if FI_PATH.exists():
        fi = json.loads(FI_PATH.read_text(encoding="utf-8")).get("top_features", [])
        top = fi[:5]
        if top:
            for item in top:
                st.write(f"- {item['feature']}: {item['importance']:.4f}")

            top_names = ", ".join([t["feature"] for t in top[:3]])
            st.info(
                "Guidance (rule-based, non-causal): Prioritize stable control and monitoring of "
                f"{top_names}. Investigate feasible agronomic adjustments through domain experts."
            )
        else:
            st.write("Feature importance not available.")
    else:
        st.write("Run training to generate feature importance in results/metrics/feature_importance.json")
