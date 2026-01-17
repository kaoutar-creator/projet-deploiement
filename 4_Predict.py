from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import (
    ARTIFACT_PATH,
    SessionKeys,
    compute_cleaned,
    get_feature_columns,
    get_raw_df,
    require_target_column,
)
from src.modeling import load_artifact


st.set_page_config(page_title="Predict • Credit Risk", layout="wide")

st.title("Predict")
st.caption("Use a saved model (preferred) or the best in-session model to predict risk for new inputs.")

raw_df = get_raw_df()
require_target_column(raw_df)

clean_df, _report = compute_cleaned(raw_df)
num_cols, cat_cols = get_feature_columns(clean_df)

# Resolve model source
pipe = None
meta: dict[str, object] = {}
numeric_ranges: dict[str, dict[str, float]] = {}

if ARTIFACT_PATH.exists():
    try:
        loaded = load_artifact(ARTIFACT_PATH)
        pipe = loaded["pipeline"]
        meta = loaded.get("metadata", {}) or {}
        numeric_ranges = (meta.get("metrics") or {}).get("numeric_ranges", {})  # type: ignore[assignment]
        st.success(f"Using saved model: `{ARTIFACT_PATH}`")
    except Exception as e:
        st.warning(f"Saved model exists but could not be loaded: {e}")

if pipe is None:
    if SessionKeys.MODEL_RESULTS not in st.session_state or SessionKeys.MODEL_PIPES not in st.session_state:
        st.info("Train and save a model first (see **Modeling** page).")
        st.stop()
    res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
    best = str(res_df.iloc[0]["model"])
    pipe = st.session_state[SessionKeys.MODEL_PIPES][best]["pipeline"]
    numeric_ranges = st.session_state[SessionKeys.MODEL_PIPES][best]["metrics"].get("numeric_ranges", {})
    st.info(f"Using in-session model: **{best}** (not saved yet)")

# Prefer metadata columns when available
numeric_for_form = meta.get("numeric_cols", num_cols) if isinstance(meta, dict) else num_cols
categorical_for_form = meta.get("categorical_cols", cat_cols) if isinstance(meta, dict) else cat_cols


def _default_numeric(col: str) -> float:
    if col in clean_df.columns:
        return float(pd.to_numeric(clean_df[col], errors="coerce").median())
    return 0.0


st.subheader("Input")

with st.form("predict_form", border=True):
    left, right = st.columns(2)
    payload: dict[str, object] = {}

    with left:
        st.caption("Numeric")
        for col in list(numeric_for_form):
            val = st.number_input(col, value=float(_default_numeric(col)))
            payload[col] = float(val)
            if col in numeric_ranges:
                r = numeric_ranges[col]
                if float(val) < r["min"] or float(val) > r["max"]:
                    st.caption(f"Note: `{col}` outside training range [{r['min']:.3g}, {r['max']:.3g}]")

    with right:
        st.caption("Categorical")
        for col in list(categorical_for_form):
            if col in clean_df.columns:
                options = sorted(clean_df[col].dropna().astype(str).unique().tolist()) or ["UNKNOWN"]
            else:
                options = ["UNKNOWN"]
            payload[col] = st.selectbox(col, options=options, index=0)

    submitted = st.form_submit_button("Predict", type="primary")


if submitted:
    x = pd.DataFrame([payload])
    try:
        pred = pipe.predict(x)[0]
        proba = pipe.predict_proba(x)[0] if hasattr(pipe, "predict_proba") else None

        if isinstance(pred, (int, float)) and int(pred) in (0, 1):
            label = "Risque Faible" if int(pred) == 1 else "Risque Elevé"
        else:
            label = str(pred)

        st.metric("Prediction", label)

        if proba is not None and len(proba) >= 2:
            out = (
                pd.DataFrame({"Class": ["Risque Elevé", "Risque Faible"], "Probability": [float(proba[0]), float(proba[1])]})
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            out["Probability"] = out["Probability"].round(3)
            st.subheader("Probabilities")
            st.table(out)
    except Exception as e:
        st.error(f"Prediction failed: {e}")


