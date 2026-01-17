from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import get_raw_df, require_target_column
from src.constants import TARGET_COL


st.set_page_config(page_title="Data • Credit Risk", layout="wide")

st.title("Data")
st.caption("Preview the dataset loaded from the fixed path and check basic summaries.")

df = get_raw_df()
require_target_column(df)

top = st.slider("Rows to preview", min_value=10, max_value=200, value=50, step=10)
st.subheader("Preview")
st.dataframe(df.head(int(top)), use_container_width=True)

st.divider()
st.subheader("Quick summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", int(df.shape[0]))
c2.metric("Columns", int(df.shape[1]))
c3.metric("Nulls", int(df.isna().sum().sum()))
c4.metric("Duplicates", int(df.duplicated().sum()))

with st.expander("Dtypes", expanded=False):
    st.json(df.dtypes.astype(str).to_dict())

with st.expander(f"Target distribution: `{TARGET_COL}`", expanded=False):
    vc = df[TARGET_COL].value_counts(dropna=False).rename_axis(TARGET_COL).reset_index(name="count")
    st.dataframe(vc, use_container_width=True)

st.divider()
st.subheader("Pandas describe")
st.dataframe(df.describe(include="all"), use_container_width=True)


