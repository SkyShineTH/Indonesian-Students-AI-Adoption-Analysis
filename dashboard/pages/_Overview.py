import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join('dashboard')))
from utils import load_data, filter_df, bar_top, corr_heatmap

st.title("Overview")

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    gender = st.multiselect("Gender", sorted(df["Gender_Label"].dropna().unique().tolist()))
    uni    = st.multiselect("University", sorted(df["University_Label"].dropna().unique().tolist()))
    edu    = st.multiselect("Education", sorted(df["Education_Label"].dropna().unique().tolist()))
    field  = st.multiselect("Field of Study", sorted(df["Field_Label"].dropna().unique().tolist()))
    ai     = st.multiselect("AI Type", sorted(df["AI_Label"].dropna().unique().tolist()))
    prov   = st.multiselect("Province", sorted(df["Province_Label"].dropna().unique().tolist()))

q = filter_df(df, gender, uni, edu, field, ai, prov)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respondents", len(q))
c2.metric("Avg PE", f"{q['PE_Score'].mean():.2f}")
c3.metric("Avg ATU", f"{q['ATU_Score'].mean():.2f}")
c4.metric("Avg AUP", f"{q['AUP_Score'].mean():.2f}")

st.markdown("### Demographics & Usage")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_top(q["Gender_Label"], "Gender Distribution"), use_container_width=True)
    st.plotly_chart(bar_top(q["Field_Label"], "Fields of Study"), use_container_width=True)
with col2:
    st.plotly_chart(bar_top(q["University_Label"], "University Type"), use_container_width=True)
    st.plotly_chart(bar_top(q["AI_Label"], "AI Tools Used"), use_container_width=True)

st.markdown("### Correlation of Composite Scores")
score_cols = ["PE_Score","CU_Score","ATU_Score","AUP_Score","MIUA_Score"]
st.plotly_chart(corr_heatmap(q, score_cols, "Correlation (PE, CU, ATU, AUP, MIUA)"), use_container_width=True)
