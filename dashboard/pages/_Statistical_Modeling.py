import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys
import os

sys.path.append(os.path.abspath(os.path.join('dashboard')))
from utils import load_data, filter_df

st.title("Statistical Modeling")

df = load_data()

with st.sidebar:
    st.header("Filters")
    gender = st.multiselect("Gender", sorted(df["Gender_Label"].dropna().unique().tolist()))
    uni    = st.multiselect("University", sorted(df["University_Label"].dropna().unique().tolist()))
    edu    = st.multiselect("Education", sorted(df["Education_Label"].dropna().unique().tolist()))
    field  = st.multiselect("Field of Study", sorted(df["Field_Label"].dropna().unique().tolist()))
    ai     = st.multiselect("AI Type", sorted(df["AI_Label"].dropna().unique().tolist()))
    prov   = st.multiselect("Province", sorted(df["Province_Label"].dropna().unique().tolist()))

q = filter_df(df, gender, uni, edu, field, ai, prov)

st.subheader("ANOVA")
dv = st.selectbox("Dependent Variable (score)", ["PE_Score","CU_Score","ATU_Score","AUP_Score","MIUA_Score"])
group = st.selectbox("Group (categorical)", ["Gender_Label","University_Label","Education_Label","Field_Label","AI_Label","Province_Label"])

if st.button("Run ANOVA"):
    # OLS & ANOVA
    formula = f"{dv} ~ C({group})"
    model = smf.ols(formula, data=q).fit()
    anova_tbl = anova_lm(model).reset_index().rename(columns={"index": "term"})
    total = anova_tbl["sum_sq"].sum()
    anova_tbl["eta_sq"] = (anova_tbl["sum_sq"] / total).round(3)

    st.write(f"**Formula:** `{formula}`")
    st.dataframe(anova_tbl)

    st.markdown("**Model Summary (short):**")
    st.text(model.summary2().tables[0].to_string())

    st.subheader("Post-hoc: Tukey HSD")
    try:
        tuk = pairwise_tukeyhsd(endog=q[dv], groups=q[group], alpha=0.05)
        st.dataframe(pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0]))
    except Exception as e:
        st.info(f"Tukey รันไม่ได้ (อาจมีหมวดหมู่ < 2 หรือตัวอย่างน้อย): {e}")

st.markdown("---")
st.subheader("Regression (OLS)")
reg_dv = st.selectbox("DV (score)", ["AUP_Score","ATU_Score"], index=0)
ivs = st.multiselect("IVs (scores)", ["PE_Score","CU_Score","ATU_Score","AUP_Score","MIUA_Score"], default=["PE_Score","CU_Score","ATU_Score"])
if ivs:
    formula = f"{reg_dv} ~ " + " + ".join(ivs)
    if st.button("Run OLS"):
        model = smf.ols(formula, data=q).fit()
        st.write(f"**Formula:** `{formula}`")
        st.text(model.summary().as_text())
