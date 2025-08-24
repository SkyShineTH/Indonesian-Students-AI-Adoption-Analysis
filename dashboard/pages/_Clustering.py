import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys
import os

sys.path.append(os.path.abspath(os.path.join('dashboard')))
from utils import load_data, filter_df

st.title("Clustering Analysis")

df = load_data()
# Filters
with st.sidebar:
    st.header("Filters")
    gender = st.multiselect("Gender", sorted(df["Gender_Label"].dropna().unique().tolist()))
    uni    = st.multiselect("University", sorted(df["University_Label"].dropna().unique().tolist()))
    edu    = st.multiselect("Education", sorted(df["Education_Label"].dropna().unique().tolist()))
    field  = st.multiselect("Field of Study", sorted(df["Field_Label"].dropna().unique().tolist()))
    ai     = st.multiselect("AI Type", sorted(df["AI_Label"].dropna().unique().tolist()))
    prov   = st.multiselect("Province", sorted(df["Province_Label"].dropna().unique().tolist()))

q = filter_df(df, gender, uni, edu, field, ai, prov)

score_cols = ["PE_Score","CU_Score","ATU_Score","AUP_Score","MIUA_Score"]
X = q[score_cols].copy().fillna(q[score_cols].mean())
X_scaled = StandardScaler().fit_transform(X)

st.subheader("Select k")
k = st.slider("Number of clusters (k)", 2, 8, 2)

# Silhouette scan quick view
if st.button("Compute Silhouette for k=2..6"):
    rows = []
    for kk in range(2, 7):
        km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        rows.append({"k": kk, "silhouette": silhouette_score(X_scaled, labels)})
    st.dataframe(pd.DataFrame(rows))

# Fit KMeans
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(X_scaled)
q["Cluster"] = labels

st.markdown("### Cluster Profiles (Mean Scores)")
profile = q.groupby("Cluster")[score_cols].mean().round(2).reset_index()
st.dataframe(profile)

# PCA 2D plot
pca = PCA(n_components=2, random_state=42)
emb = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame(emb, columns=["PC1","PC2"])
plot_df["Cluster"] = labels
st.scatter_chart(plot_df, x="PC1", y="PC2", color="Cluster", use_container_width=True)

st.markdown("**Tips:** โดยทั่วไปจากผล Silhouette ที่คุณรันก่อนหน้า k=2 มักดีที่สุด → ตีความเป็น High vs Low adoption personas.")
