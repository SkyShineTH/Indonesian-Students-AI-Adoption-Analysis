import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import ensure_report_dirs

def elbow_method(X: np.ndarray, k_min=2, k_max=10, reports_base="reports", fig_name="elbow_method"):
    inertia = []
    Ks = range(k_min, k_max + 1)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertia.append(km.inertia_)

    paths = ensure_report_dirs(reports_base)
    plt.figure()
    plt.plot(list(Ks), inertia, "o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    out_path = os.path.join(paths["figures"], f"{fig_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return pd.DataFrame({"k": list(Ks), "inertia": inertia})

def silhouette_scan(X: np.ndarray, k_min=2, k_max=10) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)

def fit_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    return km, labels

def cluster_profile(df_with_scores: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df_with_scores.copy()
    df["Cluster"] = labels
    score_cols = [c for c in df.columns if c.endswith("_Score")]
    profile = df.groupby("Cluster")[score_cols].mean().reset_index()
    return profile

def save_cluster_profile(profile_df: pd.DataFrame, reports_base="reports", name="cluster_profile"):
    paths = ensure_report_dirs(reports_base)
    out_csv = os.path.join(paths["tables"], f"{name}.csv")
    profile_df.to_csv(out_csv, index=False)
    return out_csv
