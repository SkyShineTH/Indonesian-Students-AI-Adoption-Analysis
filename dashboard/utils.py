import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from functools import lru_cache

DATA_PATH = "data/processed/clean_dataset.csv"
REPORTS_BASE = "reports"

def ensure_dirs():
    figs = os.path.join(REPORTS_BASE, "figures")
    tabs = os.path.join(REPORTS_BASE, "tables")
    res = os.path.join(REPORTS_BASE, "results")
    for p in (figs, tabs, res):
        os.makedirs(p, exist_ok=True)
    return {"figures": figs, "tables": tabs, "results": res}

@lru_cache(maxsize=1)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Gender_Label","University_Label","Education_Label","Province_Label","Field_Label","AI_Label"]:
        if col not in df.columns: df[col] = np.nan
    for p in ["PE","CU","ATU","AUP","MIUA"]:
        if f"{p}_Score" not in df.columns:
            score_cols = [c for c in df.columns if c.startswith(p)]
            if score_cols:
                df[f"{p}_Score"] = df[score_cols].mean(axis=1)
    return df

def filter_df(df: pd.DataFrame, gender, uni, edu, field, ai, provinces):
    q = df.copy()
    if gender:    q = q[q["Gender_Label"].isin(gender)]
    if uni:       q = q[q["University_Label"].isin(uni)]
    if edu:       q = q[q["Education_Label"].isin(edu)]
    if field:     q = q[q["Field_Label"].isin(field)]
    if ai:        q = q[q["AI_Label"].isin(ai)]
    if provinces: q = q[q["Province_Label"].isin(provinces)]
    return q

def bar_top(series: pd.Series, title: str, n=15):
    vc = series.value_counts().head(n).reset_index()
    vc.columns = [series.name or "Category", "Count"]
    fig = px.bar(vc, x=vc.columns[0], y="Count", title=title)
    fig.update_layout(xaxis_tickangle=-30)
    return fig

def corr_heatmap(df: pd.DataFrame, cols, title="Correlation"):
    c = df[cols].corr()
    z = c.values
    fig = ff.create_annotated_heatmap(
        z=z, x=list(c.columns), y=list(c.index),
        colorscale="RdBu", reversescale=True, showscale=True, zmin=-1, zmax=1
    )
    fig.update_layout(title=title)
    return fig
