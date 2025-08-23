import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import ensure_report_dirs

def save_fig(fig, name: str, reports_base="reports"):
    paths = ensure_report_dirs(reports_base)
    out = os.path.join(paths["figures"], f"{name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_distribution(df: pd.DataFrame, col: str, title=None, name=None, reports_base="reports"):
    fig, ax = plt.subplots(figsize=(6,4))
    vc = df[col].value_counts().sort_values(ascending=False)
    sns.barplot(x=vc.index, y=vc.values, ax=ax)
    ax.set_title(title or f"Distribution: {col}")
    ax.set_ylabel("Count")
    ax.set_xlabel(col)

    ax.tick_params(axis='x', labelrotation=45)

    for label in ax.get_xticklabels():
        label.set_ha('right')

    if name:
        return save_fig(fig, name, reports_base)
    return fig

def plot_crosstab_heatmap(df: pd.DataFrame, row: str, col: str, title=None, name=None, reports_base="reports"):
    ct = pd.crosstab(df[row], df[col])
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title or f"Crosstab: {row} vs {col}")
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    if name:
        return save_fig(fig, name, reports_base)
    return fig

def plot_correlation(df: pd.DataFrame, cols: list[str], title="Correlation Matrix", name=None, reports_base="reports"):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    if name:
        return save_fig(fig, name, reports_base)
    return fig
