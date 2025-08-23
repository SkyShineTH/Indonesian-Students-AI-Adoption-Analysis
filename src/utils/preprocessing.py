import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .mapping import (
    gender_map, university_map, education_map,
    province_map, field_map, ai_map
)

def ensure_report_dirs(base="reports"):
    figures = os.path.join(base, "figures")
    tables = os.path.join(base, "tables")
    results = os.path.join(base, "results")
    for p in (figures, tables, results):
        os.makedirs(p, exist_ok=True)
    return {"figures": figures, "tables": tables, "results": results}

def save_table(df: pd.DataFrame, name: str, base="reports"):
    paths = ensure_report_dirs(base)
    out_csv = os.path.join(paths["tables"], f"{name}.csv")
    df.to_csv(out_csv, index=False)
    return out_csv
# -------------------------------------

LIKERT_PREFIXES = ["PE", "CU", "ATU", "AUP", "MIUA"]

def build_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in LIKERT_PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            df[f"{prefix}_Score"] = df[cols].mean(axis=1, skipna=True)
    return df

def map_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df: df["Gender_Label"] = df["Gender"].map(gender_map)
    if "University" in df: df["University_Label"] = df["University"].map(university_map)
    if "Level of Education" in df: df["Education_Label"] = df["Level of Education"].map(education_map)
    if "Province" in df: df["Province_Label"] = df["Province"].map(province_map)
    if "Fields of Study" in df:
        df["Fields of Study"] = df["Fields of Study"].replace(0, np.nan)
        df["Field_Label"] = df["Fields of Study"].map(field_map)
    if "Type of AI" in df: df["AI_Label"] = df["Type of AI"].map(ai_map)
    return df

def preprocess_data(raw_path: str, save_path: str = "data/processed/clean_dataset.csv") -> pd.DataFrame:
    df = pd.read_excel(raw_path) if raw_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(raw_path)

    # Basic cleaning
    df = map_categoricals(df)
    # Fill missing by mode for categorical / mean for numeric
    df_numeric = df.select_dtypes(include=[np.number])
    df_nonnum = df.select_dtypes(exclude=[np.number])

    if not df_nonnum.empty:
        df_nonnum = df_nonnum.fillna(df_nonnum.mode().iloc[0])
    if not df_numeric.empty:
        df_numeric = df_numeric.fillna(df_numeric.mean())

    df = pd.concat([df_numeric, df_nonnum], axis=1)

    # Composite scores
    df = build_composite_scores(df)

    # Scaled composites (optional for later ML)
    score_cols = [c for c in df.columns if c.endswith("_Score")]
    if score_cols:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[score_cols])
        df[[f"{c}_Scaled" for c in score_cols]] = scaled

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def clean_for_clustering(df: pd.DataFrame, use_scaled=True) -> pd.DataFrame:
    score_cols = [c for c in df.columns if c.endswith("_Score")]
    if use_scaled:
        score_cols = [f"{c}_Scaled" for c in score_cols]
    # Fallback if scaled not present
    if not set(score_cols).issubset(df.columns):
        score_cols = [c for c in df.columns if c.endswith("_Score")]
    return df[score_cols].copy().fillna(df[score_cols].mean())
