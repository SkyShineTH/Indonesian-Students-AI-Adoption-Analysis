import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import ensure_report_dirs

def run_anova(df: pd.DataFrame, formula: str, reports_base="reports", name="anova"):
    """
    formula example: 'PE_Score ~ C(Gender_Label)'  or  'AUP_Score ~ C(University_Label)'
    """
    model = smf.ols(formula, data=df).fit()
    anova_tbl = anova_lm(model).reset_index().rename(columns={"index": "term"})
    # effect size (eta squared)
    total_ss = anova_tbl["sum_sq"].sum()
    anova_tbl["eta_sq"] = anova_tbl["sum_sq"] / total_ss

    # save outputs
    paths = ensure_report_dirs(reports_base)
    anova_path = os.path.join(paths["tables"], f"{name}.csv")
    anova_tbl.to_csv(anova_path, index=False)

    summary_path = os.path.join(paths["results"], f"{name}_model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    return {"anova_csv": anova_path, "summary_txt": summary_path, "table": anova_tbl, "model": model}

def run_tukey(df: pd.DataFrame, dv: str, group: str, reports_base="reports", name="tukey"):
    """
    dv: dependent variable (e.g., 'PE_Score')
    group: categorical group (e.g., 'Gender_Label')
    """
    tuk = pairwise_tukeyhsd(endog=df[dv], groups=df[group], alpha=0.05)
    # Convert to DataFrame
    tuk_df = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    paths = ensure_report_dirs(reports_base)
    out_csv = os.path.join(paths["tables"], f"{name}_{dv}_by_{group}.csv")
    tuk_df.to_csv(out_csv, index=False)
    return {"tukey_csv": out_csv, "table": tuk_df}

def run_ols(df: pd.DataFrame, formula: str, reports_base="reports", name="ols"):
    """
    formula example: 'AUP_Score ~ PE_Score + CU_Score + ATU_Score'
    """
    model = smf.ols(formula, data=df).fit()
    paths = ensure_report_dirs(reports_base)
    # Save summary
    summary_path = os.path.join(paths["results"], f"{name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # Save coefficients as table
    coefs = model.params.rename("coef").to_frame()
    coefs["pvalue"] = model.pvalues
    coefs["stderr"] = model.bse
    out_csv = os.path.join(paths["tables"], f"{name}_coefficients.csv")
    coefs.to_csv(out_csv)
    return {"summary_txt": summary_path, "coeffs_csv": out_csv, "coeffs": coefs, "model": model}
