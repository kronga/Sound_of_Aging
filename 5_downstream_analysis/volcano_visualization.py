#!/usr/bin/env python3
"""
Age volcano analysis from prediction tables
Config-driven version (no argparse)

Assumptions:
- Predictions table follows the age_study notebook conventions:
    columns: true_values, predictions, subject_number, research_stage
    and optionally visit_priority (if USE_VISIT_PRIORITY=True)
- Combined risk_factors table already contains all relevant columns, including:
    subject_number, research_stage, age, gender (0/1), BMI, and feature columns
- Index joining key is: "<subject_number>_<research_stage>" (created in both tables)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text


# =========================
# CONFIGURATION
# =========================

MODALITY = "blood_test"  # e.g., 'dexa', 'diet', 'nmr', 'sleep', 'ultrasound', 'fundus', 'microbiome'
# ---- Input paths ----
# Note: PREDICTIONS_PATH will be constructed dynamically per gender in main()
BASE_PREDICTIONS_PATH = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/LGBM_stuff_new/multi_seed_lgbm_age_prediction_{modality}/gender_{gender}/predictions_averaged.csv"
# BASE_PREDICTIONS_PATH = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/Ridge_stuff_new/multi_seed_ridge_age_prediction_{modality}/gender_{gender}/predictions_averaged.csv"

COMBINED_RISK_FACTORS_PATH = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/combined_risk_factors.csv"

# ---- Output ----
OUTDIR = f"/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/volacno_plots/{MODALITY}/"
RUN_SUFFIX = ""

# ---- Stratification ----
GENDERS = ["male", "female"]  # Run for both genders
GENDER_COL = "gender"         # expects 1=male, 0=female (if string labels, adapt below)
USE_VISIT_PRIORITY = False

# ---- Age binning ----
MIN_AGE = 40
MAX_AGE = 72
BIN_WIDTH = 2
PERCENTILE = 0.25

# ---- Volcano parameters ----
ALPHA = 0.1
FC_THRESHOLD = 0.0       # effect threshold on z-scale

# ---- Feature cleanup ----
# Columns to exclude from feature testing (identifiers, targets, metadata, etc.)
NON_FEATURE_COLS = [
    "subject_number",
    "research_stage",
    "visit_priority",
    "research_stage",
    "RegistrationCode",
    "subject_id",
    "age",
    "accu_age",
    "gender",
    # "BMI",
    "Date",
    "visit_date",
    "days_from_today",
]

# Modality-specific feature groups to exclude
MODALITY_EXCLUSIONS = {
    "dexa": [
        # DXA features
        "Android tissue fat percent (DXA)",
        "Total Bone Density",
        "Total fat mass (DXA)",
        "Scanned VAT mass (DXA)",
        "vat/fat",
        "Waist circumference",
        "Neck Circumference",
        "BMI"
    ],
    "diet": [
        # Diet features
        "Median daily caloric intake (DL)",
        "Median daily carbohydrate caloric intake  (DL)",
        "Median daily lipid caloric intake(DL)",
        "Median daily protein caloric intake  (DL)",
        "Median Daily Sodium",
        "Daily Fiber",
        "Daily Folate",
    ],
    "nmr": [
        # NMR features
        "LDL cholesterol (BT)",
        "Albumin (BT)",
        "Triglycerides (BT)",
        "Total cholesterol (BT)",
        "HDL cholesterol (BT)",
        "LDL cholesterol (BT)"
    ],
    "sleep": [
        # Sleep features
        "SleepEfficiancy",
        "Total Sleep Time (SM)",
        "Snore DB",
        "RDI (SM)",
        "Rem Latency",
        "ODI (SM)",
        "AHI (SM)",
        "Total Wake Time (SM)",
        "Mean Oxygen Saturation (SM)",
    ],
    "ultrasound": [
        # Ultrasound features
        "Liver viscosity (US)",
        "Liver elasticity (US)",
        "Liver attenuation (US)",
        "Carotid - intima media thickness (US)",
        "Liver sound speed (US)",
    ],
    "retina": [
        # Fundus imaging features
        "Average width (FI)",
        "Fractal dimension (FI)",
        "Vessel density (FI)",
    ],
}

def get_features_to_exclude(modality: str) -> list:
    """
    Get list of features to exclude based on modality.
    
    Parameters
    ----------
    modality : str
        The modality type (e.g., 'dxa', 'diet', 'nmr', 'sleep', 'ultrasound', 'fundus')
    
    Returns
    -------
    list
        List of feature names to exclude from analysis
    """
    modality_lower = modality.lower()
    return MODALITY_EXCLUSIONS.get(modality_lower, [])


# =========================
# CORE FUNCTIONS
# =========================

def analyze_age_predictions(
    predictions_table: pd.DataFrame,
    *,
    min_age: int,
    max_age: int,
    percentile: float,
    bin_width: int,
    use_visit_priority: bool = False,
    multi_seed: bool = False,
    
):
    """
    Bin by true age and pick top/bottom percentile of predicted age within each bin.
    Returns:
      aggregated_bottom_ids, aggregated_top_ids  (ids in subject_visit format)
    """
    df = predictions_table.copy()

    if multi_seed:
        # ename mean_predictions column to predictions, group to subject_number, 
        if 'mean_predictions' not in df.columns:
            raise ValueError("Predictions table missing 'mean_predictions' column for multi-seed analysis.")
        df = df.rename(columns={'mean_predictions': 'predictions',
                                'group': 'subject_number'})

    needed = ["true_values", "predictions", "subject_number", "research_stage"]
    if use_visit_priority:
        needed.append("visit_priority")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions table missing required columns: {missing}")

    df = df.dropna(subset=needed).copy()

    # Keep only last visit per subject
    if use_visit_priority:
        idx = df.groupby("subject_number")["visit_priority"].idxmin()
        df = df.loc[idx].copy()
    else:
        idx = df.groupby("subject_number")["research_stage"].idxmax()
        df = df.loc[idx].copy()

    # Make subject_visit index
    df.index = (
        df["subject_number"].astype(int).astype(str)
        + "_"
        + df["research_stage"]
    )

    agg_top, agg_bottom = [], []

    for a in range(min_age, max_age, bin_width):
        b = a + bin_width - 1
        bin_df = df[(df["true_values"] >= a) & (df["true_values"] <= b)]
        if len(bin_df) < 2:
            continue

        low = bin_df["predictions"].quantile(percentile)
        high = bin_df["predictions"].quantile(1 - percentile)

        agg_bottom.extend(bin_df[bin_df["predictions"] <= low].index.tolist())
        agg_top.extend(bin_df[bin_df["predictions"] >= high].index.tolist())

    return agg_bottom, agg_top


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, NaNs preserved."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size
    q_sorted = pv_sorted * (m / (np.arange(1, m + 1)))
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    qvals = np.empty_like(pv_sorted)
    qvals[order] = np.clip(q_sorted, 0, 1)
    q[mask] = qvals
    return q


def compare_tables_and_plot_volcano(
    bottom: pd.DataFrame,
    top: pd.DataFrame,
    *,
    labels: tuple[str, str],
    save_prefix: str,
    alpha: float,
    fc_threshold: float,
    modality: str = "",
    gender: str = "",
):
    """
    For z-scored features:
      effect = mean(top) - mean(bottom)
    MWU test per feature + BH-FDR using statsmodels.
    Saves: PNG, PDF, results CSV.
    Enhanced with seaborn styling, adjust_text for labels, and FDR-BH threshold line.
    
    Parameters
    ----------
    modality : str
        Optional modality name to include in plot title
    gender : str
        Optional gender label to include in plot title
    """
    common = bottom.columns.intersection(top.columns)
    if common.empty:
        raise ValueError("No common feature columns between bottom and top tables.")

    # -------------------------------------------------------------- #
    # 1. Per feature comparison
    # -------------------------------------------------------------- #
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        for f in common:
            x, y = bottom[f].dropna(), top[f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue

            effect = float(y.mean() - x.mean())
            p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            results.append({
                "feature": f,
                "delta_z": effect,
                "p_value": p,
                "n_bottom": len(x),
                "n_top": len(y)
            })

    if not results:
        raise ValueError("No valid comparisons")

    res = pd.DataFrame(results)

    # -------------------------------------------------------------- #
    # 2. Multiple-testing correction using statsmodels
    # -------------------------------------------------------------- #
    p_series = pd.to_numeric(res["p_value"], errors="coerce")
    mask = p_series.notna().values
    adj_p = np.ones(len(res), dtype=float)

    if mask.any():
        pvals = p_series.values[mask].astype(float)
        pvals[~np.isfinite(pvals)] = np.nan
        keep = ~np.isnan(pvals)
        if keep.any():
            qvals = np.full(pvals.shape, np.nan, dtype=float)
            rej, qvals[keep], _, _ = multipletests(
                pvals[keep],
                alpha=float(alpha),
                method="fdr_bh",
                is_sorted=False,
                returnsorted=False
            )
            adj_p[mask] = qvals

    res["adj_p_value"] = adj_p
    res["-log10_p"] = -np.log10(p_series)
    res["-log10_q"] = -np.log10(res["adj_p_value"])

    # Clean effect column
    col = res["delta_z"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    
    def _to_scalar(x):
        if isinstance(x, (np.ndarray, list, pd.Series)):
            arr = np.asarray(x).ravel()
            return float(arr[0]) if arr.size else np.nan
        return x
    
    col = col.map(_to_scalar)
    col = pd.to_numeric(col, errors="coerce")
    res["delta_z"] = col

    # -------------------------------------------------------------- #
    # 3. Significance flags
    # -------------------------------------------------------------- #
    fc_threshold = float(fc_threshold)
    effect_abs = col.abs().values
    comp_effect = np.greater(effect_abs, fc_threshold)
    comp_effect = np.where(np.isnan(effect_abs), False, comp_effect)
    
    lhs = (res["adj_p_value"] < float(alpha)).values
    signif_vec = np.logical_and(lhs, comp_effect)
    res["significant"] = pd.Series(signif_vec, index=res.index)

    res["regulation"] = np.select(
        [
            res["significant"] & (res["delta_z"] > 0),
            res["significant"] & (res["delta_z"] < 0),
        ],
        ["upregulated", "downregulated"],
        default="not_significant",
    )

    # -------------------------------------------------------------- #
    # 4. Plot with seaborn styling
    # -------------------------------------------------------------- #
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    def scatter(sub, color, edge, label):
        if sub.empty:
            return
        sns.scatterplot(
            x="delta_z", y="-log10_p", data=sub,
            s=80 if "regulated" in label else 50,
            alpha=0.9, color=color, edgecolor=edge, linewidth=0.8,
            label=label
        )

    scatter(res[res["regulation"] == "not_significant"],
            "grey", "dimgrey", "Not significant")
    scatter(res[res["regulation"] == "upregulated"],
            '#FF5252', "darkred",
            f"Upregulated (n={sum(res['regulation']=='upregulated')})")
    scatter(res[res["regulation"] == "downregulated"],
            '#4CAF50', "darkgreen",
            f"Downregulated (n={sum(res['regulation']=='downregulated')})")

    # Label significant points with adjust_text
    texts = []
    for _, r in res[res["significant"]].iterrows():
        texts.append(plt.text(r["delta_z"], r["-log10_p"], r["feature"],
                              fontsize=8, weight="bold"))
    if texts:
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Thresholds
    plt.axhline(-np.log10(alpha), ls="--", c="dimgrey", lw=1)
    
    # Compute BH effective cutoff
    m = int(mask.sum())
    p_sorted = np.sort(res.loc[mask, 'p_value'].values)
    bh_thresholds = (np.arange(1, m + 1) / max(m, 1)) * alpha if m > 0 else np.array([])
    passed = np.where(p_sorted <= bh_thresholds)[0] if m > 0 else np.array([])
    
    if passed.size > 0:
        k_star = int(passed.max() + 1)
        p_bh_cut = (k_star / m) * alpha
        bh_note = f"(q<{alpha})"
    else:
        k_star = 0
        p_bh_cut = alpha / m if m > 0 else np.nan
        bh_note = f"(no discoveries, first step; m={m})"
    
    if not np.isnan(p_bh_cut):
        fdr_line = -np.log10(p_bh_cut)
        plt.axhline(fdr_line, ls="-.", c="dimgrey", lw=1.2,
                    label=f"FDR-BH threshold {bh_note}")
    
    plt.axvline(fc_threshold,  ls="--", c="dimgrey", lw=1)
    plt.axvline(-fc_threshold, ls="--", c="dimgrey", lw=1)

    plt.xlabel("$\Delta$SD", fontsize=10)
    plt.ylabel(r"$-\log_{10}$(p)", fontsize=10)
    n1, n2 = len(bottom), len(top)
    modality_str = f" - {modality}" if modality else ""
    gender_str = f" ({gender})" if gender else ""
    plt.title(
        f"{labels[1]} (n={n2}) vs {labels[0]} (n={n1}){modality_str}{gender_str}",
        weight="bold"
    )
    plt.legend(frameon=True, framealpha=0.85, fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plt.savefig(save_prefix + ".png", dpi=300)
    plt.savefig(save_prefix + ".pdf")
    plt.close()

    res.sort_values("adj_p_value").to_csv(save_prefix + "_results.csv")
    return res


def make_subject_visit_index(df: pd.DataFrame, subject_col: str, visit_col: str) -> pd.Index:
    return (
        df[subject_col].astype(int).astype(str)
        + "_"
        + df[visit_col]
        )


def filter_gender(rf: pd.DataFrame, gender: str | None, gender_col: str) -> pd.DataFrame:
    if gender is None:
        return rf
    if gender_col not in rf.columns:
        raise ValueError(f"gender_col='{gender_col}' not found in combined risk_factors table.")

    # default convention: 1=male, 0=female
    if gender == "male":
        return rf[rf[gender_col] == 1].copy()
    if gender == "female":
        return rf[rf[gender_col] == 0].copy()

    raise ValueError("gender must be None, 'male', or 'female'")


# =========================
# MAIN WORKFLOW
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 2) Load combined risk factors (once)
    rf = pd.read_csv(COMBINED_RISK_FACTORS_PATH)

    needed_cols = ["subject_number", "research_stage"]
    miss = [c for c in needed_cols if c not in rf.columns]
    if miss:
        raise ValueError(f"Combined risk_factors table missing required columns: {miss}")

    # 3) Create index to match predictions
    rf = rf.copy()
    rf.index = make_subject_visit_index(rf, "subject_number", "research_stage")
    rf = rf[~rf.index.duplicated(keep="first")].copy()

    # Loop through each gender
    for gender in GENDERS:
        print(f"\n{'='*60}")
        print(f"Processing: {gender.upper()}")
        print(f"{'='*60}")
        
        # 1) Load predictions for this gender
        predictions_path = BASE_PREDICTIONS_PATH.format(modality=MODALITY, gender=gender)
        print(f"Loading predictions from: {predictions_path}")
        pred = pd.read_csv(predictions_path)
        
        bottom_ids, top_ids = analyze_age_predictions(
            pred,
            min_age=MIN_AGE,
            max_age=MAX_AGE,
            percentile=PERCENTILE,
            bin_width=BIN_WIDTH,
            use_visit_priority=USE_VISIT_PRIORITY,
            multi_seed=True,
        )
        
        # 4) Optional gender filter
        rf_gender = filter_gender(rf, gender, GENDER_COL)

        # 5) Build feature matrix: keep numeric, drop non-features, z-score globally
        # Get modality-specific exclusions
        modality_exclusions = get_features_to_exclude(MODALITY)
        drop_cols = [c for c in NON_FEATURE_COLS if c in rf_gender.columns] + modality_exclusions
        X = rf_gender.drop(columns=drop_cols, errors="ignore")
        
        print(f"Excluding {len(modality_exclusions)} {MODALITY}-related features from comparison")

        # keep numeric only
        X = X.select_dtypes(include=[np.number]).copy()

        # align with selected IDs
        bottom_ids_in = [i for i in bottom_ids if i in X.index]
        top_ids_in = [i for i in top_ids if i in X.index]

        if len(bottom_ids_in) < 10 or len(top_ids_in) < 10:
            print(f"Warning: Too few matched samples for {gender}: bottom={len(bottom_ids_in)}, top={len(top_ids_in)}. Skipping.")
            continue

        # scale across all rows (not per group) to match typical notebook behavior
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X.astype(float)),
            index=X.index,
            columns=X.columns,
        )

        bottom_tbl = X_scaled.loc[bottom_ids_in].copy()
        top_tbl = X_scaled.loc[top_ids_in].copy()

        # 6) Volcano
        percent = int(PERCENTILE * 100)
        labels = (f"Bottom {percent}%", f"Top {percent}%")

        gender_tag = gender
        save_prefix = os.path.join(
            OUTDIR,
            f"volcano_age_{gender_tag}_p{percent}_{RUN_SUFFIX}",
        )

        compare_tables_and_plot_volcano(
            bottom_tbl,
            top_tbl,
            labels=labels,
            save_prefix=save_prefix,
            alpha=ALPHA,
            fc_threshold=FC_THRESHOLD,
            modality=MODALITY,
            gender=gender,
        )

        # 7) Save the exact tables used (useful for debugging / replotting)
        bottom_tbl.to_csv(os.path.join(OUTDIR, f"bottom_scaled_{gender_tag}_p{percent}_{RUN_SUFFIX}.csv"))
        top_tbl.to_csv(os.path.join(OUTDIR, f"top_scaled_{gender_tag}_p{percent}_{RUN_SUFFIX}.csv"))

        print(f"Done with {gender}.")
        print(f"Saved: {save_prefix}.png/.pdf and {save_prefix}_results.csv")

    print(f"\n{'='*60}")
    print("All genders processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
