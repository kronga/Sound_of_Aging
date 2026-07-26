#!/usr/bin/env python3
"""
Lollipop plot of phenome-wide associations using age-residualized ∆VA.

Pipeline:
  1. Load ten-seed-averaged out-of-fold predictions.
  2. Residualize Voice Age against chronological age within sex.
  3. Define global top/bottom 25% groups from residualized Delta VA.
  4. Compute standardized mean differences between groups.
  5. Estimate 95% percentile intervals by participant-level bootstrap.

Significance flags: load from step5_volcano/voice_residualized/ CSV produced
by age_bias_check.py (female). For male the significance is computed here inline
from averaged predictions.

Output: step5_volcano/voice_residualized/lollipop_combined_p25_.png/pdf
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D

from phenotype_enrichment import add_vat_area


def _clean_feature_label(name: str) -> str:
    """Strip parenthetical suffixes like (SM), (BT), (DXA), (DL), (US), (FI).
    Also fix backslash to forward slash, collapse whitespace, and use compact
    publication labels."""
    name = name.replace("\\", "/")
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    compact_labels = {
        "Scanned VAT area": "VAT area",
        "VAT area": "VAT area",
        "Median daily caloric intake": "Daily energy intake",
        "Median daily carbohydrate caloric intake": "Daily carbohydrate energy",
        "Median daily lipid caloric intake": "Daily fat energy",
        "Median daily protein caloric intake": "Daily protein energy",
        "Median Daily Sodium": "Daily sodium",
        "Carotid - intima media thickness": "Carotid IMT",
        "Android tissue fat percent": "Android tissue fat %",
        "HbA1C": "HbA1c",
        "Rem Latency": "REM latency",
        "SleepEfficiancy": "Sleep efficiency",
        "Hand Grip Left": "Hand grip left",
        "Hand Grip Right": "Hand grip right",
        "Neck Circumference": "Neck circumference",
        "Mean Oxygen Saturation": "Mean oxygen saturation",
        "Total Bone Density": "Total bone density",
        "Total Wake Time": "Total wake time",
        "Total Sleep Time": "Total sleep time",
        "Sitting BP diastolic": "Sitting BP diastolic",
        "Sitting BP systolic": "Sitting BP systolic",
        "Snore DB": "Snore dB",
    }
    return compact_labels.get(name, name)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

# ── config ────────────────────────────────────────────────────────────────────
OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()


def tagged_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


SEED_PREDICTIONS_BASE = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"analysis_outputs/{STEP3_DIR}/"
    "gender_{gender}"
)
AVERAGED_PRED_BASE = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"analysis_outputs/{STEP3_DIR}/"
    "gender_{gender}/predictions_averaged.csv"
)
SUBJECT_DETAILS_CSV = (
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/subject_details_df_Oct25.csv"
)
RESID_VOLCANO_DIR = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"analysis_outputs/step5_volcano/{tagged_name('voice_residualized')}/"
)
OUTDIR = RESID_VOLCANO_DIR

GENDERS = ["male", "female"]
MIN_AGE, MAX_AGE = 40, 70
PERCENTILE = 0.25
ALPHA = 0.1
CI_LEVEL = 0.95
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260723
percent = int(PERCENTILE * 100)

# ── helpers ───────────────────────────────────────────────────────────────────

def _stage_rank(stage: str) -> tuple[int, str]:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def load_latest_subject_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    df = pred.rename(columns={"group": "subject_number"}).copy()
    if "research_stage" not in df.columns:
        sd = pd.read_csv(
            SUBJECT_DETAILS_CSV,
            usecols=["filename", "visit_number"],
        ).rename(columns={"visit_number": "research_stage"})
        df = df.merge(sd, left_on="index", right_on="filename", how="left")
    df = df.dropna(subset=["true_values", "predictions", "subject_number", "research_stage"]).copy()
    df["_stage_rank"] = df["research_stage"].map(_stage_rank)
    df = df.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    df["_key"] = (
        df["subject_number"].astype(int).astype(str)
        + "_"
        + df["research_stage"].astype(str)
    )
    return df.reset_index(drop=True)

def residualize_series(age: pd.Series, delta_va: pd.Series) -> tuple[pd.Series, float]:
    """OLS delta_va ~ age. Returns (residuals, r²)."""
    slope, intercept, r, _, _ = linregress(age, delta_va)
    resid = delta_va - (slope * age + intercept)
    return resid, r ** 2


def stratify_by_resid(df: pd.DataFrame) -> tuple[list, list]:
    """Global top/bottom PERCENTILE by delta_va_resid within age range.
    Returns subject_visit keys."""
    subset = df[(df["true_values"] >= MIN_AGE) & (df["true_values"] <= MAX_AGE)].copy()
    lo = subset["delta_va_resid"].quantile(PERCENTILE)
    hi = subset["delta_va_resid"].quantile(1 - PERCENTILE)
    bottom = subset.loc[subset["delta_va_resid"] <= lo, "_key"].astype(str).tolist()
    top    = subset.loc[subset["delta_va_resid"] >= hi, "_key"].astype(str).tolist()
    return bottom, top


def compute_effect_sizes(bottom_tbl: pd.DataFrame, top_tbl: pd.DataFrame) -> pd.Series:
    common = bottom_tbl.columns.intersection(top_tbl.columns)
    effects = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in common:
            x, y = bottom_tbl[f].dropna(), top_tbl[f].dropna()
            effects[f] = float(y.mean() - x.mean()) if len(x) >= 3 and len(y) >= 3 else np.nan
    return pd.Series(effects)


def bootstrap_effect_summary(
    bottom_tbl: pd.DataFrame,
    top_tbl: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Participant-level nonparametric bootstrap of top-minus-bottom effects."""
    common = bottom_tbl.columns.intersection(top_tbl.columns)
    bottom = bottom_tbl.loc[:, common].to_numpy(dtype=float)
    top = top_tbl.loc[:, common].to_numpy(dtype=float)
    point = np.nanmean(top, axis=0) - np.nanmean(bottom, axis=0)

    rng = np.random.default_rng(seed)
    effects = np.empty((N_BOOTSTRAPS, len(common)), dtype=np.float32)
    chunk_size = 250
    for start in range(0, N_BOOTSTRAPS, chunk_size):
        stop = min(start + chunk_size, N_BOOTSTRAPS)
        size = stop - start
        bottom_idx = rng.integers(0, len(bottom), size=(size, len(bottom)))
        top_idx = rng.integers(0, len(top), size=(size, len(top)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            effects[start:stop] = (
                np.nanmean(top[top_idx], axis=1)
                - np.nanmean(bottom[bottom_idx], axis=1)
            )

    alpha = (1 - CI_LEVEL) / 2
    return pd.DataFrame(
        {
            "mean": point,
            "ci_low": np.nanquantile(effects, alpha, axis=0),
            "ci_high": np.nanquantile(effects, 1 - alpha, axis=0),
        },
        index=common,
    )


def run_averaged_significance(gender: str, rf_scaled: pd.DataFrame) -> list[str]:
    """Compute significant features from averaged predictions using OLS residualization."""
    # Reuse a result table only when it contains exactly the current phenotype
    # set. This automatically invalidates the legacy VAT-mass/vat-to-fat output
    # after visit-matched VAT area is introduced.
    resid_csv = os.path.join(RESID_VOLCANO_DIR, f"volcano_age_{gender}_p25__results.csv")
    if os.path.exists(resid_csv):
        res = pd.read_csv(resid_csv)
        if (
            "feature" in res.columns
            and set(res["feature"]) == set(rf_scaled.columns)
        ):
            return res.loc[res["significant"] == True, "feature"].tolist()
        print(
            f"  Recomputing {gender} significance: phenotype columns changed."
        )

    # Compute from scratch using averaged predictions
    pred = pd.read_csv(AVERAGED_PRED_BASE.format(gender=gender))
    pred = pred.rename(columns={"mean_predictions": "predictions"})
    pred = load_latest_subject_predictions(pred)
    pred["delta_va"] = pred["predictions"] - pred["true_values"]
    slope, intercept, *_ = linregress(pred["true_values"], pred["delta_va"])
    pred["delta_va_resid"] = pred["delta_va"] - (slope * pred["true_values"] + intercept)
    pred = pred[(pred["true_values"] >= MIN_AGE) & (pred["true_values"] <= MAX_AGE)].copy()
    lo = pred["delta_va_resid"].quantile(PERCENTILE)
    hi = pred["delta_va_resid"].quantile(1 - PERCENTILE)
    bottom_keys = pred.loc[pred["delta_va_resid"] <= lo, "_key"].tolist()
    top_keys    = pred.loc[pred["delta_va_resid"] >= hi, "_key"].tolist()

    bt = rf_scaled.loc[[k for k in bottom_keys if k in rf_scaled.index]]
    tp = rf_scaled.loc[[k for k in top_keys    if k in rf_scaled.index]]

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in bt.columns.intersection(tp.columns):
            x, y = bt[f].dropna(), tp[f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            results.append({"feature": f,
                             "delta_z": float(y.mean() - x.mean()),
                             "p_value": float(mannwhitneyu(x, y, alternative="two-sided").pvalue)})
    res = pd.DataFrame(results)
    pvals = res["p_value"].values.astype(float)
    keep = np.isfinite(pvals)
    adj_p = np.ones(len(res))
    if keep.any():
        _, qvals, _, _ = multipletests(pvals[keep], alpha=ALPHA, method="fdr_bh")
        adj_p[keep] = qvals
    res["adj_p_value"] = adj_p
    res["significant"] = res["adj_p_value"] < ALPHA

    os.makedirs(RESID_VOLCANO_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESID_VOLCANO_DIR, f"volcano_age_{gender}_p25__results.csv"),
               index=False)
    return res.loc[res["significant"], "feature"].tolist()


def process_gender(
    gender: str,
    rf: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], tuple[int, int]]:
    gender_val = 1 if gender == "male" else 0
    rf_g = rf[rf["gender"] == gender_val].copy()
    drop_cols = [c for c in vv.NON_FEATURE_COLS if c in rf_g.columns]
    X = rf_g.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X.astype(float)),
                            index=X.index, columns=X.columns)

    pred = pd.read_csv(AVERAGED_PRED_BASE.format(gender=gender))
    pred = pred.rename(columns={"mean_predictions": "predictions"})
    df = load_latest_subject_predictions(pred)
    df["delta_va"] = df["predictions"] - df["true_values"]
    df["delta_va_resid"], age_r2 = residualize_series(
        df["true_values"], df["delta_va"]
    )
    bottom_ids, top_ids = stratify_by_resid(df)
    bottom_ids = [key for key in bottom_ids if key in X_scaled.index]
    top_ids = [key for key in top_ids if key in X_scaled.index]
    if len(bottom_ids) < 10 or len(top_ids) < 10:
        raise RuntimeError(
            f"Too few phenotype-matched participants for {gender}: "
            f"{len(bottom_ids)}/{len(top_ids)}"
        )

    summary = bootstrap_effect_summary(
        X_scaled.loc[bottom_ids],
        X_scaled.loc[top_ids],
        BOOTSTRAP_SEED + gender_val,
    )
    print(
        f"  bottom={len(bottom_ids)}, top={len(top_ids)}, "
        f"residualized age R²={age_r2:.3g}, bootstraps={N_BOOTSTRAPS:,}"
    )
    sig = run_averaged_significance(gender, X_scaled)
    print(f"  Significant features: {len(sig)}")
    return summary, sig, (len(bottom_ids), len(top_ids))


def plot_lollipop_combined(
    summary: dict[str, pd.DataFrame],
    sig_features: dict[str, list[str]],
    group_counts: dict[str, tuple[int, int]],
):
    FONT = 7
    LABEL_FONT = 6
    TITLE_FONT = 8
    FIG_WIDTH = 125 / 25.4
    FIG_HEIGHT = 220 / 25.4
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT,
            "axes.labelsize": FONT,
            "axes.titlesize": TITLE_FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": LABEL_FONT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    male_sdf   = summary["male"]
    female_sdf = summary["female"]
    all_features = sorted(
        set(male_sdf.index) | set(female_sdf.index),
        key=lambda f: female_sdf.loc[f, "mean"] if f in female_sdf.index else 0.0,
    )
    # Display labels: strip parenthetical suffixes and fix backslashes
    display_labels = [_clean_feature_label(f) for f in all_features]
    n_feat = len(all_features)
    y_pos  = np.arange(n_feat)
    interval_limits = []
    for gender in ("female", "male"):
        gender_summary = summary[gender].reindex(all_features)
        interval_limits.extend(
            gender_summary[["ci_low", "ci_high"]].to_numpy().ravel().tolist()
        )
    finite_limits = np.asarray(interval_limits, dtype=float)
    finite_limits = finite_limits[np.isfinite(finite_limits)]
    shared_bound = max(float(np.max(np.abs(finite_limits))) * 1.08, 0.1)
    shared_xlim = (-shared_bound, shared_bound)

    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT)
    )
    sns.set_style("whitegrid")

    for gender, ax in [("female", axes[0]), ("male", axes[1])]:
        for i in range(n_feat):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)

        sdf     = summary[gender]
        sig_set = set(sig_features.get(gender, []))
        means = np.array([sdf.loc[f, "mean"] if f in sdf.index else np.nan for f in all_features])
        ci_low = np.array(
            [sdf.loc[f, "ci_low"] if f in sdf.index else np.nan for f in all_features]
        )
        ci_high = np.array(
            [sdf.loc[f, "ci_high"] if f in sdf.index else np.nan for f in all_features]
        )
        is_sig = np.array([f in sig_set for f in all_features])
        colors = [("#FF5252" if m > 0 else "#4CAF50") if s else "grey"
                  for m, s in zip(means, is_sig)]
        edges  = ["black" if s else "none" for s in is_sig]
        sizes  = [18 if s else 8 for s in is_sig]

        valid = ~np.isnan(means)
        asymmetric_ci = np.vstack(
            [means[valid] - ci_low[valid], ci_high[valid] - means[valid]]
        )
        ax.errorbar(means[valid], y_pos[valid], xerr=asymmetric_ci,
                    fmt="none", ecolor="lightgrey", elinewidth=0.5, capsize=1.5, zorder=3)
        for sig_val in [False, True]:
            mask = is_sig == sig_val
            if not (mask & valid).any():
                continue
            ax.scatter(means[mask & valid], y_pos[mask & valid],
                       c=[colors[i] for i in np.where(mask & valid)[0]],
                       s=[sizes[i]  for i in np.where(mask & valid)[0]],
                       marker="o",
                       edgecolors=[edges[i] for i in np.where(mask & valid)[0]],
                       linewidths=0.4, zorder=5)
        ax.axvline(0, color="dimgrey", lw=0.5, ls="--", zorder=2)
        ax.set_xlabel(r"$\Delta$SD (95% bootstrap interval)", fontsize=FONT)
        bottom_n, top_n = group_counts.get(gender, ("?", "?"))
        ax.set_title(f"{gender.capitalize()}  (n={bottom_n}/{top_n})",
                     fontsize=TITLE_FONT, weight="normal")
        ax.tick_params(axis="both", labelsize=FONT)
        ax.set_xlim(shared_xlim)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(
        display_labels, fontsize=LABEL_FONT, fontweight="normal"
    )
    axes[0].tick_params(axis="y", length=0)

    fig.subplots_adjust(
        left=0.27,
        right=0.985,
        bottom=0.045,
        top=0.97,
        wspace=0.16,
    )
    fig.text(
        0.01,
        0.995,
        "a",
        ha="left",
        va="top",
        fontsize=TITLE_FONT,
        fontweight="bold",
    )
    os.makedirs(OUTDIR, exist_ok=True)
    out_prefix = os.path.join(OUTDIR, f"lollipop_combined_p{percent}_")
    plt.savefig(out_prefix + ".png", dpi=300)
    plt.savefig(out_prefix + ".pdf")

    plt.close()
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"Saved combined lollipop → {out_prefix}.png/.pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    rf = add_vat_area(pd.read_csv(vv.COMBINED_RISK_FACTORS_PATH))
    if "subject_number" not in rf.columns and "subject_id" in rf.columns:
        rf = rf.rename(columns={"subject_id": "subject_number"})
    rf = rf.copy()
    rf.index = (rf["subject_number"].astype(int).astype(str)
                + "_" + rf["research_stage"].astype(str))
    rf = rf[~rf.index.duplicated(keep="first")]

    summary, all_sig, group_counts = {}, {}, {}
    for gender in GENDERS:
        print(f"\n{'='*60}\nProcessing: {gender.upper()}\n{'='*60}")
        gender_summary, sig, counts = process_gender(gender, rf)
        summary[gender] = gender_summary
        all_sig[gender] = sig
        group_counts[gender] = counts

    print(f"\n{'='*60}\nPlotting combined lollipop\n{'='*60}")
    plot_lollipop_combined(summary, all_sig, group_counts)
    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == "__main__":
    main()
