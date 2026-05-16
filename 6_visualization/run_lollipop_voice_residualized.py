#!/usr/bin/env python3
"""
Lollipop plot of phenome-wide associations using age-residualized ∆VA.

Per-seed pipeline:
  1. Load per-seed out-of-fold predictions from step3_voice_age_ridge
  2. Per subject: delta_va = predictions - true_values
  3. OLS: delta_va ~ true_values (fit per seed)
  4. Residualize: delta_va_resid = delta_va - (slope * age + intercept)
  5. Global top/bottom 25% by delta_va_resid within age range
  6. Compute delta_z (mean_top - mean_bottom) per feature

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
from scipy import stats
from scipy.stats import linregress, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D


def _clean_feature_label(name: str) -> str:
    """Strip parenthetical suffixes like (SM), (BT), (DXA), (DL), (US), (FI).
    Also fix backslash to forward slash and collapse extra whitespace."""
    name = name.replace("\\", "/")
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

# ── config ────────────────────────────────────────────────────────────────────
OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()


def tagged_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


SEED_PREDICTIONS_BASE = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"paper_revision_outputs/{STEP3_DIR}/"
    "gender_{gender}"
)
AVERAGED_PRED_BASE = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"paper_revision_outputs/{STEP3_DIR}/"
    "gender_{gender}/predictions_averaged.csv"
)
SUBJECT_DETAILS_CSV = (
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/subject_details_df_Oct25.csv"
)
RESID_VOLCANO_DIR = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"paper_revision_outputs/step5_volcano/{tagged_name('voice_residualized')}/"
)
OUTDIR = RESID_VOLCANO_DIR

GENDERS = ["male", "female"]
MIN_AGE, MAX_AGE = 40, 72
PERCENTILE = 0.25
ALPHA = 0.1
CI_LEVEL = 0.95
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
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, usecols=["filename", "visit_number"])
    sd = sd.rename(columns={"visit_number": "research_stage"})
    df = pred.rename(columns={"group": "subject_number"}).copy()
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

def load_seed_predictions(gender_dir: str) -> dict[str, pd.DataFrame]:
    seed_preds = {}
    for entry in sorted(os.listdir(gender_dir)):
        if not entry.startswith("seed_"):
            continue
        path = os.path.join(gender_dir, entry, "predictions.csv")
        if os.path.exists(path):
            seed_preds[entry] = pd.read_csv(path)
    return seed_preds


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


def compute_summary(effects_df: pd.DataFrame) -> pd.DataFrame:
    n = len(effects_df)
    t_crit = stats.t.ppf((1 + CI_LEVEL) / 2, df=max(n - 1, 1))
    mean_e = effects_df.mean()
    ci = t_crit * effects_df.std() / np.sqrt(n)
    return pd.DataFrame({"mean": mean_e, "ci": ci})


def run_averaged_significance(gender: str, rf_scaled: pd.DataFrame) -> list[str]:
    """Compute significant features from averaged predictions using OLS residualization."""
    # Try loading pre-computed results first
    resid_csv = os.path.join(RESID_VOLCANO_DIR, f"volcano_age_{gender}_p25__results.csv")
    if os.path.exists(resid_csv):
        res = pd.read_csv(resid_csv, index_col=0)
        sig_col = "feature" if "feature" in res.columns else res.index.name
        if "feature" in res.columns:
            return res.loc[res["significant"] == True, "feature"].tolist()
        else:
            return res.index[res["significant"] == True].tolist()

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


def process_gender(gender: str, rf: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    gender_dir = SEED_PREDICTIONS_BASE.format(gender=gender)
    seed_preds = load_seed_predictions(gender_dir)
    print(f"  Found {len(seed_preds)} seeds: {list(seed_preds.keys())}")

    gender_val = 1 if gender == "male" else 0
    rf_g = rf[rf["gender"] == gender_val].copy()
    drop_cols = [c for c in vv.NON_FEATURE_COLS if c in rf_g.columns]
    X = rf_g.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X.astype(float)),
                            index=X.index, columns=X.columns)

    all_effects = {}
    final_counts = None
    for seed_name, pred in seed_preds.items():
        df = load_latest_subject_predictions(pred)
        df["delta_va"] = df["predictions"] - df["true_values"]
        df["delta_va_resid"], r2 = residualize_series(df["true_values"], df["delta_va"])
        bottom_ids, top_ids = stratify_by_resid(df)
        bottom_ids = [i for i in bottom_ids if i in X_scaled.index]
        top_ids    = [i for i in top_ids if i in X_scaled.index]

        if len(bottom_ids) < 10 or len(top_ids) < 10:
            print(f"  Skipping {seed_name}: too few matched ({len(bottom_ids)}/{len(top_ids)})")
            continue

        all_effects[seed_name] = compute_effect_sizes(
            X_scaled.loc[bottom_ids], X_scaled.loc[top_ids])
        final_counts = (len(bottom_ids), len(top_ids))
        print(f"  {seed_name}: bottom={len(bottom_ids)}, top={len(top_ids)},  r²_age={r2:.3f}")

    if len(all_effects) < 2:
        raise RuntimeError(f"Not enough seeds for {gender}.")

    effects_df = pd.DataFrame(all_effects).T.dropna(axis=1, how="all")
    if final_counts is not None:
        effects_df.attrs["group_counts"] = final_counts
    sig = run_averaged_significance(gender, X_scaled)
    print(f"  Significant features: {len(sig)}")
    return effects_df, sig


def plot_lollipop_combined(
    summary: dict[str, pd.DataFrame],
    sig_features: dict[str, list[str]],
    n_seeds: dict[str, int],
    group_counts: dict[str, tuple[int, int]],
):
    FONT = 8
    FIG_WIDTH = 6.30
    plt.rcParams.update({"font.size": FONT})

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

    line_height_inch = FONT / 72 * 1.5
    fig_height = max(6, n_feat * line_height_inch + 1.5)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(FIG_WIDTH, fig_height))
    sns.set_style("whitegrid")

    for gender, ax in [("female", axes[0]), ("male", axes[1])]:
        for i in range(n_feat):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)

        sdf     = summary[gender]
        sig_set = set(sig_features.get(gender, []))
        means = np.array([sdf.loc[f, "mean"] if f in sdf.index else np.nan for f in all_features])
        cis   = np.array([sdf.loc[f, "ci"]   if f in sdf.index else np.nan for f in all_features])
        is_sig = np.array([f in sig_set for f in all_features])
        colors = [("#FF5252" if m > 0 else "#4CAF50") if s else "grey"
                  for m, s in zip(means, is_sig)]
        edges  = ["black" if s else "none" for s in is_sig]
        sizes  = [18 if s else 8 for s in is_sig]

        valid = ~np.isnan(means)
        ax.errorbar(means[valid], y_pos[valid], xerr=cis[valid],
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
        ax.set_xlabel(r"$\Delta$SD (mean $\pm$ 95% CI)", fontsize=FONT)
        bottom_n, top_n = group_counts.get(gender, ("?", "?"))
        ax.set_title(f"{gender.capitalize()}  (n={bottom_n}/{top_n})",
                     fontsize=FONT, weight="bold")
        ax.tick_params(axis="both", labelsize=FONT)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(display_labels, fontsize=FONT)
    axes[0].tick_params(axis="y", length=0)

    fig.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF5252',
               markeredgecolor='black', markersize=4, label='Higher in old-predicted (sig.)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4CAF50',
               markeredgecolor='black', markersize=4, label='Lower in old-predicted (sig.)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
               markersize=3.5, label='Not significant'),
    ], fontsize=FONT, frameon=True, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(OUTDIR, exist_ok=True)
    out_prefix = os.path.join(OUTDIR, f"lollipop_combined_p{percent}_")
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")

    plt.close()
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"Saved combined lollipop → {out_prefix}.png/.pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    rf = pd.read_csv(vv.COMBINED_RISK_FACTORS_PATH)
    if "subject_number" not in rf.columns and "subject_id" in rf.columns:
        rf = rf.rename(columns={"subject_id": "subject_number"})
    rf = rf.copy()
    rf.index = (rf["subject_number"].astype(int).astype(str)
                + "_" + rf["research_stage"].astype(str))
    rf = rf[~rf.index.duplicated(keep="first")]

    all_effects, all_sig, group_counts = {}, {}, {}
    for gender in GENDERS:
        print(f"\n{'='*60}\nProcessing: {gender.upper()}\n{'='*60}")
        effects_df, sig = process_gender(gender, rf)
        all_effects[gender] = effects_df
        all_sig[gender]     = sig
        group_counts[gender] = effects_df.attrs.get("group_counts", ("?", "?"))

    print(f"\n{'='*60}\nPlotting combined lollipop\n{'='*60}")
    summary  = {g: compute_summary(all_effects[g]) for g in GENDERS}
    n_seeds  = {g: len(all_effects[g]) for g in GENDERS}
    plot_lollipop_combined(summary, all_sig, n_seeds, group_counts)
    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == "__main__":
    main()
