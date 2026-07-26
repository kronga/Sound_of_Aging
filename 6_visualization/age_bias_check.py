"""
Supplementary figure: age-bias sensitivity analysis.

Three panels (female, voice model):
  A — Raw ∆VA vs. chronological age: shows regression-to-the-mean artifact + slope
  B — Residualized ∆VA vs. chronological age: flat after OLS deconfounding
  C — Phenome-wide effect sizes: main analysis vs. residualized-∆VA analysis,
      showing direction and significance of top hits are preserved
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, linregress
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

# ── paths ─────────────────────────────────────────────────────────────────────
REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE  = os.path.join(REPO, "..", "analysis_outputs")
OUT_TAG   = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()
PRED_PATH = os.path.join(OUT_BASE, "step3_voice_age_ridge", "gender_female",
                         "predictions_averaged.csv")
MAIN_VOLCANO_CSV = os.path.join(OUT_BASE, "step5_volcano", "voice",
                                "volcano_age_female_p25__results.csv")
SUBJECT_DETAILS_CSV = ("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
                       "Oct25_voice_full_length/subject_details_df_Oct25.csv")

MIN_AGE, MAX_AGE = 40, 70
PERCENTILE = 0.25
ALPHA = 0.1


def prefixed_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


def residual_volcano_dir() -> str:
    return os.path.join(OUT_BASE, "step5_volcano", prefixed_name("voice_residualized"))


PRED_PATH = os.path.join(OUT_BASE, STEP3_DIR, "gender_female", "predictions_averaged.csv")


def _stage_rank(stage: str) -> tuple[int, str]:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


# ── data loading ──────────────────────────────────────────────────────────────

def load_enriched_predictions() -> pd.DataFrame:
    """Load voice predictions and add research_stage via subject_details join."""
    pred = pd.read_csv(PRED_PATH)
    pred = pred.rename(columns={"group": "subject_number",
                                "mean_predictions": "predictions"})
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, usecols=["filename", "visit_number"])
    sd = sd.rename(columns={"visit_number": "research_stage"})
    pred = pred.merge(sd, left_on="index", right_on="filename", how="left")
    pred = pred.dropna(subset=["true_values", "predictions", "subject_number", "research_stage"])
    pred["_stage_rank"] = pred["research_stage"].map(_stage_rank)
    pred = pred.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    pred["delta_va"] = pred["predictions"] - pred["true_values"]
    return pred.reset_index(drop=True)


def build_subject_visit_key(df: pd.DataFrame) -> pd.Series:
    return (df["subject_number"].astype(int).astype(str)
            + "_"
            + df["research_stage"].astype(str))


# ── core analysis ─────────────────────────────────────────────────────────────

def residualize(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float, float]:
    """OLS: delta_va ~ true_values. Adds delta_va_resid column. Returns slope/intercept/r/p."""
    slope, intercept, r, p, _ = linregress(df["true_values"], df["delta_va"])
    df = df.copy()
    df["delta_va_resid"] = df["delta_va"] - (slope * df["true_values"] + intercept)
    return df, slope, intercept, r, p


def stratify_global(df: pd.DataFrame, value_col: str) -> tuple[list, list]:
    """Global top/bottom PERCENTILE within age range. Returns (bottom_keys, top_keys) as subject_visit strings."""
    subset = df[(df["true_values"] >= MIN_AGE) & (df["true_values"] <= MAX_AGE)].copy()
    subset["_key"] = build_subject_visit_key(subset)
    lo = subset[value_col].quantile(PERCENTILE)
    hi = subset[value_col].quantile(1 - PERCENTILE)
    bottom_keys = subset.loc[subset[value_col] <= lo, "_key"].tolist()
    top_keys    = subset.loc[subset[value_col] >= hi, "_key"].tolist()
    return bottom_keys, top_keys


def run_phenome_wide(bottom_keys: list, top_keys: list) -> pd.DataFrame:
    """Load combined_risk_factors (female), z-score, MWU per feature + BH-FDR."""
    rf = pd.read_csv(vv.COMBINED_RISK_FACTORS_PATH)
    if "subject_number" not in rf.columns and "subject_id" in rf.columns:
        rf = rf.rename(columns={"subject_id": "subject_number"})

    rf = rf[rf["gender"] == 0].copy()   # female
    rf.index = (rf["subject_number"].astype(int).astype(str)
                + "_"
                + rf["research_stage"].astype(str))
    rf = rf[~rf.index.duplicated(keep="first")]

    drop_cols = [c for c in vv.NON_FEATURE_COLS if c in rf.columns]
    X = rf.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X.astype(float)),
                            index=X.index, columns=X.columns)

    bottom_tbl = X_scaled.loc[[k for k in bottom_keys if k in X_scaled.index]]
    top_tbl    = X_scaled.loc[[k for k in top_keys    if k in X_scaled.index]]

    if len(bottom_tbl) < 10 or len(top_tbl) < 10:
        raise ValueError(f"Too few matched samples: bottom={len(bottom_tbl)}, top={len(top_tbl)}")

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f in bottom_tbl.columns.intersection(top_tbl.columns):
            x, y = bottom_tbl[f].dropna(), top_tbl[f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            results.append({
                "feature": f,
                "delta_z": float(y.mean() - x.mean()),
                "p_value": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
            })

    res = pd.DataFrame(results)
    pvals = res["p_value"].values.astype(float)
    keep  = np.isfinite(pvals)
    adj_p = np.ones(len(res))
    if keep.any():
        _, qvals, _, _ = multipletests(pvals[keep], alpha=ALPHA,
                                       method="fdr_bh", is_sorted=False)
        adj_p[keep] = qvals
    res["adj_p_value"] = adj_p
    res["significant"] = res["adj_p_value"] < ALPHA
    return res.set_index("feature")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.42)

    pred_df = load_enriched_predictions()
    pred_df, slope, intercept, r_a, p_a = residualize(pred_df)

    # ── Panel A: raw ∆VA vs. age ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    age_v   = pred_df["true_values"]
    delta_v = pred_df["delta_va"]
    x_line  = np.linspace(age_v.min(), age_v.max(), 200)

    ax_a.scatter(age_v, delta_v, alpha=0.2, s=6, color="steelblue", rasterized=True)
    ax_a.plot(x_line, slope * x_line + intercept, color="crimson", lw=2)
    ax_a.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_a.set_xlabel("Chronological age (years)", fontsize=12)
    ax_a.set_ylabel("∆VA  (predicted − actual, years)", fontsize=12)
    ax_a.set_title("A  Raw ∆VA vs. Chronological Age\n(regression-to-the-mean artifact)",
                   fontsize=11, fontweight="bold", loc="left")
    p_a_str = f"{p_a:.1e}" if p_a < 1e-3 else f"{p_a:.3f}"
    ax_a.text(0.97, 0.97,
              f"slope = {slope:.3f} yr/yr\nr = {r_a:.2f},  p = {p_a_str}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

    # ── Panel B: residualized ∆VA vs. age ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    resid_v = pred_df["delta_va_resid"]
    r_b, p_b = pearsonr(age_v, resid_v)
    slope_b, intercept_b = np.polyfit(age_v, resid_v, 1)

    ax_b.scatter(age_v, resid_v, alpha=0.2, s=6, color="steelblue", rasterized=True)
    ax_b.plot(x_line, slope_b * x_line + intercept_b, color="crimson", lw=2)
    ax_b.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_b.set_xlabel("Chronological age (years)", fontsize=12)
    ax_b.set_ylabel("Residualized ∆VA (years)", fontsize=12)
    ax_b.set_title("B  Age-Residualized ∆VA vs. Chronological Age\n(age effect removed)",
                   fontsize=11, fontweight="bold", loc="left")
    p_b_str = f"{p_b:.1e}" if p_b < 1e-3 else f"{p_b:.3f}"
    ax_b.text(0.97, 0.97,
              f"r = {r_b:.3f},  p = {p_b_str}",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

    # ── Panel C: phenome-wide sensitivity ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[2])

    bottom_keys, top_keys = stratify_global(pred_df, "delta_va_resid")
    print(f"Panel C  stratification: bottom n={len(bottom_keys)}, top n={len(top_keys)}")

    resid_res = run_phenome_wide(bottom_keys, top_keys)

    main_res = (pd.read_csv(MAIN_VOLCANO_CSV, index_col=0)
                  .set_index("feature")[["delta_z", "significant"]])
    merged = main_res.join(resid_res[["delta_z", "significant"]],
                           lsuffix="_main", rsuffix="_resid").dropna()

    sig_both = merged["significant_main"] & merged["significant_resid"]
    sig_one  = merged["significant_main"] | merged["significant_resid"]
    colors   = np.where(sig_both, "crimson",
               np.where(sig_one,  "darkorange", "steelblue"))

    ax_c.scatter(merged["delta_z_main"], merged["delta_z_resid"],
                 c=colors, alpha=0.7, s=30, zorder=3)
    lim = np.abs(merged[["delta_z_main", "delta_z_resid"]].values).max() * 1.1
    ax_c.plot([-lim, lim], [-lim, lim], "k--", lw=1, zorder=2)
    ax_c.axhline(0, color="grey", lw=0.5)
    ax_c.axvline(0, color="grey", lw=0.5)
    r_c, _ = pearsonr(merged["delta_z_main"], merged["delta_z_resid"])
    ax_c.text(0.05, 0.95, f"r = {r_c:.3f}", transform=ax_c.transAxes,
              va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))
    ax_c.set_xlabel("Effect size — main analysis (within-bin ∆VA)", fontsize=12)
    ax_c.set_ylabel("Effect size — sensitivity (age-residualized ∆VA)", fontsize=12)
    ax_c.set_title(f"C  Feature Associations Preserved\n(female, voice, n={len(merged)} features)",
                   fontsize=11, fontweight="bold", loc="left")
    ax_c.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='crimson',    markersize=7, label='Sig. in both'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='darkorange', markersize=7, label='Sig. in one'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',  markersize=7, label='Not significant'),
    ], fontsize=9, loc="lower right")

    # ── save 3-panel figure ────────────────────────────────────────────────────
    plt.suptitle("Age-bias sensitivity: OLS residualization removes age confound; "
                 "phenome-wide associations preserved",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    os.makedirs(OUT_BASE, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_BASE, f"{prefixed_name('supplementary_age_bias_check')}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()

    # ── save 2-panel figure (A + B only) ──────────────────────────────────────
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 5))
    fig2.subplots_adjust(wspace=0.38)

    ax2a.scatter(age_v, delta_v, alpha=0.2, s=6, color="steelblue", rasterized=True)
    ax2a.plot(x_line, slope * x_line + intercept, color="crimson", lw=2)
    ax2a.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2a.set_xlabel("Chronological age (years)", fontsize=12)
    ax2a.set_ylabel("∆VA  (predicted − actual, years)", fontsize=12)
    ax2a.set_title("A  Raw ∆VA vs. Chronological Age\n(regression-to-the-mean artifact)",
                   fontsize=11, fontweight="bold", loc="left")
    ax2a.text(0.97, 0.97,
              f"slope = {slope:.3f} yr/yr\nr = {r_a:.2f},  p = {p_a_str}",
              transform=ax2a.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

    ax2b.scatter(age_v, resid_v, alpha=0.2, s=6, color="steelblue", rasterized=True)
    ax2b.plot(x_line, slope_b * x_line + intercept_b, color="crimson", lw=2)
    ax2b.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2b.set_xlabel("Chronological age (years)", fontsize=12)
    ax2b.set_ylabel("Residualized ∆VA (years)", fontsize=12)
    ax2b.set_title("B  Age-Residualized ∆VA vs. Chronological Age\n(age effect removed)",
                   fontsize=11, fontweight="bold", loc="left")
    ax2b.text(0.97, 0.97,
              f"r = {r_b:.3f},  p = {p_b_str}",
              transform=ax2b.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

    fig2.suptitle("Age-bias correction: OLS residualization of ∆VA",
                  fontsize=12, y=1.01)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(OUT_BASE, f"{prefixed_name('supplementary_age_bias_check_AB')}.{ext}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # ── save residualized volcano results CSV (for lollipop script) ───────────
    resid_out_dir = residual_volcano_dir()
    os.makedirs(resid_out_dir, exist_ok=True)
    resid_res.reset_index().to_csv(
        os.path.join(resid_out_dir, "volcano_age_female_p25__results.csv"))
    print(f"Residualized results → {resid_out_dir}/volcano_age_female_p25__results.csv")

    print(f"Panel A  slope = {slope:.4f} yr/yr,  r = {r_a:.3f},  p = {p_a:.2e}")
    print(f"Panel B  r(resid ∆VA, age) = {r_b:.4f},  p = {p_b:.3f}")
    print(f"Panel C  r(main vs resid effect sizes) = {r_c:.3f}  ({len(merged)} features)")
    print(f"Saved → {OUT_BASE}/{prefixed_name('supplementary_age_bias_check')}.png/pdf")
    print(f"Saved → {OUT_BASE}/{prefixed_name('supplementary_age_bias_check_AB')}.png/pdf")


if __name__ == "__main__":
    main()
