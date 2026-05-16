#!/usr/bin/env python3
"""
Combined 3×2 boxplot figure: AHI, VAT mass, Hand grip right × Female/Male.
Output: 9 cm wide × 12 cm tall, all fonts 7-8 pt.
"""

import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

# ── paths ──────────────────────────────────────────────────────────────────
STEP3_DIR = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step3_voice_age_ridge_one_per_subject"
AVERAGED_PRED_PATH = os.path.join(STEP3_DIR, "gender_{gender}/predictions_averaged.csv")
SUBJECT_DETAILS_CSV = (
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/subject_details_df_Oct25.csv"
)
OUTDIR = "/home/davidkro/PycharmProjects/DeepVoice/voice_age_manuscript/final_figs"

MIN_AGE, MAX_AGE = 40, 72
PERCENTILE = 0.25
FONT_SIZE = 7

# ── feature specs ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PlotSpec:
    column_name: str
    ylabel: str
    threshold: float | None = None


FEATURES = [
    PlotSpec("AHI (SM)", "AHI (events/hr)", 15),
    PlotSpec("Scanned VAT mass (DXA)", "VAT mass (kg)", None),
    PlotSpec("Hand Grip Right", "Hand grip right (kg)", None),
]

GENDERS = ["female", "male"]
GENDER_LABELS = {"female": "Female", "male": "Male"}
GENDER_INT = {"female": 0, "male": 1}

# ── colours (match existing single-panel style) ────────────────────────────
PALETTE = ["lightgreen", "salmon"]


def _stage_rank(stage):
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def load_rf() -> pd.DataFrame:
    rf = pd.read_csv(vv.COMBINED_RISK_FACTORS_PATH)
    if "subject_number" not in rf.columns and "subject_id" in rf.columns:
        rf = rf.rename(columns={"subject_id": "subject_number"})
    rf = rf.copy()
    rf.index = (
        rf["subject_number"].astype(int).astype(str)
        + "_"
        + rf["research_stage"].astype(str)
    )
    return rf[~rf.index.duplicated(keep="first")]


def load_groups(gender: str) -> tuple[list[str], list[str]]:
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, usecols=["filename", "visit_number"])
    sd = sd.rename(columns={"visit_number": "research_stage"})
    pred = pd.read_csv(AVERAGED_PRED_PATH.format(gender=gender))
    pred = pred.rename(columns={"group": "subject_number", "mean_predictions": "predictions"})
    pred = pred.merge(sd, left_on="index", right_on="filename", how="left")
    pred = pred.dropna(subset=["true_values", "predictions", "subject_number", "research_stage"]).copy()
    pred["_stage_rank"] = pred["research_stage"].map(_stage_rank)
    pred = pred.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    pred["delta_va"] = pred["predictions"] - pred["true_values"]
    slope, intercept, *_ = linregress(pred["true_values"], pred["delta_va"])
    pred["delta_va_resid"] = pred["delta_va"] - (slope * pred["true_values"] + intercept)
    pred = pred[(pred["true_values"] >= MIN_AGE) & (pred["true_values"] <= MAX_AGE)].copy()
    pred["_key"] = (
        pred["subject_number"].astype(int).astype(str)
        + "_"
        + pred["research_stage"].astype(str)
    )
    lo = pred["delta_va_resid"].quantile(PERCENTILE)
    hi = pred["delta_va_resid"].quantile(1 - PERCENTILE)
    bottom = pred.loc[pred["delta_va_resid"] <= lo, "_key"].tolist()
    top = pred.loc[pred["delta_va_resid"] >= hi, "_key"].tolist()
    return bottom, top


def sig_label(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    return "ns"


def draw_panel(ax, rf_gender: pd.DataFrame, bottom_keys: list, top_keys: list, spec: PlotSpec):
    top_vals = rf_gender.loc[
        [k for k in top_keys if k in rf_gender.index], spec.column_name
    ].dropna()
    bot_vals = rf_gender.loc[
        [k for k in bottom_keys if k in rf_gender.index], spec.column_name
    ].dropna()

    _, pvalue = mannwhitneyu(top_vals, bot_vals, alternative="two-sided")
    label = sig_label(pvalue)

    plot_data = pd.DataFrame({
        spec.column_name: pd.concat([bot_vals, top_vals]),
        "Group": ["Bottom 25%"] * len(bot_vals) + ["Top 25%"] * len(top_vals),
    })

    sns.boxplot(
        data=plot_data, x="Group", y=spec.column_name,
        palette=PALETTE, ax=ax, width=0.55,
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False,
        linewidth=0.8,
    )
    sns.swarmplot(
        data=plot_data, x="Group", y=spec.column_name,
        color="black", alpha=0.3, size=1.2, ax=ax,
    )

    y_max = plot_data[spec.column_name].max()
    y_min = plot_data[spec.column_name].min()
    y_range = max(y_max - y_min, 1.0)
    y_sig = y_max + 0.05 * y_range
    h = 0.02 * y_range
    ax.plot([0, 0, 1, 1], [y_sig, y_sig + h, y_sig + h, y_sig], lw=1.0, c="black")
    ax.text(0.5, y_sig + h, label, ha="center", va="bottom",
            fontsize=FONT_SIZE + 1, fontweight="bold")

    if spec.threshold is not None:
        ax.axhline(y=spec.threshold, color="darkblue", linestyle="--", linewidth=0.8, zorder=10)

    ax.set_ylabel(spec.ylabel, fontsize=FONT_SIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.set_xticks([])


def main():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "font.family": "sans-serif",
    })
    sns.set_style("whitegrid")

    rf = load_rf()

    # pre-load groups for both sexes
    groups = {}
    for gender in GENDERS:
        groups[gender] = load_groups(gender)

    fig_w_cm, fig_h_cm = 9, 12 * 1.3
    fig, axes = plt.subplots(
        nrows=len(FEATURES), ncols=len(GENDERS),
        figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
    )

    for row, spec in enumerate(FEATURES):
        for col, gender in enumerate(GENDERS):
            ax = axes[row][col]
            bottom_keys, top_keys = groups[gender]
            rf_gender = rf[rf["gender"] == GENDER_INT[gender]].copy()

            if spec.column_name not in rf_gender.columns:
                ax.set_visible(False)
                continue

            draw_panel(ax, rf_gender, bottom_keys, top_keys, spec)


    # single shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[0], label="Bottom 25%"),
        Patch(facecolor=PALETTE[1], label="Top 25%"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(pad=0.5, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(bottom=0.07)

    out_prefix = os.path.join(OUTDIR, "fig2_combined_boxplots")
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
