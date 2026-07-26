#!/usr/bin/env python3
"""Boxplot panel: AHI, waist circumference, and VAT area by sex."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

sys.path.insert(0, os.path.dirname(__file__))
from run_boxplots_voice_residualized import (
    PlotSpec,
    VAT_AREA_COLUMN,
    load_rf,
    load_residualized_groups,
)

OUT_TAG   = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()

def tagged_name(name):
    return f"{name}_{OUT_TAG}" if OUT_TAG else name

OUTDIR = os.path.join(
    "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step5_volcano",
    tagged_name("voice_residualized"),
    "boxplots",
)

PANEL_SPECS = [
    PlotSpec(
        "AHI (SM)",
        "ahi",
        "AHI (events/hour)",
        threshold=15,
    ),
    PlotSpec(
        "Waist circumference",
        "waist",
        "Waist circumference (cm)",
        threshold={"female": 88, "male": 102},
    ),
    PlotSpec(
        VAT_AREA_COLUMN,
        "vat_area",
        "VAT area (cm²)",
        threshold=None,
    ),
]

GENDERS = ["female", "male"]
PALETTE = {"Bottom 25%": "#90EE90", "Top 25%": "#FA8072"}

# Together with the 125-mm lollipop and a 3-mm gap, this yields a
# 180-mm-wide, 220-mm-high combined figure.
FIG_WIDTH = 52 / 25.4
FIG_HEIGHT = 220 / 25.4
FONT = 7
TITLE_FONT = 8
SMALL_FONT = 6


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def draw_subplot(ax, top_vals, bot_vals, spec, gender, title, show_ylabel):
    plot_data = pd.DataFrame({
        "value": pd.concat([bot_vals, top_vals]),
        "Group": ["Bottom 25%"] * len(bot_vals) + ["Top 25%"] * len(top_vals),
    })

    sns.boxplot(
        data=plot_data, x="Group", y="value",
        hue="Group", palette=PALETTE, legend=False, ax=ax, width=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False, order=["Bottom 25%", "Top 25%"],
    )
    sns.stripplot(
        data=plot_data, x="Group", y="value",
        color="black", alpha=0.22, size=0.6, jitter=0.22, ax=ax,
        order=["Bottom 25%", "Top 25%"],
    )

    threshold = spec.threshold_for(gender)
    if threshold is not None:
        ax.axhline(
            threshold,
            color="darkblue",
            linestyle="--",
            linewidth=0.8,
            zorder=10,
        )

    y_max   = plot_data["value"].max()
    y_min   = plot_data["value"].min()
    y_range = max(y_max - y_min, 1.0)
    y_sig   = y_max + 0.04 * y_range
    h       = 0.015 * y_range
    _, pval = mannwhitneyu(top_vals, bot_vals, alternative="two-sided")
    stars   = sig_stars(pval)

    ax.plot([0, 0, 1, 1], [y_sig, y_sig + h, y_sig + h, y_sig], lw=1.0, c="black")
    weight = "bold" if stars != "ns" else "normal"
    ax.text(0.5, y_sig + h, stars, ha="center", va="bottom",
            fontsize=FONT, fontweight=weight)

    ax.set_title(title, fontsize=TITLE_FONT, fontweight="normal")
    ax.set_ylabel(
        spec.ylabel if show_ylabel else "",
        fontsize=FONT,
        labelpad=2,
    )
    ax.set_xlabel("")
    n_bot, n_top = len(bot_vals), len(top_vals)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"n={n_bot}", f"n={n_top}"], fontsize=SMALL_FONT
    )
    ax.tick_params(axis="y", labelsize=FONT)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    sns.despine(ax=ax)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT,
            "axes.labelsize": FONT,
            "axes.titlesize": TITLE_FONT,
            "xtick.labelsize": SMALL_FONT,
            "ytick.labelsize": FONT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rf = load_rf()

    groups = {}
    for gender in GENDERS:
        gender_val = 1 if gender == "male" else 0
        rf_g = rf[rf["gender"] == gender_val].copy()
        _, bot_keys, top_keys = load_residualized_groups(gender)
        groups[gender] = (rf_g, bot_keys, top_keys)

    n_rows = len(PANEL_SPECS)
    n_cols = len(GENDERS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey="row",
    )
    sns.set_style("whitegrid")

    for row, spec in enumerate(PANEL_SPECS):
        for col, gender in enumerate(GENDERS):
            ax = axes[row][col]
            rf_g, bot_keys, top_keys = groups[gender]

            if spec.column_name not in rf_g.columns:
                ax.set_visible(False)
                continue

            top_vals = rf_g.loc[[k for k in top_keys if k in rf_g.index], spec.column_name].dropna() * spec.scale
            bot_vals = rf_g.loc[[k for k in bot_keys if k in rf_g.index], spec.column_name].dropna() * spec.scale

            title      = gender.capitalize() if row == 0 else ""
            show_ylabel = col == 0
            draw_subplot(
                ax,
                top_vals,
                bot_vals,
                spec,
                gender,
                title,
                show_ylabel,
            )

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        bottom=0.035,
        top=0.97,
        wspace=0.15,
        hspace=0.30,
    )
    fig.text(
        0.015,
        0.995,
        "b",
        ha="left",
        va="top",
        fontsize=TITLE_FONT,
        fontweight="bold",
    )
    os.makedirs(OUTDIR, exist_ok=True)
    out_prefix = os.path.join(OUTDIR, "boxplots_panel")
    plt.savefig(out_prefix + ".png", dpi=300)
    plt.savefig(out_prefix + ".pdf")
    plt.close(fig)
    print(f"Saved panel → {out_prefix}.png/.pdf")


if __name__ == "__main__":
    main()
