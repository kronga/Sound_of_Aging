"""
Generate main-text WavLM probe figure (Fig. 3).

Reads QC-passed probe results from:
  analysis_outputs/step_p5_wavlm_probe_qc/probe_results_full_qc.csv

Selects top-2 features per acoustic category (Praat HNR excluded),
produces a clean horizontal grouped bar chart with female/male bars,
colour-coded by category.

Output: voice_age_manuscript/final_figs/fig3_wavlm_probe_top.pdf/.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parents[2]
INPUT_CSV = ROOT / "analysis_outputs" / "step_p5_wavlm_probe_qc" / "probe_results_full_qc.csv"
OUT_DIR   = ROOT / "voice_age_manuscript" / "final_figs"

# ── Category display order and colour palette ─────────────────────────────────
CATEGORY_ORDER = [
    "Loudness",
    "Harmonicity",
    "Cepstral",
    "Formants",
    "Spectral / tilt",
    "Voicing / breaks",
    "Perturbation",
    "Glottal quality",
    "F0 dynamics",
]

CATEGORY_COLORS = {
    "Loudness":         "#5B8DB8",
    "Harmonicity":      "#956CB4",
    "Cepstral":         "#EE854A",
    "Formants":         "#D65F5F",
    "Spectral / tilt":  "#6ACC65",
    "Voicing / breaks": "#DC7EC0",
    "Perturbation":     "#4878D0",
    "Glottal quality":  "#82C0CC",
    "F0 dynamics":      "#8C613C",
}

# Praat HNR features to exclude (near-zero / negative R²)
EXCLUDE_FEATURES = {"praat_hnr_mean", "praat_hnr_std"}

TOP_N_PER_CATEGORY = 2


def select_top_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Praat HNR, then take top-N by r2_all within each category."""
    df = df[~df["feature"].isin(EXCLUDE_FEATURES)].copy()
    selected = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat].sort_values("r2_all", ascending=False)
        selected.append(sub.head(TOP_N_PER_CATEGORY))
    return pd.concat(selected, ignore_index=True)


def plot(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort: category order, then descending R² within category
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df = df.copy()
    df["_cat_rank"] = df["category"].map(cat_rank).fillna(99)
    df = df.sort_values(["_cat_rank", "r2_all"], ascending=[True, False]).reset_index(drop=True)
    df["label"] = df["label"].replace(
        {"CPP": "Cepstral peak prominence (CPPS)"}
    )

    labels    = df["label"].tolist()
    r2_female = df["r2_female"].tolist()
    r2_male   = df["r2_male"].tolist()
    cats      = df["category"].tolist()

    n      = len(labels)
    y_pos  = np.arange(n)
    bar_h  = 0.35

    FONT = 8
    plt.rcParams.update({
        "font.size":        FONT,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    width_mm = 180.0
    height_mm = min(225.0, max(150.0, n * 10.7 + 18.0))
    fig, ax = plt.subplots(
        figsize=(width_mm / 25.4, height_mm / 25.4)
    )

    bar_colors = [CATEGORY_COLORS.get(c, "#aaaaaa") for c in cats]

    # Female bars (solid), male bars (hatched)
    bars_f = ax.barh(
        y_pos + bar_h / 2, r2_female, bar_h,
        color=bar_colors, alpha=0.85, label="Female",
        edgecolor="white", linewidth=0.3,
    )
    bars_m = ax.barh(
        y_pos - bar_h / 2, r2_male, bar_h,
        color=bar_colors, alpha=0.45, hatch="///", label="Male",
        edgecolor="white", linewidth=0.3,
    )

    # Annotate R² values
    for i, (rf, rm) in enumerate(zip(r2_female, r2_male)):
        if np.isfinite(rf):
            ax.text(rf + 0.005, y_pos[i] + bar_h / 2,
                    f"{rf:.2f}", va="center", ha="left", fontsize=6)
        if np.isfinite(rm):
            ax.text(rm + 0.005, y_pos[i] - bar_h / 2,
                    f"{rm:.2f}", va="center", ha="left", fontsize=6)

    # Category separator lines
    prev_cat = None
    for i, cat in enumerate(cats):
        if cat != prev_cat and i > 0:
            ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
        prev_cat = cat

    # Alternating row background
    for i in range(n):
        ax.axhspan(i - 0.5, i + 0.5,
                   color="#f7f7f7" if i % 2 == 0 else "white", zorder=0)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT)
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_xlabel("Out-of-fold R²", fontsize=FONT)

    # Category colour legend on right side
    cat_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c)
        for c in CATEGORY_ORDER
    ]
    # Sex legend
    sex_legend = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, label="Female"),
        mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="///", label="Male"),
    ]

    leg1 = ax.legend(
        handles=sex_legend,
        fontsize=FONT - 1, loc="upper right",
        frameon=True, framealpha=0.9,
        title="Sex", title_fontsize=FONT - 1,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=cat_patches,
        fontsize=FONT - 1.5,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=True, framealpha=0.9,
        title="Category", title_fontsize=FONT - 1,
        ncol=1,
        borderaxespad=0,
    )

    # Keep labels, bars and both legends within a fixed 180-mm journal canvas.
    fig.subplots_adjust(left=0.31, right=0.75, bottom=0.08, top=0.985)
    for ext in (".pdf", ".png"):
        out = OUT_DIR / f"fig4_wavlm_probe_top{ext}"
        fig.savefig(out, dpi=300 if ext == ".png" else None)
        print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df_top = select_top_features(df)
    print(f"Selected {len(df_top)} features across {df_top['category'].nunique()} categories")
    print(df_top[["label", "category", "r2_all", "r2_female", "r2_male"]].to_string(index=False))
    plot(df_top)


if __name__ == "__main__":
    main()
