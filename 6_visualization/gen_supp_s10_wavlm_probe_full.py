"""
Generate supplementary WavLM probe figure (Supp Fig S6).

All 56 acoustic features (Praat HNR excluded), female/male bars,
colour-coded by category, split into two side-by-side feature panels.

Output: voice_age_manuscript/final_figs/supp_fig_S6_wavlm_probe_gender.pdf/.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT      = Path(__file__).parents[2]
INPUT_CSV = ROOT / "paper_revision_outputs" / "step_p5_wavlm_probe_qc" / "probe_results_full_qc.csv"
OUT_DIR   = ROOT / "voice_age_manuscript" / "final_figs"

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

EXCLUDE_FEATURES = {"praat_hnr_mean", "praat_hnr_std"}


def select_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["feature"].isin(EXCLUDE_FEATURES)].copy()
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df["_cat_rank"] = df["category"].map(cat_rank).fillna(99)
    return df.sort_values(["_cat_rank", "r2_all"], ascending=[True, False]).reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bar_h  = 0.35

    FONT = 7
    plt.rcParams.update({
        "font.size":        FONT,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    # Preserve the original top-to-bottom order while cutting the long list in half.
    split_idx = int(np.ceil(len(df) / 2))
    panels = [df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()]
    max_panel_n = max(len(panel) for panel in panels)
    fig_height = max(8.0, max_panel_n * 0.34 + 1.2)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.8, fig_height),
        sharex=True,
        gridspec_kw={"wspace": 0.62},
    )

    for ax, panel in zip(axes, panels):
        labels    = panel["label"].tolist()
        r2_female = panel["r2_female"].tolist()
        r2_male   = panel["r2_male"].tolist()
        cats      = panel["category"].tolist()

        n     = len(labels)
        y_pos = np.arange(n)
        bar_colors = [CATEGORY_COLORS.get(c, "#aaaaaa") for c in cats]

        for i in range(n):
            ax.axhspan(i - 0.5, i + 0.5,
                       color="#f7f7f7" if i % 2 == 0 else "white", zorder=0)

        ax.barh(
            y_pos + bar_h / 2, r2_female, bar_h,
            color=bar_colors, alpha=0.85, label="Female",
            edgecolor="white", linewidth=0.3,
        )
        ax.barh(
            y_pos - bar_h / 2, r2_male, bar_h,
            color=bar_colors, alpha=0.45, hatch="///", label="Male",
            edgecolor="white", linewidth=0.3,
        )

        for i, (rf, rm) in enumerate(zip(r2_female, r2_male)):
            if np.isfinite(rf):
                ax.text(rf + 0.004, y_pos[i] + bar_h / 2,
                        f"{rf:.2f}", va="center", ha="left", fontsize=5)
            if np.isfinite(rm):
                ax.text(rm + 0.004, y_pos[i] - bar_h / 2,
                        f"{rm:.2f}", va="center", ha="left", fontsize=5)

        prev_cat = None
        for i, cat in enumerate(cats):
            if cat != prev_cat and i > 0:
                ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
            prev_cat = cat

        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=FONT)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_xlim(left=-0.05, right=1.10)
        ax.set_xlabel("Out-of-fold R²", fontsize=FONT)

    cat_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c)
        for c in CATEGORY_ORDER
    ]
    sex_legend = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, label="Female"),
        mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="///", label="Male"),
    ]

    leg1 = axes[1].legend(
        handles=sex_legend,
        fontsize=FONT - 1, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=True, framealpha=0.9,
        title="Sex", title_fontsize=FONT - 1,
        borderaxespad=0,
    )
    axes[1].add_artist(leg1)
    axes[1].legend(
        handles=cat_patches,
        fontsize=FONT - 1.5,
        loc="upper left", bbox_to_anchor=(1.02, 0.86),
        frameon=True, framealpha=0.9,
        title="Category", title_fontsize=FONT - 1,
        ncol=1,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.11, right=0.82, top=0.98, bottom=0.08, wspace=0.62)
    for ext in (".pdf", ".png"):
        out = OUT_DIR / f"supp_fig_S6_wavlm_probe_gender{ext}"
        fig.savefig(out, dpi=300 if ext == ".png" else None, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df_all = select_all_features(df)
    print(f"Plotting {len(df_all)} features across {df_all['category'].nunique()} categories")
    plot(df_all)


if __name__ == "__main__":
    main()
