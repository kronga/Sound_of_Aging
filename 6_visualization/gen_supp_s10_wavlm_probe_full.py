"""
Generate the full supplementary WavLM probe figure.

All 56 acoustic features (Praat HNR excluded), female/male bars,
color-coded by category and split across two columns.

Output: voice_age_manuscript/final_figs/supp_fig_S5_wavlm_probe_gender.pdf/.png
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
INPUT_CSV = ROOT / "analysis_outputs" / "step_p5_wavlm_probe_qc" / "probe_results_full_qc.csv"
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

    df = df.copy()
    df["label"] = df["label"].replace({"CPP": "Cepstral peak prominence (CPPS)"})

    FONT = 6
    plt.rcParams.update({
        "font.size":        FONT,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

    width_in = 180 / 25.4
    height_in = 225 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))

    split_at = int(np.ceil(len(df) / 2))
    panels = (df.iloc[:split_at], df.iloc[split_at:])
    bar_h = 0.36

    for ax, panel in zip(axes, panels):
        labels = panel["label"].tolist()
        r2_female = panel["r2_female"].to_numpy()
        r2_male = panel["r2_male"].to_numpy()
        cats = panel["category"].tolist()
        y_pos = np.arange(len(panel))
        bar_colors = [CATEGORY_COLORS.get(c, "#aaaaaa") for c in cats]

        for i in range(len(panel)):
            ax.axhspan(
                i - 0.5,
                i + 0.5,
                color="#f7f7f7" if i % 2 == 0 else "white",
                zorder=0,
            )

        ax.barh(
            y_pos + bar_h / 2,
            r2_female,
            bar_h,
            color=bar_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.barh(
            y_pos - bar_h / 2,
            r2_male,
            bar_h,
            color=bar_colors,
            alpha=0.45,
            hatch="///",
            edgecolor="white",
            linewidth=0.3,
        )

        for i, (rf, rm) in enumerate(zip(r2_female, r2_male)):
            if np.isfinite(rf):
                ax.text(
                    rf + 0.012,
                    y_pos[i] + bar_h / 2,
                    f"{rf:.2f}",
                    va="center",
                    ha="left",
                    fontsize=FONT,
                )
            if np.isfinite(rm):
                ax.text(
                    rm + 0.012,
                    y_pos[i] - bar_h / 2,
                    f"{rm:.2f}",
                    va="center",
                    ha="left",
                    fontsize=FONT,
                )

        prev_cat = None
        for i, cat in enumerate(cats):
            if cat != prev_cat and i > 0:
                ax.axhline(
                    i - 0.5,
                    color="gray",
                    linewidth=0.5,
                    linestyle="--",
                    alpha=0.6,
                )
            prev_cat = cat

        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=FONT)
        ax.tick_params(axis="both", labelsize=FONT, length=2.5, width=0.6)
        ax.set_xlim(left=0, right=1.12)
        ax.set_ylim(len(panel) - 0.5, -0.5)
        ax.set_xlabel("Out-of-fold R²", fontsize=FONT)

    cat_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c)
        for c in CATEGORY_ORDER
    ]
    sex_legend = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, label="Female"),
        mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="///", label="Male"),
    ]

    fig.legend(
        handles=sex_legend + cat_patches,
        fontsize=FONT,
        loc="lower center",
        frameon=False,
        ncol=6,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2,
        columnspacing=1.0,
    )

    fig.subplots_adjust(
        left=0.220,
        right=0.985,
        top=0.985,
        bottom=0.095,
        wspace=0.78,
    )
    for ext in (".pdf", ".png"):
        out = OUT_DIR / f"supp_fig_S5_wavlm_probe_gender{ext}"
        fig.savefig(out, dpi=300 if ext == ".png" else None)
        print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df_all = select_all_features(df)
    print(f"Plotting {len(df_all)} features across {df_all['category'].nunique()} categories")
    plot(df_all)


if __name__ == "__main__":
    main()
