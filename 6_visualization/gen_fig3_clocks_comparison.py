"""
Generate Fig 3 panels as separate files (to be combined later):
  fig3a_bar.png/pdf       — vertical grouped bar chart of R² per modality
  fig3b_heatmap_female.png/pdf — voice-row correlation heatmap, females
  fig3c_heatmap_male.png/pdf   — voice-row correlation heatmap, males

Blood-test modality excluded (sparse). No colourbars. "Voice Age" labelled as "Voice".
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Output paths ───────────────────────────────────────────────────────────────
OUT_DIR   = Path("/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step6_visualization")
FINAL_DIR = Path("/home/davidkro/PycharmProjects/DeepVoice/voice_age_manuscript/final_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ── R² data: 10-seed mean ± SD ─────────────────────────────────────────────────
# Voice: canonical from plan.md (step3_voice_age_ridge_one_per_subject, 10 seeds)
# Others: 10-seed means from step4_multimodality_lgbm_hpo (LGBM) or
#         step4_multimodality_ridge (Ridge, for NMR and Retina where Ridge wins)
# Lifestyle: single seed only (no SD available)
DATA = {
    "MS Metabolomics": {"female_r2": 0.6293, "male_r2": 0.5043, "female_sd": 0.0035, "male_sd": 0.0048},
    "Voice":           {"female_r2": 0.5390, "male_r2": 0.4390, "female_sd": 0.0020, "male_sd": 0.0010},
    "Lifestyle":       {"female_r2": 0.5103, "male_r2": 0.4023, "female_sd": None,   "male_sd": None  },
    "Sleep":           {"female_r2": 0.5072, "male_r2": 0.4256, "female_sd": 0.0019, "male_sd": 0.0023},
    "DEXA":            {"female_r2": 0.4605, "male_r2": 0.4064, "female_sd": 0.0020, "male_sd": 0.0018},
    "Diet":            {"female_r2": 0.2955, "male_r2": 0.2691, "female_sd": 0.0042, "male_sd": 0.0052},
    "NMR Metabolomics":{"female_r2": 0.2551, "male_r2": 0.1908, "female_sd": 0.0008, "male_sd": 0.0014},
    "Microbiome":      {"female_r2": 0.1751, "male_r2": 0.1613, "female_sd": 0.0030, "male_sd": 0.0025},
    "Retina":          {"female_r2": 0.1439, "male_r2": 0.1551, "female_sd": 0.0000, "male_sd": 0.0001},
}

# ── Voice-row correlations ─────────────────────────────────────────────────────
CORR_FEMALE = {
    "MS Metabolomics":  0.579,
    "Voice":            1.000,
    "Lifestyle":        0.504,
    "Sleep":            0.548,
    "DEXA":             0.487,
    "Diet":             0.379,
    "NMR Metabolomics": 0.347,
    "Microbiome":       0.275,
    "Retina":           0.295,
}
CORR_MALE = {
    "MS Metabolomics":  0.430,
    "Voice":            1.000,
    "Lifestyle":        0.400,
    "Sleep":            0.430,
    "DEXA":             0.382,
    "Diet":             0.340,
    "NMR Metabolomics": 0.277,
    "Microbiome":       0.201,
    "Retina":           0.321,
}

FEMALE_COLOR = "#E07B7B"
MALE_COLOR   = "#6E9FC2"
VOICE_LABEL  = "Voice"


def _save(fig: plt.Figure, stem: str) -> None:
    for fmt in ("png", "pdf"):
        for base in (OUT_DIR, FINAL_DIR):
            p = base / f"{stem}.{fmt}"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Saved: {p}")


def make_bar_panel() -> None:
    df = pd.DataFrame(DATA).T.sort_values("female_r2", ascending=False)
    x = np.arange(len(df))
    bar_w = 0.35

    f_sd = df["female_sd"].where(df["female_sd"].notna(), 0).astype(float).values
    m_sd = df["male_sd"].where(df["male_sd"].notna(), 0).astype(float).values

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.set_style("whitegrid")

    err_kw = dict(ecolor="#333333", capsize=3, capthick=1, elinewidth=1, zorder=4)

    female_bars = ax.bar(x - bar_w / 2, df["female_r2"], width=bar_w,
                         color=FEMALE_COLOR, label="Female", zorder=3,
                         yerr=f_sd, error_kw=err_kw)
    male_bars   = ax.bar(x + bar_w / 2, df["male_r2"],   width=bar_w,
                         color=MALE_COLOR,   label="Male",   zorder=3,
                         yerr=m_sd, error_kw=err_kw)

    # Bold outline on Voice bars
    voice_idx = list(df.index).index(VOICE_LABEL)
    for b in [female_bars[voice_idx], male_bars[voice_idx]]:
        b.set_edgecolor("#222222")
        b.set_linewidth(1.8)

    # Value labels above bars (offset by SD so they clear the error bar)
    for bar, val, sd in zip(female_bars, df["female_r2"], f_sd):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (sd or 0) + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    for bar, val, sd in zip(male_bars, df["male_r2"], m_sd):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (sd or 0) + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Out-of-fold R²", fontsize=9)
    ax.set_ylim(0, 0.78)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, "fig3a_bar")
    plt.close()


def make_heatmap_panel(corr_dict: dict[str, float], sex: str, stem: str) -> None:
    # Same column order as bar chart (all modalities, including Voice)
    df_order = pd.DataFrame(DATA).T.sort_values("female_r2", ascending=False)
    ordered = [k for k in df_order.index if k in corr_dict]

    vals = np.array([[corr_dict[k] for k in ordered]])
    df_plot = pd.DataFrame(vals, columns=ordered)

    n_cols = len(ordered)
    fig, ax = plt.subplots(figsize=(n_cols * 0.85, 1.6))
    sns.set_style("white")

    sns.heatmap(
        df_plot,
        ax=ax,
        cmap="rocket_r",
        vmin=0, vmax=0.7,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5},
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        square=True,
    )
    ax.set_yticks([0.5])
    ax.set_yticklabels([sex], rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    _save(fig, stem)
    plt.close()


def main() -> None:
    make_bar_panel()
    make_heatmap_panel(CORR_FEMALE, "Female", "fig3b_heatmap_female")
    make_heatmap_panel(CORR_MALE,   "Male",   "fig3c_heatmap_male")


if __name__ == "__main__":
    main()
