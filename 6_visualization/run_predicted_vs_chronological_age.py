#!/usr/bin/env python3
"""
Sanity-check scatter plots: predicted age vs chronological age.

Reads averaged step-3 predictions for female and male, then writes tagged
output figures without overwriting prior runs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()


def tagged_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


BASE = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs"
AVERAGED_PRED_BASE = os.path.join(BASE, STEP3_DIR, "gender_{gender}", "predictions_averaged.csv")
OUT_BASE = BASE
GENDERS = ["female", "male"]
COLORS = {"female": "lightgreen", "male": "salmon"}


def _stage_rank(stage: str) -> tuple[int, str]:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def load_predictions(gender: str) -> pd.DataFrame:
    sd = pd.read_csv(
        "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/subject_details_df_Oct25.csv",
        usecols=["filename", "visit_number"],
    ).rename(columns={"visit_number": "research_stage"})
    df = pd.read_csv(AVERAGED_PRED_BASE.format(gender=gender))
    pred_col = "mean_predictions" if "mean_predictions" in df.columns else "predictions"
    df = df.dropna(subset=["true_values", pred_col]).rename(columns={pred_col: "predicted_age", "group": "subject_number"})
    df = df.merge(sd, left_on="index", right_on="filename", how="left")
    df = df.dropna(subset=["subject_number", "research_stage"]).copy()
    df["_stage_rank"] = df["research_stage"].map(_stage_rank)
    df = df.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    return df.reset_index(drop=True)


def plot_panel(ax: plt.Axes, df: pd.DataFrame, gender: str) -> None:
    x = df["true_values"].to_numpy()
    y = df["predicted_age"].to_numpy()
    r, p = pearsonr(x, y)
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    sns.scatterplot(x=x, y=y, ax=ax, s=12, alpha=0.35, color=COLORS[gender], edgecolor=None)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="dimgrey", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title(f"{gender.capitalize()}  (n={len(df)})", fontweight="bold")
    p_str = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
    ax.text(
        0.97,
        0.03,
        f"r = {r:.3f}\nR² = {r2:.3f}\nMAE = {mae:.2f}\np = {p_str}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.85),
    )


def main() -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharex=False, sharey=False)

    summaries = []
    for ax, gender in zip(axes, GENDERS):
        df = load_predictions(gender)
        plot_panel(ax, df, gender)
        summaries.append({"gender": gender, "n": len(df)})

    plt.tight_layout()
    out_prefix = os.path.join(OUT_BASE, tagged_name("predicted_vs_chronological_age"))
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_prefix}.png/.pdf")


if __name__ == "__main__":
    main()
