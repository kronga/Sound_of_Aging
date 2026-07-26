#!/usr/bin/env python3
"""
Recreate the gradient predicted-vs-chronological-age figure from age_study.ipynb
using the latest available recording per subject.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()


def tagged_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs"
AVERAGED_PRED_BASE = os.path.join(BASE, STEP3_DIR, "gender_{gender}", "predictions_averaged.csv")
SUBJECT_DETAILS_CSV = (
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/subject_details_df_Oct25.csv"
)
GENDERS = ["female", "male"]
X_LIM_MIN = 39
X_LIM_MAX = 71
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 13
COLORBAR_FONTSIZE = 16
METRICS_BOX_FONTSIZE = 15


def _stage_rank(stage: str) -> tuple[int, str]:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def load_latest_predictions(gender: str) -> pd.DataFrame:
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, usecols=["filename", "visit_number"])
    sd = sd.rename(columns={"visit_number": "research_stage"})
    df = pd.read_csv(AVERAGED_PRED_BASE.format(gender=gender))
    pred_col = "mean_predictions" if "mean_predictions" in df.columns else "predictions"
    df = df.rename(columns={pred_col: "predictions", "group": "subject_number"})
    df = df.dropna(subset=["true_values", "predictions"]).copy()
    df = df.merge(sd, left_on="index", right_on="filename", how="left")
    df = df.dropna(subset=["subject_number", "research_stage"]).copy()
    df["_stage_rank"] = df["research_stage"].map(_stage_rank)
    df = df.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    return df.reset_index(drop=True)


def add_gradient_values(df: pd.DataFrame) -> pd.DataFrame:
    df_plot = df.copy()
    df_plot["age_year"] = df_plot["true_values"].round()
    df_plot["residual"] = df_plot["predictions"] - df_plot["true_values"]

    color_values = []
    for _, row in df_plot.iterrows():
        age_year = row["age_year"]
        age_year_data = df_plot.loc[df_plot["age_year"] == age_year, "residual"]
        if len(age_year_data) < 4:
            color_values.append(0.0)
            continue

        q1 = age_year_data.quantile(0.25)
        q3 = age_year_data.quantile(0.75)
        residual = row["residual"]

        if residual >= q3:
            normalized = (residual - q3) / (age_year_data.max() - q3 + 1e-10)
            color_values.append(0.5 + 0.5 * normalized)
        elif residual <= q1:
            normalized = (residual - age_year_data.min()) / (q1 - age_year_data.min() + 1e-10)
            color_values.append(-1.0 + 0.5 * normalized)
        else:
            normalized = (residual - q1) / (q3 - q1 + 1e-10)
            color_values.append(-0.15 + 0.3 * normalized)

    df_plot["color_value"] = color_values
    return df_plot


def build_cmap() -> LinearSegmentedColormap:
    colors_list = [
        "#228B22",
        "#90EE90",
        "#E8F5E9",
        "#FFFFFF",
        "#FFFFFF",
        "#FFE8E8",
        "#FFA07A",
        "#DC143C",
    ]
    return LinearSegmentedColormap.from_list("green_white_red", colors_list, N=256)


def plot_gender(df: pd.DataFrame, gender: str) -> None:
    df_plot = add_gradient_values(df)
    true_values = df_plot["true_values"]
    predictions = df_plot["predictions"]
    mae = mean_absolute_error(true_values, predictions)
    rmse = float(np.sqrt(mean_squared_error(true_values, predictions)))
    r2 = r2_score(true_values, predictions)
    correlation, p = pearsonr(true_values, predictions)

    cmap = build_cmap()
    min_age = min(true_values.min(), predictions.min()) - 2
    max_age = max(true_values.max(), predictions.max()) + 2

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        df_plot["true_values"],
        df_plot["predictions"],
        c=df_plot["color_value"],
        cmap=cmap,
        alpha=0.7,
        s=40,
        edgecolors="gray",
        linewidth=0.5,
        vmin=-1,
        vmax=1,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_ticks([-0.75, 0.75])
    cbar.set_ticklabels(['Bottom 25%\n"younger voice"', 'Top 25%\n"older voice"'])
    cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE)

    z = np.polyfit(df_plot["true_values"], df_plot["predictions"], 3)
    pfit = np.poly1d(z)
    x_sorted = np.linspace(df_plot["true_values"].min(), df_plot["true_values"].max(), 300)
    ax.plot(
        x_sorted,
        pfit(x_sorted),
        color="darkblue",
        alpha=0.8,
        linewidth=2.5,
        label="Median trendline",
        zorder=5,
    )
    ax.plot(
        [X_LIM_MIN, X_LIM_MAX],
        [X_LIM_MIN, X_LIM_MAX],
        linestyle="--",
        color="grey",
        linewidth=1.4,
        alpha=0.98,
        label="Identity line",
        zorder=4,
    )

    ax.set_xlim(X_LIM_MIN, X_LIM_MAX)
    ax.set_ylim(min_age, max_age)
    ax.set_xlabel("Chronological age (years)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Predicted age (years)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=LEGEND_FONTSIZE)

    metrics_text = (
        f"n = {len(df_plot):,} people\n"
        f"MAE = {mae:.2f} years\n"
        f"RMSE = {rmse:.2f} years\n"
        f"R² = {r2:.3f}"
    )
    ax.text(
        0.98,
        0.03,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=METRICS_BOX_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.85),
    )

    out_prefix = os.path.join(BASE, tagged_name(f"predicted_vs_chronological_age_gradient_{gender}"))
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_prefix}.png/.pdf")


def main() -> None:
    for gender in GENDERS:
        plot_gender(load_latest_predictions(gender), gender)


if __name__ == "__main__":
    main()
