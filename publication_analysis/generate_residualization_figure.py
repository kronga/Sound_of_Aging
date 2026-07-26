#!/usr/bin/env python3
"""Generate the ΔVA residualization diagnostic on the analysis cohort."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
INPUT_ROOT = (
    ROOT
    / "analysis_outputs"
    / "repeated_analysis"
    / "final"
    / "primary_voice"
)
OUTPUT_ROOT = (
    ROOT
    / "analysis_outputs"
    / "repeated_analysis"
    / "figures"
)
COLORS = {"female": "#D97777", "male": "#6B9DC1"}
LABELS = {"female": "Female", "male": "Male"}


def load_predictions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected = {"female": 3631, "male": 3348}
    for sex in ("female", "male"):
        path = INPUT_ROOT / f"gender_{sex}" / "predictions_averaged.csv"
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"mean_predictions": "voice_age"})
        frame["sex"] = sex
        frame = frame.loc[
            frame["true_values"].between(40, 70),
            ["group", "true_values", "voice_age", "sex"],
        ].dropna()
        if len(frame) != expected[sex] or frame["group"].nunique() != expected[sex]:
            raise RuntimeError(
                f"{sex}: expected {expected[sex]} unique participants, "
                f"observed {len(frame)} rows and {frame['group'].nunique()} IDs."
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def residualize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output: list[pd.DataFrame] = []
    rows: list[dict[str, float | str | int]] = []
    for sex, subset in frame.groupby("sex", sort=False):
        subset = subset.copy()
        subset["raw_delta_va"] = subset["voice_age"] - subset["true_values"]
        raw_fit = linregress(subset["true_values"], subset["raw_delta_va"])
        subset["delta_va"] = subset["raw_delta_va"] - (
            raw_fit.intercept + raw_fit.slope * subset["true_values"]
        )
        residual_fit = linregress(subset["true_values"], subset["delta_va"])
        rows.append(
            {
                "sex": sex,
                "n": len(subset),
                "raw_slope": raw_fit.slope,
                "raw_r": raw_fit.rvalue,
                "residual_slope": residual_fit.slope,
                "residual_r": residual_fit.rvalue,
            }
        )
        output.append(subset)
    return pd.concat(output, ignore_index=True), pd.DataFrame(rows)


def make_figure(frame: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 76 / 25.4),
        sharex=True,
    )
    for panel, axis, value, title in (
        ("a", axes[0], "raw_delta_va", "Raw Voice Age difference"),
        ("b", axes[1], "delta_va", "Age-residualized ΔVA"),
    ):
        for sex in ("female", "male"):
            subset = frame.loc[frame["sex"] == sex]
            axis.scatter(
                subset["true_values"],
                subset[value],
                s=3,
                alpha=0.16,
                linewidth=0,
                color=COLORS[sex],
                label=LABELS[sex],
                rasterized=True,
            )
            fit = linregress(subset["true_values"], subset[value])
            x_values = np.linspace(40, 70, 200)
            axis.plot(
                x_values,
                fit.intercept + fit.slope * x_values,
                color=COLORS[sex],
                linewidth=1.1,
            )
        axis.axhline(0, color="#555555", linestyle="--", linewidth=0.7)
        axis.set_xlim(39.5, 70.5)
        axis.set_xlabel("Chronological age (years)")
        axis.set_ylabel(
            "Predicted minus chronological age (years)"
            if value == "raw_delta_va"
            else "ΔVA (years)"
        )
        axis.set_title(title, loc="center", fontweight="bold")
        axis.text(
            -0.13,
            1.04,
            panel,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
    axes[0].legend(frameon=False, loc="lower left")
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.17,
        top=0.92,
        wspace=0.27,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = OUTPUT_ROOT / f"supp_fig_S4_residualization.{extension}"
        figure.savefig(
            destination,
            dpi=300 if extension == "png" else None,
        )
        print(f"Saved: {destination}")
    plt.close(figure)
    diagnostics.to_csv(
        OUTPUT_ROOT / "supp_fig_S4_residualization_metrics.csv",
        index=False,
    )


def main() -> None:
    predictions = load_predictions()
    residualized, diagnostics = residualize(predictions)
    make_figure(residualized, diagnostics)
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
