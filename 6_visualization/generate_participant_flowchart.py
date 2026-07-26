#!/usr/bin/env python3
"""Generate the publication-ready Supplementary Figure 6 cohort flowchart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FINAL_FIGURES = ROOT / "voice_age_manuscript" / "final_figs"

BLUE = "#4C78A8"
BLUE_DARK = "#2F5F8F"
BLUE_LIGHT = "#EEF4F8"
FEMALE = "#D97777"
MALE = "#6B9DC1"
INK = "#202124"
MUTED = "#62666A"
GRAY_FILL = "#F2F2F2"
GRAY_EDGE = "#B8B8B8"


def rounded_box(
    axis: plt.Axes,
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.9,
    radius: float = 0.018,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )
    )


def stage_box(
    axis: plt.Axes,
    center_x: float,
    heading: str,
    details: str,
    *,
    final: bool = False,
) -> None:
    center_y = 0.67
    width = 0.205
    height = 0.245
    rounded_box(
        axis,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        facecolor="#E5EFF6" if final else BLUE_LIGHT,
        edgecolor=BLUE_DARK if final else BLUE,
        linewidth=1.15 if final else 0.9,
    )
    axis.text(
        center_x,
        center_y + 0.047,
        heading,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        linespacing=1.15,
        zorder=3,
    )
    axis.text(
        center_x,
        center_y - 0.055,
        details,
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
        linespacing=1.28,
        zorder=3,
    )


def horizontal_arrow(axis: plt.Axes, start: float, end: float) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (start, 0.67),
            (end, 0.67),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=MUTED,
            zorder=1,
        )
    )


def exclusion_box(
    axis: plt.Axes,
    center_x: float,
    heading: str,
    detail: str,
) -> None:
    rounded_box(
        axis,
        center_x=center_x,
        center_y=0.36,
        width=0.195,
        height=0.125,
        facecolor=GRAY_FILL,
        edgecolor=GRAY_EDGE,
        linewidth=0.7,
        radius=0.014,
    )
    axis.plot(
        [center_x, center_x],
        [0.605, 0.435],
        color=GRAY_EDGE,
        linewidth=0.7,
        linestyle=(0, (2.0, 2.0)),
        zorder=1,
    )
    axis.text(
        center_x,
        0.383,
        heading,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        center_x,
        0.338,
        detail,
        ha="center",
        va="center",
        fontsize=6,
        color=MUTED,
        linespacing=1.15,
    )


def sex_box(
    axis: plt.Axes,
    center_x: float,
    label: str,
    count: str,
    color: str,
) -> None:
    rounded_box(
        axis,
        center_x=center_x,
        center_y=0.145,
        width=0.105,
        height=0.105,
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        radius=0.014,
    )
    axis.text(
        center_x,
        0.164,
        label,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=color,
    )
    axis.text(
        center_x,
        0.125,
        count,
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
    )


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(180 / 25.4, 100 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    centers = (0.115, 0.365, 0.615, 0.865)
    stage_box(
        axis,
        centers[0],
        "Voice recordings\ncollected",
        "10,434 recordings\n8,800 participants",
    )
    stage_box(
        axis,
        centers[1],
        "Quality control\npassed",
        "7,784 recordings\n7,081 participants",
    )
    stage_box(
        axis,
        centers[2],
        "Latest clinic visit\nretained",
        "7,081 recordings\n7,081 participants",
    )
    stage_box(
        axis,
        centers[3],
        "Final analytic\ncohort",
        "6,979 participants\nages 40–70 years",
        final=True,
    )

    box_half_width = 0.205 / 2
    for left, right in zip(centers[:-1], centers[1:]):
        horizontal_arrow(
            axis,
            left + box_half_width + 0.008,
            right - box_half_width - 0.008,
        )

    exclusion_box(
        axis,
        0.240,
        "Excluded",
        "2,650 technically\nfaulty recordings",
    )
    exclusion_box(
        axis,
        0.490,
        "Earlier visits removed",
        "703 repeated\nrecordings",
    )
    exclusion_box(
        axis,
        0.740,
        "Outside age window",
        "102 participants",
    )

    branch_y = 0.225
    axis.plot(
        [centers[3], centers[3]],
        [0.535, branch_y],
        color=MUTED,
        linewidth=0.8,
        zorder=1,
    )
    axis.plot(
        [0.795, 0.935],
        [branch_y, branch_y],
        color=MUTED,
        linewidth=0.8,
        zorder=1,
    )
    for x_value in (0.795, 0.935):
        axis.add_patch(
            FancyArrowPatch(
                (x_value, branch_y),
                (x_value, 0.201),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=MUTED,
                zorder=1,
            )
        )
    sex_box(axis, 0.795, "Females", "n = 3,631", FEMALE)
    sex_box(axis, 0.935, "Males", "n = 3,348", MALE)

    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.97)
    return figure


def main() -> None:
    figure = make_figure()
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    destinations = (FINAL_FIGURES / "supp_fig_S6_participant_flowchart",)
    for stem in destinations:
        figure.savefig(stem.with_suffix(".png"), dpi=600)
        figure.savefig(stem.with_suffix(".pdf"))
        print(f"Saved: {stem}.png/.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
