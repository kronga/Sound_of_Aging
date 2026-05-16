#!/usr/bin/env python3
"""
Supplementary Fig S4 — Participant flowchart.
Shows filtration from raw collection → QC → one-per-subject → final sex split.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "paper_revision_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
BOX_MAIN   = "#2C6FAC"   # dark blue — main pipeline boxes
BOX_EXCL   = "#B0B0B0"   # grey — excluded
BOX_FEMALE = "#C2527A"   # rose — female
BOX_MALE   = "#3A7EBF"   # medium blue — male
TEXT_WHITE = "white"
TEXT_DARK  = "#1A1A1A"

def rounded_box(ax, x, y, w, h, text, bg, tc=TEXT_WHITE, fontsize=10.5):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=bg, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight="normal",
            linespacing=1.5, zorder=4)

def arrow(ax, x, y_top, y_bot, color="#555555"):
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.8, mutation_scale=14),
        zorder=2,
    )

def excl_box(ax, x_main, y_mid, text, h=0.10):
    ax.annotate(
        "", xy=(x_main + 0.18, y_mid), xytext=(x_main + 0.05, y_mid),
        arrowprops=dict(arrowstyle="-|>", color=BOX_EXCL,
                        lw=1.5, mutation_scale=12),
        zorder=2,
    )
    rounded_box(ax, x_main + 0.36, y_mid, 0.34, h,
                text, BOX_EXCL, tc=TEXT_DARK, fontsize=9)

# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

cx = 0.42   # main column x-centre

# ── boxes (top → bottom) ──────────────────────────────────────────────────────

# Box 1 — collected
rounded_box(ax, cx, 0.88, 0.60, 0.11,
            "All voice recordings collected\n"
            "10,434 recordings  ·  8,800 participants",
            BOX_MAIN, fontsize=10.5)

arrow(ax, cx, 0.825, 0.745)

# Box 2 — after QC
rounded_box(ax, cx, 0.71, 0.60, 0.12,
            "After quality control\n"
            "(RF classifier; faulty recordings removed)\n"
            "7,784 recordings  ·  7,081 unique participants",
            BOX_MAIN, fontsize=10)
excl_box(ax, cx + 0.30, 0.795,
         "Excluded: 2,650\nrecordings")

arrow(ax, cx, 0.650, 0.560)

# Box 3 — one per subject (most recent visit)
rounded_box(ax, cx, 0.52, 0.60, 0.12,
            "Most recent clinic visit retained per participant\n"
            "(repeated assessments, not duplicates;\n"
            "earlier visits discarded)\n"
            "6,979 participants  ·  age 40–70 years",
            BOX_MAIN, fontsize=10)
excl_box(ax, cx + 0.30, 0.52,
         "Discarded: 703 recordings\n(earlier clinic visits)\n+ 102 outside age range",
         h=0.13)

arrow(ax, cx, 0.460, 0.375)

# Box 4 — final dataset
rounded_box(ax, cx, 0.34, 0.60, 0.10,
            "Final analytic cohort\n6,979 participants",
            "#1A4F80", fontsize=11, tc=TEXT_WHITE)

# Split arrows to female / male
ax.annotate("", xy=(0.21, 0.225), xytext=(cx - 0.05, 0.295),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5, mutation_scale=13), zorder=2)
ax.annotate("", xy=(0.63, 0.225), xytext=(cx + 0.05, 0.295),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5, mutation_scale=13), zorder=2)

# Sex boxes
rounded_box(ax, 0.21, 0.18, 0.30, 0.10,
            "Females\nn = 3,631",
            BOX_FEMALE, fontsize=12)
rounded_box(ax, 0.63, 0.18, 0.30, 0.10,
            "Males\nn = 3,348",
            BOX_MALE, fontsize=12)

# ── title ──────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97,
        "Supplementary Fig. S4 — Participant flow diagram",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TEXT_DARK)

plt.tight_layout(pad=0.3)

for ext in ("png", "pdf"):
    out = OUTPUT_DIR / f"supp_fig_S4_participant_flowchart.{ext}"
    dpi = 300 if ext == "png" else None
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {out}")

plt.close()
