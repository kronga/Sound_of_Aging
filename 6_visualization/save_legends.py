#!/usr/bin/env python3
"""Save standalone legend figures for the lollipop and boxplot panel."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
def tagged_name(name):
    return f"{name}_{OUT_TAG}" if OUT_TAG else name

OUTDIR = os.path.join(
    "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step5_volcano",
    tagged_name("voice_residualized"),
)
FONT = 8


def save_legend(handles, out_stem, ncol=1):
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.set_visible(False)
    legend = fig.legend(handles=handles, fontsize=FONT, frameon=True,
                        loc="center", ncol=ncol)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches=bbox)
    fig.savefig(out_stem + ".pdf", bbox_inches=bbox)
    plt.close(fig)
    print(f"Saved → {out_stem}.png/.pdf")


# Lollipop legend
lollipop_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF5252',
           markeredgecolor='black', markersize=5, label='Higher in old-predicted (sig.)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
           markeredgecolor='black', markersize=5, label='Lower in old-predicted (sig.)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markersize=4, label='Not significant'),
]
save_legend(lollipop_handles, os.path.join(OUTDIR, "legend_lollipop"))

# Boxplot legend
boxplot_handles = [
    mpatches.Patch(facecolor="#90EE90", edgecolor="grey", label="Bottom 25%"),
    mpatches.Patch(facecolor="#FA8072", edgecolor="grey", label="Top 25%"),
]
save_legend(boxplot_handles, os.path.join(OUTDIR, "legend_boxplots"))
