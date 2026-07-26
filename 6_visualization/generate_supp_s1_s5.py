"""
Generate the engineered-feature comparison (Supplementary Figure 3) and
partition-stability plot (Supplementary Figure 1).

S3: Grouped barplot comparing age prediction performance (R², MAE) across
    acoustic feature sets × model type (Ridge and LightGBM) for each sex.

S1: Partition-stability dot plot for the final WavLM Ridge model.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs"
OUT_DIR = BASE

SEEDS = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_17",
         "seed_42", "seed_99", "seed_123", "seed_256", "seed_512"]

SEXES = ["female", "male"]
FEATURES = ["egemaps", "emobase", "praat", "compare2016"]
FEATURE_LABELS = {
    "wavlm": "WavLM-Large",
    "egemaps": "eGeMAPS",
    "emobase": "emobase",
    "praat": "Praat",
    "compare2016": "ComParE 2016",
}

# Features for which only LightGBM results are meaningful
# (Ridge degrades with high-dimensional sparse feature sets)
LGBM_ONLY_FEATURES = {"compare2016"}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json_safe(path):
    """Return parsed JSON or None if file missing."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def collect_ridge_wavlm(sex):
    """
    Collect per-seed R² and MAE for WavLM Ridge (filtered latest-subject).
    Returns (list_r2, list_mae).
    """
    r2_vals, mae_vals = [], []
    base = os.path.join(BASE, "step3_voice_age_ridge_filtered", f"gender_{sex}")
    for seed in SEEDS:
        path = os.path.join(base, seed, "metrics.json")
        d = load_json_safe(path)
        if d is not None:
            r2_vals.append(d["oof_R2"])
            mae_vals.append(d["oof_MAE"])
    return r2_vals, mae_vals


def collect_ridge_classical(feature, sex):
    """
    Collect per-seed R² and MAE for a classical feature with Ridge.
    Returns (list_r2, list_mae).
    """
    r2_vals, mae_vals = [], []
    base = os.path.join(BASE, "step4_classical_ridge", feature, f"gender_{sex}")
    for seed in SEEDS:
        path = os.path.join(base, seed, "metrics.json")
        d = load_json_safe(path)
        if d is not None:
            r2_vals.append(d["oof_R2"])
            mae_vals.append(d["oof_MAE"])
    return r2_vals, mae_vals


def collect_lgbm_wavlm(sex):
    """
    Collect R² and MAE for WavLM LightGBM.
    Prefers fold-level JSONs; falls back to metrics_{sex}_seed42.json.
    Returns (list_r2, list_mae) — may be single-element list if only seed42.
    """
    wavlm_dir = os.path.join(BASE, "step4_classical_boosting", "lgbm", "wavlm")
    fold_files = [f for f in os.listdir(wavlm_dir)
                  if f.startswith(f"fold_{sex}_") and f.endswith(".json")]
    if fold_files:
        r2_vals, mae_vals = [], []
        for fname in sorted(fold_files):
            d = load_json_safe(os.path.join(wavlm_dir, fname))
            if d is not None:
                r2_vals.append(d["R2"])
                mae_vals.append(d["MAE"])
        return r2_vals, mae_vals
    # fallback
    path = os.path.join(wavlm_dir, f"metrics_{sex}_seed42.json")
    d = load_json_safe(path)
    if d is not None:
        return [d["R2"]], [d["MAE"]]
    return [], []


def collect_lgbm_classical(feature, sex):
    """
    Collect per-seed R² and MAE for classical feature with LightGBM.
    Reads seed_{s}/metrics_by_gender.json → {sex}.oof_R2 / {sex}.oof_MAE
    Returns (list_r2, list_mae).
    """
    r2_vals, mae_vals = [], []
    base = os.path.join(BASE, "step4_classical_lgbm", f"lgbm_{feature}_age")
    for seed in SEEDS:
        path = os.path.join(base, seed, "metrics_by_gender.json")
        d = load_json_safe(path)
        if d is not None and sex in d:
            r2_vals.append(d[sex]["oof_R2"])
            mae_vals.append(d[sex]["oof_MAE"])
    return r2_vals, mae_vals


# ---------------------------------------------------------------------------
# Collect all data
# ---------------------------------------------------------------------------

def collect_all_data():
    """
    Returns nested dict:
        data[sex][feature][model] = {"r2": [...], "mae": [...]}
    """
    data = {sex: {} for sex in SEXES}
    missing_report = []

    for sex in SEXES:
        # WavLM Ridge
        r2, mae = collect_ridge_wavlm(sex)
        if not r2:
            missing_report.append(f"Ridge WavLM {sex}: NO DATA")
        data[sex]["wavlm"] = {"Ridge": {"r2": r2, "mae": mae}}

        # WavLM LightGBM
        r2, mae = collect_lgbm_wavlm(sex)
        if not r2:
            missing_report.append(f"LightGBM WavLM {sex}: NO DATA")
        data[sex]["wavlm"]["LightGBM"] = {"r2": r2, "mae": mae}

        for feat in FEATURES:
            data[sex][feat] = {}
            # Ridge classical — skip for features where Ridge is not meaningful
            if feat in LGBM_ONLY_FEATURES:
                data[sex][feat]["Ridge"] = {"r2": [], "mae": []}
            else:
                r2, mae = collect_ridge_classical(feat, sex)
                if not r2:
                    missing_report.append(f"Ridge {feat} {sex}: NO DATA")
                data[sex][feat]["Ridge"] = {"r2": r2, "mae": mae}
            # LightGBM classical
            r2, mae = collect_lgbm_classical(feat, sex)
            if not r2:
                missing_report.append(f"LightGBM {feat} {sex}: NO DATA")
            data[sex][feat]["LightGBM"] = {"r2": r2, "mae": mae}

    return data, missing_report


# ---------------------------------------------------------------------------
# Supplementary Figure 3
# ---------------------------------------------------------------------------

def plot_s3(data):
    feature_order = ["wavlm", "egemaps", "emobase", "praat", "compare2016"]
    model_order = ["Ridge", "LightGBM"]
    model_colors = {"Ridge": "#4682B4", "LightGBM": "#FF7F50"}

    sex_titles = {"female": "Female", "male": "Male"}
    metric_ylabels = {"r2": "Out-of-fold R²", "mae": "MAE (years)"}
    metric_ylims = {"r2": (-0.10, 0.65), "mae": (3.5, 7.0)}
    metric_keys = ["r2", "mae"]

    n_feats = len(feature_order)
    n_models = len(model_order)
    bar_width = 0.35
    group_gap = 0.1
    group_width = n_models * bar_width + group_gap
    x_positions = np.arange(n_feats) * group_width

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plt.style.use("seaborn-v0_8-whitegrid")

    summary_rows = []  # for printing

    for row_idx, metric in enumerate(metric_keys):
        for col_idx, sex in enumerate(SEXES):
            ax = axes[row_idx][col_idx]

            for feat_idx, feat in enumerate(feature_order):
                for mod_idx, model in enumerate(model_order):
                    vals = data[sex][feat][model][metric]
                    if not vals:
                        continue
                    mean_val = np.mean(vals)
                    std_val = np.std(vals) if len(vals) > 1 else 0.0

                    x = x_positions[feat_idx] + mod_idx * bar_width

                    # Color logic: red + hatch for negative R²
                    if metric == "r2" and mean_val < 0:
                        color = "#CC0000"
                        hatch = "//"
                    else:
                        color = model_colors[model]
                        hatch = None

                    bar = ax.bar(
                        x, mean_val, bar_width * 0.9,
                        color=color,
                        hatch=hatch,
                        yerr=std_val if std_val > 0 else None,
                        capsize=4,
                        error_kw={"elinewidth": 1.2, "ecolor": "black"},
                        label=model if feat_idx == 0 else "_nolegend_",
                        zorder=3,
                    )

                    # Value label
                    label_y = mean_val + (std_val if std_val > 0 else 0) + (
                        0.005 if metric == "r2" else 0.03
                    )
                    fmt = f"{mean_val:.2f}" if metric == "r2" else f"{mean_val:.1f}"
                    ax.text(
                        x + bar_width * 0.45, label_y, fmt,
                        ha="center", va="bottom", fontsize=7.5, rotation=0,
                    )

                    summary_rows.append({
                        "sex": sex, "feature": FEATURE_LABELS.get(feat, feat),
                        "model": model, "metric": metric,
                        "mean": mean_val, "std": std_val, "n_seeds": len(vals),
                    })

            # Dashed line at R²=0
            if metric == "r2":
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

            # x ticks at group centers
            tick_centers = x_positions + bar_width * (n_models - 1) / 2
            ax.set_xticks(tick_centers)
            ax.set_xticklabels(
                [FEATURE_LABELS.get(f, f) for f in feature_order],
                fontsize=10,
            )
            ax.set_ylim(metric_ylims[metric])
            ax.set_ylabel(metric_ylabels[metric], fontsize=11)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

            if row_idx == 0:
                ax.set_title(sex_titles[sex], fontsize=13, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Feature Set", fontsize=11)

            # Legend only in top-left
            if row_idx == 0 and col_idx == 0:
                handles = [
                    mpatches.Patch(color=model_colors[m], label=m)
                    for m in model_order
                ]
                ax.legend(handles=handles, fontsize=10, loc="upper right")

    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "supp_fig_S3_feature_comparison.png")
    out_pdf = os.path.join(OUT_DIR, "supp_fig_S3_feature_comparison.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    return summary_rows


# ---------------------------------------------------------------------------
# Supplementary Figure 1
# ---------------------------------------------------------------------------

def plot_s1(data):
    """
    Four-panel partition-stability plot for WavLM Ridge.
    Panels: Female R², Female MAE, Male R², Male MAE
    """
    sex_labels = {"female": "Female", "male": "Male"}
    metrics = [
        ("r2", "Cross-validated R²"),
        ("mae", "MAE (years)"),
    ]
    # Suggested y-limits per panel
    ylims = {
        ("female", "r2"): (0.53, 0.58),
        ("female", "mae"): (3.85, 3.95),
        ("male", "r2"): (0.43, 0.46),
        ("male", "mae"): (4.35, 4.40),
    }

    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    plt.style.use("seaborn-v0_8-whitegrid")

    panel_idx = 0
    for sex in SEXES:
        for metric, ylabel in metrics:
            ax = axes[panel_idx]
            vals = np.array(data[sex]["wavlm"]["Ridge"][metric])

            if len(vals) == 0:
                ax.set_title(f"{sex_labels[sex]} {ylabel}\nNO DATA")
                panel_idx += 1
                continue

            mean_val = np.mean(vals)
            std_val = np.std(vals)

            # Adaptive y-limits: use suggested if values fit, else auto-expand
            key = (sex, metric)
            y_lo, y_hi = ylims.get(key, (None, None))
            margin = std_val * 2.5 + 0.005
            if y_lo is None or (min(vals) < y_lo + 0.005) or (max(vals) > y_hi - 0.005):
                y_lo = min(vals) - margin
                y_hi = max(vals) + margin

            # Shaded std band
            ax.axhspan(mean_val - std_val, mean_val + std_val,
                       color="#4682B4", alpha=0.15, zorder=1)
            # Mean line
            ax.axhline(mean_val, color="#4682B4", linewidth=1.5,
                       linestyle="-", zorder=2, label="Mean")

            # Jittered dots
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(jitter, vals, color="#4682B4", s=40, zorder=3,
                       edgecolors="white", linewidths=0.5)

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{sex_labels[sex]}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.4)

            # Annotate mean ± std
            annot_fmt = ".3f" if metric == "r2" else ".2f"
            ax.text(
                0.5, 0.05,
                f"Mean={mean_val:{annot_fmt}}\nStd={std_val:{annot_fmt}}",
                transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8),
            )

            # Small metric label inside
            metric_disp = "R²" if metric == "r2" else "MAE"
            ax.text(
                0.5, 0.95, metric_disp,
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
            )

            panel_idx += 1

    fig.suptitle(
        "Supplementary Figure 1 | Stability of Voice Age model across data partitions",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "supp_fig_S1_seed_stability.png")
    out_pdf = os.path.join(OUT_DIR, "supp_fig_S1_seed_stability.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(summary_rows):
    """Print a formatted table of mean ± std for R² and MAE."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE — S3 data (mean ± SD across partitions)")
    print("=" * 90)
    # Pivot: row = sex+feature+model, cols = R2, MAE
    from collections import defaultdict
    pivot = defaultdict(dict)
    for row in summary_rows:
        key = (row["sex"], row["feature"], row["model"])
        pivot[key][row["metric"]] = (row["mean"], row["std"], row["n_seeds"])

    header = f"{'Sex':<8}{'Feature':<16}{'Model':<12}{'R² (mean±SD)':<22}{'MAE (mean±SD)':<22}{'Partitions'}"
    print(header)
    print("-" * 90)
    feat_order_labels = ["WavLM-Large", "eGeMAPS", "emobase", "Praat", "ComParE 2016"]
    for sex in SEXES:
        for feat_label in feat_order_labels:
            for model in ["Ridge", "LightGBM"]:
                key = (sex, feat_label, model)
                if key not in pivot:
                    continue
                r2_data = pivot[key].get("r2", (None, None, 0))
                mae_data = pivot[key].get("mae", (None, None, 0))
                r2_str = (f"{r2_data[0]:.4f}±{r2_data[1]:.4f}"
                          if r2_data[0] is not None else "N/A")
                mae_str = (f"{mae_data[0]:.4f}±{mae_data[1]:.4f}"
                           if mae_data[0] is not None else "N/A")
                n = r2_data[2]
                print(f"{sex:<8}{feat_label:<16}{model:<12}{r2_str:<22}{mae_str:<22}{n}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Collecting data...")
    data, missing = collect_all_data()

    if missing:
        print("\nMISSING / unavailable data:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All expected data found.")

    print("\nGenerating Supplementary Figure 3...")
    summary_rows = plot_s3(data)

    print("\nGenerating Supplementary Figure 1...")
    plot_s1(data)

    print_summary_table(summary_rows)
    print("\nDone.")
