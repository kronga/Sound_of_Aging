#!/usr/bin/env python3
"""
Recreate highlighted risk-factor boxplots using age-residualized voice groups.

This uses the updated combined_risk_factors.csv table and the same averaged
prediction residualization logic used by the residualized lollipop analysis.
Outputs are written to a tagged directory when VOICE_RESID_OUT_TAG is set.
"""

import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "5_downstream_analysis"))
import volcano_visualization as vv

from phenotype_enrichment import VAT_AREA_COLUMN, add_vat_area

OUT_TAG = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()


def tagged_name(name: str) -> str:
    return f"{name}_{OUT_TAG}" if OUT_TAG else name


OUTDIR = os.path.join(
    "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step5_volcano",
    tagged_name("voice_residualized"),
    "boxplots",
)
AVERAGED_PRED_BASE = (
    "/home/davidkro/PycharmProjects/DeepVoice/"
    f"analysis_outputs/{STEP3_DIR}/"
    "gender_{gender}/predictions_averaged.csv"
)
SUBJECT_DETAILS_CSV = (
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/subject_details_df_Oct25.csv"
)
GENDERS = ["female", "male"]
MIN_AGE, MAX_AGE = 40, 70
PERCENTILE = 0.25


@dataclass(frozen=True)
class PlotSpec:
    column_name: str
    output_stub: str
    ylabel: str
    threshold: float | dict[str, float] | None = None
    scale: float = 1.0

    def threshold_for(self, gender: str) -> float | None:
        if isinstance(self.threshold, dict):
            return self.threshold.get(gender)
        return self.threshold


PLOT_SPECS = [
    # Adiposity
    PlotSpec("BMI", "bmi", "BMI (kg/m²)", 30),
    PlotSpec("Waist circumference", "waist", "Waist circumference (cm)",
             {"female": 88, "male": 102}),
    PlotSpec(VAT_AREA_COLUMN, "vat_area", "VAT area (cm²)", None),
    # Sleep
    PlotSpec("AHI (SM)", "ahi", "AHI (events/hour)", 15),
    PlotSpec("Snore DB", "snore_db", "Snore Intensity (dB)", None),
    PlotSpec("Mean Oxygen Saturation (SM)", "mean_o2_sat", "Mean O₂ Saturation (%)", 95),
    # Cardiometabolic
    PlotSpec("HbA1C (BT)", "hba1c", "HbA1c (%)", 5.7),
    PlotSpec("Sitting BP diastolic", "sitting_bp_diastolic", "Sitting BP Diastolic (mmHg)", 80),
    PlotSpec("Liver viscosity (US)", "liver_viscosity", "Liver viscosity (kPa)", 2.0),
    PlotSpec("Carotid - intima media thickness (US)", "carotid_imt", "Carotid IMT (mm)", 0.9),
    # Functional capacity (higher in younger-sounding)
    PlotSpec("Hand Grip Left", "hand_grip_left", "Hand Grip Left (kg)", None, scale=1/2.205),
    PlotSpec("Hand Grip Right", "hand_grip_right", "Hand Grip Right (kg)", None, scale=1/2.205),
    PlotSpec("Albumin (BT)", "albumin", "Albumin (g/dL)", 3.5),
]


def _stage_rank(stage: str) -> tuple[int, str]:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def load_rf() -> pd.DataFrame:
    rf = add_vat_area(pd.read_csv(vv.COMBINED_RISK_FACTORS_PATH))
    if "subject_number" not in rf.columns and "subject_id" in rf.columns:
        rf = rf.rename(columns={"subject_id": "subject_number"})
    rf = rf.copy()
    rf.index = (
        rf["subject_number"].astype(int).astype(str)
        + "_"
        + rf["research_stage"].astype(str)
    )
    rf = rf[~rf.index.duplicated(keep="first")]
    return rf


def load_residualized_groups(gender: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    pred = pd.read_csv(AVERAGED_PRED_BASE.format(gender=gender))
    pred = pred.rename(columns={"group": "subject_number", "mean_predictions": "predictions"})
    if "research_stage" not in pred.columns:
        sd = pd.read_csv(
            SUBJECT_DETAILS_CSV,
            usecols=["filename", "visit_number"],
        ).rename(columns={"visit_number": "research_stage"})
        pred = pred.merge(sd, left_on="index", right_on="filename", how="left")
    pred = pred.dropna(subset=["true_values", "predictions", "subject_number", "research_stage"]).copy()
    pred["_stage_rank"] = pred["research_stage"].map(_stage_rank)
    pred = pred.sort_values(["subject_number", "_stage_rank", "index"]).drop_duplicates(
        subset=["subject_number"], keep="last"
    )
    pred["delta_va"] = pred["predictions"] - pred["true_values"]
    slope, intercept, *_ = linregress(pred["true_values"], pred["delta_va"])
    pred["delta_va_resid"] = pred["delta_va"] - (slope * pred["true_values"] + intercept)
    pred = pred[(pred["true_values"] >= MIN_AGE) & (pred["true_values"] <= MAX_AGE)].copy()
    pred["_key"] = (
        pred["subject_number"].astype(int).astype(str)
        + "_"
        + pred["research_stage"].astype(str)
    )
    lo = pred["delta_va_resid"].quantile(PERCENTILE)
    hi = pred["delta_va_resid"].quantile(1 - PERCENTILE)
    bottom_keys = pred.loc[pred["delta_va_resid"] <= lo, "_key"].tolist()
    top_keys = pred.loc[pred["delta_va_resid"] >= hi, "_key"].tolist()
    return pred, bottom_keys, top_keys


def create_biomarker_comparison_plot(
    table: pd.DataFrame,
    top_keys: list[str],
    bottom_keys: list[str],
    spec: PlotSpec,
    gender: str,
) -> dict:
    top_values = table.loc[[k for k in top_keys if k in table.index], spec.column_name].dropna() * spec.scale
    bottom_values = table.loc[[k for k in bottom_keys if k in table.index], spec.column_name].dropna() * spec.scale

    statistic, pvalue = mannwhitneyu(top_values, bottom_values, alternative="two-sided")
    if pvalue < 0.001:
        sig_text = "***"
    elif pvalue < 0.01:
        sig_text = "**"
    elif pvalue < 0.1:
        sig_text = "*"
    else:
        sig_text = "ns"

    plot_data = pd.DataFrame({
        spec.column_name: pd.concat([bottom_values, top_values]),
        "Group": ["Bottom 25%"] * len(bottom_values) + ["Top 25%"] * len(top_values),
    })

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(60 / 25.4, 80 / 25.4))
    sns.boxplot(
        data=plot_data,
        x="Group",
        y=spec.column_name,
        hue="Group",
        palette={"Bottom 25%": "lightgreen", "Top 25%": "salmon"},
        legend=False,
        ax=ax,
        width=0.5,
        medianprops=dict(color="black", linewidth=1.2),
        showfliers=False,
    )
    sns.stripplot(
        data=plot_data,
        x="Group",
        y=spec.column_name,
        color="black",
        alpha=0.22,
        size=0.8,
        jitter=0.22,
        ax=ax,
    )

    y_max = plot_data[spec.column_name].max()
    y_min = plot_data[spec.column_name].min()
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_sig = y_max + 0.05 * y_range
    h = 0.02 * y_range
    ax.plot([0, 0, 1, 1], [y_sig, y_sig + h, y_sig + h, y_sig], lw=1.5, c="black")
    ax.text(
        0.5,
        y_sig + h,
        sig_text,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

    threshold = spec.threshold_for(gender)
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="darkblue",
            linestyle="--",
            linewidth=1.0,
            zorder=10,
        )

    ax.set_ylabel(spec.ylabel, fontsize=8)
    ax.set_xlabel("")
    ax.set_title("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Bottom 25%\n(n={len(bottom_values)})",
            f"Top 25%\n(n={len(top_values)})",
        ],
        fontsize=7,
    )
    ax.tick_params(axis="y", labelsize=7)

    os.makedirs(OUTDIR, exist_ok=True)
    out_prefix = os.path.join(OUTDIR, f"{spec.output_stub}_{gender}_comparison_boxplot")
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=600)
    plt.savefig(out_prefix + ".pdf")
    plt.close(fig)

    return {
        "gender": gender,
        "feature": spec.column_name,
        "top_n": len(top_values),
        "bottom_n": len(bottom_values),
        "top_mean": float(top_values.mean()),
        "bottom_mean": float(bottom_values.mean()),
        "p_value": float(pvalue),
        "significance": sig_text,
        "png": out_prefix + ".png",
        "pdf": out_prefix + ".pdf",
    }


def main() -> None:
    rf = load_rf()
    results = []
    for gender in GENDERS:
        gender_val = 1 if gender == "male" else 0
        rf_gender = rf[rf["gender"] == gender_val].copy()
        _, bottom_keys, top_keys = load_residualized_groups(gender)
        for spec in PLOT_SPECS:
            if spec.column_name not in rf_gender.columns:
                print(f"[SKIP] {gender}: missing column {spec.column_name}")
                continue
            res = create_biomarker_comparison_plot(rf_gender, top_keys, bottom_keys, spec, gender)
            results.append(res)
            print(
                f"{gender} | {spec.column_name}: bottom={res['bottom_n']} top={res['top_n']} "
                f"p={res['p_value']:.4g} -> {res['png']}"
            )

    if results:
        out_csv = os.path.join(OUTDIR, "boxplot_summary.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved summary → {out_csv}")


if __name__ == "__main__":
    main()
