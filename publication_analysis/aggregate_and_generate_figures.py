#!/usr/bin/env python3
"""Aggregate repeated analyses and generate synchronized publication figures."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
RESULTS = ROOT / "analysis_outputs" / "repeated_analysis" / "final"
SUMMARY_DIR = RESULTS / "summary"
FINAL_FIGURES = ROOT / "voice_age_manuscript" / "final_figs"
FINAL_FINAL = FINAL_FIGURES / "final_final"
SUBMISSION_FIGURES = ROOT / "voice_age_manuscript" / "submission" / "Figures"
OLD_LGBM = ROOT / "analysis_outputs" / "step4_multimodality_lgbm_hpo"
OLD_RIDGE = ROOT / "analysis_outputs" / "step4_multimodality_ridge"
CONDITIONED_SUMMARY = (
    ROOT
    / "analysis_outputs"
    / "step4_voice_conditioned_holdout_fullpool_oof_cal"
    / "voice_conditioned_holdout_oof_fullpool_cal_summary.csv"
)
CONDITIONED_ROOT = CONDITIONED_SUMMARY.parent

SEEDS = (42, 1, 2, 3, 4, 17, 99, 123, 256, 512)
SEXES = ("female", "male")
REPRESENTATIONS = ("wavlm", "egemaps", "emobase", "praat", "compare2016")
METRICS = ("Pearson_r", "R2", "MAE", "RMSE")
REPRESENTATION_LABELS = {
    "wavlm": "WavLM-Large",
    "egemaps": "eGeMAPS",
    "emobase": "emobase",
    "praat": "Praat",
    "compare2016": "ComParE 2016",
}
CLOCKS = (
    "metabolomics",
    "voice",
    "lifestyle",
    "sleep",
    "DEXA",
    "diet",
    "NMR",
    "retina",
    "microbiome",
)
CLOCK_LABELS = {
    "metabolomics": "MS metabolomics",
    "voice": "Voice",
    "lifestyle": "Lifestyle",
    "sleep": "Sleep",
    "DEXA": "DXA",
    "diet": "Diet",
    "NMR": "NMR metabolomics",
    "retina": "Retina",
    "microbiome": "Microbiome",
}
SELECTED_MODELS = {
    "metabolomics": "lgbm",
    "lifestyle": "lgbm",
    "sleep": "lgbm",
    "DEXA": "lgbm",
    "diet": "lgbm",
    "NMR": "ridge",
    "retina": "lgbm",
    "microbiome": "lgbm",
}
FEMALE_COLOR = "#D97777"
MALE_COLOR = "#6B9DC1"
OLDER_COLOR = "#D95F70"
YOUNGER_COLOR = "#4E9F8A"
GRAY = "#888888"


def configure_style() -> None:
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
    sns.set_style("whitegrid")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_file_rows(paths: list[Path], context: dict[str, object]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = load_json(path)
        overall = payload.get("overall", payload)
        seed = int(path.parent.name.split("_")[-1])
        rows.append(
            {
                **context,
                "seed": seed,
                "Pearson_r": overall.get(
                    "Pearson_r", overall.get("oof_Pearson_r")
                ),
                "R2": overall.get("R2", overall.get("oof_R2")),
                "MAE": overall.get("MAE", overall.get("oof_MAE")),
                "RMSE": overall.get("RMSE", overall.get("oof_RMSE")),
                "n": payload.get(
                    "n_rows",
                    payload.get("n_samples", overall.get("n_samples")),
                ),
                "n_participants": payload.get("n_participants", np.nan),
            }
        )
    return rows


def existing_output_metric_rows(
    base: Path,
    context: dict[str, object],
) -> list[dict]:
    """Load either seed directories or the compact LightGBM seed table."""
    paths = [base / f"seed_{seed}" / "metrics.json" for seed in SEEDS]
    if all(path.exists() for path in paths):
        return metric_file_rows(paths, context)

    seed_table_path = base / "metrics_per_seed.csv"
    averaged_path = base / "metrics_averaged.json"
    if not seed_table_path.exists() or not averaged_path.exists():
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(
            "\n".join([*missing, str(seed_table_path), str(averaged_path)])
        )
    seed_table = pd.read_csv(seed_table_path)
    averaged = load_json(averaged_path)
    expected_columns = {"seed", "Pearson_r", "R2", "MAE", "RMSE"}
    if not expected_columns.issubset(seed_table.columns):
        raise RuntimeError(
            f"Unexpected metric columns in {seed_table_path}: "
            f"{seed_table.columns.tolist()}"
        )
    return [
        {
            **context,
            "seed": int(row["seed"]),
            "Pearson_r": float(row["Pearson_r"]),
            "R2": float(row["R2"]),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "n": int(averaged["n_samples"]),
            "n_participants": np.nan,
        }
        for _, row in seed_table.iterrows()
    ]


def ensure_ten_seeds(frame: pd.DataFrame, group_cols: list[str]) -> None:
    counts = frame.groupby(group_cols)["seed"].nunique()
    incomplete = counts[counts != len(SEEDS)]
    if not incomplete.empty:
        raise RuntimeError(f"Incomplete ten-seed outputs:\n{incomplete}")


def summarize_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
    aggregations["n_seeds"] = ("seed", "nunique")
    aggregations["n"] = ("n", "max")
    if "n_participants" in frame:
        aggregations["n_participants"] = ("n_participants", "max")
    return (
        frame.groupby(group_cols, as_index=False)
        .agg(**aggregations)
        .sort_values(group_cols)
    )


def aggregate_predictions(
    seed_dirs: list[Path],
    destination: Path,
    primary_voice: bool,
) -> pd.DataFrame:
    frames = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        frame = pd.read_csv(seed_dir / "predictions.csv")
        frames.append((seed, frame))
    frames.sort(key=lambda pair: SEEDS.index(pair[0]))
    reference = frames[0][1].copy()
    metadata_columns = [
        column
        for column in reference.columns
        if column not in {"predictions", "fold"}
    ]
    prediction_columns: dict[str, np.ndarray] = {}
    for seed, frame in frames:
        if len(frame) != len(reference):
            raise RuntimeError(f"Prediction length mismatch for seed {seed}.")
        for column in metadata_columns:
            left = reference[column].astype(str).fillna("")
            right = frame[column].astype(str).fillna("")
            if not left.equals(right):
                raise RuntimeError(
                    f"Prediction metadata mismatch for seed {seed}, {column}."
                )
        prediction_columns[f"pred_seed_{seed}"] = frame["predictions"].to_numpy()

    output = reference.loc[:, metadata_columns].copy()
    prediction_frame = pd.DataFrame(prediction_columns)
    output["mean_predictions"] = prediction_frame.mean(axis=1)
    output["pred_std"] = prediction_frame.std(axis=1, ddof=1)
    output = pd.concat([output, prediction_frame], axis=1)
    if primary_voice:
        output = output.rename(
            columns={
                "filename": "index",
                "subject_number": "group",
            }
        )
    else:
        output = output.rename(columns={"subject_number": "group"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return output


def aggregate_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict] = []
    for sex in SEXES:
        base = RESULTS / "primary_voice" / f"gender_{sex}"
        paths = [base / f"seed_{seed}" / "metrics.json" for seed in SEEDS]
        if not all(path.exists() for path in paths):
            missing = [str(path) for path in paths if not path.exists()]
            raise FileNotFoundError("\n".join(missing))
        primary_rows.extend(metric_file_rows(paths, {"sex": sex}))
        aggregate_predictions(
            [base / f"seed_{seed}" for seed in SEEDS],
            base / "predictions_averaged.csv",
            primary_voice=True,
        )
    primary = pd.DataFrame(primary_rows)
    ensure_ten_seeds(primary, ["sex"])
    primary.to_csv(SUMMARY_DIR / "primary_voice_metrics_by_seed.csv", index=False)
    primary_summary = summarize_metrics(primary, ["sex"])
    primary_summary.to_csv(
        SUMMARY_DIR / "primary_voice_metrics_summary.csv", index=False
    )

    acoustic_rows: list[dict] = []
    # Reuse the identical WavLM-Ridge primary run.
    for _, row in primary.iterrows():
        acoustic_rows.append(
            {
                "representation": "wavlm",
                "model": "ridge",
                **row.to_dict(),
            }
        )
    for representation in REPRESENTATIONS:
        models = ("lgbm",) if representation == "wavlm" else ("ridge", "lgbm")
        for model in models:
            for sex in SEXES:
                base = (
                    RESULTS
                    / "acoustic_benchmark"
                    / representation
                    / model
                    / f"gender_{sex}"
                )
                paths = [
                    base / f"seed_{seed}" / "metrics.json" for seed in SEEDS
                ]
                if not all(path.exists() for path in paths):
                    missing = [str(path) for path in paths if not path.exists()]
                    raise FileNotFoundError("\n".join(missing))
                acoustic_rows.extend(
                    metric_file_rows(
                        paths,
                        {
                            "representation": representation,
                            "model": model,
                            "sex": sex,
                        },
                    )
                )
    acoustic = pd.DataFrame(acoustic_rows)
    ensure_ten_seeds(acoustic, ["representation", "model", "sex"])
    acoustic.to_csv(
        SUMMARY_DIR / "acoustic_benchmark_metrics_by_seed.csv", index=False
    )
    acoustic_summary = summarize_metrics(
        acoustic, ["representation", "model", "sex"]
    )
    acoustic_summary.to_csv(
        SUMMARY_DIR / "acoustic_benchmark_metrics_summary.csv", index=False
    )

    lifestyle_rows: list[dict] = []
    for model in ("ridge", "lgbm"):
        for sex in SEXES:
            base = (
                RESULTS
                / "comparison_clocks"
                / "lifestyle"
                / model
                / f"gender_{sex}"
            )
            paths = [base / f"seed_{seed}" / "metrics.json" for seed in SEEDS]
            if not all(path.exists() for path in paths):
                missing = [str(path) for path in paths if not path.exists()]
                raise FileNotFoundError("\n".join(missing))
            lifestyle_rows.extend(
                metric_file_rows(
                    paths,
                    {
                        "clock": "lifestyle",
                        "model": model,
                        "sex": sex,
                    },
                )
            )
            aggregate_predictions(
                [base / f"seed_{seed}" for seed in SEEDS],
                base / "predictions_averaged.csv",
                primary_voice=False,
            )
    lifestyle = pd.DataFrame(lifestyle_rows)
    ensure_ten_seeds(lifestyle, ["clock", "model", "sex"])
    lifestyle.to_csv(
        SUMMARY_DIR / "lifestyle_metrics_by_seed.csv", index=False
    )
    lifestyle_summary = summarize_metrics(
        lifestyle, ["clock", "model", "sex"]
    )
    lifestyle_summary.to_csv(
        SUMMARY_DIR / "lifestyle_metrics_summary.csv", index=False
    )
    return primary_summary, acoustic_summary, lifestyle_summary


def collect_clock_model_results(
    lifestyle_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for clock in (
        "metabolomics",
        "sleep",
        "DEXA",
        "diet",
        "NMR",
        "retina",
        "microbiome",
    ):
        for model, root, prefix in (
            ("ridge", OLD_RIDGE, "ridge"),
            ("lgbm", OLD_LGBM, "lgbm"),
        ):
            for sex in SEXES:
                base = root / f"{prefix}_{clock}_age" / f"gender_{sex}"
                rows.extend(
                    existing_output_metric_rows(
                        base,
                        {"clock": clock, "model": model, "sex": sex},
                    )
                )
    clock_by_seed = pd.DataFrame(rows)
    ensure_ten_seeds(clock_by_seed, ["clock", "model", "sex"])
    clock_summary = summarize_metrics(
        clock_by_seed, ["clock", "model", "sex"]
    )
    clock_summary = pd.concat(
        [clock_summary, lifestyle_summary],
        ignore_index=True,
    )
    clock_summary.to_csv(
        SUMMARY_DIR / "comparison_clock_model_family_summary.csv",
        index=False,
    )
    clock_by_seed.to_csv(
        SUMMARY_DIR / "comparison_clock_model_family_by_seed.csv",
        index=False,
    )
    return clock_by_seed, clock_summary


def primary_prediction_path(sex: str) -> Path:
    return (
        RESULTS
        / "primary_voice"
        / f"gender_{sex}"
        / "predictions_averaged.csv"
    )


def clock_prediction_path(clock: str, sex: str) -> Path:
    model = SELECTED_MODELS[clock]
    if clock == "lifestyle":
        return (
            RESULTS
            / "comparison_clocks"
            / "lifestyle"
            / model
            / f"gender_{sex}"
            / "predictions_averaged.csv"
        )
    root = OLD_LGBM if model == "lgbm" else OLD_RIDGE
    return (
        root
        / f"{model}_{clock}_age"
        / f"gender_{sex}"
        / "predictions_averaged.csv"
    )


def prediction_stage_rank(value: object) -> tuple[int, str]:
    text = str(value)
    if text == "baseline":
        return (-1, text)
    try:
        return (int(text.split("_", 1)[0]), text)
    except (TypeError, ValueError):
        return (-2, text)


def keyed_predictions(path: Path, voice: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"group": "subject_number"})
    frame["subject_number"] = pd.to_numeric(
        frame["subject_number"], errors="coerce"
    )
    frame["true_values"] = pd.to_numeric(
        frame["true_values"], errors="coerce"
    )
    frame = frame.dropna(
        subset=["subject_number", "true_values", "mean_predictions"]
    ).copy()
    frame["subject_number"] = frame["subject_number"].astype(np.int64)
    # Prediction tables can contain repeated assessments for a participant.
    # Retain the most recent one within each clock before making a pairwise
    # participant join.  Most tables expose research_stage; NMR does not, so
    # chronological age provides the ordering fallback for that table.
    if "research_stage" in frame:
        frame["_assessment_rank"] = frame["research_stage"].map(
            prediction_stage_rank
        )
    else:
        frame["_assessment_rank"] = list(
            zip(frame["true_values"], frame.index.astype(str))
        )
    frame = (
        frame.sort_values(["subject_number", "_assessment_rank"])
        .drop_duplicates("subject_number", keep="last")
        .copy()
    )
    return frame.loc[
        :,
        ["subject_number", "true_values", "mean_predictions"],
    ]


def pairwise_voice_clock_correlations() -> pd.DataFrame:
    rows: list[dict] = []
    for sex in SEXES:
        voice = keyed_predictions(primary_prediction_path(sex), voice=True)
        voice = voice.rename(
            columns={
                "true_values": "voice_assessment_age",
                "mean_predictions": "voice_prediction",
            }
        )
        for clock in SELECTED_MODELS:
            other = keyed_predictions(clock_prediction_path(clock, sex))
            other = other.rename(
                columns={
                    "true_values": "clock_assessment_age",
                    "mean_predictions": "clock_prediction",
                }
            )
            merged = voice.merge(
                other,
                on="subject_number",
                how="inner",
                validate="one_to_one",
            )
            age_difference = (
                merged["voice_assessment_age"]
                - merged["clock_assessment_age"]
            ).abs()
            rows.append(
                {
                    "sex": sex,
                    "clock": clock,
                    "model": SELECTED_MODELS[clock],
                    "n": len(merged),
                    "median_absolute_age_difference": age_difference.median(),
                    "p95_absolute_age_difference": age_difference.quantile(0.95),
                    "Pearson_r": merged[
                        ["voice_prediction", "clock_prediction"]
                    ].corr().iloc[0, 1],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        SUMMARY_DIR / "pairwise_voice_clock_correlations.csv", index=False
    )
    return result


def selected_clock_summary(
    primary_summary: pd.DataFrame,
    clock_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sex in SEXES:
        voice = primary_summary.loc[primary_summary["sex"] == sex].iloc[0]
        rows.append(
            {
                "clock": "voice",
                "model": "ridge",
                "sex": sex,
                **{
                    column: voice[column]
                    for column in voice.index
                    if column.endswith("_mean")
                    or column.endswith("_sd")
                    or column in {"n", "n_seeds"}
                },
            }
        )
        for clock, model in SELECTED_MODELS.items():
            row = clock_summary.loc[
                (clock_summary["clock"] == clock)
                & (clock_summary["model"] == model)
                & (clock_summary["sex"] == sex)
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Expected one selected row for {clock}/{model}/{sex}."
                )
            rows.append(row.iloc[0].to_dict())
    selected = pd.DataFrame(rows)
    selected.to_csv(
        SUMMARY_DIR / "selected_comparison_clock_summary.csv", index=False
    )
    return selected


def save_figure(
    figure: plt.Figure,
    stem: str,
    destinations: list[tuple[Path, str]] | None = None,
    dpi: int = 600,
) -> None:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    png = FINAL_FIGURES / f"{stem}.png"
    pdf = FINAL_FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf)
    if destinations:
        for directory, name in destinations:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, directory / f"{name}.png")
            shutil.copy2(pdf, directory / f"{name}.pdf")
    print(f"Saved {png} and {pdf}")


def make_figure1(primary_summary: pd.DataFrame) -> None:
    configure_style()
    female = pd.read_csv(primary_prediction_path("female"))
    male = pd.read_csv(primary_prediction_path("male"))
    female["sex"] = "Female"
    male["sex"] = "Male"
    combined = pd.concat([female, male], ignore_index=True)

    figure = plt.figure(figsize=(180 / 25.4, 78 / 25.4))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[0.58, 1.25, 1.25],
        wspace=0.34,
        left=0.055,
        right=0.985,
        bottom=0.16,
        top=0.90,
    )
    pyramid = figure.add_subplot(grid[0, 0])
    bins = np.arange(40, 75, 5)
    centres = (bins[:-1] + bins[1:]) / 2
    female_counts, _ = np.histogram(
        female["true_values"], bins=bins
    )
    male_counts, _ = np.histogram(male["true_values"], bins=bins)
    pyramid.barh(
        centres,
        -male_counts,
        height=4.4,
        color=MALE_COLOR,
        edgecolor="white",
        label="Male",
    )
    pyramid.barh(
        centres,
        female_counts,
        height=4.4,
        color=FEMALE_COLOR,
        edgecolor="white",
        label="Female",
    )
    pyramid.axvline(0, color="black", linewidth=0.7)
    pyramid.set_yticks(centres)
    pyramid_labels = [
        f"{int(a)}–{int(b - 1)}" for a, b in zip(bins[:-1], bins[1:])
    ]
    pyramid_labels[-1] = "65–70"
    pyramid.set_yticklabels(pyramid_labels)
    pyramid.set_xticks([-1000, 0, 1000])
    pyramid.set_xticklabels(["1,000", "0", "1,000"])
    pyramid.set_xlabel("Participants")
    pyramid.set_ylabel("Age (years)")
    pyramid.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#BBBBBB",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.4,
    )

    for panel_index, (sex, data) in enumerate(
        (("female", female), ("male", male)), start=1
    ):
        axis = figure.add_subplot(grid[0, panel_index])
        slope, intercept, *_ = linregress(
            data["true_values"], data["mean_predictions"]
        )
        data = data.copy()
        data["delta_va"] = data["mean_predictions"] - (
            slope * data["true_values"] + intercept
        )
        lower = data["delta_va"].quantile(0.25)
        upper = data["delta_va"].quantile(0.75)
        middle = data["delta_va"].between(lower, upper, inclusive="neither")
        bottom = data["delta_va"] <= lower
        top = data["delta_va"] >= upper
        axis.scatter(
            data.loc[middle, "true_values"],
            data.loc[middle, "mean_predictions"],
            s=4,
            color="#D5D5D5",
            alpha=0.45,
            linewidth=0,
            rasterized=True,
        )
        axis.scatter(
            data.loc[bottom, "true_values"],
            data.loc[bottom, "mean_predictions"],
            s=5,
            color=YOUNGER_COLOR,
            alpha=0.55,
            linewidth=0,
            label="Bottom ΔVA quartile",
            rasterized=True,
        )
        axis.scatter(
            data.loc[top, "true_values"],
            data.loc[top, "mean_predictions"],
            s=5,
            color=OLDER_COLOR,
            alpha=0.55,
            linewidth=0,
            label="Top ΔVA quartile",
            rasterized=True,
        )
        coefficients = np.polyfit(
            data["true_values"], data["mean_predictions"], 3
        )
        x_values = np.linspace(40, 70, 300)
        axis.plot(
            x_values,
            np.polyval(coefficients, x_values),
            color="#2E3FA3",
            linewidth=1.2,
            label="Cubic polynomial fit",
        )
        axis.plot(
            [40, 70],
            [40, 70],
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            label="Identity line",
        )
        axis.set_xlim(39, 71)
        axis.set_ylim(
            min(38, math.floor(data["mean_predictions"].min())),
            max(72, math.ceil(data["mean_predictions"].max())),
        )
        axis.set_title(sex.capitalize())
        axis.set_xlabel("Chronological age (years)")
        axis.set_ylabel("Predicted age (years)")
        summary = primary_summary.loc[primary_summary["sex"] == sex].iloc[0]
        text = (
            f"n = {int(summary['n']):,}\n"
            f"r = {summary['Pearson_r_mean']:.3f} ± {summary['Pearson_r_sd']:.3f}\n"
            f"R² = {summary['R2_mean']:.3f} ± {summary['R2_sd']:.3f}\n"
            f"MAE = {summary['MAE_mean']:.2f} ± {summary['MAE_sd']:.2f} yr\n"
            f"RMSE = {summary['RMSE_mean']:.2f} ± {summary['RMSE_sd']:.2f} yr"
        )
        axis.text(
            0.98,
            0.03,
            text,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.9,
            },
        )
        if sex == "female":
            axis.legend(
                frameon=False,
                loc="upper left",
                handlelength=1.5,
                markerscale=1.5,
            )

    for label, axis in zip(
        ("a", "b", "c"),
        figure.axes,
    ):
        axis.text(
            -0.17,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
    save_figure(
        figure,
        "Figure1",
        destinations=[
            (FINAL_FINAL, "Fig1_final"),
            (SUBMISSION_FIGURES, "Figure_1"),
        ],
    )
    plt.close(figure)


def make_supplementary_acoustic_figure(
    acoustic_summary: pd.DataFrame,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(180 / 25.4, 112 / 25.4),
        sharex=True,
    )
    models = ("ridge", "lgbm")
    colors = {"ridge": "#4C78A8", "lgbm": "#F28E6B"}
    width = 0.34
    x = np.arange(len(REPRESENTATIONS))
    for row_index, metric in enumerate(("R2", "MAE")):
        for column_index, sex in enumerate(SEXES):
            axis = axes[row_index, column_index]
            for model_index, model in enumerate(models):
                means = []
                errors = []
                for representation in REPRESENTATIONS:
                    row = acoustic_summary.loc[
                        (acoustic_summary["representation"] == representation)
                        & (acoustic_summary["model"] == model)
                        & (acoustic_summary["sex"] == sex)
                    ].iloc[0]
                    means.append(row[f"{metric}_mean"])
                    errors.append(row[f"{metric}_sd"])
                positions = x + (model_index - 0.5) * width
                if metric == "R2":
                    plotted = np.maximum(means, -0.08)
                    plotted_errors = [
                        0.0 if raw < -0.08 else error
                        for raw, error in zip(means, errors)
                    ]
                else:
                    plotted = means
                    plotted_errors = errors
                axis.bar(
                    positions,
                    plotted,
                    width=width * 0.9,
                    yerr=plotted_errors,
                    color=colors[model],
                    label={"ridge": "Ridge", "lgbm": "LightGBM"}[model],
                    capsize=1.5,
                    linewidth=0,
                )
                if metric == "R2":
                    for position, raw, error in zip(
                        positions, means, errors
                    ):
                        if raw < -0.08:
                            axis.scatter(
                                position,
                                -0.075,
                                marker="v",
                                s=12,
                                color="#A33A3A",
                                zorder=5,
                            )
            axis.set_title(sex.capitalize())
            axis.set_ylabel(
                "Out-of-fold R²" if metric == "R2" else "MAE (years)"
            )
            if metric == "R2":
                axis.set_ylim(-0.10, 0.66)
                axis.axhline(0, color="black", linewidth=0.5)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [REPRESENTATION_LABELS[value] for value in REPRESENTATIONS],
                rotation=25,
                ha="right",
            )
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, ncol=2)
    figure.text(
        0.01,
        0.98,
        "a",
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.01,
        0.49,
        "b",
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.16,
        top=0.94,
        hspace=0.34,
        wspace=0.22,
    )
    save_figure(figure, "supp_fig_S3_feature_comparison")
    plt.close(figure)


def make_seed_stability_figure(primary_by_seed: pd.DataFrame) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(180 / 25.4, 48 / 25.4),
    )
    settings = (
        ("female", "R2", "R²"),
        ("female", "MAE", "MAE (years)"),
        ("male", "R2", "R²"),
        ("male", "MAE", "MAE (years)"),
    )
    for axis, (sex, metric, label) in zip(axes, settings):
        values = primary_by_seed.loc[
            primary_by_seed["sex"] == sex, metric
        ].to_numpy()
        axis.scatter(
            np.arange(len(values)),
            values,
            color=FEMALE_COLOR if sex == "female" else MALE_COLOR,
            s=14,
            zorder=3,
        )
        mean = values.mean()
        sd = values.std(ddof=1)
        axis.axhline(mean, color="#333333", linewidth=0.8)
        axis.axhspan(mean - sd, mean + sd, color="#999999", alpha=0.18)
        axis.set_title(sex.capitalize())
        axis.set_ylabel(label)
        axis.set_xlabel("Partition seed")
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(
            [str(seed) for seed in SEEDS],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        decimals = 3 if metric == "R2" else 2
        data_span = max(float(values.max() - values.min()), 1e-4)
        axis.set_ylim(
            float(values.min()) - 0.16 * data_span,
            float(values.max()) + 0.55 * data_span,
        )
        axis.text(
            0.98,
            0.96,
            f"mean ± SD\n{mean:.{decimals}f} ± {sd:.{decimals}f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#999999",
                "linewidth": 0.6,
                "alpha": 0.94,
            },
        )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.28,
        top=0.86,
        wspace=0.42,
    )
    save_figure(figure, "supp_fig_S1_seed_stability")
    plt.close(figure)


def make_clock_model_family_figure(clock_summary: pd.DataFrame) -> None:
    configure_style()
    clock_order = (
        "metabolomics",
        "sleep",
        "lifestyle",
        "DEXA",
        "diet",
        "NMR",
        "retina",
        "microbiome",
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 72 / 25.4),
        sharey=True,
    )
    width = 0.35
    x = np.arange(len(clock_order))
    for axis, sex in zip(axes, SEXES):
        for model_index, (model, color) in enumerate(
            (("ridge", "#4C78A8"), ("lgbm", "#F28E6B"))
        ):
            subset = clock_summary.loc[
                (clock_summary["sex"] == sex)
                & (clock_summary["model"] == model)
            ].set_index("clock")
            means = [subset.loc[clock, "R2_mean"] for clock in clock_order]
            errors = [subset.loc[clock, "R2_sd"] for clock in clock_order]
            positions = x + (model_index - 0.5) * width
            plotted = np.maximum(means, -0.08)
            plotted_errors = [
                0.0 if raw < -0.08 else error
                for raw, error in zip(means, errors)
            ]
            axis.bar(
                positions,
                plotted,
                width=width * 0.9,
                yerr=plotted_errors,
                color=color,
                capsize=1.5,
                label={"ridge": "Ridge", "lgbm": "LightGBM"}[model],
            )
            for position, raw, error in zip(positions, means, errors):
                if raw < -0.08:
                    axis.scatter(
                        position,
                        -0.075,
                        marker="v",
                        s=12,
                        color="#A33A3A",
                        zorder=5,
                    )
                    axis.text(
                        position,
                        -0.055,
                        f"{raw:.2f} ± {error:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )
        axis.set_title(sex.capitalize())
        axis.set_xticks(x)
        axis.set_xticklabels(
            [CLOCK_LABELS[clock] for clock in clock_order],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Out-of-fold R²")
        axis.set_ylim(-0.10, 0.70)
        axis.axhline(0, color="black", linewidth=0.5)
    axes[0].legend(frameon=False, ncol=2)
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.31,
        top=0.90,
        wspace=0.18,
    )
    save_figure(figure, "supp_fig_S7_clock_model_families")
    plt.close(figure)


def plot_conditioned_panel(axis: plt.Axes) -> list[mlines.Line2D]:
    data = pd.read_csv(CONDITIONED_SUMMARY).rename(
        columns={
            "baseline_R2_std": "baseline_R2_sd",
            "cond_R2_mean": "conditioned_R2_mean",
            "cond_R2_std": "conditioned_R2_sd",
            "voice_R2_std": "voice_R2_sd",
        }
    )
    gain_rows = []
    for row in data.itertuples(index=False):
        gains = []
        for seed in SEEDS:
            base = CONDITIONED_ROOT / row.modality / f"seed_{seed}" / (
                f"gender_{row.sex}"
            )
            baseline = load_json(base / "baseline_metrics.json")["test_R2"]
            voice = load_json(base / "voice_metrics.json")["test_R2"]
            conditioned = load_json(base / "conditioned_metrics.json")["test_R2"]
            gains.append(conditioned - max(baseline, voice))
        gain_rows.append(
            {
                "modality": row.modality,
                "sex": row.sex,
                "gain_over_stronger_mean": float(np.mean(gains)),
                "gain_over_stronger_sd": float(np.std(gains, ddof=1)),
            }
        )
    data = data.merge(
        pd.DataFrame(gain_rows),
        on=["modality", "sex"],
        how="left",
        validate="one_to_one",
    )
    data.to_csv(
        SUMMARY_DIR / "voice_conditioned_summary_stronger_baseline.csv",
        index=False,
    )
    female_order = (
        data.loc[data["sex"] == "female"]
        .sort_values("gain_over_stronger_mean", ascending=True)["modality"]
        .tolist()
    )
    centres = {modality: index for index, modality in enumerate(female_order)}
    for index, modality in enumerate(female_order):
        if index % 2 == 0:
            axis.axhspan(index - 0.48, index + 0.48, color="#F1F1F1")
        for sex, color, offset in (
            ("female", FEMALE_COLOR, 0.17),
            ("male", MALE_COLOR, -0.17),
        ):
            row = data.loc[
                (data["modality"] == modality) & (data["sex"] == sex)
            ].iloc[0]
            baseline = row["baseline_R2_mean"]
            voice = row["voice_R2_mean"]
            combined = row["conditioned_R2_mean"]
            y = index + offset
            axis.plot(
                [min(baseline, voice), combined],
                [y, y],
                color=GRAY,
                linewidth=0.5,
            )
            axis.plot(
                [max(baseline, voice), combined],
                [y, y],
                color=color,
                linewidth=2,
            )
            axis.errorbar(
                baseline,
                y,
                xerr=row["baseline_R2_sd"],
                fmt="s",
                markersize=3.6,
                color=GRAY,
                capsize=1.3,
                linewidth=0.6,
                zorder=3,
            )
            axis.errorbar(
                voice,
                y,
                xerr=row["voice_R2_sd"],
                fmt="D",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=color,
                color=color,
                capsize=1.3,
                linewidth=0.8,
                zorder=3,
            )
            axis.errorbar(
                combined,
                y,
                xerr=row["conditioned_R2_sd"],
                fmt="o",
                markersize=3.8,
                color=color,
                capsize=1.5,
                linewidth=0.7,
                zorder=4,
            )
    axis.set_yticks(range(len(female_order)))
    conditioned_labels = {
        "metabolomics": "MS\nmetabolomics",
        "NMR": "NMR\nmetabolomics",
    }
    axis.set_yticklabels(
        [
            conditioned_labels.get(
                value,
                CLOCK_LABELS.get(value, value),
            )
            for value in female_order
        ]
    )
    axis.set_xlim(0, 0.76)
    axis.set_xlabel("Held-out test-set R²")
    axis.grid(False)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    legend = [
        mlines.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=GRAY,
            markeredgecolor=GRAY,
            markersize=4,
            label="Modality only",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markersize=4,
            label="Voice only",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GRAY,
            markeredgecolor=GRAY,
            markersize=4,
            label="Combined",
        ),
        mlines.Line2D(
            [0], [0], color=FEMALE_COLOR, linewidth=2, label="Female gain"
        ),
        mlines.Line2D(
            [0], [0], color=MALE_COLOR, linewidth=2, label="Male gain"
        ),
    ]
    return legend


def make_figure3(
    selected: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    configure_style()
    female_selected = (
        selected.loc[selected["sex"] == "female"]
        .set_index("clock")
        .loc[list(CLOCKS)]
    )
    order = female_selected["R2_mean"].sort_values(ascending=False).index.tolist()
    figure = plt.figure(figsize=(180 / 25.4, 100 / 25.4))
    panel_a_grid = figure.add_gridspec(
        2,
        height_ratios=[2.5, 0.75],
        left=0.065,
        right=0.535,
        bottom=0.22,
        top=0.92,
        hspace=0.08,
    )
    bar_axis = figure.add_subplot(panel_a_grid[0, 0])
    heat_axis = figure.add_subplot(panel_a_grid[1, 0])
    conditioned_axis = figure.add_axes([0.645, 0.22, 0.34, 0.70])
    conditioned_legend_axis = figure.add_axes([0.645, 0.015, 0.34, 0.10])

    x = np.arange(len(order))
    width = 0.34
    for offset, sex, color in (
        (-width / 2, "female", FEMALE_COLOR),
        (width / 2, "male", MALE_COLOR),
    ):
        subset = selected.loc[selected["sex"] == sex].set_index("clock")
        values = np.array([subset.loc[clock, "R2_mean"] for clock in order])
        errors = np.array([subset.loc[clock, "R2_sd"] for clock in order])
        bars = bar_axis.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=1.3,
            color=color,
            label=sex.capitalize(),
            linewidth=0,
        )
        voice_index = order.index("voice")
        bars[voice_index].set_edgecolor("#222222")
        bars[voice_index].set_linewidth(1.1)
    bar_axis.set_ylabel("Out-of-fold R²")
    bar_axis.set_ylim(0, 0.73)
    bar_axis.set_xticks([])
    bar_axis.legend(frameon=False, ncol=2, loc="upper right")
    bar_axis.set_axisbelow(True)
    bar_axis.grid(False)
    bar_axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    correlation_matrix = np.empty((2, len(order)))
    for row_index, sex in enumerate(SEXES):
        lookup = correlations.loc[
            correlations["sex"] == sex
        ].set_index("clock")
        for column_index, clock in enumerate(order):
            correlation_matrix[row_index, column_index] = (
                1.0 if clock == "voice" else lookup.loc[clock, "Pearson_r"]
            )
    sns.heatmap(
        correlation_matrix,
        ax=heat_axis,
        cmap="rocket_r",
        vmin=0,
        vmax=0.70,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        linewidths=0.4,
        linecolor="white",
    )
    heat_axis.set_yticks([0.5, 1.5])
    heat_axis.set_yticklabels(["Female", "Male"], rotation=0)
    heat_axis.set_xticks(np.arange(len(order)) + 0.5)
    heat_axis.set_xticklabels(
        [CLOCK_LABELS[clock] for clock in order],
        rotation=38,
        ha="right",
    )
    heat_axis.tick_params(length=0)
    heat_axis.set_xlabel("")
    heat_axis.set_ylabel("")

    conditioned_legend = plot_conditioned_panel(conditioned_axis)
    conditioned_legend_axis.axis("off")
    conditioned_legend_axis.legend(
        # Matplotlib fills multi-column legends column-wise. This ordering
        # places the three model markers on the first row and the two gain
        # lines on the second row.
        handles=[
            conditioned_legend[0],
            conditioned_legend[3],
            conditioned_legend[1],
            conditioned_legend[4],
            conditioned_legend[2],
        ],
        frameon=True,
        fancybox=True,
        framealpha=1,
        facecolor="white",
        edgecolor="#999999",
        loc="center",
        ncol=3,
        columnspacing=0.9,
        labelspacing=0.55,
        handletextpad=0.4,
        borderaxespad=0.2,
        borderpad=0.5,
    )
    figure.text(
        0.018,
        0.955,
        "a",
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.598,
        0.955,
        "b",
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    save_figure(
        figure,
        "Figure3",
        destinations=[
            (FINAL_FINAL, "Figure3_final_combined"),
            (SUBMISSION_FIGURES, "Figure_3"),
        ],
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate result tables without producing figures.",
    )
    args = parser.parse_args()
    primary_summary, acoustic_summary, lifestyle_summary = aggregate_outputs()
    primary_by_seed = pd.read_csv(
        SUMMARY_DIR / "primary_voice_metrics_by_seed.csv"
    )
    _, clock_summary = collect_clock_model_results(lifestyle_summary)
    correlations = pairwise_voice_clock_correlations()
    selected = selected_clock_summary(primary_summary, clock_summary)
    if args.aggregate_only:
        return
    make_figure1(primary_summary)
    make_supplementary_acoustic_figure(acoustic_summary)
    make_seed_stability_figure(primary_by_seed)
    make_clock_model_family_figure(clock_summary)
    make_figure3(selected, correlations)


if __name__ == "__main__":
    main()
