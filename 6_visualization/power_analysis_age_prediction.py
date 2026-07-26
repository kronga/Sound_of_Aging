#!/usr/bin/env python3
"""Repeated-CV learning curve for WavLM Voice Age prediction.

The analysis deliberately mirrors the final Figure 1 pipeline:

* the same 6,979 participants and latest QC-passed recordings;
* sex-stratified models;
* the same ten shuffled participant-level five-fold outer partitions;
* a fold-specific participant-level 20% inner tuning holdout;
* median imputation, no WavLM feature standardization, and the same ridge
  alpha grid used for the primary model.

For every non-full learning-curve point, ``n`` participants are sampled from
each outer training fold. The resulting predictions are aggregated over the
five held-out folds before performance is calculated. The full-capacity point
uses the existing primary Figure 1 result for the same sex and partition seed,
which is exactly the same model/evaluation design without subsampling.

The cluster worker has 20 tasks (two sexes x ten partition seeds). Combine the
completed task outputs with ``--combine``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-davidkro")

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold


ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
VOICE_BASE = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length"
)
WAVLM_CSV = VOICE_BASE / "WavLM_features_filtered_with_RF.csv"
PRIMARY_RESULTS = (
    ROOT
    / "analysis_outputs"
    / "repeated_analysis"
    / "final"
    / "primary_voice"
)
OUTPUT_DIR = Path(
    os.environ.get(
        "VOICE_POWER_OUTPUT_DIR",
        ROOT
        / "analysis_outputs"
        / "repeated_analysis"
        / "power_analysis_cv",
    )
)
FINAL_FIGURES = ROOT / "voice_age_manuscript" / "final_figs"

MIN_AGE, MAX_AGE = 40, 70
SEXES = ("female", "male")
SEX_VALUE = {"female": 0.0, "male": 1.0}
PARTITION_SEEDS = (42, 1, 2, 3, 4, 17, 99, 123, 256, 512)
N_OUTER_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.20
ALPHA_CANDIDATES = (0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0)
SAMPLE_SIZES = (50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500)


def stage_rank(value: object) -> tuple[int, str]:
    text = str(value)
    if text == "baseline":
        return (-1, text)
    try:
        return (int(text.split("_", 1)[0]), text)
    except (TypeError, ValueError):
        return (-2, text)


def load_data(sex: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the exact final one-recording-per-participant WavLM cohort."""
    source = pd.read_csv(WAVLM_CSV, index_col=0)
    source.index = source.index.astype(str)
    source.index.name = "filename"
    metadata = source.loc[
        :,
        ["age", "gender", "subject_number", "research_stage"],
    ].dropna(subset=["age", "gender", "subject_number"])
    metadata["_stage_rank"] = metadata["research_stage"].map(stage_rank)
    metadata["_filename"] = metadata.index
    metadata = (
        metadata.sort_values(
            ["subject_number", "_stage_rank", "_filename"]
        )
        .drop_duplicates("subject_number", keep="last")
        .drop(columns=["_stage_rank", "_filename"])
    )
    metadata = metadata.loc[
        metadata["age"].between(MIN_AGE, MAX_AGE, inclusive="both")
    ].copy()
    if (
        len(metadata) != 6979
        or metadata["subject_number"].nunique() != 6979
    ):
        raise RuntimeError(
            "Expected 6,979 latest QC-passed participants, observed "
            f"{len(metadata)} rows and "
            f"{metadata['subject_number'].nunique()} "
            "unique participants."
        )
    feature_names = [
        column for column in source.columns if column.startswith("feature_")
    ]
    features = source.loc[:, feature_names]
    features = features.loc[
        ~features.index.duplicated(keep="last")
    ].copy()
    frame = metadata.join(
        features.loc[metadata.index, feature_names],
        how="left",
    )
    frame = frame.loc[frame["gender"] == SEX_VALUE[sex]].copy()
    x = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    y = frame["age"].astype(float)
    groups = frame["subject_number"]
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    expected = 3631 if sex == "female" else 3348
    if len(frame) != expected or groups.nunique() != expected:
        raise RuntimeError(
            f"Expected {expected} {sex} participants, observed {len(frame)}."
        )
    return x, y, groups


def random_group_folds(
    groups: pd.Series,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the shuffled participant folds used for final Figure 1."""
    unique_groups = np.asarray(sorted(pd.unique(groups)))
    splitter = KFold(
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_idx, test_group_idx in splitter.split(unique_groups):
        train_groups = set(unique_groups[train_group_idx])
        test_groups = set(unique_groups[test_group_idx])
        train_idx = np.flatnonzero(groups.isin(train_groups).to_numpy())
        test_idx = np.flatnonzero(groups.isin(test_groups).to_numpy())
        folds.append((train_idx, test_idx))
    return folds


def sample_outer_training_participants(
    train_idx: np.ndarray,
    groups: pd.Series,
    n: int,
    seed: int,
    fold_index: int,
) -> np.ndarray:
    """Sample exactly ``n`` participants from one outer training fold."""
    train_groups = np.asarray(sorted(pd.unique(groups.iloc[train_idx])))
    if n > len(train_groups):
        raise ValueError(
            f"Requested n={n} from an outer training fold containing "
            f"{len(train_groups)} participants."
        )
    seed_sequence = np.random.SeedSequence([seed, fold_index, n])
    rng = np.random.default_rng(seed_sequence)
    selected_groups = set(rng.choice(train_groups, n, replace=False))
    selected_mask = groups.iloc[train_idx].isin(selected_groups).to_numpy()
    selected_idx = train_idx[selected_mask]
    if groups.iloc[selected_idx].nunique() != n:
        raise RuntimeError("Participant-level subsampling produced wrong n.")
    return selected_idx


def choose_alpha(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
) -> float:
    """Fold-specific inner-holdout tuning matching the primary pipeline."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed,
    )
    train_idx, validation_idx = next(
        splitter.split(np.zeros(len(groups)), groups=groups)
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x.iloc[train_idx])
    x_validation = imputer.transform(x.iloc[validation_idx])
    best_alpha = ALPHA_CANDIDATES[0]
    best_score = -np.inf
    for alpha in ALPHA_CANDIDATES:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y.iloc[train_idx])
        score = r2_score(
            y.iloc[validation_idx],
            model.predict(x_validation),
        )
        if score > best_score:
            best_alpha = alpha
            best_score = float(score)
    return float(best_alpha)


def fit_predict(
    x: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_seed: int,
    groups: pd.Series,
) -> tuple[np.ndarray, float]:
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]
    best_alpha = choose_alpha(
        x_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        groups_train.reset_index(drop=True),
        inner_seed,
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x.iloc[test_idx])
    model = Ridge(alpha=best_alpha)
    model.fit(x_train_imputed, y_train)
    return model.predict(x_test_imputed), best_alpha


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def primary_full_row(
    sex: str,
    seed: int,
    full_n: int,
) -> dict[str, object]:
    """Use the exact corresponding final Figure 1 partition result."""
    path = (
        PRIMARY_RESULTS
        / f"gender_{sex}"
        / f"seed_{seed}"
        / "metrics.json"
    )
    payload = json.loads(path.read_text())
    overall = payload["overall"]
    return {
        "sex": sex,
        "seed": seed,
        "n": full_n,
        "is_full": True,
        "R2": float(overall["R2"]),
        "Pearson_r": float(overall["Pearson_r"]),
        "MAE": float(overall["MAE"]),
        "RMSE": float(overall["RMSE"]),
        "n_participants": int(payload["n_participants"]),
        "n_outer_folds": int(len(payload["folds"])),
        "mean_selected_alpha": float(
            np.mean(
                [
                    fold["selected_hyperparameters"]["alpha"]
                    for fold in payload["folds"]
                ]
            )
        ),
    }


def run_partition(
    sex: str,
    seed: int,
    *,
    smoke: bool = False,
) -> pd.DataFrame:
    x, y, groups = load_data(sex)
    folds = random_group_folds(groups, seed)
    full_n = int(round(np.mean([len(train_idx) for train_idx, _ in folds])))
    sample_sizes = (50, 500) if smoke else SAMPLE_SIZES
    rows: list[dict[str, object]] = []
    for n in sample_sizes:
        if any(n > len(train_idx) for train_idx, _ in folds):
            continue
        oof = np.full(len(y), np.nan)
        selected_alphas: list[float] = []
        for fold_index, (outer_train_idx, test_idx) in enumerate(
            folds,
            start=1,
        ):
            sampled_train_idx = sample_outer_training_participants(
                outer_train_idx,
                groups,
                n,
                seed,
                fold_index,
            )
            predicted, best_alpha = fit_predict(
                x,
                y,
                sampled_train_idx,
                test_idx,
                seed * 100 + fold_index,
                groups,
            )
            oof[test_idx] = predicted
            selected_alphas.append(best_alpha)
        if np.isnan(oof).any():
            raise RuntimeError("OOF predictions are incomplete.")
        row = {
            "sex": sex,
            "seed": seed,
            "n": n,
            "is_full": False,
            **calculate_metrics(y.to_numpy(), oof),
            "n_participants": int(groups.nunique()),
            "n_outer_folds": N_OUTER_FOLDS,
            "mean_selected_alpha": float(np.mean(selected_alphas)),
        }
        rows.append(row)
        print(
            f"{sex} seed={seed} n={n}: "
            f"R²={row['R2']:.3f}, MAE={row['MAE']:.2f}",
            flush=True,
        )
    if not smoke:
        rows.append(primary_full_row(sex, seed, full_n))
    return pd.DataFrame(rows)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def task_list() -> list[tuple[str, int]]:
    return [
        (sex, seed)
        for sex in SEXES
        for seed in PARTITION_SEEDS
    ]


def task_output(sex: str, seed: int, *, smoke: bool = False) -> Path:
    base = OUTPUT_DIR / ("smoke" if smoke else "partitions")
    return base / f"power_{sex}_seed_{seed}.csv"


def run_task(job_index: int, *, smoke: bool = False) -> Path:
    tasks = task_list()
    if not 0 <= job_index < len(tasks):
        raise ValueError(
            f"job-index must be 0–{len(tasks) - 1}, received {job_index}."
        )
    sex, seed = tasks[job_index]
    destination = task_output(sex, seed, smoke=smoke)
    if destination.exists():
        print(f"[SKIP] Existing result: {destination}")
        return destination
    result = run_partition(sex, seed, smoke=smoke)
    atomic_csv(destination, result)
    print(f"Saved: {destination}")
    return destination


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = ("R2", "Pearson_r", "MAE", "RMSE")
    for (sex, n, is_full), group in raw.groupby(
        ["sex", "n", "is_full"],
        sort=False,
    ):
        row: dict[str, object] = {
            "sex": sex,
            "n": int(n),
            "is_full": bool(is_full),
            "n_partitions": int(group["seed"].nunique()),
        }
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            half_width = 1.96 * sd / np.sqrt(len(values))
            prefix = {
                "R2": "r2",
                "Pearson_r": "r",
                "MAE": "mae",
                "RMSE": "rmse",
            }[metric]
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_sd"] = sd
            row[f"{prefix}_ci_lo"] = mean - half_width
            row[f"{prefix}_ci_hi"] = mean + half_width
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["sex", "n"])
        .reset_index(drop=True)
    )


def plot(summary: pd.DataFrame) -> None:
    font = 8
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"female": "#D97777", "male": "#6B9DC1"}
    metrics = (
        ("r2", "R²"),
        ("mae", "MAE (years)"),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 78 / 25.4),
    )
    for axis, (prefix, ylabel) in zip(axes, metrics):
        for sex in SEXES:
            group = summary.loc[summary["sex"] == sex].sort_values("n")
            color = colors[sex]
            mean = group[f"{prefix}_mean"].to_numpy()
            low = group[f"{prefix}_ci_lo"].to_numpy()
            high = group[f"{prefix}_ci_hi"].to_numpy()
            axis.plot(
                group["n"],
                mean,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=sex.capitalize(),
            )
            axis.fill_between(
                group["n"],
                low,
                high,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            full = group.loc[group["is_full"]]
            axis.errorbar(
                full["n"],
                full[f"{prefix}_mean"],
                yerr=np.vstack(
                    [
                        full[f"{prefix}_mean"] - full[f"{prefix}_ci_lo"],
                        full[f"{prefix}_ci_hi"] - full[f"{prefix}_mean"],
                    ]
                ),
                fmt="D",
                color=color,
                markerfacecolor="white",
                markersize=5,
                markeredgewidth=1.0,
                linewidth=0.8,
                capsize=2,
                zorder=5,
            )
        axis.set_xlabel("Training participants per outer fold")
        axis.set_ylabel(ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    r2_low = min(-0.05, float(summary["r2_ci_lo"].min()) - 0.02)
    r2_high = max(0.58, float(summary["r2_ci_hi"].max()) + 0.02)
    axes[0].set_ylim(r2_low, r2_high)
    handles = [
        mlines.Line2D(
            [0],
            [0],
            color=colors[sex],
            marker="o",
            markersize=3,
            label=sex.capitalize(),
        )
        for sex in SEXES
    ]
    handles.append(
        mlines.Line2D(
            [0],
            [0],
            color="#555555",
            marker="D",
            markerfacecolor="white",
            markersize=5,
            linewidth=0,
            label="Full outer-training fold",
        )
    )
    axes[0].legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        handlelength=1.5,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.18,
        top=0.96,
        wspace=0.28,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    destinations = (
        OUTPUT_DIR / "power_analysis_learning_curve",
        FINAL_FIGURES / "supp_fig_S2_power_analysis",
    )
    for stem in destinations:
        figure.savefig(stem.with_suffix(".pdf"))
        figure.savefig(stem.with_suffix(".png"), dpi=600)
        print(f"Saved: {stem}.pdf/.png")
    plt.close(figure)


def combine() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [
        task_output(sex, seed)
        for sex, seed in task_list()
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing partition results:\n"
            + "\n".join(str(path) for path in missing)
        )
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for sex in SEXES:
        observed = set(raw.loc[raw["sex"] == sex, "seed"].unique())
        if observed != set(PARTITION_SEEDS):
            raise RuntimeError(
                f"Incomplete partition seeds for {sex}: {sorted(observed)}"
            )
    summary = summarise(raw)
    atomic_csv(OUTPUT_DIR / "power_analysis_results_by_partition.csv", raw)
    atomic_csv(OUTPUT_DIR / "power_analysis_results.csv", summary)
    plot(summary)
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-index",
        type=int,
        default=int(os.environ.get("JOB_INDEX", "0")),
    )
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    if args.list_tasks:
        for index, (sex, seed) in enumerate(task_list()):
            print(index, sex, seed)
        return
    if args.combine:
        _, summary = combine()
        print(summary.to_string(index=False))
        return
    if args.plot_only:
        plot(pd.read_csv(OUTPUT_DIR / "power_analysis_results.csv"))
        return
    run_task(args.job_index, smoke=args.smoke)


if __name__ == "__main__":
    main()
