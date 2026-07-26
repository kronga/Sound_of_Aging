#!/usr/bin/env python3
"""Repeated model evaluation on explicitly defined analysis cohorts.

The worker is designed for Elysium/SGE execution.  Each job handles one
representation × model × sex × seed combination so that seed-specific outer
partitions are genuinely independent for the voice and acoustic-benchmark
analyses.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-davidkro")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
VOICE_BASE = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length"
)
CLOCK_BASE = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "age_prediction_new_pipeline/data"
)
OUTPUT_BASE = ROOT / "analysis_outputs" / "repeated_analysis"

WAVLM_CSV = VOICE_BASE / "WavLM_features_filtered_with_RF.csv"
CLASSICAL_PARQUETS = {
    "praat": VOICE_BASE / "features_praat" / "all_features.parquet",
    "egemaps": VOICE_BASE / "features_egemaps" / "all_features.parquet",
    "emobase": VOICE_BASE / "features_emobase" / "all_features.parquet",
    "compare2016": VOICE_BASE / "features_compare2016" / "all_features.parquet",
}
SEEDS = (42, 1, 2, 3, 4, 17, 99, 123, 256, 512)
SEXES = ("female", "male")
SEX_VALUE = {"female": 0.0, "male": 1.0}
N_OUTER_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.20
RIDGE_ALPHAS = (0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0)
ENGINEERED_RIDGE_ALPHAS = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
)
CLOCK_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
N_LGBM_CANDIDATES = 30


@dataclass(frozen=True)
class Task:
    family: str
    representation: str
    model: str
    sex: str
    seed: int


def ridge_tasks() -> list[Task]:
    tasks: list[Task] = []
    for sex in SEXES:
        for seed in SEEDS:
            tasks.append(Task("primary_voice", "wavlm", "ridge", sex, seed))
    for representation in CLASSICAL_PARQUETS:
        for sex in SEXES:
            for seed in SEEDS:
                tasks.append(
                    Task(
                        "acoustic_benchmark",
                        representation,
                        "ridge",
                        sex,
                        seed,
                    )
                )
    for sex in SEXES:
        for seed in SEEDS:
            tasks.append(
                Task("comparison_clock", "lifestyle", "ridge", sex, seed)
            )
    return tasks


def lgbm_tasks() -> list[Task]:
    tasks: list[Task] = []
    for representation in ("wavlm", *CLASSICAL_PARQUETS):
        for sex in SEXES:
            for seed in SEEDS:
                tasks.append(
                    Task(
                        "acoustic_benchmark",
                        representation,
                        "lgbm",
                        sex,
                        seed,
                    )
                )
    for sex in SEXES:
        for seed in SEEDS:
            tasks.append(
                Task("comparison_clock", "lifestyle", "lgbm", sex, seed)
            )
    return tasks


def stage_rank(value: object) -> tuple[int, str]:
    text = str(value)
    if text == "baseline":
        return (-1, text)
    try:
        return (int(text.split("_", 1)[0]), text)
    except (TypeError, ValueError):
        return (-2, text)


def canonical_voice_metadata() -> pd.DataFrame:
    """Return the exact final 6,979-participant, latest-QC-passed cohort."""
    metadata = pd.read_csv(
        WAVLM_CSV,
        index_col=0,
        usecols=[
            "index",
            "age",
            "gender",
            "subject_number",
            "research_stage",
        ],
    )
    metadata.index = metadata.index.astype(str)
    metadata.index.name = "filename"
    metadata = metadata.dropna(subset=["age", "gender", "subject_number"])
    metadata["_stage_rank"] = metadata["research_stage"].map(stage_rank)
    metadata = (
        metadata.sort_values(["subject_number", "_stage_rank", "filename"])
        .drop_duplicates("subject_number", keep="last")
        .drop(columns="_stage_rank")
    )
    metadata = metadata.loc[metadata["age"].between(40, 70)].copy()
    if len(metadata) != 6979:
        raise RuntimeError(
            f"Canonical cohort has {len(metadata)} rows; expected 6,979."
        )
    counts = metadata["gender"].value_counts().to_dict()
    if counts.get(0.0) != 3631 or counts.get(1.0) != 3348:
        raise RuntimeError(f"Unexpected sex counts in canonical cohort: {counts}")
    return metadata


def load_voice_representation(representation: str) -> tuple[pd.DataFrame, list[str]]:
    metadata = canonical_voice_metadata()
    if representation == "wavlm":
        feature_names = [
            f"feature_{idx}" for idx in range(1024)
        ]
        features = pd.read_csv(
            WAVLM_CSV,
            index_col=0,
            usecols=["index", *feature_names],
        )
    else:
        features = pd.read_parquet(CLASSICAL_PARQUETS[representation])
        feature_names = list(features.columns)
    features.index = features.index.astype(str)
    features.index.name = "filename"
    # The source tables contain four exact duplicate filename rows.  Collapse
    # them before indexing so each canonical participant contributes exactly
    # one recording rather than allowing ``.loc``/``join`` to expand rows.
    features = features.loc[~features.index.duplicated(keep="last")].copy()
    missing = metadata.index.difference(features.index)
    if len(missing):
        raise RuntimeError(
            f"{representation} is missing {len(missing)} canonical recordings."
        )
    data = metadata.join(features.loc[metadata.index, feature_names], how="left")
    if len(data) != len(metadata) or data["subject_number"].duplicated().any():
        raise RuntimeError(
            f"{representation} produced {len(data)} rows for "
            f"{metadata['subject_number'].nunique()} participants."
        )
    return data, feature_names


def load_lifestyle() -> tuple[pd.DataFrame, list[str]]:
    x_path = CLOCK_BASE / "X_lifestyle_age.csv"
    y_path = CLOCK_BASE / "Y_lifestyle_age.csv"
    features = pd.read_csv(x_path, index_col=[0, 1])
    outcomes = pd.read_csv(y_path, index_col=[0, 1])
    if "age" not in features:
        features = features.join(outcomes[["age"]], how="inner")
    features = features.reset_index()
    features = features.dropna(subset=["age", "gender", "subject_number"])
    feature_names = [
        column
        for column in features.columns
        if column
        not in {
            "subject_number",
            "RegistrationCode",
            "age",
            "gender",
            "research_stage",
        }
    ]
    features = features.loc[:, [
        "subject_number",
        "research_stage",
        "age",
        "gender",
        *feature_names,
    ]]
    return features, feature_names


def random_group_folds(
    groups: pd.Series, seed: int, n_splits: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Shuffle unique participants, then assign them to outer folds."""
    unique_groups = np.asarray(sorted(pd.unique(groups)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_group_idx, test_group_idx in splitter.split(unique_groups):
        train_groups = set(unique_groups[train_group_idx])
        test_groups = set(unique_groups[test_group_idx])
        train_idx = np.flatnonzero(groups.isin(train_groups).to_numpy())
        test_idx = np.flatnonzero(groups.isin(test_groups).to_numpy())
        yield train_idx, test_idx


def fixed_group_folds(
    x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(x, y, groups)


def inner_group_split(
    groups: pd.Series, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed,
    )
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "Pearson_r": r,
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def ridge_pipeline(
    alpha: float,
    standardize: bool,
    solver: str = "auto",
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        )
    ]
    if standardize:
        steps.append(("scale", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=alpha, solver=solver)))
    return Pipeline(steps)


def choose_ridge_alpha(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    candidates: tuple[float, ...],
    standardize: bool,
    seed: int,
    solver: str = "auto",
) -> tuple[float, float]:
    train_idx, validation_idx = inner_group_split(groups, seed)
    best_alpha = candidates[0]
    best_score = -np.inf
    for alpha in candidates:
        model = ridge_pipeline(alpha, standardize, solver)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        score = r2_score(
            y.iloc[validation_idx],
            model.predict(x.iloc[validation_idx]),
        )
        if score > best_score:
            best_alpha = alpha
            best_score = float(score)
    return float(best_alpha), float(best_score)


def sampled_lgbm_parameters(seed: int) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    sampled: list[dict[str, float | int]] = []
    for _ in range(N_LGBM_CANDIDATES):
        sampled.append(
            {
                "num_leaves": int(rng.integers(20, 150)),
                "max_depth": int(rng.integers(3, 15)),
                "learning_rate": float(rng.uniform(0.01, 0.21)),
                "n_estimators": int(rng.integers(100, 1000)),
                "min_child_samples": int(rng.integers(10, 100)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "reg_alpha": float(rng.uniform(0.0, 1.0)),
                "reg_lambda": float(rng.uniform(0.0, 1.0)),
            }
        )
    return sampled


def make_lgbm(
    params: dict[str, float | int],
    seed: int,
    threads: int,
) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        verbosity=-1,
        random_state=seed,
        n_jobs=threads,
        subsample_freq=1,
        **params,
    )


def choose_lgbm_parameters(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    threads: int,
) -> tuple[dict[str, float | int], float]:
    train_idx, validation_idx = inner_group_split(groups, seed)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x.iloc[train_idx])
    x_validation = imputer.transform(x.iloc[validation_idx])
    best_params: dict[str, float | int] | None = None
    best_score = -np.inf
    for params in sampled_lgbm_parameters(seed):
        model = make_lgbm(params, seed, threads)
        model.fit(x_train, y.iloc[train_idx])
        score = r2_score(
            y.iloc[validation_idx],
            model.predict(x_validation),
        )
        if score > best_score:
            best_params = params
            best_score = float(score)
    if best_params is None:
        raise RuntimeError("LightGBM hyperparameter selection produced no model.")
    return best_params, best_score


def output_directory(task: Task, smoke: bool) -> Path:
    base = OUTPUT_BASE / ("smoke" if smoke else "final")
    if task.family == "primary_voice":
        return base / "primary_voice" / f"gender_{task.sex}" / f"seed_{task.seed}"
    if task.family == "acoustic_benchmark":
        return (
            base
            / "acoustic_benchmark"
            / task.representation
            / task.model
            / f"gender_{task.sex}"
            / f"seed_{task.seed}"
        )
    return (
        base
        / "comparison_clocks"
        / task.representation
        / task.model
        / f"gender_{task.sex}"
        / f"seed_{task.seed}"
    )


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_task(task: Task, smoke: bool = False) -> None:
    out_dir = output_directory(task, smoke)
    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "predictions.csv"
    if metrics_path.exists() and predictions_path.exists():
        print(f"[SKIP] Complete output already exists: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if task.family == "comparison_clock":
        data, feature_names = load_lifestyle()
        shuffle_outer = False
    else:
        data, feature_names = load_voice_representation(task.representation)
        shuffle_outer = True

    data = data.loc[data["gender"] == SEX_VALUE[task.sex]].copy()
    if smoke:
        keep_groups = pd.unique(data["subject_number"])[:300]
        data = data.loc[data["subject_number"].isin(keep_groups)].copy()

    x = data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    y = data["age"].astype(float).reset_index(drop=True)
    groups = data["subject_number"].reset_index(drop=True)
    x = x.reset_index(drop=True)
    metadata_columns = [
        column
        for column in ["filename", "subject_number", "research_stage", "age"]
        if column in data.reset_index().columns
    ]
    metadata = data.reset_index().loc[:, metadata_columns].reset_index(drop=True)

    n_splits = 2 if smoke else N_OUTER_FOLDS
    if shuffle_outer:
        outer_folds = list(random_group_folds(groups, task.seed, n_splits))
    else:
        outer_folds = list(fixed_group_folds(x, y, groups, n_splits))

    oof = np.full(len(data), np.nan)
    fold_assignment = np.full(len(data), -1, dtype=int)
    fold_details: list[dict[str, object]] = []
    threads = int(os.environ.get("ANALYSIS_LGBM_THREADS", "8"))

    for fold_index, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        inner_seed = task.seed * 100 + fold_index
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]

        if task.model == "ridge":
            standardize = task.representation != "wavlm"
            if task.representation == "wavlm":
                candidates = RIDGE_ALPHAS
            elif task.family == "acoustic_benchmark":
                candidates = ENGINEERED_RIDGE_ALPHAS
            else:
                candidates = CLOCK_RIDGE_ALPHAS
            ridge_solver = (
                "lsqr"
                if task.family == "acoustic_benchmark"
                and task.representation != "wavlm"
                else "auto"
            )
            best_alpha, inner_score = choose_ridge_alpha(
                x_train,
                y_train,
                groups_train,
                candidates,
                standardize,
                inner_seed,
                ridge_solver,
            )
            model = ridge_pipeline(best_alpha, standardize, ridge_solver)
            model.fit(x_train, y_train)
            predicted = model.predict(x.iloc[test_idx])
            selected: object = {"alpha": best_alpha}
        else:
            best_params, inner_score = choose_lgbm_parameters(
                x_train,
                y_train,
                groups_train,
                inner_seed,
                threads,
            )
            imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )
            x_train_imputed = imputer.fit_transform(x_train)
            x_test_imputed = imputer.transform(x.iloc[test_idx])
            model = make_lgbm(best_params, inner_seed, threads)
            model.fit(x_train_imputed, y_train)
            predicted = model.predict(x_test_imputed)
            selected = best_params

        oof[test_idx] = predicted
        fold_assignment[test_idx] = fold_index
        fold_details.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_participants": int(groups.iloc[train_idx].nunique()),
                "test_participants": int(groups.iloc[test_idx].nunique()),
                "inner_validation_R2": float(inner_score),
                "selected_hyperparameters": selected,
                "test_metrics": metrics(
                    y.iloc[test_idx].to_numpy(),
                    np.asarray(predicted),
                ),
            }
        )

    if np.isnan(oof).any() or (fold_assignment < 1).any():
        raise RuntimeError("OOF predictions are incomplete.")

    result_metrics = {
        **asdict(task),
        "n_rows": int(len(data)),
        "n_participants": int(groups.nunique()),
        "n_features": int(len(feature_names)),
        "outer_folds_shuffled": shuffle_outer,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "overall": metrics(y.to_numpy(), oof),
        "folds": fold_details,
    }
    predictions = metadata.copy()
    predictions["true_values"] = y
    predictions["predictions"] = oof
    predictions["fold"] = fold_assignment
    atomic_csv(predictions_path, predictions)
    atomic_json(metrics_path, result_metrics)
    print(json.dumps(result_metrics["overall"], indent=2))
    print(f"Saved: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("ridge", "lgbm"), required=True)
    parser.add_argument(
        "--job-index",
        type=int,
        default=int(os.environ.get("JOB_INDEX", "0")),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    tasks = ridge_tasks() if args.kind == "ridge" else lgbm_tasks()
    if args.list_tasks:
        for index, task in enumerate(tasks):
            print(index, asdict(task))
        return
    if not 0 <= args.job_index < len(tasks):
        raise IndexError(
            f"job-index {args.job_index} outside 0..{len(tasks) - 1}"
        )
    run_task(tasks[args.job_index], smoke=args.smoke)


if __name__ == "__main__":
    main()
