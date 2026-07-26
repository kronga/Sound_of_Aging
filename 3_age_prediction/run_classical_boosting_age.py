"""
LightGBM / XGBoost age prediction from classical acoustic feature sets.

5-fold GroupKFold CV, Optuna HPO (50 trials) on inner 20% validation split,
single seed. Designed to be launched one job per (model, feature_set) pair.

Usage
-----
  # Run one job (cluster dispatch target)
  python run_classical_boosting_age.py --model lgbm --feature-set egemaps

  # Launch all 8 jobs in parallel locally
  python run_classical_boosting_age.py --submit

  # Print summary table after all jobs complete
  python run_classical_boosting_age.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─────────────────────────── config ──────────────────────────────────────── #

PYTHON_HIMEM8 = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
PYTHON_HIMEM7 = "/net/mraid20/export/jasmine/david/anaconda3/envs/lgbm_compat/bin/python"

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
WAVLM_FILTERED_CSV  = BASE / "WavLM_features_filtered_with_RF.csv"

FEATURE_PARQUETS: dict[str, Path] = {
    "praat":       BASE / "features_praat"       / "all_features.parquet",
    "egemaps":     BASE / "features_egemaps"     / "all_features.parquet",
    "compare2016": BASE / "features_compare2016" / "all_features.parquet",
    "emobase":     BASE / "features_emobase"     / "all_features.parquet",
}

OUTPUT_BASE  = Path(__file__).parents[2] / "analysis_outputs" / "step4_classical_boosting"
RIDGE_WAVLM  = Path(__file__).parents[2] / "analysis_outputs" / "step3_voice_age_ridge"
RF_CLASS_DIR = Path(__file__).parents[2] / "analysis_outputs" / "step4_classical_rf"
RIDGE_CLASS  = Path(__file__).parents[2] / "analysis_outputs" / "step4_classical_ridge"

MIN_AGE, MAX_AGE = 40, 70
N_SPLITS         = 5
INNER_VAL_FRAC   = 0.20
N_TRIALS         = 50
SEED             = 42
N_JOBS           = 25  # match SGE slot allocation — do not use -1

# ─────────────────────────── data loading ────────────────────────────────── #

def _load_wavlm() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X, y, gender, subject_number arrays from QC-filtered WavLM CSV."""
    df = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0)
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]
    embed_cols = [c for c in df.columns if c.startswith("feature_")]
    X = SimpleImputer(strategy="median").fit_transform(df[embed_cols].to_numpy(dtype=float))
    print(f"  WavLM: {len(df)} subjects, {len(embed_cols)} features")
    return X, df["age"].to_numpy(dtype=float), df["gender"].to_numpy(), df["subject_number"].to_numpy()


def _load(feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(FEATURE_PARQUETS[feature_set])
    feats.index.name = "filename"

    qc_files = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0, usecols=[0]).index
    feats = feats[feats.index.isin(qc_files)]

    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])
    df = feats.join(sd, how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    feat_cols = [c for c in df.columns if c not in {"age", "gender", "subject_number"}]
    print(f"  Recordings: {len(df)}  Features: {len(feat_cols)}")
    return df, feat_cols


# ─────────────────────────── Optuna objectives ───────────────────────────── #

def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "verbosity": -1, "n_jobs": N_JOBS,
        "objective": "regression", "metric": "mae",
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return float(np.mean(np.abs(y_val - model.predict(X_val))))


def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "verbosity": 0, "n_jobs": N_JOBS, "objective": "reg:absoluteerror",
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma":            trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }
    model = xgb.XGBRegressor(**params, random_state=SEED)
    model.fit(X_tr, y_tr)
    return float(np.mean(np.abs(y_val - model.predict(X_val))))


# ─────────────────────────── CV ──────────────────────────────────────── #

def _run_single_fold(model_name: str, X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, gender: str,
                     fold_idx: int, n_jobs: int) -> dict:
    """Run one fold of GroupKFold CV. Saves y_true/y_pred for later OOF aggregation."""
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    tr_idx, te_idx = splits[fold_idx]
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    rng = np.random.default_rng(SEED + fold_idx)
    n_val    = max(10, int(len(y_tr) * INNER_VAL_FRAC))
    val_idx  = rng.choice(len(y_tr), n_val, replace=False)
    val_mask = np.zeros(len(y_tr), dtype=bool)
    val_mask[val_idx] = True
    X_itr, X_ival = X_tr[~val_mask], X_tr[val_mask]
    y_itr, y_ival = y_tr[~val_mask], y_tr[val_mask]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED + fold_idx),
    )
    study.optimize(
        lambda trial: _lgbm_objective(trial, X_itr, y_itr, X_ival, y_ival),
        n_trials=N_TRIALS, show_progress_bar=False,
    )

    best = study.best_params
    best.update({"verbosity": -1, "n_jobs": n_jobs,
                 "objective": "regression", "metric": "mae"})
    print(f"  fold {fold_idx}: val MAE={study.best_value:.3f}  "
          f"leaves={best.get('num_leaves')}  lr={best.get('learning_rate'):.3f}")

    model = lgb.LGBMRegressor(**best)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    ss_res = np.sum((y_te - pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2) + 1e-10
    r2  = float(1.0 - ss_res / ss_tot)
    r   = float(pearsonr(y_te, pred)[0])
    mae = float(np.mean(np.abs(y_te - pred)))

    print(f"  {gender.upper()} fold {fold_idx}: R²={r2:.4f}  r={r:.4f}  MAE={mae:.3f}")
    return {"fold": fold_idx, "R2": r2, "r": r, "MAE": mae,
            "y_true": y_te.tolist(), "y_pred": pred.tolist()}


def _run_cv(model_name: str, X: np.ndarray, y: np.ndarray,
            groups: np.ndarray, gender: str, n_jobs: int = N_JOBS) -> dict:
    """Run full CV sequentially (used for classical feature sets)."""
    oof_true, oof_pred = [], []
    fold_r2, fold_r, fold_mae = [], [], []

    for fold_idx in range(N_SPLITS):
        res = _run_single_fold(model_name, X, y, groups, gender, fold_idx, n_jobs)
        fold_r2.append(res["R2"])
        fold_r.append(res["r"])
        fold_mae.append(res["MAE"])
        oof_true.extend(res["y_true"])
        oof_pred.extend(res["y_pred"])

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)
    ss_res = np.sum((oof_true - oof_pred) ** 2)
    ss_tot = np.sum((oof_true - oof_true.mean()) ** 2) + 1e-10
    r2  = float(1.0 - ss_res / ss_tot)
    r   = float(pearsonr(oof_true, oof_pred)[0])
    mae = float(np.mean(np.abs(oof_true - oof_pred)))

    r2_std  = float(np.std(fold_r2, ddof=1))
    r_std   = float(np.std(fold_r,  ddof=1))
    mae_std = float(np.std(fold_mae, ddof=1))

    print(f"  {gender.upper()} OOF: R²={r2:.4f}±{r2_std:.4f}  "
          f"r={r:.4f}±{r_std:.4f}  MAE={mae:.3f}±{mae_std:.3f}"
          f"  pred=[{oof_pred.min():.1f}, {oof_pred.max():.1f}]")
    return {"R2": r2, "R2_std": r2_std, "r": r, "r_std": r_std,
            "MAE": mae, "MAE_std": mae_std,
            "pred_min": float(oof_pred.min()), "pred_max": float(oof_pred.max())}


# ─────────────────────────── single job ──────────────────────────────────── #

def run_one(model_name: str, feature_set: str,
            fold_idx: int | None = None, gender_filter: str | None = None,
            n_jobs: int = N_JOBS) -> None:
    label = f"fold={fold_idx}" if fold_idx is not None else "all folds"
    print(f"\n{'='*60}\n{model_name.upper()}  ×  {feature_set.upper()}  {label}\n{'='*60}")

    if feature_set == "wavlm":
        X_all, y_all, g_all, grp = _load_wavlm()
    else:
        df, feat_cols = _load(feature_set)
        imputer = SimpleImputer(strategy="median")
        X_all = imputer.fit_transform(df[feat_cols].to_numpy(dtype=float))
        y_all = df["age"].to_numpy(dtype=float)
        g_all = df["gender"].to_numpy()
        grp   = df["subject_number"].to_numpy()

    out_dir = OUTPUT_BASE / model_name / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    gender_pairs = [(0, "female"), (1, "male")]
    if gender_filter is not None:
        gender_pairs = [(v, l) for v, l in gender_pairs if l == gender_filter]

    for gender_val, gender_label in gender_pairs:
        mask = g_all == gender_val
        print(f"\n  {gender_label.upper()}  (n={mask.sum()})")

        if fold_idx is not None:
            result = _run_single_fold(model_name, X_all[mask], y_all[mask],
                                      grp[mask], gender_label, fold_idx, n_jobs)
            out_file = out_dir / f"fold_{gender_label}_{fold_idx}.json"
            with open(out_file, "w") as f:
                json.dump({"model": model_name, "feature_set": feature_set,
                           "gender": gender_label, **result}, f, indent=2)
        else:
            metrics = _run_cv(model_name, X_all[mask], y_all[mask],
                              grp[mask], gender_label, n_jobs)
            with open(out_dir / f"metrics_{gender_label}_seed{SEED}.json", "w") as f:
                json.dump({"model": model_name, "feature_set": feature_set,
                           "gender": gender_label, "seed": SEED, **metrics}, f, indent=2)


# ─────────────────────────── summary ─────────────────────────────────────── #

def print_summary() -> None:
    rows = []

    for model_name in ("lgbm", "xgboost"):
        for fs in FEATURE_PARQUETS:
            row = {"model": model_name.upper(), "features": fs}
            for g in ("female", "male"):
                p = OUTPUT_BASE / model_name / fs / f"metrics_{g}_seed{SEED}.json"
                if p.exists():
                    m = json.loads(p.read_text())
                    row[f"{g}_R2"]  = round(m["R2"], 4)
                    row[f"{g}_r"]   = round(m["r"], 4)
                    row[f"{g}_MAE"] = round(m["MAE"], 3)
            rows.append(row)

    # RF classical
    for fs in FEATURE_PARQUETS:
        row = {"model": "RF", "features": fs}
        for g in ("female", "male"):
            p = RF_CLASS_DIR / fs / f"gender_{g}" / f"metrics_seed{SEED}.json"
            if p.exists():
                m = json.loads(p.read_text())
                row[f"{g}_R2"]  = round(m["R2"], 4)
                row[f"{g}_r"]   = round(m["r"], 4)
                row[f"{g}_MAE"] = round(m["MAE"], 3)
        rows.append(row)

    # Ridge WavLM
    wavlm = {"model": "Ridge", "features": "WavLM-Large"}
    for g in ("female", "male"):
        p = RIDGE_WAVLM / f"gender_{g}" / "metrics_averaged.json"
        if p.exists():
            m = json.loads(p.read_text())
            wavlm[f"{g}_R2"]  = round(m["averaged_R2"], 4)
            wavlm[f"{g}_r"]   = round(m["averaged_Pearson_r"], 4)
            wavlm[f"{g}_MAE"] = round(m["averaged_MAE"], 3)
    rows.append(wavlm)

    df = pd.DataFrame(rows).set_index(["model", "features"])
    out = OUTPUT_BASE / "summary_boosting_all.csv"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print("\n" + "="*70)
    print("SUMMARY: LightGBM / XGBoost / RF × classical features vs WavLM Ridge")
    print("="*70)
    print(df.to_string())
    print(f"\nSaved → {out}")


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model",              choices=["lgbm", "xgboost"],
                   help="Run a single model × feature-set job")
    g.add_argument("--submit",             action="store_true", help="Launch all 8 classical jobs (legacy)")
    g.add_argument("--submit-wavlm",       action="store_true", help="Launch 10 lgbm×wavlm fold jobs")
    g.add_argument("--submit-classical",   action="store_true", help="Launch lgbm classical fold jobs (per fold×gender)")
    g.add_argument("--merge-wavlm",        action="store_true", help="Aggregate wavlm fold results")
    g.add_argument("--merge-classical",    action="store_true", help="Aggregate all classical fold results")
    g.add_argument("--summary",            action="store_true", help="Print completed results table")
    p.add_argument("--feature-set",  choices=list(FEATURE_PARQUETS) + ["wavlm"])
    p.add_argument("--fold",      type=int, default=None, help="Run single fold (0-indexed)")
    p.add_argument("--gender",    choices=["female", "male"], default=None)
    p.add_argument("--n-jobs",    type=int, default=N_JOBS, help="LightGBM n_jobs")
    p.add_argument("--partition", choices=["himem7.q", "himem8.q"], default="himem8.q")
    args = p.parse_args()

    if args.summary:
        print_summary()
    elif args.submit:
        _submit()
    elif args.submit_wavlm:
        _submit_wavlm(args.partition)
    elif args.submit_classical:
        _submit_classical_folds(args.partition)
    elif args.merge_wavlm:
        merge_wavlm_folds()
    elif args.merge_classical:
        for fs in FEATURE_PARQUETS:
            merge_folds(fs)
    else:
        if not args.feature_set:
            p.error("--feature-set required with --model")
        run_one(args.model, args.feature_set, fold_idx=args.fold,
                gender_filter=args.gender, n_jobs=args.n_jobs)


def _submit() -> None:
    import time as _time
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parents[2]))
    from elysium import ClusterParams, Elysium, JobStatus
    from datetime import datetime

    PYTHON     = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
    PARTITION  = "himem8.q"
    CPU        = 25
    MEM        = "64G"

    script  = Path(__file__).resolve()
    ts      = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / "distribute" / ts

    collections = {}
    for model in ("lgbm", "xgboost"):
        for fs in FEATURE_PARQUETS:
            key = f"{model}_{fs}"
            remote_dir = run_dir / key
            cluster_params = ClusterParams(host_name="mcluster02",
                                           remote_dir=str(remote_dir))
            elysium = Elysium(cluster_params)
            elysium.start()

            cmd = f"{PYTHON} {script} --model {model} --feature-set {fs}"
            col = elysium.run(
                command=cmd, total_jobs=1,
                cpu=CPU, mem=MEM, partition=PARTITION, gpu=0,
                cwd=str(script.parent),
            )
            collections[key] = col
            print(f"Queued {key}  collection_id={col.id}")

    import json as _json
    manifest = run_dir / "collections.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(_json.dumps(
        {k: {"collection_id": v.id} for k, v in collections.items()}, indent=2))

    # Wait for daemon to submit all jobs to SGE
    print("Waiting for SGE submission...", end="", flush=True)
    deadline = _time.time() + 120
    while _time.time() < deadline:
        if all(len(c.get_jobs(JobStatus.PENDING)) == 0 for c in collections.values()):
            break
        print(".", end="", flush=True)
        _time.sleep(2)
    print()

    for key, col in collections.items():
        print(f"  {key}: {dict(col.get_stats())}")

    print(f"\nManifest → {manifest}")
    print(f"When done: python {script.name} --summary")


def merge_folds(feature_set: str, model_name: str = "lgbm") -> dict:
    """Aggregate per-fold JSON files into OOF metrics. Saves metrics_*_seed42.json."""
    out_dir = OUTPUT_BASE / model_name / feature_set
    results = {}
    for gender_label in ("female", "male"):
        folds = []
        for fi in range(N_SPLITS):
            p = out_dir / f"fold_{gender_label}_{fi}.json"
            if not p.exists():
                print(f"  Missing: {p}")
                continue
            folds.append(json.loads(p.read_text()))

        if len(folds) != N_SPLITS:
            print(f"  {gender_label}: only {len(folds)}/{N_SPLITS} folds available — skipping")
            continue

        oof_true = np.array([v for f in folds for v in f["y_true"]])
        oof_pred = np.array([v for f in folds for v in f["y_pred"]])
        fold_r2  = [f["R2"]  for f in folds]
        fold_r   = [f["r"]   for f in folds]
        fold_mae = [f["MAE"] for f in folds]

        ss_res = np.sum((oof_true - oof_pred) ** 2)
        ss_tot = np.sum((oof_true - oof_true.mean()) ** 2) + 1e-10
        r2  = float(1.0 - ss_res / ss_tot)
        r   = float(pearsonr(oof_true, oof_pred)[0])
        mae = float(np.mean(np.abs(oof_true - oof_pred)))

        metrics = {
            "R2": r2,  "R2_std":  float(np.std(fold_r2,  ddof=1)),
            "r":  r,   "r_std":   float(np.std(fold_r,   ddof=1)),
            "MAE": mae, "MAE_std": float(np.std(fold_mae, ddof=1)),
            "pred_min": float(oof_pred.min()), "pred_max": float(oof_pred.max()),
        }
        with open(out_dir / f"metrics_{gender_label}_seed{SEED}.json", "w") as f:
            json.dump({"model": model_name, "feature_set": feature_set,
                       "gender": gender_label, "seed": SEED, **metrics}, f, indent=2)

        print(f"  {feature_set} {gender_label.upper()} OOF: R²={r2:.4f}±{metrics['R2_std']:.4f}  "
              f"r={r:.4f}±{metrics['r_std']:.4f}  MAE={mae:.3f}±{metrics['MAE_std']:.3f}")
        results[gender_label] = metrics
    return results


def merge_wavlm_folds(model_name: str = "lgbm") -> dict:
    return merge_folds("wavlm", model_name)


def _submit_wavlm(partition: str = "himem8.q") -> None:
    import time as _time
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parents[2]))
    from elysium import ClusterParams, Elysium, JobStatus
    from datetime import datetime

    PYTHON    = PYTHON_HIMEM7 if partition == "himem7.q" else PYTHON_HIMEM8
    PARTITION = partition
    CPU       = 10
    MEM       = "32G"
    N_FOLD_JOBS = 10  # threads per fold job

    script  = Path(__file__).resolve()
    ts      = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / "distribute" / f"wavlm_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    collections = {}
    for fold_idx in range(N_SPLITS):
        for gender_label in ("female", "male"):
            key = f"lgbm_wavlm_fold{fold_idx}_{gender_label}"
            cluster_params = ClusterParams(host_name="mcluster02",
                                           remote_dir=str(run_dir / f"fold{fold_idx}_{gender_label}"))
            elysium = Elysium(cluster_params)
            elysium.start()

            cmd = (f"{PYTHON} {script} --model lgbm --feature-set wavlm "
                   f"--fold {fold_idx} --gender {gender_label} --n-jobs {N_FOLD_JOBS}")
            col = elysium.run(
                command=cmd, total_jobs=1,
                cpu=CPU, mem=MEM, partition=PARTITION, gpu=0,
                cwd=str(script.parent),
            )
            collections[key] = col
            print(f"Queued {key}  collection_id={col.id}")

    import json as _json
    (run_dir / "collections.json").write_text(
        _json.dumps({k: {"collection_id": v.id} for k, v in collections.items()}, indent=2))

    print("\nWaiting for SGE submission...", end="", flush=True)
    deadline = _time.time() + 120
    while _time.time() < deadline:
        if all(len(c.get_jobs(JobStatus.PENDING)) == 0 for c in collections.values()):
            break
        print(".", end="", flush=True)
        _time.sleep(2)
    print()
    for key, col in collections.items():
        print(f"  {key}: {dict(col.get_stats())}")
    print(f"\nRun dir → {run_dir}")
    print(f"When done, merge: python {script.name} --merge-wavlm")


def _submit_classical_folds(partition: str = "himem8.q") -> None:
    import time as _time
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parents[2]))
    from elysium import ClusterParams, Elysium, JobStatus
    from datetime import datetime

    PYTHON    = PYTHON_HIMEM7 if partition == "himem7.q" else PYTHON_HIMEM8
    PARTITION = partition
    MEM       = "16G"
    CPU_DEFAULT     = 10
    CPU_COMPARE2016 = 20

    script  = Path(__file__).resolve()
    ts      = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / "distribute" / f"classical_folds_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    collections = {}
    for fs in FEATURE_PARQUETS:
        cpu = CPU_COMPARE2016 if fs == "compare2016" else CPU_DEFAULT
        n_jobs = cpu
        for fold_idx in range(N_SPLITS):
            for gender_label in ("female", "male"):
                key = f"lgbm_{fs}_fold{fold_idx}_{gender_label}"
                cluster_params = ClusterParams(
                    host_name="mcluster02",
                    remote_dir=str(run_dir / fs / f"fold{fold_idx}_{gender_label}"),
                )
                elysium = Elysium(cluster_params)
                elysium.start()

                cmd = (f"{PYTHON} {script} --model lgbm --feature-set {fs} "
                       f"--fold {fold_idx} --gender {gender_label} --n-jobs {n_jobs}")
                col = elysium.run(
                    command=cmd, total_jobs=1,
                    cpu=cpu, mem=MEM, partition=PARTITION, gpu=0,
                    cwd=str(script.parent),
                )
                collections[key] = col
                print(f"Queued {key}  (cpu={cpu})  collection_id={col.id}")
                _time.sleep(1)  # throttle SSH connections

    (run_dir / "collections.json").write_text(
        json.dumps({k: {"collection_id": v.id} for k, v in collections.items()}, indent=2))

    print(f"\nWaiting for SGE submission ({len(collections)} jobs)...", end="", flush=True)
    deadline = _time.time() + 180
    while _time.time() < deadline:
        if all(len(c.get_jobs(JobStatus.PENDING)) == 0 for c in collections.values()):
            break
        print(".", end="", flush=True)
        _time.sleep(3)
    print()

    by_fs: dict[str, dict] = {}
    for key, col in collections.items():
        fs = key.split("_")[1]
        by_fs.setdefault(fs, {})
        stats = dict(col.get_stats())
        for status, count in stats.items():
            by_fs[fs][status] = by_fs[fs].get(status, 0) + count

    for fs, stats in by_fs.items():
        print(f"  {fs}: {stats}")
    print(f"\nRun dir → {run_dir}")
    print(f"When done: python {script.name} --merge-classical")


if __name__ == "__main__":
    main()
