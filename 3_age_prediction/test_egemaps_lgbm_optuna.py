"""
Quick test: eGeMAPSv02 age prediction with LightGBM + Optuna HPO.

GroupKFold (5 folds, grouped by subject_number). Within each training fold,
Optuna optimizes LightGBM hyperparameters (50 trials) on an inner 20%
validation split. Best model evaluated on the held-out test fold.

Usage
-----
  python test_egemaps_lgbm_optuna.py
  python test_egemaps_lgbm_optuna.py --seed 1
"""
from __future__ import annotations

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─────────────────────────── config ──────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
EGEMAPS_PARQUET    = BASE / "features_egemaps" / "all_features.parquet"
SUBJECT_CSV        = BASE / "subject_details_df_Oct25.csv"
WAVLM_FILTERED_CSV = BASE / "WavLM_features_filtered_with_RF.csv"

MIN_AGE, MAX_AGE = 40, 70
N_SPLITS         = 5
INNER_VAL_FRAC   = 0.20
N_TRIALS         = 50

# ─────────────────────────── data ────────────────────────────────────────── #

def load_data() -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(EGEMAPS_PARQUET)
    feats.index.name = "filename"

    qc_files = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0, usecols=[0]).index
    feats = feats[feats.index.isin(qc_files)]

    sd = pd.read_csv(SUBJECT_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])
    df = feats.join(sd, how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    feat_cols = [c for c in df.columns if c not in {"age", "gender", "subject_number"}]
    print(f"Loaded: {len(df)} subjects, {len(feat_cols)} features")
    return df, feat_cols


# ─────────────────────────── optuna objective ────────────────────────────── #

def _make_objective(X_tr, y_tr, val_mask):
    X_itr, y_itr = X_tr[~val_mask], y_tr[~val_mask]
    X_ival, y_ival = X_tr[val_mask],  y_tr[val_mask]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity":       -1,
            "n_jobs":          -1,
            "objective":       "regression",
            "metric":          "mae",
            "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_itr, y_itr)
        pred = model.predict(X_ival)
        return float(np.mean(np.abs(y_ival - pred)))  # minimize MAE

    return objective


# ─────────────────────────── CV ──────────────────────────────────────────── #

def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           seed: int, gender: str) -> dict:
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_true, oof_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Inner validation mask
        n_val = max(10, int(len(y_tr) * INNER_VAL_FRAC))
        val_idx = rng.choice(len(y_tr), n_val, replace=False)
        val_mask = np.zeros(len(y_tr), dtype=bool)
        val_mask[val_idx] = True

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed + fold))
        study.optimize(_make_objective(X_tr, y_tr, val_mask),
                       n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        best.update({"verbosity": -1, "n_jobs": -1,
                     "objective": "regression", "metric": "mae"})
        print(f"    fold {fold+1}: best MAE={study.best_value:.3f}  "
              f"leaves={best.get('num_leaves')}  lr={best.get('learning_rate'):.3f}  "
              f"n_est={best.get('n_estimators')}")

        model = lgb.LGBMRegressor(**best)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        oof_true.extend(y_te.tolist())
        oof_pred.extend(pred.tolist())

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)

    ss_res = np.sum((oof_true - oof_pred) ** 2)
    ss_tot = np.sum((oof_true - oof_true.mean()) ** 2) + 1e-10
    r2  = float(1.0 - ss_res / ss_tot)
    r   = float(pearsonr(oof_true, oof_pred)[0])
    mae = float(np.mean(np.abs(oof_true - oof_pred)))

    print(f"  {gender.upper()} OOF: R²={r2:.4f}  r={r:.4f}  MAE={mae:.3f}"
          f"  pred=[{oof_pred.min():.1f}, {oof_pred.max():.1f}]")
    return {"R2": r2, "r": r, "MAE": mae,
            "pred_min": float(oof_pred.min()), "pred_max": float(oof_pred.max())}


# ─────────────────────────── main ────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df, feat_cols = load_data()

    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(df[feat_cols].to_numpy(dtype=float))
    y_all = df["age"].to_numpy(dtype=float)
    g_all = df["gender"].to_numpy()
    grp   = df["subject_number"].to_numpy()

    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        mask = g_all == gender_val
        print(f"\n{'='*60}\n{gender_label.upper()}  (n={mask.sum()})  seed={args.seed}\n{'='*60}")
        run_cv(X_all[mask], y_all[mask], grp[mask],
               seed=args.seed, gender=gender_label)


if __name__ == "__main__":
    main()
