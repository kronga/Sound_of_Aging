"""
Random Forest age prediction from classical acoustic feature sets.

GroupKFold (5 folds, grouped by subject_number). Within each training fold,
a held-out validation set (20%) is used for random-search HPO. Final model
trained on the full fold training set with best hyperparameters, evaluated
on the held-out test fold.

Outputs
-------
  paper_revision_outputs/step4_classical_rf/{feature_set}/gender_{female|male}/
  paper_revision_outputs/step4_classical_rf/summary_rf_vs_wavlm.csv

Usage
-----
  python run_classical_rf_age.py --seed 42              # single seed, all 4 sets
  python run_classical_rf_age.py --feature-sets egemaps praat --seed 42
  python run_classical_rf_age.py --smoke                # tiny data, fast HPO
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── config ──────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
WAVLM_FILTERED_CSV  = BASE / "WavLM_features_filtered_with_RF.csv"

FEATURE_PARQUETS: dict[str, Path] = {
    "praat":       BASE / "features_praat"       / "all_features.parquet",
    "egemaps":     BASE / "features_egemaps"     / "all_features.parquet",
    "compare2016": BASE / "features_compare2016" / "all_features.parquet",
    "emobase":     BASE / "features_emobase"     / "all_features.parquet",
}

OUTPUT_BASE = Path(__file__).parents[2] / "paper_revision_outputs" / "step4_classical_rf"
WAVLM_RIDGE = Path(__file__).parents[2] / "paper_revision_outputs" / "step3_voice_age_ridge"

MIN_AGE, MAX_AGE  = 40, 70
N_SPLITS          = 5
INNER_VAL_FRAC    = 0.20
N_ITER_SEARCH     = 30

RF_PARAM_DIST = {
    "n_estimators":    [100, 200, 300, 500],
    "max_depth":       [None, 10, 20, 30],
    "min_samples_leaf":[1, 2, 5, 10, 20],
    "max_features":    ["sqrt", "log2", 0.1, 0.2, 0.3],
}

# ─────────────────────────── data loading ────────────────────────────────── #

def _load_dataset(feature_set: str, smoke: bool) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(FEATURE_PARQUETS[feature_set])
    feats.index.name = "filename"

    qc_files = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0, usecols=[0]).index
    feats = feats[feats.index.isin(qc_files)]

    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])
    df = feats.join(sd, how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

    if "visit_number" in df.columns:
        df = df.sort_values("visit_number").drop_duplicates("subject_number", keep="first")
    else:
        df = df[~df["subject_number"].duplicated(keep="first")]

    if smoke:
        df = df.groupby("gender").head(150).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in {"age", "gender", "subject_number"}]
    return df, feat_cols


# ─────────────────────────── HPO + eval ──────────────────────────────────── #

def _best_rf_params(
    X_tr: np.ndarray, y_tr: np.ndarray,
    rng: np.random.Generator,
    n_iter: int,
) -> dict:
    """Random search on a 20% inner validation split."""
    n_val = max(10, int(len(y_tr) * INNER_VAL_FRAC))
    val_mask = np.zeros(len(y_tr), dtype=bool)
    val_idx  = rng.choice(len(y_tr), n_val, replace=False)
    val_mask[val_idx] = True

    X_itr, X_ival = X_tr[~val_mask], X_tr[val_mask]
    y_itr, y_ival = y_tr[~val_mask], y_tr[val_mask]

    best_r2, best_params = -np.inf, {}
    for params in ParameterSampler(RF_PARAM_DIST, n_iter=n_iter,
                                   random_state=int(rng.integers(1e6))):
        rf = RandomForestRegressor(n_jobs=-1, random_state=0, **params)
        rf.fit(X_itr, y_itr)
        pred = rf.predict(X_ival)
        ss_res = np.sum((y_ival - pred) ** 2)
        ss_tot = np.sum((y_ival - y_ival.mean()) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_params = r2, params

    return best_params


def _run_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    seed: int, n_iter: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_true, oof_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        params = _best_rf_params(X_tr, y_tr, rng, n_iter)
        print(f"    fold {fold+1}: best={params}")

        rf = RandomForestRegressor(n_jobs=-1, random_state=seed, **params)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)

        oof_true.extend(y_te.tolist())
        oof_pred.extend(pred.tolist())

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)

    ss_res = np.sum((oof_true - oof_pred) ** 2)
    ss_tot = np.sum((oof_true - oof_true.mean()) ** 2) + 1e-10
    r2  = float(1.0 - ss_res / ss_tot)
    r   = float(pearsonr(oof_true, oof_pred)[0])
    mae = float(np.mean(np.abs(oof_true - oof_pred)))

    print(f"    OOF: R²={r2:.4f}  r={r:.4f}  MAE={mae:.3f}"
          f"  pred=[{oof_pred.min():.1f}, {oof_pred.max():.1f}]")
    return {"R2": r2, "r": r, "MAE": mae,
            "pred_min": float(oof_pred.min()), "pred_max": float(oof_pred.max())}


# ─────────────────────────── per feature set ─────────────────────────────── #

def run_one(feature_set: str, seed: int, smoke: bool) -> dict:
    print(f"\n{'='*60}\n{feature_set.upper()}  seed={seed}\n{'='*60}")
    df, feat_cols = _load_dataset(feature_set, smoke)
    print(f"  Recordings: {len(df)}  |  Features: {len(feat_cols)}")

    out_dir = OUTPUT_BASE / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    n_iter = 5 if smoke else N_ITER_SEARCH
    row = {"modality": feature_set}

    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val]
        X = sub[feat_cols].to_numpy(dtype=float)
        y = sub["age"].to_numpy(dtype=float)
        groups = sub["subject_number"].to_numpy()

        print(f"\n  {gender_label.upper()}  (n={len(sub)})")
        metrics = _run_cv(X, y, groups, seed=seed, n_iter=n_iter)

        row[f"{gender_label}_R2"]       = round(metrics["R2"], 4)
        row[f"{gender_label}_r"]        = round(metrics["r"], 4)
        row[f"{gender_label}_MAE"]      = round(metrics["MAE"], 3)
        row[f"{gender_label}_pred_min"] = round(metrics["pred_min"], 1)
        row[f"{gender_label}_pred_max"] = round(metrics["pred_max"], 1)

        (out_dir / f"gender_{gender_label}").mkdir(exist_ok=True)
        with open(out_dir / f"gender_{gender_label}" / f"metrics_seed{seed}.json", "w") as f:
            json.dump({"seed": seed, **metrics}, f, indent=2)

    return row


# ─────────────────────────── summary ─────────────────────────────────────── #

def _build_summary(rows: list[dict]) -> pd.DataFrame:
    wavlm = {"modality": "WavLM-Large"}
    for g in ("female", "male"):
        p = WAVLM_RIDGE / f"gender_{g}" / "metrics_averaged.json"
        if p.exists():
            m = json.loads(p.read_text())
            wavlm[f"{g}_R2"]  = round(m["averaged_R2"], 4)
            wavlm[f"{g}_r"]   = round(m["averaged_Pearson_r"], 4)
            wavlm[f"{g}_MAE"] = round(m["averaged_MAE"], 3)
    rows.append(wavlm)

    df = pd.DataFrame(rows).set_index("modality")
    out = OUTPUT_BASE / "summary_rf_vs_wavlm.csv"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print("\n" + "="*60)
    print("RF vs WavLM-Large")
    print("="*60)
    print(df.to_string())
    print(f"\nSaved → {out}")
    return df


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-sets", nargs="+", default=list(FEATURE_PARQUETS),
                   choices=list(FEATURE_PARQUETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    rows = []
    for fs in args.feature_sets:
        rows.append(run_one(fs, seed=args.seed, smoke=args.smoke))

    _build_summary(rows)


if __name__ == "__main__":
    main()
