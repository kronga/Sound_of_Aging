"""Legacy fold-alignment sensitivity analysis.

This worker is retained for reproducibility of the earlier comparison only.
It is not the voice-conditioned analysis reported in the final manuscript.
Use ``run_voice_conditioned_holdout.py`` for the leakage-free, calibrated,
intersection-cohort analysis used in Figure 3.

Decodes $JOB_INDEX → (modality, seed) and runs 5-fold nested CV for:
  - baseline:    modality features only
  - conditioned: modality features + voice OOF prediction

Uses the same model type + HPO protocol as the main benchmarking:
  - LightGBM + RandomizedSearchCV (50 trials, 20% inner holdout) for:
      metabolomics, sleep, DEXA, diet, microbiome, lifestyle
  - Ridge + alpha grid (inner holdout) for:
      NMR, retina

Results saved per (modality, seed, sex, condition):
  step4_voice_conditioned_hpo/<modality>/seed_<seed>/gender_<sex>/
    {baseline,conditioned}_metrics.json

Usage (via Elysium):
    python run_voice_conditioned_hpo_worker.py   # JOB_INDEX from env
    python run_voice_conditioned_hpo_worker.py --job-index 5   # for local testing
"""
from __future__ import annotations

import sys
import argparse
import json
import os
import re

# sklearn.metrics.cluster contains a compiled extension (.so) requiring GLIBC_2.23,
# absent on RHEL-7 himem compute nodes. Stub it before sklearn loads it via:
#   sklearn.model_selection → _classification_threshold → sklearn.metrics → cluster
from unittest.mock import MagicMock as _MM
for _m in [
    # sklearn.metrics.cluster — requires GLIBC_2.23 on himem8 (RHEL-7) nodes
    "sklearn.metrics.cluster",
    "sklearn.metrics.cluster._supervised",
    "sklearn.metrics.cluster._unsupervised",
    "sklearn.metrics.cluster._bicluster",
    "sklearn.metrics.cluster._expected_mutual_info_fast",
    # sklearn.neighbors._ball_tree / _kd_tree — requires GLIBC_2.23 on himem7 nodes;
    # triggered via sklearn.impute._knn → sklearn.neighbors._base → _ball_tree/_kd_tree.
    # We only use SimpleImputer (not KNNImputer), so stubbing _knn is safe.
    "sklearn.impute._knn",
    "sklearn.neighbors._ball_tree",
    "sklearn.neighbors._kd_tree",
    "sklearn.neighbors._base",
    "sklearn.neighbors._dist_metrics",
]:
    sys.modules[_m] = _MM()
del _m, _MM

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
except OSError:
    # GLIBC too old for installed LightGBM (requires 2.27); fall back to 3.3.5
    # which only needs 2.14 and is API-compatible for our usage.
    sys.path.insert(0, "/home/davidkro/PycharmProjects/DeepVoice/lgbm_compat")
    import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/age_prediction_new_pipeline/data/"
VOICE_OOF_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step3_voice_age_ridge_one_per_subject"
OUTPUT_BASE    = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_hpo"

# ── Job grid: 8 modalities × 1 seed = 8 jobs (single-seed pilot) ──────────────
MODALITIES = [
    ("NMR",          "X_NMR_age.csv",          "ridge"),
    ("metabolomics", "X_metabolomics_age.csv", "lgbm"),
    ("sleep",        "X_sleep_age.csv",        "lgbm"),
    ("DEXA",         "X_DEXA_age.csv",         "lgbm"),
    ("diet",         "X_diet_age.csv",         "lgbm"),
    ("microbiome",   "X_microbiome_age.csv",   "lgbm"),
    ("lifestyle",    "X_lifestyle_age.csv",    "lgbm"),
    ("retina",       "X_retina_age.csv",       "ridge"),
]
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
TOTAL_JOBS = len(MODALITIES) * len(SEEDS)   # 80

# ── HPO config ─────────────────────────────────────────────────────────────────
N_SPLITS          = 5
N_ITER_SEARCH     = 50
VALIDATION_FRAC   = 0.2
ALPHA_CANDIDATES  = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
N_JOBS            = 9    # leave 1 core for OS; used by both LightGBM and RandomizedSearchCV

MODALITY_DROP = {
    "sleep": ["sat_below_88", "neurokit_hrv_frequency_ulf_during_wake"],
}
MICROBIOME_MIN_PREVALENCE = 0.10
VOICE_COL = "voice_pred"


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_modality(name: str, features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(DATA_DIR, features_csv), index_col=[0, 1])
    if "RegistrationCode" in df.columns:
        df = df.drop(columns=["RegistrationCode"])

    if "age" not in df.columns:
        y_path = os.path.join(DATA_DIR, features_csv.replace("X_", "Y_"))
        if os.path.exists(y_path):
            y = pd.read_csv(y_path, index_col=[0, 1])
            df = df.join(y[["age"]], how="inner")
        else:
            raise FileNotFoundError(f"Y file not found: {y_path}")

    for col in MODALITY_DROP.get(name, []):
        if col in df.columns:
            df = df.drop(columns=[col])

    if name == "NMR":
        df = df.drop(columns=[c for c in df.columns if "_FC" in c or "_pct" in c or ":" in c])

    if name == "microbiome":
        feat_only = df.drop(columns=["age"], errors="ignore")
        prevalence = (feat_only > 0.0001).sum() / len(feat_only)
        keep = prevalence[prevalence >= MICROBIOME_MIN_PREVALENCE].index.tolist()
        df = df[[c for c in keep if c != "age"] + ["age"]]

    df = df.reset_index()
    feature_cols = [c for c in df.columns
                    if c not in ("subject_number", "RegistrationCode", "age",
                                 "gender", "research_stage")]
    return df, feature_cols


def load_voice_oof(seed: int, sex: str) -> pd.DataFrame:
    path = os.path.join(VOICE_OOF_BASE, f"gender_{sex}", f"seed_{seed}", "predictions.csv")
    df = pd.read_csv(path)[["group", "predictions"]].rename(
        columns={"group": "subject_number", "predictions": VOICE_COL}
    )
    df["subject_number"] = df["subject_number"].astype(str)
    return df


# ── CV routines ────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2  = float(r2_score(y_true, y_pred))
    r   = float(pearsonr(y_true, y_pred)[0])
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"oof_R2": r2, "oof_r": r, "oof_MAE": mae, "n": int(len(y_true))}


def _lgbm_cv(df_sex: pd.DataFrame, feat_cols: list[str], seed: int) -> dict:
    """5-fold nested CV with RandomizedSearchCV inner HPO."""
    groups = df_sex["subject_number"].astype(str).values
    imp = SimpleImputer(strategy="median")

    san = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in feat_cols]
    seen: dict[str, int] = {}
    final_san: list[str] = []
    for s in san:
        if s in seen:
            seen[s] += 1
            final_san.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            final_san.append(s)

    X_raw = imp.fit_transform(df_sex[feat_cols].values)
    y     = df_sex["age"].values
    X = pd.DataFrame(X_raw, columns=final_san)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan)

    param_dists = {
        "num_leaves":       randint(20, 150),
        "max_depth":        randint(3, 15),
        "learning_rate":    uniform(0.01, 0.2),
        "n_estimators":     randint(100, 1000),
        "min_child_samples":randint(10, 100),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha":        uniform(0, 1),
        "reg_lambda":       uniform(0, 1),
    }

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te       = X.iloc[test_idx]
        groups_tr  = groups[train_idx]

        # inner holdout split for HPO
        unique_grp = np.unique(groups_tr)
        rng = np.random.default_rng(seed + fold_idx)
        val_grp = set(rng.choice(unique_grp,
                                 size=max(1, int(len(unique_grp) * VALIDATION_FRAC)),
                                 replace=False))
        val_mask = np.array([g in val_grp for g in groups_tr])
        split_arr = np.where(val_mask, 0, -1)
        ps = PredefinedSplit(split_arr)

        est = lgb.LGBMRegressor(objective="regression", metric="rmse",
                                boosting_type="gbdt", verbosity=-1,
                                random_state=seed, n_jobs=N_JOBS)
        rs = RandomizedSearchCV(est, param_dists, n_iter=N_ITER_SEARCH,
                                scoring="r2", cv=ps, n_jobs=1,
                                random_state=seed + fold_idx, verbose=0)
        rs.fit(X_tr, y_tr)

        bp = {**rs.best_params_, "objective": "regression", "metric": "rmse",
              "boosting_type": "gbdt", "verbosity": -1, "seed": seed,
              "num_threads": N_JOBS}
        model = lgb.LGBMRegressor(**bp)
        model.fit(X_tr, y_tr)
        oof_pred[test_idx] = model.predict(X_te)

    mask = ~np.isnan(oof_pred)
    return _metrics(y[mask], oof_pred[mask])


def _ridge_cv(df_sex: pd.DataFrame, feat_cols: list[str], seed: int) -> dict:
    """5-fold nested CV with inner alpha-grid selection."""
    groups = df_sex["subject_number"].astype(str).values
    imp = SimpleImputer(strategy="median")
    X_raw = imp.fit_transform(df_sex[feat_cols].values)
    y     = df_sex["age"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups)):
        X_tr, y_tr = X_raw[train_idx], y[train_idx]
        X_te       = X_raw[test_idx]
        groups_tr  = groups[train_idx]

        unique_grp = np.unique(groups_tr)
        rng = np.random.default_rng(seed + fold_idx)
        val_grp = set(rng.choice(unique_grp,
                                 size=max(1, int(len(unique_grp) * VALIDATION_FRAC)),
                                 replace=False))
        val_mask = np.array([g in val_grp for g in groups_tr])
        X_inner_tr, y_inner_tr = X_tr[~val_mask], y_tr[~val_mask]
        X_inner_val, y_inner_val = X_tr[val_mask], y_tr[val_mask]

        sc = StandardScaler()
        X_inner_tr_s = sc.fit_transform(X_inner_tr)
        X_inner_val_s = sc.transform(X_inner_val)

        best_alpha, best_r2 = ALPHA_CANDIDATES[0], -np.inf
        for alpha in ALPHA_CANDIDATES:
            m = Ridge(alpha=alpha)
            m.fit(X_inner_tr_s, y_inner_tr)
            r2 = r2_score(y_inner_val, m.predict(X_inner_val_s))
            if r2 > best_r2:
                best_r2, best_alpha = r2, alpha

        sc_full = StandardScaler()
        X_tr_s = sc_full.fit_transform(X_tr)
        X_te_s = sc_full.transform(X_te)
        model = Ridge(alpha=best_alpha)
        model.fit(X_tr_s, y_tr)
        oof_pred[test_idx] = model.predict(X_te_s)

    mask = ~np.isnan(oof_pred)
    return _metrics(y[mask], oof_pred[mask])


# ── Main ───────────────────────────────────────────────────────────────────────

def load_voice_metrics(seed: int, sex: str) -> dict:
    path = os.path.join(VOICE_OOF_BASE, f"gender_{sex}", f"seed_{seed}", "metrics.json")
    with open(path) as f:
        d = json.load(f)
    return {"oof_R2": d["oof_R2"], "oof_r": d["oof_Pearson_r"], "oof_MAE": d["oof_MAE"],
            "n": d["n_samples"]}


def run(job_index: int, output_base: str = OUTPUT_BASE) -> None:
    mod_idx  = job_index // len(SEEDS)
    seed_idx = job_index  % len(SEEDS)
    name, feat_csv, model_type = MODALITIES[mod_idx]
    seed = SEEDS[seed_idx]

    print(f"[job {job_index}] modality={name}  seed={seed}  model={model_type}")

    df, feat_cols = load_modality(name, feat_csv)

    for sex in ("female", "male"):
        out_dir = os.path.join(output_base, name, f"seed_{seed}", f"gender_{sex}")
        b_path = os.path.join(out_dir, "baseline_metrics.json")
        c_path = os.path.join(out_dir, "conditioned_metrics.json")
        v_path = os.path.join(out_dir, "voice_metrics.json")
        if os.path.exists(b_path) and os.path.exists(c_path) and os.path.exists(v_path):
            print(f"  [{sex}] already done — skipping")
            continue

        os.makedirs(out_dir, exist_ok=True)

        sex_val = 0 if sex == "female" else 1
        df_sex = df[df["gender"] == sex_val].copy() if "gender" in df.columns else df.copy()
        df_sex["subject_number"] = df_sex["subject_number"].astype(str)

        voice_oof = load_voice_oof(seed, sex)
        df_joined = df_sex.merge(voice_oof, on="subject_number", how="inner")
        print(f"  [{sex}] n={len(df_sex)} → joined={len(df_joined)}")

        cv_fn = _lgbm_cv if model_type == "lgbm" else _ridge_cv

        # Voice-only performance (from step3, on same-sex cohort — reference)
        m_voice = load_voice_metrics(seed, sex)
        with open(v_path, "w") as f:
            json.dump(m_voice, f, indent=2)

        # Baseline: modality only (on joined subset for fair comparison)
        print(f"  [{sex}] baseline ...", flush=True)
        m_base = cv_fn(df_joined, feat_cols, seed)
        with open(b_path, "w") as f:
            json.dump(m_base, f, indent=2)

        # Conditioned: modality + voice
        print(f"  [{sex}] conditioned ...", flush=True)
        m_cond = cv_fn(df_joined, feat_cols + [VOICE_COL], seed)
        with open(c_path, "w") as f:
            json.dump(m_cond, f, indent=2)

        beats_base  = m_cond["oof_R2"] > m_base["oof_R2"]
        beats_voice = m_cond["oof_R2"] > m_voice["oof_R2"]
        print(f"  [{sex}] voice R²={m_voice['oof_R2']:.4f}  "
              f"baseline R²={m_base['oof_R2']:.4f}  "
              f"conditioned R²={m_cond['oof_R2']:.4f}  "
              f"beats_base={beats_base}  beats_voice={beats_voice}")

    print(f"[job {job_index}] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int,
                        default=int(os.environ.get("JOB_INDEX", 0)))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.job_index, output_base=args.output_dir or OUTPUT_BASE)
