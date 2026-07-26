"""
Voice-conditioned age prediction: stratified age holdout.

Fixes the data-leakage in run_voice_conditioned_hpo_worker.py where misaligned
fold splits between step-3 (voice) and step-4 (modality) caused training subjects'
voice predictions to be generated using test subjects' age labels.

Clean protocol (per modality, seed, sex):
  1. Merge voice + modality data on subject_number (inner join), one per subject
  2. Stratified 80/20 train/test split by age (5 quantile bins, at subject level)
  3. Train voice Ridge on TRAIN subjects only (alpha-grid HPO, 20% inner holdout)
     voice_pred(train) = in-sample;  voice_pred(test) = out-of-sample  → no leakage
  4. Baseline:    modality features only  → train → eval on test
  5. Conditioned: modality + voice_pred  → train → eval on test
  Model types (same as HPO-CV approach):
    LightGBM + Optuna 50 trials: metabolomics, sleep, DEXA, diet, microbiome, lifestyle
    Ridge + alpha-grid HPO:       NMR, retina

Results: analysis_outputs/step4_voice_conditioned_holdout/
Suffix:  _holdout  (so old CV results are not overwritten)

Full-pool-voice protocol (--full-pool-voice):
  Voice model for TEST prediction is trained on the full voice pool minus the
  holdout test subjects.  This prevents test-subject age labels from ever being
  seen during training, regardless of how the outer split was drawn.
  Voice OOF for TRAIN subjects is still generated within the inner-join train
  split only (GroupKFold when --oof-train, in-sample otherwise).

Usage:
    python run_voice_conditioned_holdout.py                                # all modalities, all seeds
    python run_voice_conditioned_holdout.py --modality NMR                 # single modality
    python run_voice_conditioned_holdout.py --seed 42                      # single seed
    python run_voice_conditioned_holdout.py --full-pool-voice --oof-train  # strict leakage-free
    python run_voice_conditioned_holdout.py --plot-only                    # skip training, just plot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings

# Stub incompatible sklearn extensions before they are imported.
# sklearn.metrics.cluster and sklearn.neighbors require GLIBC_2.23 which is
# absent on RHEL-7 himem nodes. We only use SimpleImputer (not KNNImputer),
# so stubbing _knn and the cluster module is safe.
from unittest.mock import MagicMock as _MM
for _m in [
    "sklearn.metrics.cluster",
    "sklearn.metrics.cluster._supervised",
    "sklearn.metrics.cluster._unsupervised",
    "sklearn.metrics.cluster._bicluster",
    "sklearn.metrics.cluster._expected_mutual_info_fast",
    "sklearn.impute._knn",
    "sklearn.neighbors._ball_tree",
    "sklearn.neighbors._kd_tree",
    "sklearn.neighbors._base",
    "sklearn.neighbors._dist_metrics",
]:
    sys.modules[_m] = _MM()
del _m, _MM

import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except OSError:
    sys.path.insert(0, "/home/davidkro/PycharmProjects/DeepVoice/lgbm_compat")
    import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/age_prediction_new_pipeline/data/"
WAVLM_CSV       = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/WavLM_features_filtered_with_RF.csv"
OUTPUT_BASE          = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_holdout"
OOF_OUTPUT_BASE      = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_holdout_oof"
FULLPOOL_OUTPUT_BASE     = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_holdout_fullpool"
FULLPOOL_OOF_OUTPUT_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_holdout_fullpool_oof"
FULLPOOL_OOF_CAL_OUTPUT_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_holdout_fullpool_oof_cal"


def _get_output_base(oof_train: bool, full_pool_voice: bool,
                     calibrate_voice: bool = False) -> str:
    if full_pool_voice and oof_train and calibrate_voice:
        return FULLPOOL_OOF_CAL_OUTPUT_BASE
    if full_pool_voice and oof_train:
        return FULLPOOL_OOF_OUTPUT_BASE
    if full_pool_voice:
        return FULLPOOL_OUTPUT_BASE
    if oof_train:
        return OOF_OUTPUT_BASE
    return OUTPUT_BASE
OLD_OUTPUT_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_hpo"

# ── Config ────────────────────────────────────────────────────────────────────
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
SEEDS                    = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
TOTAL_JOBS               = len(MODALITIES) * len(SEEDS)   # 80
N_JOBS                   = 9   # CPU threads for LightGBM (leave 1 for OS)
TEST_FRAC                = 0.20
N_AGE_BINS               = 5
INNER_VAL_FRAC           = 0.20
N_TRIALS                 = 50
ALPHA_CANDIDATES         = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
MIN_AGE, MAX_AGE         = 40, 70
VOICE_COL                = "voice_pred"
MODALITY_DROP            = {"sleep": ["sat_below_88", "neurokit_hrv_frequency_ulf_during_wake"]}
MICROBIOME_MIN_PREVALENCE = 0.10


# ── Data helpers ───────────────────────────────────────────────────────────────

def _stage_rank(stage) -> tuple:
    s = str(stage)
    if s == "baseline":
        return (-1, s)
    try:
        return (int(s.split("_", 1)[0]), s)
    except Exception:
        return (-2, s)


def _keep_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject_number, latest research_stage wins."""
    if "research_stage" not in df.columns:
        return df.drop_duplicates(subset=["subject_number"], keep="last").reset_index(drop=True)
    df = df.copy()
    df["_rank"] = df["research_stage"].map(_stage_rank)
    return (df.sort_values(["subject_number", "_rank"])
              .drop_duplicates(subset=["subject_number"], keep="last")
              .drop(columns=["_rank"])
              .reset_index(drop=True))


def load_wavlm() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(WAVLM_CSV, index_col=0)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df["subject_number"] = df["subject_number"].astype(str)
    df = _keep_latest(df)
    return df, feat_cols


def load_modality(name: str, features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(DATA_DIR, features_csv), index_col=[0, 1])
    if "RegistrationCode" in df.columns:
        df = df.drop(columns=["RegistrationCode"])
    if "age" not in df.columns:
        y_path = os.path.join(DATA_DIR, features_csv.replace("X_", "Y_"))
        y = pd.read_csv(y_path, index_col=[0, 1])
        df = df.join(y[["age"]], how="inner")
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
    df["subject_number"] = df["subject_number"].astype(str)
    df = _keep_latest(df)
    feat_cols = [c for c in df.columns
                 if c not in ("subject_number", "RegistrationCode", "age",
                               "gender", "research_stage")]
    return df, feat_cols


# ── Stratified split ───────────────────────────────────────────────────────────

def stratified_subject_split(df: pd.DataFrame, seed: int) -> tuple[set, set]:
    """80/20 split at subject level, stratified by age quantile bins."""
    subj_ages = df.groupby("subject_number")["age"].first()
    age_bins = pd.qcut(subj_ages, q=N_AGE_BINS, labels=False, duplicates="drop")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr_idx, te_idx = next(sss.split(subj_ages.index, age_bins))
    return set(subj_ages.index[tr_idx]), set(subj_ages.index[te_idx])


# ── Voice Ridge ────────────────────────────────────────────────────────────────

def train_voice_ridge(X_tr: np.ndarray, y_tr: np.ndarray,
                      groups_tr: np.ndarray, seed: int):
    """Fit voice Ridge with alpha HPO on subject-level inner holdout."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_tr)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_imp)

    unique_grp = np.unique(groups_tr)
    rng = np.random.default_rng(seed)
    val_grp = set(rng.choice(unique_grp,
                             size=max(1, int(len(unique_grp) * INNER_VAL_FRAC)),
                             replace=False))
    val_mask = np.array([g in val_grp for g in groups_tr])

    best_alpha, best_r2 = ALPHA_CANDIDATES[0], -np.inf
    for alpha in ALPHA_CANDIDATES:
        m = Ridge(alpha=alpha)
        m.fit(X_sc[~val_mask], y_tr[~val_mask])
        r2 = r2_score(y_tr[val_mask], m.predict(X_sc[val_mask]))
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    model = Ridge(alpha=best_alpha)
    model.fit(X_sc, y_tr)
    return imp, sc, model


def predict_voice(imp, sc, model, X: np.ndarray) -> np.ndarray:
    return model.predict(sc.transform(imp.transform(X)))


VOICE_INNER_SPLITS = 5

def voice_oof_for_train(X_voice_tr: np.ndarray, y_tr: np.ndarray,
                        groups_tr: np.ndarray, seed: int) -> np.ndarray:
    """Inner GroupKFold to generate OOF voice predictions for training subjects.

    Ensures voice_pred_train has the same noise level as voice_pred_test, so
    the conditioned model doesn't over-rely on artificially accurate voice signal.
    """
    gkf = GroupKFold(n_splits=VOICE_INNER_SPLITS)
    oof = np.full(len(y_tr), np.nan)
    for fold_i, (inn_tr, inn_va) in enumerate(gkf.split(X_voice_tr, y_tr, groups_tr)):
        v_imp, v_sc, v_model = train_voice_ridge(
            X_voice_tr[inn_tr], y_tr[inn_tr], groups_tr[inn_tr], seed + fold_i)
        oof[inn_va] = predict_voice(v_imp, v_sc, v_model, X_voice_tr[inn_va])
    return oof


# ── Modality Ridge HPO ────────────────────────────────────────────────────────

def train_ridge_hpo(X_tr: np.ndarray, y_tr: np.ndarray,
                    groups_tr: np.ndarray, seed: int):
    """Fit Ridge with alpha HPO on subject-level inner holdout."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_tr)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_imp)

    unique_grp = np.unique(groups_tr)
    rng = np.random.default_rng(seed)
    val_grp = set(rng.choice(unique_grp,
                             size=max(1, int(len(unique_grp) * INNER_VAL_FRAC)),
                             replace=False))
    val_mask = np.array([g in val_grp for g in groups_tr])

    best_alpha, best_r2 = ALPHA_CANDIDATES[0], -np.inf
    for alpha in ALPHA_CANDIDATES:
        m = Ridge(alpha=alpha)
        m.fit(X_sc[~val_mask], y_tr[~val_mask])
        r2 = r2_score(y_tr[val_mask], m.predict(X_sc[val_mask]))
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    model = Ridge(alpha=best_alpha)
    model.fit(X_sc, y_tr)
    return imp, sc, model


def predict_ridge(imp, sc, model, X: np.ndarray) -> np.ndarray:
    return model.predict(sc.transform(imp.transform(X)))


# ── LightGBM + Optuna ─────────────────────────────────────────────────────────

def _optuna_objective(X_itr, y_itr, X_ival, y_ival):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": -1, "n_jobs": N_JOBS,
            "objective": "regression", "metric": "mae",
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_itr, y_itr)
        return float(np.mean(np.abs(y_ival - m.predict(X_ival))))
    return objective


def train_lgbm_optuna(X_tr: np.ndarray, y_tr: np.ndarray,
                      groups_tr: np.ndarray, seed: int):
    """Fit LightGBM with Optuna HPO on subject-level inner holdout."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_tr)

    unique_grp = np.unique(groups_tr)
    rng = np.random.default_rng(seed)
    val_grp = set(rng.choice(unique_grp,
                             size=max(1, int(len(unique_grp) * INNER_VAL_FRAC)),
                             replace=False))
    val_mask = np.array([g in val_grp for g in groups_tr])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        _optuna_objective(X_imp[~val_mask], y_tr[~val_mask],
                          X_imp[val_mask],  y_tr[val_mask]),
        n_trials=N_TRIALS, show_progress_bar=False,
    )

    best_params = {**study.best_params,
                   "verbosity": -1, "n_jobs": N_JOBS,
                   "objective": "regression", "metric": "mae"}
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_imp, y_tr)
    return imp, model


def predict_lgbm(imp, model, X: np.ndarray) -> np.ndarray:
    return model.predict(imp.transform(X))


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2  = float(r2_score(y_true, y_pred))
    r   = float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan")
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"test_R2": r2, "test_r": r, "test_MAE": mae, "n": int(len(y_true))}


# ── Main pair runner ───────────────────────────────────────────────────────────

def run_pair(mod_name: str, feat_csv: str, model_type: str, seed: int,
             voice_df: pd.DataFrame, voice_feat_cols: list[str],
             oof_train: bool = False, full_pool_voice: bool = False,
             calibrate_voice: bool = False, force: bool = False) -> None:

    df_mod, feat_cols = load_modality(mod_name, feat_csv)
    base = _get_output_base(oof_train, full_pool_voice, calibrate_voice)

    for sex_val, sex_label in [(0, "female"), (1, "male")]:
        out_dir = os.path.join(base, mod_name, f"seed_{seed}", f"gender_{sex_label}")
        b_path = os.path.join(out_dir, "baseline_metrics.json")
        c_path = os.path.join(out_dir, "conditioned_metrics.json")
        v_path = os.path.join(out_dir, "voice_metrics.json")

        if not force and all(os.path.exists(p) for p in [b_path, c_path, v_path]):
            print(f"  [{sex_label}] already done — skipping")
            continue

        os.makedirs(out_dir, exist_ok=True)

        # ── filter by sex ─────────────────────────────────────────────────────
        df_sex_mod   = df_mod[df_mod["gender"] == sex_val].copy() \
                       if "gender" in df_mod.columns else df_mod.copy()
        df_sex_voice = voice_df[voice_df["gender"] == sex_val].copy()

        # ── merge: inner join on subject_number ───────────────────────────────
        voice_for_merge = df_sex_voice[["subject_number"] + voice_feat_cols]
        df_joined = df_sex_mod.merge(voice_for_merge, on="subject_number", how="inner")

        print(f"  [{sex_label}] modality n={len(df_sex_mod)}  "
              f"voice n={len(df_sex_voice)}  joined n={len(df_joined)}")

        if len(df_joined) < 30:
            print(f"  [{sex_label}] too few subjects — skipping")
            continue

        # ── stratified train/test split at subject level ───────────────────────
        train_subs, test_subs = stratified_subject_split(df_joined, seed)
        df_tr = df_joined[df_joined["subject_number"].isin(train_subs)].reset_index(drop=True)
        df_te = df_joined[df_joined["subject_number"].isin(test_subs)].reset_index(drop=True)

        y_tr  = df_tr["age"].values
        y_te  = df_te["age"].values
        grp_tr = df_tr["subject_number"].values

        X_voice_tr = df_tr[voice_feat_cols].values
        X_voice_te = df_te[voice_feat_cols].values
        X_mod_tr   = df_tr[feat_cols].values
        X_mod_te   = df_te[feat_cols].values

        print(f"    train={len(df_tr)}  test={len(df_te)}", flush=True)

        # ── voice predictions for TRAIN subjects ──────────────────────────────
        if oof_train:
            # GroupKFold OOF within inner-join train split: same noise as test
            voice_pred_tr = voice_oof_for_train(X_voice_tr, y_tr, grp_tr, seed)
        else:
            # in-sample (conservative ΔR²): train on inner-join train split only
            v_imp_tr, v_sc_tr, v_model_tr = train_voice_ridge(X_voice_tr, y_tr, grp_tr, seed)
            voice_pred_tr = predict_voice(v_imp_tr, v_sc_tr, v_model_tr, X_voice_tr)

        # ── voice predictions for TEST subjects ────────────────────────────────
        if full_pool_voice:
            # train on full voice pool minus test subjects → test ages never seen
            df_pool = df_sex_voice[~df_sex_voice["subject_number"].isin(test_subs)].reset_index(drop=True)
            print(f"    full-pool voice training: n={len(df_pool)} (vs inner-join train n={len(df_tr)})", flush=True)
            v_imp_te, v_sc_te, v_model_te = train_voice_ridge(
                df_pool[voice_feat_cols].values,
                df_pool["age"].values,
                df_pool["subject_number"].values,
                seed,
            )
        else:
            # original: train on inner-join train split only
            v_imp_te, v_sc_te, v_model_te = train_voice_ridge(X_voice_tr, y_tr, grp_tr, seed)
        voice_pred_te = predict_voice(v_imp_te, v_sc_te, v_model_te, X_voice_te)

        # ── linear calibration on train subjects (leak-free) ──────────────────
        # Fits age_train ~ a*voice_pred_train + b using train subjects only,
        # then applies the same rescaling to test predictions.
        # Fixes subpopulation mean/scale shift without touching test ages.
        if calibrate_voice and np.std(voice_pred_tr) > 0:
            a_cal, b_cal = np.polyfit(voice_pred_tr, y_tr, deg=1)
            voice_pred_te = a_cal * voice_pred_te + b_cal
            voice_pred_tr = a_cal * voice_pred_tr + b_cal

        voice_metrics = _metrics(y_te, voice_pred_te)

        # ── baseline: modality only ────────────────────────────────────────────
        print(f"    [{sex_label}] baseline ({model_type}) ...", flush=True)
        if model_type == "ridge":
            b_imp, b_sc, b_model = train_ridge_hpo(X_mod_tr, y_tr, grp_tr, seed)
            base_pred_te = predict_ridge(b_imp, b_sc, b_model, X_mod_te)
        else:
            b_imp, b_model = train_lgbm_optuna(X_mod_tr, y_tr, grp_tr, seed)
            base_pred_te = predict_lgbm(b_imp, b_model, X_mod_te)
        base_metrics = _metrics(y_te, base_pred_te)

        # ── conditioned: modality + voice_pred ────────────────────────────────
        print(f"    [{sex_label}] conditioned ({model_type}) ...", flush=True)
        X_cond_tr = np.column_stack([X_mod_tr, voice_pred_tr])
        X_cond_te = np.column_stack([X_mod_te, voice_pred_te])

        if model_type == "ridge":
            c_imp, c_sc, c_model = train_ridge_hpo(X_cond_tr, y_tr, grp_tr, seed)
            cond_pred_te = predict_ridge(c_imp, c_sc, c_model, X_cond_te)
        else:
            c_imp, c_model = train_lgbm_optuna(X_cond_tr, y_tr, grp_tr, seed)
            cond_pred_te = predict_lgbm(c_imp, c_model, X_cond_te)
        cond_metrics = _metrics(y_te, cond_pred_te)

        delta = cond_metrics["test_R2"] - base_metrics["test_R2"]
        print(f"    voice R²={voice_metrics['test_R2']:.4f}  "
              f"baseline R²={base_metrics['test_R2']:.4f}  "
              f"conditioned R²={cond_metrics['test_R2']:.4f}  "
              f"ΔR²={delta:+.4f}")

        with open(v_path, "w") as f:
            json.dump(voice_metrics, f, indent=2)
        with open(b_path, "w") as f:
            json.dump(base_metrics, f, indent=2)
        with open(c_path, "w") as f:
            json.dump(cond_metrics, f, indent=2)


# ── Aggregation ────────────────────────────────────────────────────────────────

def _load_results(base_dir: str, r2_key: str) -> pd.DataFrame:
    rows = []
    for mod_name, _, _ in MODALITIES:
        mod_dir = os.path.join(base_dir, mod_name)
        if not os.path.isdir(mod_dir):
            continue
        for seed in SEEDS:
            seed_dir = os.path.join(mod_dir, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            for sex_label in ("female", "male"):
                sex_dir = os.path.join(seed_dir, f"gender_{sex_label}")
                b_path = os.path.join(sex_dir, "baseline_metrics.json")
                c_path = os.path.join(sex_dir, "conditioned_metrics.json")
                v_path = os.path.join(sex_dir, "voice_metrics.json")
                if not all(os.path.exists(p) for p in [b_path, c_path, v_path]):
                    continue
                with open(b_path) as f:
                    bm = json.load(f)
                with open(c_path) as f:
                    cm = json.load(f)
                with open(v_path) as f:
                    vm = json.load(f)
                rows.append({
                    "modality":    mod_name,
                    "seed":        seed,
                    "sex":         sex_label,
                    "baseline_R2": bm[r2_key],
                    "cond_R2":     cm[r2_key],
                    "voice_R2":    vm[r2_key],
                    "delta_R2":    cm[r2_key] - bm[r2_key],
                })
    return pd.DataFrame(rows)


def aggregate_and_save(oof_train: bool = False, full_pool_voice: bool = False,
                       calibrate_voice: bool = False) -> pd.DataFrame:
    base = _get_output_base(oof_train, full_pool_voice, calibrate_voice)
    parts = (["_oof"] if oof_train else []) + (["_fullpool"] if full_pool_voice else []) + (["_cal"] if calibrate_voice else [])
    suffix = "".join(parts)
    df = _load_results(base, "test_R2")
    if df.empty:
        print(f"No holdout{suffix} results found yet.")
        return df

    summary = (df.groupby(["modality", "sex"])
                 .agg(
                     baseline_R2_mean=("baseline_R2", "mean"),
                     baseline_R2_std=("baseline_R2", "std"),
                     cond_R2_mean=("cond_R2", "mean"),
                     cond_R2_std=("cond_R2", "std"),
                     voice_R2_mean=("voice_R2", "mean"),
                     voice_R2_std=("voice_R2", "std"),
                     delta_R2_mean=("delta_R2", "mean"),
                     delta_R2_std=("delta_R2", "std"),
                     n_seeds=("seed", "count"),
                 ).reset_index())
    out_csv = os.path.join(base, f"voice_conditioned_holdout{suffix}_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary → {out_csv}")
    print(summary[["modality", "sex", "baseline_R2_mean", "cond_R2_mean",
                   "delta_R2_mean", "delta_R2_std", "n_seeds"]].to_string(index=False))
    return summary


# ── Comparison plot ────────────────────────────────────────────────────────────

MOD_ORDER = ["NMR", "retina", "metabolomics", "DEXA", "sleep",
             "diet", "microbiome", "lifestyle"]


def plot_comparison() -> None:
    df_old      = _load_results(OLD_OUTPUT_BASE,         "oof_R2")
    df_new      = _load_results(OUTPUT_BASE,             "test_R2")
    df_oof      = _load_results(OOF_OUTPUT_BASE,         "test_R2")
    df_fp_oof   = _load_results(FULLPOOL_OOF_OUTPUT_BASE,"test_R2")

    if df_old.empty and df_new.empty and df_oof.empty and df_fp_oof.empty:
        print("No results to plot.")
        return

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return (df.groupby(["modality", "sex"])["delta_R2"]
                  .agg(mean="mean", std="std")
                  .reset_index())

    old_agg    = _agg(df_old)
    new_agg    = _agg(df_new)
    oof_agg    = _agg(df_oof)
    fp_oof_agg = _agg(df_fp_oof)

    approaches = [
        ("HPO-CV (leaky)",              old_agg,    "#FF7043"),
        ("Holdout in-sample voice",     new_agg,    "#2196F3"),
        ("Holdout OOF voice",           oof_agg,    "#4CAF50"),
        ("Holdout full-pool OOF voice", fp_oof_agg, "#9C27B0"),
    ]
    n_bars = len(approaches)
    bar_w  = 0.18
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w
    x = np.arange(len(MOD_ORDER))

    # ── ΔR² comparison ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    for ax, sex in zip(axes, ("female", "male")):
        for (label, agg_df, color), offset in zip(approaches, offsets):
            if agg_df.empty:
                continue
            sub   = agg_df[agg_df["sex"] == sex].set_index("modality")
            means = [sub.loc[m, "mean"] if m in sub.index else np.nan for m in MOD_ORDER]
            stds  = [sub.loc[m, "std"]  if m in sub.index else 0.0   for m in MOD_ORDER]
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
                   label=label, color=color, alpha=0.85, error_kw={"linewidth": 1.0})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(MOD_ORDER, rotation=35, ha="right", fontsize=9)
        ax.set_title(sex.capitalize(), fontsize=11)
        ax.set_ylabel("ΔR² (conditioned − baseline)" if sex == "female" else "")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Voice conditioning ΔR²: leaky CV vs. holdout variants vs. full-pool OOF",
                 fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_BASE, "comparison_all_approaches.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out_path}")

    # ── absolute R² ───────────────────────────────────────────────────────────
    active = [(lbl, df_src) for lbl, df_src in [
        ("HPO-CV (leaky)",              df_old),
        ("Holdout in-sample voice",     df_new),
        ("Holdout OOF voice",           df_oof),
        ("Holdout full-pool OOF voice", df_fp_oof),
    ] if not df_src.empty]

    if not active:
        return

    n_rows = len(active)
    fig2, axes2 = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), sharey=False)
    if n_rows == 1:
        axes2 = [axes2]

    for row_axes, (label, df_src) in zip(axes2, active):
        for ax, sex in zip(row_axes, ("female", "male")):
            sub = (df_src[df_src["sex"] == sex]
                   .groupby("modality")[["baseline_R2", "cond_R2", "voice_R2"]]
                   .mean().reset_index()
                   .set_index("modality"))
            bw = 0.25
            x3 = np.arange(len(MOD_ORDER))
            for k, (col_key, clr, lbl) in enumerate([
                ("baseline_R2", "#78909C", "Baseline"),
                ("cond_R2",     "#2196F3", "Conditioned"),
                ("voice_R2",    "#4CAF50", "Voice only"),
            ]):
                vals = [sub.loc[m, col_key] if m in sub.index else np.nan for m in MOD_ORDER]
                ax.bar(x3 + (k - 1) * bw, vals, bw, label=lbl, color=clr, alpha=0.85)
            ax.set_xticks(x3)
            ax.set_xticklabels(MOD_ORDER, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{label} — {sex.capitalize()}", fontsize=10)
            ax.set_ylabel("R²")
            ax.legend(fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(axis="y", alpha=0.3)

    fig2.suptitle("Absolute R²: baseline / conditioned / voice-only", fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_BASE, "absolute_r2_all_approaches.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out2}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int,
                        default=int(os.environ.get("JOB_INDEX", -1)),
                        help="Cluster job index → (modality, seed). -1 means local mode.")
    parser.add_argument("--modality",  type=str, default=None,
                        help="Run a single modality (name must match MODALITIES list)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Run a single seed (default: all)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; aggregate and plot existing results")
    parser.add_argument("--oof-train", action="store_true",
                        help="Use inner K-fold OOF for voice predictions on train subjects")
    parser.add_argument("--full-pool-voice", action="store_true",
                        help="Train voice model for test prediction on full voice pool minus "
                             "test subjects (strictly leakage-free: test ages never seen)")
    parser.add_argument("--calibrate-voice", action="store_true",
                        help="Linearly calibrate voice predictions using train subjects "
                             "(fixes subpopulation mean/scale shift, leak-free)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing metrics instead of skipping completed jobs")
    args = parser.parse_args()

    oof_train        = args.oof_train
    full_pool_voice  = args.full_pool_voice
    calibrate_voice  = args.calibrate_voice
    os.makedirs(_get_output_base(oof_train, full_pool_voice, calibrate_voice), exist_ok=True)

    if args.plot_only:
        aggregate_and_save(oof_train, full_pool_voice, calibrate_voice)
        plot_comparison()
        return

    # ── cluster mode: one (modality, seed) per job ────────────────────────────
    if args.job_index >= 0:
        mod_idx  = args.job_index // len(SEEDS)
        seed_idx = args.job_index  % len(SEEDS)
        name, feat_csv, model_type = MODALITIES[mod_idx]
        seed = SEEDS[seed_idx]
        print(f"[job {args.job_index}] modality={name}  seed={seed}  "
              f"model={model_type}  oof_train={oof_train}  "
              f"full_pool_voice={full_pool_voice}  calibrate_voice={calibrate_voice}")
        voice_df, voice_feat_cols = load_wavlm()
        run_pair(name, feat_csv, model_type, seed, voice_df, voice_feat_cols,
                 oof_train, full_pool_voice, calibrate_voice, args.force)
        return

    # ── local mode ────────────────────────────────────────────────────────────
    print("Loading WavLM features...")
    voice_df, voice_feat_cols = load_wavlm()
    print(f"  WavLM: {len(voice_df)} subjects, {len(voice_feat_cols)} features")

    mods = [(n, c, t) for n, c, t in MODALITIES
            if args.modality is None or n == args.modality]
    seeds = [args.seed] if args.seed is not None else SEEDS

    for name, feat_csv, model_type in mods:
        for seed in seeds:
            print(f"\n=== {name}  seed={seed}  model={model_type} ===")
            try:
                run_pair(name, feat_csv, model_type, seed, voice_df, voice_feat_cols,
                         oof_train, full_pool_voice, calibrate_voice, args.force)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

    aggregate_and_save(oof_train, full_pool_voice, calibrate_voice)
    plot_comparison()


if __name__ == "__main__":
    main()
