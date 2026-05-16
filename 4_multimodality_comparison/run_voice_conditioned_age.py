"""
Voice-conditioned age prediction analysis.

For each modality, append the OOF Voice Age predictions as an extra feature
and compare R² against the baseline (modality alone).

  ΔR² = R²(modality + voice_pred) - R²(modality only)

A large ΔR² means voice captures aging signal the modality misses.
A near-zero ΔR² means voice is redundant given that modality.

Protocol
--------
- Single seed (42) for quick test; pass --seed N for another
- 5-fold GroupKFold outer CV, no HPO (default LightGBM params)
- Voice OOF predictions from step3_voice_age_ridge_one_per_subject/seed_{seed}
  are truly out-of-fold → no leakage when appended as features
- Both conditions use identical fold splits (same GroupKFold seed)
- Sex-stratified (female / male separately)

Usage
-----
    python run_voice_conditioned_age.py                  # seed 42, all modalities
    python run_voice_conditioned_age.py --seed 1         # different seed
    python run_voice_conditioned_age.py --modality sleep # single modality
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/age_prediction_new_pipeline/data/"
VOICE_OOF_BASE = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step3_voice_age_ridge_one_per_subject"
OUTPUT_BASE    = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step4_voice_conditioned"

MODALITIES = {
    "sleep":        "X_sleep_age.csv",
    "DEXA":         "X_DEXA_age.csv",
    "NMR":          "X_NMR_age.csv",
    "metabolomics": "X_metabolomics_age.csv",
    "retina":       "X_retina_age.csv",
    "diet":         "X_diet_age.csv",
    "microbiome":   "X_microbiome_age.csv",
    "lifestyle":    "X_lifestyle_age.csv",
}

MODALITY_DROP = {
    "sleep": ["sat_below_88", "neurokit_hrv_frequency_ulf_during_wake"],
}
MICROBIOME_MIN_PREVALENCE = 0.10

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="regression",
    verbosity=-1,
    n_jobs=8,
)

N_SPLITS = 5
VOICE_COL = "voice_pred"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_modality(name: str, features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(DATA_DIR, features_csv), index_col=[0, 1])
    if "RegistrationCode" in df.columns:
        df = df.drop(columns=["RegistrationCode"])

    if "age" not in df.columns:
        y_csv = features_csv.replace("X_", "Y_")
        y_path = os.path.join(DATA_DIR, y_csv)
        if os.path.exists(y_path):
            y = pd.read_csv(y_path, index_col=[0, 1])
            df = df.join(y[["age"]], how="inner")
        else:
            raise FileNotFoundError(f"Y file not found: {y_path}")

    for col in MODALITY_DROP.get(name, []):
        if col in df.columns:
            df = df.drop(columns=[col])

    if name == "NMR":
        drop_fc = [c for c in df.columns if "_FC" in c or "_pct" in c or ":" in c]
        df = df.drop(columns=drop_fc)

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


# ── CV evaluation ──────────────────────────────────────────────────────────────

def _run_cv(df_sex: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    groups = df_sex["subject_number"].astype(str).values
    X = df_sex[feature_cols].values
    y = df_sex["age"].values

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed)
        model.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X[test_idx])

    mask = ~np.isnan(oof_pred)
    r2  = r2_score(y[mask], oof_pred[mask])
    r   = float(pearsonr(y[mask], oof_pred[mask])[0])
    mae = mean_absolute_error(y[mask], oof_pred[mask])
    return {"oof_R2": r2, "oof_r": r, "oof_MAE": mae, "n": int(mask.sum())}


def run_modality_pair(name: str, feat_csv: str, seed: int) -> dict:
    print(f"\n=== {name} (seed={seed}) ===")
    df, feat_cols = load_modality(name, feat_csv)

    results = {}
    for sex in ("female", "male"):
        # Filter by sex if gender column present
        if "gender" in df.columns:
            sex_val = 0 if sex == "female" else 1
            df_sex = df[df["gender"] == sex_val].copy()
        else:
            df_sex = df.copy()

        df_sex["subject_number"] = df_sex["subject_number"].astype(str)

        # Load voice OOF and join
        voice_oof = load_voice_oof(seed, sex)
        df_with_voice = df_sex.merge(voice_oof, on="subject_number", how="inner")

        n_base   = len(df_sex)
        n_joined = len(df_with_voice)
        print(f"  {sex}: modality n={n_base}, after voice join n={n_joined}")

        # Baseline: modality features only (on joined subset for fair comparison)
        m_base = _run_cv(df_with_voice, feat_cols, seed)

        # Conditioned: modality + voice_pred
        m_cond = _run_cv(df_with_voice, feat_cols + [VOICE_COL], seed)

        delta = m_cond["oof_R2"] - m_base["oof_R2"]
        print(f"    baseline R²={m_base['oof_R2']:.4f}  conditioned R²={m_cond['oof_R2']:.4f}  ΔR²={delta:+.4f}")

        results[sex] = {
            "baseline_R2":    m_base["oof_R2"],
            "conditioned_R2": m_cond["oof_R2"],
            "delta_R2":       delta,
            "baseline_r":     m_base["oof_r"],
            "conditioned_r":  m_cond["oof_r"],
            "baseline_MAE":   m_base["oof_MAE"],
            "conditioned_MAE":m_cond["oof_MAE"],
            "n": m_base["n"],
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--modality", type=str, default=None,
                        help="Run a single modality (for cluster dispatch)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    mods = {args.modality: MODALITIES[args.modality]} if args.modality else MODALITIES

    all_results = {}
    for name, feat_csv in mods.items():
        try:
            all_results[name] = run_modality_pair(name, feat_csv, args.seed)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")

    # Save JSON
    out_json = os.path.join(OUTPUT_BASE, f"voice_conditioned_seed{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Print summary table
    rows = []
    for mod, sexes in all_results.items():
        for sex, m in sexes.items():
            rows.append({
                "modality": mod, "sex": sex,
                "baseline_R2":    round(m["baseline_R2"], 4),
                "conditioned_R2": round(m["conditioned_R2"], 4),
                "delta_R2":       round(m["delta_R2"], 4),
                "n": m["n"],
            })
    df_summary = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_BASE, f"voice_conditioned_seed{args.seed}.csv")
    df_summary.to_csv(out_csv, index=False)
    print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
