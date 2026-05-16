"""
Random Forest age prediction from WavLM-Large embeddings.

Same pipeline as run_classical_rf_age.py:
  GroupKFold (5 folds), inner 20% HPO (30 random configs), single seed.

After completion, prints a full summary table comparing:
  RF + WavLM  vs  Ridge + WavLM  vs  RF + classical sets  vs  Ridge + classical sets

Usage
-----
  python run_wavlm_rf_age.py
  python run_wavlm_rf_age.py --seed 1
  python run_wavlm_rf_age.py --summary-only   # just print the table
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(__file__))
from run_classical_rf_age import _run_cv   # reuse identical CV logic

# ─────────────────────────── config ──────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
WAVLM_FILTERED_CSV = BASE / "WavLM_features_filtered_with_RF.csv"
SUBJECT_CSV        = BASE / "subject_details_df_Oct25.csv"

OUTPUT_BASE  = Path(__file__).parents[2] / "paper_revision_outputs" / "step3_wavlm_rf"
RF_CLASS_DIR = Path(__file__).parents[2] / "paper_revision_outputs" / "step4_classical_rf"
RIDGE_CLASS  = Path(__file__).parents[2] / "paper_revision_outputs" / "step4_classical_ridge"
RIDGE_WAVLM  = Path(__file__).parents[2] / "paper_revision_outputs" / "step3_voice_age_ridge"

MIN_AGE, MAX_AGE = 40, 70
N_ITER_SEARCH    = 30

# ─────────────────────────── data ────────────────────────────────────────── #

def load_wavlm() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Loading QC-filtered WavLM embeddings …")
    df = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0)
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    embed_cols = [c for c in df.columns if c.startswith("feature_")]
    X = SimpleImputer(strategy="median").fit_transform(df[embed_cols].to_numpy(dtype=float))
    y = df["age"].to_numpy(dtype=float)
    g = df["gender"].to_numpy()
    grp = df["subject_number"].to_numpy()

    print(f"  {len(df)} subjects, {len(embed_cols)} features  "
          f"(female={( g==0).sum()}, male={(g==1).sum()})")
    return X, y, g, grp


# ─────────────────────────── run ─────────────────────────────────────────── #

def run(seed: int) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    X, y, g, grp = load_wavlm()

    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        mask = g == gender_val
        print(f"\n{'='*60}\nWavLM RF  {gender_label.upper()}  (n={mask.sum()})  seed={seed}\n{'='*60}")
        metrics = _run_cv(X[mask], y[mask], grp[mask],
                          seed=seed, n_iter=N_ITER_SEARCH)

        out = OUTPUT_BASE / f"gender_{gender_label}"
        out.mkdir(exist_ok=True)
        with open(out / f"metrics_seed{seed}.json", "w") as f:
            json.dump({"seed": seed, **metrics}, f, indent=2)


# ─────────────────────────── summary table ───────────────────────────────── #

def build_summary(seed: int) -> None:
    rows = []

    # ── WavLM RF ─────────────────────────────────────────────────────────── #
    row = {"model": "RF", "features": "WavLM-Large"}
    for g in ("female", "male"):
        p = OUTPUT_BASE / f"gender_{g}" / f"metrics_seed{seed}.json"
        if p.exists():
            m = json.loads(p.read_text())
            row[f"{g}_R2"]  = round(m["R2"], 4)
            row[f"{g}_r"]   = round(m["r"], 4)
            row[f"{g}_MAE"] = round(m["MAE"], 3)
    rows.append(row)

    # ── WavLM Ridge ──────────────────────────────────────────────────────── #
    row = {"model": "Ridge", "features": "WavLM-Large"}
    for g in ("female", "male"):
        p = RIDGE_WAVLM / f"gender_{g}" / "metrics_averaged.json"
        if p.exists():
            m = json.loads(p.read_text())
            row[f"{g}_R2"]  = round(m["averaged_R2"], 4)
            row[f"{g}_r"]   = round(m["averaged_Pearson_r"], 4)
            row[f"{g}_MAE"] = round(m["averaged_MAE"], 3)
    rows.append(row)

    # ── Classical RF ─────────────────────────────────────────────────────── #
    for fs in ["egemaps", "emobase", "praat", "compare2016"]:
        row = {"model": "RF", "features": fs}
        for g in ("female", "male"):
            p = RF_CLASS_DIR / fs / f"gender_{g}" / f"metrics_seed{seed}.json"
            if p.exists():
                m = json.loads(p.read_text())
                row[f"{g}_R2"]  = round(m["R2"], 4)
                row[f"{g}_r"]   = round(m["r"], 4)
                row[f"{g}_MAE"] = round(m["MAE"], 3)
        rows.append(row)

    # ── Classical Ridge ───────────────────────────────────────────────────── #
    for fs in ["egemaps", "emobase", "praat", "compare2016"]:
        row = {"model": "Ridge", "features": fs}
        for g in ("female", "male"):
            p = RIDGE_CLASS / fs / f"gender_{g}" / "predictions_averaged.csv"
            if p.exists():
                df = pd.read_csv(p)
                y_true = df["true_values"].values
                y_pred = df["mean_predictions"].values
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-10
                row[f"{g}_R2"]  = round(1 - ss_res / ss_tot, 4)
                row[f"{g}_r"]   = round(pearsonr(y_true, y_pred)[0], 4)
                row[f"{g}_MAE"] = round(np.mean(np.abs(y_true - y_pred)), 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["model", "features"])
    out = OUTPUT_BASE / "summary_rf_vs_ridge_all.csv"
    df.to_csv(out)

    print("\n" + "="*70)
    print("SUMMARY: RF vs Ridge × WavLM vs Classical features")
    print("="*70)
    print(df.to_string())
    print(f"\nSaved → {out}")


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore")

    if not args.summary_only:
        run(args.seed)

    build_summary(args.seed)


if __name__ == "__main__":
    main()
