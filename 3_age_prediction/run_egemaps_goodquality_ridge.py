"""
Ridge age prediction on audio_features_egemaps_good_quality.csv.

Same pipeline as the classical ridge runs:
  10 seeds, 5-fold GroupKFold (grouped by subject_number = filename prefix),
  nested alpha HPO on inner 20% validation split.

Usage
-----
  python run_egemaps_goodquality_ridge.py
  python run_egemaps_goodquality_ridge.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ridge_regression import run_multi_seed_ridge

# ─────────────────────────── config ──────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
FEATS_CSV = BASE / "audio_features_egemaps_good_quality.csv"
META_CSV  = BASE / "audio_features_egemaps_metadata_good_quality.csv"

OUTPUT_BASE = Path(__file__).parents[2] / "analysis_outputs" / "step4_egemaps_goodquality_ridge"

MIN_AGE, MAX_AGE  = 40, 70
SEEDS             = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS          = 5
ALPHA_CANDIDATES  = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ─────────────────────────── data ────────────────────────────────────────── #

def load_data(smoke: bool) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_csv(FEATS_CSV)
    meta  = pd.read_csv(META_CSV, usecols=["filename", "age", "gender"])

    df = feats.merge(meta, on="filename", how="inner")
    df = df.dropna(subset=["age"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

    # derive subject_number from filename prefix (e.g. 9635947353_20250925 → 9635947353)
    df["subject_number"] = df["filename"].str.split("_").str[0]
    df = df[~df["subject_number"].duplicated(keep="first")].reset_index(drop=True)

    if smoke:
        df = df.groupby("gender").head(150).reset_index(drop=True)

    feat_cols = [c for c in df.columns
                 if c not in {"filename", "age", "gender", "subject_number"}]
    print(f"Subjects: {len(df)}  (female={(df['gender']==0).sum()}, male={(df['gender']==1).sum()})")
    print(f"Features: {len(feat_cols)}")
    return df, feat_cols


# ─────────────────────────── main ────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_data(args.smoke)

    results = {}
    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val].copy()
        print(f"\n{'='*60}\n{gender_label.upper()}  (n={len(sub)})\n{'='*60}")
        out_dir = str(OUTPUT_BASE / f"gender_{gender_label}")

        metrics = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=out_dir,
            seeds=[42] if args.smoke else SEEDS,
            columns=feat_cols,
            handle_nans="impute",
            impute_strategy="median",
            n_splits=2 if args.smoke else N_SPLITS,
            alpha=1.0,
            standardize=True,
            optimize_alpha=True,
            alpha_candidates=ALPHA_CANDIDATES,
            validation_fraction=0.2,
            save_plots=not args.smoke,
        )

        results[gender_label] = metrics
        print(f"  R²={metrics['averaged_R2']:.4f}  "
              f"r={metrics['averaged_Pearson_r']:.4f}  "
              f"MAE={metrics['averaged_MAE']:.3f}")

    print("\n" + "="*60)
    print("SUMMARY  egemaps good-quality CSV  (ridge, 10 seeds)")
    print("="*60)
    for g, m in results.items():
        print(f"  {g:6s}: R²={m['averaged_R2']:.4f}  "
              f"r={m['averaged_Pearson_r']:.4f}  MAE={m['averaged_MAE']:.3f}")


if __name__ == "__main__":
    main()
