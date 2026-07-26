"""
Ridge age prediction using all MFCC-related features from the emobase parquet
(MFCCs 1–12, 456 columns including delta features and statistics).

Same pipeline: 10 seeds, 5-fold GroupKFold, nested alpha HPO.

Usage
-----
  python run_emobase_mfcc_ridge.py
  python run_emobase_mfcc_ridge.py --smoke
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

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
EMOBASE_PARQUET     = BASE / "features_emobase" / "all_features.parquet"
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
WAVLM_FILTERED_CSV  = BASE / "WavLM_features_filtered_with_RF.csv"

OUTPUT_BASE = Path(__file__).parents[2] / "analysis_outputs" / "step4_emobase_mfcc_ridge"

MIN_AGE, MAX_AGE = 40, 70
SEEDS            = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS         = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def load_data(smoke: bool) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(EMOBASE_PARQUET)
    feats.index.name = "filename"

    mfcc_cols = [c for c in feats.columns if "mfcc" in c.lower()]
    feats = feats[mfcc_cols]

    qc_files = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0, usecols=[0]).index
    feats = feats[feats.index.isin(qc_files)]

    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])
    df = feats.join(sd, how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    if smoke:
        df = df.groupby("gender").head(150).reset_index(drop=True)

    print(f"Subjects: {len(df)}  (female={(df['gender']==0).sum()}, male={(df['gender']==1).sum()})")
    print(f"MFCC features: {len(mfcc_cols)}")
    return df, mfcc_cols


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

        metrics = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=str(OUTPUT_BASE / f"gender_{gender_label}"),
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
    print("SUMMARY  emobase MFCC features (456 cols)  ridge, 10 seeds")
    print("="*60)
    for g, m in results.items():
        print(f"  {g:6s}: R²={m['averaged_R2']:.4f}  "
              f"r={m['averaged_Pearson_r']:.4f}  MAE={m['averaged_MAE']:.3f}")


if __name__ == "__main__":
    main()
