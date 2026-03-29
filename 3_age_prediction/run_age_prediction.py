"""
Run voice-age prediction for a single modality (WavLM embeddings).

Edit the CONFIG section at the top and run:
    python run_age_prediction.py

Outputs land in OUTPUT_DIR (one sub-directory per gender when
SPLIT_GENDER=True, or a single flat directory otherwise).
"""

import os
import pandas as pd
from ridge_regression import run_multi_seed_ridge

# ============================================================
# CONFIG — edit these paths before running
# ============================================================

WAVLM_FEATURES_CSV = "/path/to/WavLM_features.csv"         # (n_recordings × d) embeddings
SUBJECT_DETAILS_CSV = "/path/to/subject_details_df.csv"     # must contain: age, gender, subject_number

OUTPUT_DIR = "output/voice_age_ridge"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]            # 10 seeds for bagging

# Ridge config
N_SPLITS = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
OPTIMIZE_ALPHA = True
STANDARDIZE = False          # WavLM features are already normalised in practice
MIN_AGE, MAX_AGE = 40, 70    # cohort age filter

# ============================================================

def main():
    # Load embeddings
    wavlm = pd.read_csv(WAVLM_FEATURES_CSV, index_col=0)
    subject_details = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename")

    # Merge
    df = wavlm.join(subject_details[["age", "gender", "subject_number"]], how="inner")
    df = df.dropna(subset=["age", "subject_number"])

    # Cohort filter
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

    # Keep first visit per subject
    if "visit_number" in df.columns:
        df = df.sort_values("visit_number").drop_duplicates("subject_number", keep="first")

    feature_cols = wavlm.columns.tolist()

    print(f"Dataset: {len(df)} recordings, {len(feature_cols)} WavLM features")
    print(f"Male: {(df['gender']==1).sum()}  Female: {(df['gender']==0).sum()}")

    # Run separately for each gender (paper convention)
    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val].copy()
        print(f"\n{'='*60}")
        print(f"Running {gender_label.upper()}  (n={len(sub)})")
        print(f"{'='*60}")

        metrics = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=os.path.join(OUTPUT_DIR, f"gender_{gender_label}"),
            seeds=SEEDS,
            columns=feature_cols,
            handle_nans="impute",
            impute_strategy="median",
            n_splits=N_SPLITS,
            alpha=0.5,
            standardize=STANDARDIZE,
            optimize_alpha=OPTIMIZE_ALPHA,
            alpha_candidates=ALPHA_CANDIDATES,
            validation_fraction=0.2,
            save_plots=True,
        )

        print(f"\n{gender_label.upper()} results:")
        print(f"  Averaged R²   = {metrics['averaged_R2']:.4f}")
        print(f"  Averaged MAE  = {metrics['averaged_MAE']:.2f} years")
        print(f"  Pearson r     = {metrics['averaged_Pearson_r']:.4f}")


if __name__ == "__main__":
    main()
