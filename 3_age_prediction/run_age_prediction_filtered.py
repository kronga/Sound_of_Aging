"""
Step 3 re-run using quality-filtered WavLM embeddings.
Also writes fold_metrics_summary.csv per gender for easy std calculation.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from ridge_regression import run_multi_seed_ridge

# ============================================================
# CONFIG
# ============================================================
WAVLM_FEATURES_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/WavLM_features_filtered_with_RF.csv"
SUBJECT_DETAILS_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/subject_details_df_Oct25.csv"

OUTPUT_DIR = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step3_voice_age_ridge_filtered"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]

N_SPLITS = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
OPTIMIZE_ALPHA = True
STANDARDIZE = False
MIN_AGE, MAX_AGE = 40, 70
# ============================================================


def build_fold_metrics_summary(gender_out_dir: str) -> pd.DataFrame:
    """
    Collect per-seed × per-fold R², Pearson r, MAE from seed_*/metrics.json
    and save as fold_metrics_summary.csv.
    Returns the DataFrame.
    """
    rows = []
    for entry in sorted(os.listdir(gender_out_dir)):
        if not entry.startswith("seed_"):
            continue
        seed = int(entry.split("_")[1])
        metrics_path = os.path.join(gender_out_dir, entry, "metrics.json")
        pred_path = os.path.join(gender_out_dir, entry, "predictions.csv")
        if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        pred = pd.read_csv(pred_path)
        for fold in sorted(pred["fold"].unique()):
            fold_pred = pred[pred["fold"] == fold]
            yt = fold_pred["true_values"]
            yp = fold_pred["predictions"]
            mask = yt.notna() & yp.notna()
            yt, yp = yt[mask], yp[mask]
            rows.append({
                "seed": seed,
                "fold": fold,
                "n": len(yt),
                "R2": float(r2_score(yt, yp)),
                "Pearson_r": float(pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
                "MAE": float(mean_absolute_error(yt, yp)),
            })

    df = pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)
    out_path = os.path.join(gender_out_dir, "fold_metrics_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved fold_metrics_summary.csv → {out_path}")
    return df


def print_fold_stats(df: pd.DataFrame, gender: str) -> None:
    print(f"\n  {gender.upper()} — per-fold stats (across {df['seed'].nunique()} seeds × {df['fold'].nunique()} folds):")
    for metric in ["R2", "Pearson_r", "MAE"]:
        mean = df[metric].mean()
        std_across_folds = df.groupby("fold")[metric].mean().std()
        std_across_seeds = df.groupby("seed")[metric].mean().std()
        print(f"    {metric}: mean={mean:.4f}  std(folds)={std_across_folds:.4f}  std(seeds)={std_across_seeds:.4f}")


def main():
    # Filtered CSV already contains age, gender, subject_number alongside feature_* columns
    df = pd.read_csv(WAVLM_FEATURES_CSV, index_col=0)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

    print(f"Dataset (filtered): {len(df)} recordings, {len(feature_cols)} WavLM features")
    print(f"Male: {(df['gender']==1).sum()}  Female: {(df['gender']==0).sum()}")

    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val].copy()
        print(f"\n{'='*60}\nRunning {gender_label.upper()}  (n={len(sub)})\n{'='*60}")

        gender_out = os.path.join(OUTPUT_DIR, f"gender_{gender_label}")
        metrics = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=gender_out,
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

        print(f"\n{gender_label.upper()} averaged results:")
        print(f"  R²   = {metrics['averaged_R2']:.4f}")
        print(f"  MAE  = {metrics['averaged_MAE']:.2f} yrs")
        print(f"  r    = {metrics['averaged_Pearson_r']:.4f}")

        fold_df = build_fold_metrics_summary(gender_out)
        print_fold_stats(fold_df, gender_label)


if __name__ == "__main__":
    main()
