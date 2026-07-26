"""
Compare voice-age Ridge predictions with vs without per-fold linear calibration.

Runs the same multi-seed pipeline as run_age_prediction.py with
calibrate_predictions=True so each fold fits a linear correction
(polyfit on in-sample training predictions) before evaluating on the
validation set.

Outputs land in:
    analysis_outputs/step3_voice_age_ridge_cal/gender_{female,male}/

Results are printed side-by-side with the existing uncalibrated metrics.
"""

import json
import os
import pandas as pd
from ridge_regression import run_multi_seed_ridge

WAVLM_FEATURES_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/WavLM_features.csv"
SUBJECT_DETAILS_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/subject_details_df_Oct25.csv"

OUTPUT_DIR_ORIG = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step3_voice_age_ridge"
OUTPUT_DIR_CAL  = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step3_voice_age_ridge_cal"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]

N_SPLITS = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
OPTIMIZE_ALPHA = True
STANDARDIZE = False
MIN_AGE, MAX_AGE = 40, 70


def _load_orig_metrics(gender_label: str) -> dict:
    path = os.path.join(OUTPUT_DIR_ORIG, f"gender_{gender_label}", "metrics_averaged.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    wavlm = pd.read_csv(WAVLM_FEATURES_CSV, index_col=0)
    subject_details = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename")
    df = wavlm.join(subject_details[["age", "gender", "subject_number"]], how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    if "visit_number" in df.columns:
        df = df.sort_values("visit_number").drop_duplicates("subject_number", keep="first")

    feature_cols = wavlm.columns.tolist()
    print(f"Dataset: {len(df)} recordings, {len(feature_cols)} WavLM features")

    results = {}
    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val].copy()
        print(f"\n{'='*60}")
        print(f"Running {gender_label.upper()}  (n={len(sub)})  — with calibration")
        print(f"{'='*60}")

        m = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=os.path.join(OUTPUT_DIR_CAL, f"gender_{gender_label}"),
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
            save_plots=False,
            calibrate_predictions=True,
        )
        results[gender_label] = m

    print("\n" + "=" * 70)
    print("CALIBRATION COMPARISON  (10-seed averaged OOF, GroupKFold-5)")
    print("=" * 70)
    print(f"{'Metric':<22} {'Female (no cal)':>16} {'Female (cal)':>14} {'Male (no cal)':>14} {'Male (cal)':>12}")
    print("-" * 80)

    for label, key_raw, key_cal in [
        ("R²",    "averaged_R2",        "averaged_cal_R2"),
        ("Pearson r", "averaged_Pearson_r", "averaged_cal_Pearson_r"),
        ("MAE",   "averaged_MAE",       "averaged_cal_MAE"),
        ("RMSE",  "averaged_RMSE",      "averaged_cal_RMSE"),
    ]:
        orig_f = _load_orig_metrics("female")
        orig_m = _load_orig_metrics("male")
        cal_f  = results.get("female", {})
        cal_m  = results.get("male", {})
        v_f0 = orig_f.get(key_raw, float("nan"))
        v_f1 = cal_f.get(key_cal, float("nan"))
        v_m0 = orig_m.get(key_raw, float("nan"))
        v_m1 = cal_m.get(key_cal, float("nan"))
        print(f"{label:<22} {v_f0:>16.4f} {v_f1:>14.4f} {v_m0:>14.4f} {v_m1:>12.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
