"""
Run age prediction across 8 biological modalities using the same Ridge + nested-CV
alpha-tuning protocol as the Voice Age pipeline (step 3).

This replaces the LightGBM comparison to give apples-to-apples benchmarking:
  - Same 5-fold GroupKFold outer CV
  - Same inner alpha selection (20% holdout from training groups)
  - Same alpha candidate grid
  - Same 10-seed bagging
  - Same per-gender split

Outputs go to analysis_outputs/step4_multimodality_ridge/
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "3_age_prediction"))

from ridge_regression import run_multi_seed_ridge

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/age_prediction_new_pipeline/data/"
OUTPUT_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_multimodality_ridge"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
OPTIMIZE_ALPHA = True
STANDARDIZE = True

MODALITIES = {
    "sleep":        "X_sleep_age.csv",
    "blood_test":   "X_blood_test_age.csv",
    "DEXA":         "X_DEXA_age.csv",
    "NMR":          "X_NMR_age.csv",
    "metabolomics": "X_metabolomics_age.csv",
    "retina":       "X_retina_age.csv",
    "diet":         "X_diet_age.csv",
    "microbiome":   "X_microbiome_age.csv",
}

MODALITY_DROP = {
    "sleep": ["sat_below_88", "neurokit_hrv_frequency_ulf_during_wake"],
}

MICROBIOME_MIN_PREVALENCE = 0.10
# ============================================================


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
            raise FileNotFoundError(f"Expected Y file not found: {y_path}")

    drop = MODALITY_DROP.get(name, [])
    df = df.drop(columns=[c for c in drop if c in df.columns])

    if name == "NMR":
        fc_pct = [c for c in df.columns if "_FC" in c or "_pct" in c or ":" in c]
        df = df.drop(columns=fc_pct)

    if name == "microbiome":
        prevalence = (df > 0.0001).sum() / len(df)
        keep = prevalence[prevalence >= MICROBIOME_MIN_PREVALENCE].index.tolist()
        df = df[keep]

    df = df.reset_index()
    feature_cols = [c for c in df.columns
                    if c not in ("subject_number", "RegistrationCode", "age",
                                 "gender", "research_stage")]
    return df, feature_cols


def _load_completed_metrics(out_dir: str) -> dict:
    row = {}
    for g in ("female", "male"):
        mpath = os.path.join(out_dir, f"gender_{g}", "metrics_averaged.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            row[f"{g}_R2"] = m.get("averaged_R2")
            row[f"{g}_r"] = m.get("averaged_Pearson_r")
            row[f"{g}_MAE"] = m.get("averaged_MAE")
    return row


def main():
    summary_rows = []

    for name, feat_csv in MODALITIES.items():
        path = os.path.join(DATA_DIR, feat_csv)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue

        out_dir = os.path.join(OUTPUT_BASE, f"ridge_{name}_age")

        # Resume: skip modalities where both genders have averaged predictions
        both_done = all(
            os.path.exists(os.path.join(out_dir, f"gender_{g}", "predictions_averaged.csv"))
            for g in ("female", "male")
        )
        if both_done:
            print(f"\n[RESUME] {name.upper()} — already complete, loading metrics")
            row = {"modality": name, **_load_completed_metrics(out_dir)}
            summary_rows.append(row)
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name.upper()}")
        print(f"{'='*60}")

        df, feature_cols = load_modality(name, feat_csv)
        print(f"  Shape: {df.shape}  Features: {len(feature_cols)}")

        if "subject_number" not in df.columns:
            print(f"[SKIP] {name}: no subject_number column")
            continue

        row = {"modality": name}
        for gender_val, gender_label in [(0, "female"), (1, "male")]:
            sub = df[df["gender"] == gender_val].copy() if "gender" in df.columns else df.copy()
            if sub.empty:
                print(f"  No rows for gender={gender_label}, skipping")
                continue

            gender_out = os.path.join(out_dir, f"gender_{gender_label}")
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
                save_plots=False,
            )
            row[f"{gender_label}_R2"] = metrics.get("averaged_R2")
            row[f"{gender_label}_r"] = metrics.get("averaged_Pearson_r")
            row[f"{gender_label}_MAE"] = metrics.get("averaged_MAE")
            print(f"  {gender_label.upper()}: R²={metrics.get('averaged_R2'):.4f}  "
                  f"r={metrics.get('averaged_Pearson_r'):.4f}  "
                  f"MAE={metrics.get('averaged_MAE'):.2f}")

        summary_rows.append(row)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        summary.to_csv(os.path.join(OUTPUT_BASE, "summary_all_modalities.csv"), index=False)
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
