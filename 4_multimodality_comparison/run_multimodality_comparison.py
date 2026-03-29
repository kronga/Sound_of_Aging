"""
Run age prediction across multiple biological modalities and compare with voice.

Modalities supported: voice (WavLM), sleep, blood_test, DEXA, NMR,
                      metabolomics, retina, diet, microbiome, lifestyle.

Edit the CONFIG section and run:
    python run_multimodality_comparison.py

Each modality produces its own output sub-directory under OUTPUT_BASE.
Cross-modality correlation is computed by modalities_correlations.py
(step 6_visualization).
"""

import os
import pandas as pd
from lightgbm_regression import run_multi_seed_lightgbm

# ============================================================
# CONFIG — edit these paths before running
# ============================================================

DATA_DIR = "/path/to/age_prediction_data/"   # CSVs named X_<modality>_age.csv
OUTPUT_BASE = "output/multimodality_lgbm"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS = 5
SPLIT_GENDER = True

# Map modality name  →  (features CSV, age target CSV)
MODALITIES = {
    "sleep":        ("X_sleep_age.csv",        None),   # age col expected inside
    "blood_test":   ("X_blood_test_age.csv",   None),
    "DEXA":         ("X_DEXA_age.csv",         None),
    "NMR":          ("X_NMR_age.csv",          None),
    "metabolomics": ("X_metabolomics_age.csv", None),
    "retina":       ("X_retina_age.csv",       None),
    "diet":         ("X_diet_age.csv",         None),
    "microbiome":   ("X_microbiome_age.csv",   None),
}

# Columns to drop from each modality (leakage / redundancy)
MODALITY_DROP = {
    "sleep":    ["sat_below_88", "neurokit_hrv_frequency_ulf_during_wake"],
    "NMR":      [],   # FC/pct columns are filtered programmatically below
}

# Minimum species prevalence for microbiome filtering
MICROBIOME_MIN_PREVALENCE = 0.10

# ============================================================

def load_modality(name: str, features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(DATA_DIR, features_csv), index_col=[0, 1])
    if "RegistrationCode" in df.columns:
        df = df.drop(columns=["RegistrationCode"])

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


def main():
    summary_rows = []

    for name, (feat_csv, _) in MODALITIES.items():
        path = os.path.join(DATA_DIR, feat_csv)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name.upper()}")
        print(f"{'='*60}")

        df, feature_cols = load_modality(name, feat_csv)
        print(f"  Shape: {df.shape}  Features: {len(feature_cols)}")

        metrics = run_multi_seed_lightgbm(
            df=df,
            target_col="age",
            group_col="subject_number",
            output_dir=os.path.join(OUTPUT_BASE, f"lgbm_{name}_age"),
            seeds=SEEDS,
            columns=feature_cols,
            handle_nans="impute",
            impute_strategy="median",
            n_splits=N_SPLITS,
            split_gender=SPLIT_GENDER,
            save_plots=True,
        )

        row = {"modality": name}
        if SPLIT_GENDER and isinstance(metrics, dict):
            for gender, gm in metrics.items():
                row[f"{gender}_R2"] = gm.get("averaged_R2")
                row[f"{gender}_r"] = gm.get("averaged_Pearson_r")
        else:
            row["R2"] = metrics.get("averaged_R2")
            row["Pearson_r"] = metrics.get("averaged_Pearson_r")
        summary_rows.append(row)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(OUTPUT_BASE, "summary_all_modalities.csv"), index=False)
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
