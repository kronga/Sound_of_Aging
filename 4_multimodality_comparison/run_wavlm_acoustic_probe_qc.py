"""
WavLM -> acoustic feature probe with QC filter applied, sex-split.

Same probe design as run_wavlm_acoustic_probe.py, but uses
WavLM_features_filtered_with_RF.csv (recordings classified as 'good' by
the QC random forest). Runs three subsets: All, Female, Male, and writes
a single results table plus a paired (no-QC vs QC) comparison CSV.

Outputs
-------
  paper_revision_outputs/step_p5_wavlm_probe_qc/probe_results_full_qc.csv
  paper_revision_outputs/step_p5_wavlm_probe_qc/probe_qc_vs_noqc_comparison.csv
  paper_revision_outputs/step_p5_wavlm_probe_qc/probe_r2_gender_split_qc.pdf/.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from run_wavlm_acoustic_probe import (  # reuse feature registry + core probe
    PROBE_FEATURES,
    CATEGORY_ORDER,
    FEATURE_CATEGORY,
    probe_one_feature,
)

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
WAVLM_FILTERED_CSV  = BASE / "WavLM_features_filtered_with_RF.csv"
PRAAT_PARQUET       = BASE / "features_praat"   / "all_features.parquet"
EGEMAPS_PARQUET     = BASE / "features_egemaps" / "all_features.parquet"
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
NOQC_RESULTS_CSV    = Path(__file__).parents[2] / "paper_revision_outputs" / "step_p5_wavlm_probe" / "probe_results_full.csv"

OUTPUT_DIR = Path(__file__).parents[2] / "paper_revision_outputs" / "step_p5_wavlm_probe_qc"


def _probe_subset(X_full, df, mask, groups, label):
    X_sub  = X_full[mask]
    grp_sub = groups[mask]
    results = {}
    for col, display, _ in PROBE_FEATURES:
        if col not in df.columns:
            continue
        y_all = df[col].to_numpy().astype(float)
        y_sub = y_all[mask]
        valid = np.isfinite(y_sub)
        if valid.sum() < 50:
            print(f"  [SKIP-{label}] {col}: only {valid.sum()} valid rows")
            continue
        m = probe_one_feature(X_sub[valid], y_sub[valid], grp_sub[valid])
        results[col] = m
        print(f"  [{label}] {display:<35} R2={m['r2']:+.3f}  r={m['r']:+.3f}  n={m['n']}")
    return results


def run(smoke: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading QC-filtered WavLM embeddings from {WAVLM_FILTERED_CSV.name}")
    wavlm = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0)
    embed_cols = [c for c in wavlm.columns if c.startswith("feature_")]
    wavlm = wavlm[embed_cols].astype("float32")
    wavlm.index.name = "filename"
    print(f"  {len(wavlm)} QC-passed recordings, {len(embed_cols)} embedding dims")

    print("Loading Praat features...")
    praat = pd.read_parquet(PRAAT_PARQUET)
    praat.index.name = "filename"

    print("Loading eGeMAPSv02 features...")
    egemaps = pd.read_parquet(EGEMAPS_PARQUET)
    egemaps.index.name = "filename"

    print("Loading subject details...")
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "gender", "subject_number"])

    df = (wavlm
          .join(praat,   how="inner")
          .join(egemaps, how="inner")
          .join(sd,      how="inner")
          .copy())
    print(f"Matched recordings after QC + feature joins: {len(df)}")

    if smoke:
        df = df.iloc[:300].copy()

    df["subject"] = df["subject_number"].astype(str)
    groups = df["subject"].to_numpy()

    X = SimpleImputer(strategy="median").fit_transform(df[embed_cols].to_numpy())

    mask_all    = np.ones(len(df), dtype=bool)
    mask_female = (df["gender"].to_numpy() == 0)
    mask_male   = (df["gender"].to_numpy() == 1)

    print(f"\n{'='*60}\nAll subjects (QC, n={mask_all.sum()})\n{'='*60}")
    res_all = _probe_subset(X, df, mask_all, groups, "ALL")
    print(f"\n{'='*60}\nFemale (QC, n={mask_female.sum()})\n{'='*60}")
    res_f   = _probe_subset(X, df, mask_female, groups, "F")
    print(f"\n{'='*60}\nMale (QC, n={mask_male.sum()})\n{'='*60}")
    res_m   = _probe_subset(X, df, mask_male, groups, "M")

    rows = []
    for col, display, _ in PROBE_FEATURES:
        if col not in res_all:
            continue
        row = {
            "feature":   col,
            "label":     display,
            "category":  FEATURE_CATEGORY.get(col, "Other"),
            "r2_all":    res_all[col]["r2"],
            "r_all":     res_all[col]["r"],
            "p_all":     res_all[col]["p"],
            "n_all":     res_all[col]["n"],
        }
        for key, res in (("female", res_f), ("male", res_m)):
            if col in res:
                row[f"r2_{key}"] = res[col]["r2"]
                row[f"r_{key}"]  = res[col]["r"]
                row[f"p_{key}"]  = res[col]["p"]
                row[f"n_{key}"]  = res[col]["n"]
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("r2_all", ascending=False)
    out_csv = OUTPUT_DIR / "probe_results_full_qc.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nQC probe results -> {out_csv}")
    print(results_df[["label", "r2_all", "r2_female", "r2_male"]].to_string(index=False))

    # Paired comparison with no-QC run
    if NOQC_RESULTS_CSV.exists():
        noqc = pd.read_csv(NOQC_RESULTS_CSV)
        merged = noqc.merge(results_df, on=["feature", "label", "category"],
                            suffixes=("_noqc", "_qc"))
        merged["delta_r2_all"]    = merged["r2_all_qc"]    - merged["r2_all_noqc"]
        merged["delta_r2_female"] = merged["r2_female_qc"] - merged["r2_female_noqc"]
        merged["delta_r2_male"]   = merged["r2_male_qc"]   - merged["r2_male_noqc"]
        merged["sex_gap_qc"]      = merged["r2_female_qc"] - merged["r2_male_qc"]
        merged["sex_gap_noqc"]    = merged["r2_female_noqc"] - merged["r2_male_noqc"]
        comp_csv = OUTPUT_DIR / "probe_qc_vs_noqc_comparison.csv"
        merged.to_csv(comp_csv, index=False)
        print(f"\nQC vs no-QC comparison -> {comp_csv}")

        print("\nSummary stats (R^2):")
        for col in ["r2_all_noqc", "r2_all_qc",
                    "r2_female_noqc", "r2_female_qc",
                    "r2_male_noqc", "r2_male_qc"]:
            print(f"  {col:22s}  mean={merged[col].mean():+.3f}  median={merged[col].median():+.3f}")
        print(f"\n  mean delta_r2_all    = {merged['delta_r2_all'].mean():+.4f}")
        print(f"  mean delta_r2_female = {merged['delta_r2_female'].mean():+.4f}")
        print(f"  mean delta_r2_male   = {merged['delta_r2_male'].mean():+.4f}")
        print(f"  mean sex gap (F-M) QC   = {merged['sex_gap_qc'].mean():+.4f}")
        print(f"  mean sex gap (F-M) noQC = {merged['sex_gap_noqc'].mean():+.4f}")

    return results_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
