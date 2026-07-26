"""
Age prediction from classical acoustic feature sets, compared to WavLM.

Runs the same 10-seed GroupKFold ridge pipeline used for WavLM embeddings
on each of the four classical feature sets produced by extract_classical_features.py:
  praat / egemaps / compare2016 / emobase

Outputs
-------
  analysis_outputs/step4_classical_ridge/{feature_set}/gender_{female|male}/
  analysis_outputs/step4_classical_ridge/summary_classical_vs_wavlm.csv

Usage
-----
  python run_classical_age_prediction.py                  # all four sets
  python run_classical_age_prediction.py --feature-sets praat egemaps
  python run_classical_age_prediction.py --smoke          # 300 rows, no plots
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ridge_regression import run_multi_seed_ridge

# ─────────────────────────── paths & config ──────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
WAVLM_FILTERED_CSV  = BASE / "WavLM_features_filtered_with_RF.csv"  # QC-passed filenames

FEATURE_PARQUETS: dict[str, Path] = {
    "praat":       BASE / "features_praat"       / "all_features.parquet",
    "egemaps":     BASE / "features_egemaps"     / "all_features.parquet",
    "compare2016": BASE / "features_compare2016" / "all_features.parquet",
    "emobase":     BASE / "features_emobase"     / "all_features.parquet",
}

WAVLM_SUMMARY = Path(__file__).parents[2] / "analysis_outputs" / "step3_voice_age_ridge"
OUTPUT_BASE   = Path(__file__).parents[2] / "analysis_outputs" / "step4_classical_ridge"

SEEDS            = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS         = 5
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MIN_AGE, MAX_AGE = 40, 70

# ─────────────────────────── helpers ─────────────────────────────────────── #

def _load_dataset(parquet_path: Path, smoke: bool) -> pd.DataFrame:
    feats = pd.read_parquet(parquet_path)
    feats.index.name = "filename"

    # Restrict to QC-filtered recordings (same set used for WavLM evaluation)
    qc_files = pd.read_csv(WAVLM_FILTERED_CSV, index_col=0, usecols=[0]).index
    feats = feats[feats.index.isin(qc_files)]

    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])

    df = feats.join(sd, how="inner")
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

    # keep first visit per subject (same as WavLM pipeline)
    if "visit_number" in df.columns:
        df = df.sort_values("visit_number").drop_duplicates("subject_number", keep="first")
    else:
        df = df[~df["subject_number"].duplicated(keep="first")]

    if smoke:
        df = df.iloc[:300]

    return df


def _wavlm_summary() -> dict[str, dict]:
    """Read per-gender averaged metrics from existing WavLM ridge outputs."""
    result = {}
    for gender in ("female", "male"):
        metrics_csv = WAVLM_SUMMARY / f"gender_{gender}" / "averaged_metrics.csv"
        if metrics_csv.exists():
            m = pd.read_csv(metrics_csv).iloc[0].to_dict()
            result[gender] = m
    return result


def _run_one_feature_set(
    name: str,
    parquet_path: Path,
    smoke: bool,
) -> dict[str, dict]:
    print(f"\n{'='*60}")
    print(f"Feature set: {name.upper()}")
    print(f"{'='*60}")

    df = _load_dataset(parquet_path, smoke)
    feat_cols = [c for c in df.columns if c not in {"age", "gender", "subject_number"}]
    print(f"  Recordings: {len(df)}  |  Features: {len(feat_cols)}")

    gender_results: dict[str, dict] = {}
    for gender_val, gender_label in [(0, "female"), (1, "male")]:
        sub = df[df["gender"] == gender_val].copy()
        print(f"\n  {gender_label.upper()}  (n={len(sub)})")
        out_dir = str(OUTPUT_BASE / name / f"gender_{gender_label}")

        metrics = run_multi_seed_ridge(
            df=sub,
            target_col="age",
            group_col="subject_number",
            output_dir=out_dir,
            seeds=SEEDS,
            columns=feat_cols,
            handle_nans="impute",
            impute_strategy="median",
            n_splits=N_SPLITS,
            alpha=1.0,
            standardize=True,
            optimize_alpha=True,
            alpha_candidates=ALPHA_CANDIDATES,
            validation_fraction=0.2,
            save_plots=not smoke,
        )

        print(f"    R²={metrics['averaged_R2']:.4f}  "
              f"r={metrics['averaged_Pearson_r']:.4f}  "
              f"MAE={metrics['averaged_MAE']:.2f} yr")
        gender_results[gender_label] = metrics

    return gender_results


def _build_summary(all_results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for feat_set, gender_dict in all_results.items():
        row = {"modality": feat_set}
        for gender_label, metrics in gender_dict.items():
            row[f"{gender_label}_R2"]  = metrics.get("averaged_R2")
            row[f"{gender_label}_r"]   = metrics.get("averaged_Pearson_r")
            row[f"{gender_label}_MAE"] = metrics.get("averaged_MAE")
        rows.append(row)

    # Append WavLM row from existing outputs
    wavlm = _wavlm_summary()
    if wavlm:
        row = {"modality": "WavLM-Large"}
        for gender_label, m in wavlm.items():
            row[f"{gender_label}_R2"]  = m.get("averaged_R2")
            row[f"{gender_label}_r"]   = m.get("averaged_Pearson_r")
            row[f"{gender_label}_MAE"] = m.get("averaged_MAE")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("modality")
    return df.sort_values("female_R2", ascending=False)


# ─────────────────────────── main ────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-sets", nargs="+", default=list(FEATURE_PARQUETS),
                   choices=list(FEATURE_PARQUETS))
    p.add_argument("--smoke", action="store_true", help="300-row test run")
    args = p.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict]] = {}
    for name in args.feature_sets:
        all_results[name] = _run_one_feature_set(name, FEATURE_PARQUETS[name], args.smoke)

    summary = _build_summary(all_results)
    out_csv = OUTPUT_BASE / "summary_classical_vs_wavlm.csv"
    summary.to_csv(out_csv)

    print("\n" + "="*60)
    print("SUMMARY: Classical features vs. WavLM-Large (female R²)")
    print("="*60)
    print(summary.to_string())
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
