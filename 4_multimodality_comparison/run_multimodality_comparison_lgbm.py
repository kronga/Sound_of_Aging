"""
Run age prediction across 8 biological modalities using LightGBM + RandomizedSearch HPO.

Pipeline:
  - 5-fold GroupKFold outer CV
  - Inner HPO: 20% validation holdout from training groups, 30-iteration RandomizedSearch
  - 10-seed bagging (same seeds as Ridge baseline)
  - Per-gender split

Outputs go to paper_revision_outputs/step4_multimodality_lgbm_hpo/

Usage
-----
  # Smoke test (fast, local):
  python run_multimodality_comparison_lgbm.py --smoke

  # Single modality (for cluster dispatch):
  python run_multimodality_comparison_lgbm.py --modality sleep

  # Submit all modalities to mcluster02:
  python run_multimodality_comparison_lgbm.py --submit

  # Run all modalities locally (sequential):
  python run_multimodality_comparison_lgbm.py
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from lightgbm_regression import run_multi_seed_lightgbm

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/age_prediction_new_pipeline/data/"
OUTPUT_BASE = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step4_multimodality_lgbm_hpo"
SEEDS = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS = 5
N_ITER_SEARCH = 30
VALIDATION_FRACTION = 0.2

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


def run_modality(name: str, feat_csv: str, seeds: list[int], n_splits: int,
                 n_iter: int, val_frac: float, output_base: str,
                 n_jobs: int = 8) -> dict:
    out_dir = os.path.join(output_base, f"lgbm_{name}_age")

    both_done = all(
        os.path.exists(os.path.join(out_dir, f"gender_{g}", "predictions_averaged.csv"))
        for g in ("female", "male")
    )
    if both_done:
        print(f"\n[RESUME] {name.upper()} — already complete, loading metrics")
        return {"modality": name, **_load_completed_metrics(out_dir)}

    print(f"\n{'='*60}")
    print(f"Running: {name.upper()}")
    print(f"{'='*60}")

    df, feature_cols = load_modality(name, feat_csv)
    print(f"  Shape: {df.shape}  Features: {len(feature_cols)}")

    if "subject_number" not in df.columns:
        print(f"[SKIP] {name}: no subject_number column")
        return {"modality": name}

    metrics = run_multi_seed_lightgbm(
        df=df,
        target_col="age",
        group_col="subject_number",
        output_dir=out_dir,
        seeds=seeds,
        columns=feature_cols,
        handle_nans="impute",
        impute_strategy="median",
        n_splits=n_splits,
        split_gender=True,
        save_plots=False,
        optimize_hyperparams=True,
        n_iter_search=n_iter,
        validation_fraction=val_frac,
        n_jobs=n_jobs,
    )

    row = {"modality": name}
    if isinstance(metrics, dict):
        for gender, gm in metrics.items():
            row[f"{gender}_R2"] = gm.get("averaged_R2")
            row[f"{gender}_r"] = gm.get("averaged_Pearson_r")
            row[f"{gender}_MAE"] = gm.get("averaged_MAE")
            print(f"  {gender.upper()}: R²={gm.get('averaged_R2'):.4f}  "
                  f"r={gm.get('averaged_Pearson_r'):.4f}  "
                  f"MAE={gm.get('averaged_MAE'):.2f}")
    return row


def smoke_test():
    """1 seed, 2 folds, 5 HPO iterations on sleep — just confirms the pipeline runs."""
    print("=== SMOKE TEST ===")
    name = "sleep"
    feat_csv = MODALITIES[name]
    path = os.path.join(DATA_DIR, feat_csv)
    if not os.path.exists(path):
        print(f"[ERROR] Data not found: {path}")
        sys.exit(1)

    df, feature_cols = load_modality(name, feat_csv)
    # small subset: first 200 rows per gender to keep it fast
    df = df.groupby("gender").head(200).reset_index(drop=True)
    print(f"  Subset shape: {df.shape}  Features: {len(feature_cols)}")

    out_dir = os.path.join(OUTPUT_BASE, "smoke_test_lgbm")
    metrics = run_multi_seed_lightgbm(
        df=df,
        target_col="age",
        group_col="subject_number",
        output_dir=out_dir,
        seeds=[42],
        columns=feature_cols,
        handle_nans="impute",
        impute_strategy="median",
        n_splits=2,
        split_gender=True,
        save_plots=False,
        optimize_hyperparams=True,
        n_iter_search=5,
        validation_fraction=0.2,
    )
    print("\nSmoke test PASSED. Metrics:", metrics)


def submit_to_cluster():
    """Submit one SGE job per modality to mcluster02."""
    this_script = os.path.abspath(__file__)
    repo_dir = os.path.dirname(this_script)
    dist_root = os.path.join(repo_dir, "distribute", time.strftime("%y%m%d%H%M%S"))
    os.makedirs(dist_root, exist_ok=True)

    # Verify SSH
    ret = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "mcluster02", "echo ok"],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print("[ERROR] Cannot reach mcluster02. Run: ssh-copy-id mcluster02")
        sys.exit(1)

    # Detect python interpreter
    python_bin = sys.executable

    submitted = []
    for name in MODALITIES:
        job_dir = os.path.join(dist_root, name)
        os.makedirs(job_dir, exist_ok=True)

        script = f"""#!/bin/bash
#$ -N lgbm_{name}
#$ -q himem8.q
#$ -pe threads 10
#$ -l h_vmem=8G
#$ -cwd
#$ -o {job_dir}/stdout.log
#$ -e {job_dir}/stderr.log

JOBDIR="{job_dir}"

{python_bin} {this_script} --modality {name} --n-jobs 10
"""
        script_path = os.path.join(job_dir, "script.sh")
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        ret = subprocess.run(
            ["ssh", "mcluster02", f"cd {repo_dir} && qsub {script_path}"],
            capture_output=True, text=True,
        )
        if ret.returncode == 0:
            job_id = ret.stdout.strip()
            print(f"  Submitted {name}: {job_id}")
            submitted.append({"modality": name, "job_id": job_id, "job_dir": job_dir})
        else:
            print(f"  [ERROR] {name}: {ret.stderr.strip()}")

    manifest = os.path.join(dist_root, "submitted_jobs.json")
    with open(manifest, "w") as f:
        json.dump(submitted, f, indent=2)
    print(f"\nManifest: {manifest}")
    print(f"Monitor: ssh mcluster02 'qstat -u $USER'")


def main_all():
    summary_rows = []
    for name, feat_csv in MODALITIES.items():
        path = os.path.join(DATA_DIR, feat_csv)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        row = run_modality(name, feat_csv, SEEDS, N_SPLITS, N_ITER_SEARCH,
                           VALIDATION_FRACTION, OUTPUT_BASE)
        summary_rows.append(row)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        summary.to_csv(os.path.join(OUTPUT_BASE, "summary_all_modalities.csv"), index=False)
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--submit", action="store_true", help="Submit jobs to mcluster02")
    parser.add_argument("--modality", type=str, help="Run a single modality")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Threads for LightGBM (default: 8)")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.submit:
        submit_to_cluster()
    elif args.modality:
        name = args.modality
        if name not in MODALITIES:
            print(f"[ERROR] Unknown modality '{name}'. Choose from: {list(MODALITIES)}")
            sys.exit(1)
        feat_csv = MODALITIES[name]
        path = os.path.join(DATA_DIR, feat_csv)
        if not os.path.exists(path):
            print(f"[ERROR] Data not found: {path}")
            sys.exit(1)
        run_modality(name, feat_csv, SEEDS, N_SPLITS, N_ITER_SEARCH,
                     VALIDATION_FRACTION, OUTPUT_BASE, n_jobs=args.n_jobs)
    else:
        main_all()
