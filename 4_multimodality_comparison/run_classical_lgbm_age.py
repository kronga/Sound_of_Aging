"""
LightGBM + HPO age prediction from classical acoustic feature sets.

One job per feature set, submitted to mcluster02 via Elysium.
Each job runs run_multi_seed_lightgbm with optimize_hyperparams=True
and 20 parallel workers — same pipeline as the other biological modalities.

Usage
-----
  # Submit all 4 feature sets to cluster
  python run_classical_lgbm_age.py --submit

  # Run a single feature set locally (cluster dispatch target)
  python run_classical_lgbm_age.py --feature-set egemaps

  # Smoke test locally (300 rows, 1 seed, 2 folds, 5 HPO iters)
  python run_classical_lgbm_age.py --smoke

  # Print final summary (after all jobs complete)
  python run_classical_lgbm_age.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lightgbm_regression import run_multi_seed_lightgbm

# ─────────────────────────── config ──────────────────────────────────────── #

PYTHON = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
BASE   = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"

FEATURE_PARQUETS: dict[str, Path] = {
    "praat":       BASE / "features_praat"       / "all_features.parquet",
    "egemaps":     BASE / "features_egemaps"     / "all_features.parquet",
    "compare2016": BASE / "features_compare2016" / "all_features.parquet",
    "emobase":     BASE / "features_emobase"     / "all_features.parquet",
}

OUTPUT_BASE   = Path(__file__).parents[2] / "analysis_outputs" / "step4_classical_lgbm"
DISTRIBUTE_DIR = Path(__file__).parent / "distribute_classical_lgbm"

SEEDS             = [42, 1, 2, 3, 4, 17, 99, 123, 256, 512]
N_SPLITS          = 5
N_ITER_SEARCH     = 30
VALIDATION_FRAC   = 0.2
N_JOBS            = 20     # LightGBM parallel threads per job
MIN_AGE, MAX_AGE  = 40, 70

# Cluster
PARTITION  = "himem8.q"
CPU        = N_JOBS
MEM        = "64G"

# ─────────────────────────── data loading ────────────────────────────────── #

def _load_dataset(feature_set: str, smoke: bool = False) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(FEATURE_PARQUETS[feature_set])
    feats.index.name = "filename"

    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "age", "gender", "subject_number"])

    df = feats.join(sd, how="inner").copy()
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    if smoke:
        df = df.groupby("gender").head(150).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in {"age", "gender", "subject_number"}]
    return df, feat_cols


# ─────────────────────────── single feature-set run ──────────────────────── #

def run_one(feature_set: str, smoke: bool = False) -> dict:
    out_dir = str(OUTPUT_BASE / f"lgbm_{feature_set}_age")

    # Resume if already complete
    both_done = all(
        (Path(out_dir) / f"gender_{g}" / "predictions_averaged.csv").exists()
        for g in ("female", "male")
    )
    if both_done:
        print(f"[RESUME] {feature_set} — already complete")
        return _load_metrics(out_dir, feature_set)

    print(f"\n{'='*60}\nFeature set: {feature_set.upper()}\n{'='*60}")
    df, feat_cols = _load_dataset(feature_set, smoke)
    print(f"  Recordings: {len(df)}  |  Features: {len(feat_cols)}")

    metrics = run_multi_seed_lightgbm(
        df=df,
        target_col="age",
        group_col="subject_number",
        output_dir=out_dir,
        seeds=[42] if smoke else SEEDS,
        columns=feat_cols,
        handle_nans="impute",
        impute_strategy="median",
        n_splits=2 if smoke else N_SPLITS,
        split_gender=True,
        save_plots=not smoke,
        optimize_hyperparams=True,
        n_iter_search=5 if smoke else N_ITER_SEARCH,
        validation_fraction=VALIDATION_FRAC,
        n_jobs=N_JOBS,
    )

    row = {"modality": feature_set}
    if isinstance(metrics, dict):
        for gender, gm in metrics.items():
            row[f"{gender}_R2"]  = gm.get("averaged_R2")
            row[f"{gender}_r"]   = gm.get("averaged_Pearson_r")
            row[f"{gender}_MAE"] = gm.get("averaged_MAE")
            print(f"  {gender.upper()}: R²={gm.get('averaged_R2'):.4f}  "
                  f"r={gm.get('averaged_Pearson_r'):.4f}  "
                  f"MAE={gm.get('averaged_MAE'):.2f}")
    return row


def _load_metrics(out_dir: str, name: str) -> dict:
    row = {"modality": name}
    for g in ("female", "male"):
        p = Path(out_dir) / f"gender_{g}" / "metrics_averaged.json"
        if p.exists():
            m = json.loads(p.read_text())
            row[f"{g}_R2"]  = m.get("averaged_R2")
            row[f"{g}_r"]   = m.get("averaged_Pearson_r")
            row[f"{g}_MAE"] = m.get("averaged_MAE")
    return row


# ─────────────────────────── summary ─────────────────────────────────────── #

def print_summary() -> None:
    rows = []
    for fs in FEATURE_PARQUETS:
        out_dir = str(OUTPUT_BASE / f"lgbm_{fs}_age")
        rows.append(_load_metrics(out_dir, fs))

    # Add WavLM from ridge results (best available comparison)
    wavlm_base = Path(__file__).parents[2] / "analysis_outputs" / "step3_voice_age_ridge"
    wavlm_row = {"modality": "WavLM-Large (ridge)"}
    for g in ("female", "male"):
        p = wavlm_base / f"gender_{g}" / "averaged_metrics.csv"
        if p.exists():
            m = pd.read_csv(p).iloc[0].to_dict()
            wavlm_row[f"{g}_R2"]  = m.get("averaged_R2")
            wavlm_row[f"{g}_r"]   = m.get("averaged_Pearson_r")
            wavlm_row[f"{g}_MAE"] = m.get("averaged_MAE")
    rows.append(wavlm_row)

    df = pd.DataFrame(rows).set_index("modality")
    df = df.sort_values("female_R2", ascending=False)
    out_csv = OUTPUT_BASE / "summary_classical_lgbm_vs_wavlm.csv"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print("\n" + "="*60)
    print("Classical LGBM vs. WavLM-Large  (sorted by female R²)")
    print("="*60)
    print(df.to_string())
    print(f"\nSaved → {out_csv}")


# ─────────────────────────── cluster submission ───────────────────────────── #

def submit() -> None:
    import time as _time
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from elysium import ClusterParams, Elysium, JobStatus

    from datetime import datetime
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = DISTRIBUTE_DIR / ts

    manifest: dict[str, dict] = {}
    script = Path(__file__).resolve()
    collections = {}

    for fs in FEATURE_PARQUETS:
        fs_remote_dir = run_dir / fs
        cluster_params = ClusterParams(host_name="mcluster02", remote_dir=str(fs_remote_dir))
        elysium = Elysium(cluster_params)
        elysium.start()

        cmd = f"{PYTHON} {script} --feature-set {fs}"
        col = elysium.run(
            command=cmd,
            total_jobs=1,
            cpu=CPU,
            mem=MEM,
            partition=PARTITION,
            gpu=0,
            cwd=str(script.parent),
        )
        collections[fs] = col
        manifest[fs] = {"remote_dir": str(fs_remote_dir), "collection_id": col.id}
        print(f"Queued {fs}  collection_id={col.id}")

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "collections.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Wait until all PENDING jobs are actually submitted to SGE (seconds, not hours)
    print("Waiting for daemon to submit jobs to SGE...", end="", flush=True)
    deadline = _time.time() + 120
    while _time.time() < deadline:
        all_submitted = all(
            len(col.get_jobs(JobStatus.PENDING)) == 0
            for col in collections.values()
        )
        if all_submitted:
            break
        print(".", end="", flush=True)
        _time.sleep(2)
    print()

    for fs, col in collections.items():
        stats = col.get_stats()
        print(f"  {fs}: {dict(stats)}")

    print(f"\nManifest → {manifest_path}")
    print(f"When done: python {script.name} --summary")


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--feature-set", choices=list(FEATURE_PARQUETS),
                   help="Run a single feature set (used by cluster jobs)")
    g.add_argument("--submit",  action="store_true", help="Distribute all 4 to mcluster02")
    g.add_argument("--smoke",   action="store_true", help="Quick local test (all 4, small data)")
    g.add_argument("--summary", action="store_true", help="Print completed results table")
    p.add_argument("--no-submit-wait", action="store_true")
    args = p.parse_args()

    if args.submit:
        submit()
    elif args.summary:
        print_summary()
    elif args.smoke:
        for fs in FEATURE_PARQUETS:
            run_one(fs, smoke=True)
        print_summary()
    else:
        run_one(args.feature_set, smoke=False)


if __name__ == "__main__":
    main()
