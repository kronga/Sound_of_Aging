"""
Submit classical feature extraction jobs to the SGE cluster (himem8.q).

Creates one job per (feature_set, chunk).  Each job processes CHUNK_SIZE
recordings in parallel with N_WORKERS worker processes.

Usage
-----
  python submit_classical_features.py            # submit all 4 feature sets
  python submit_classical_features.py --feature-sets praat egemaps
  python submit_classical_features.py --dry-run  # print commands, don't submit
  python submit_classical_features.py --merge     # merge chunks after jobs finish
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
SCRIPT = Path(__file__).parent / "extract_classical_features.py"
INPUT_DIR = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/Raw_voice"
)
DISTRIBUTE_DIR = Path(__file__).parent / "distribute_classical"

ALL_FEATURE_SETS = ["praat", "egemaps", "compare2016", "emobase"]

CHUNK_SIZE  = 500
N_WORKERS   = 4   # parallel processes per job
QUEUE       = "himem8.q"
VMEM_PER_SLOT = "8G"   # h_vmem per thread slot
THREADS     = N_WORKERS

# Estimated runtimes per chunk (minutes) – used only for informational output
EST_MINUTES = {"praat": 35, "egemaps": 10, "compare2016": 20, "emobase": 10}


def get_audio_files(input_dir: Path) -> list[str]:
    exts = {".flac", ".wav", ".mp3"}
    return sorted(str(p) for p in input_dir.iterdir() if p.suffix.lower() in exts)


def make_job_script(
    job_dir: Path,
    feature_set: str,
    chunk_start: int,
    chunk_end: int,
) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    script = job_dir / "run.sh"
    script.write_text(f"""\
#!/bin/bash
#$ -N feat_{feature_set}_{chunk_start}
#$ -q {QUEUE}
#$ -pe threads {THREADS}
#$ -l h_vmem={VMEM_PER_SLOT}
#$ -cwd
#$ -o /dev/null
#$ -e /dev/null

exec > "{job_dir}/stdout.log" 2> "{job_dir}/stderr.log"

{PYTHON} {SCRIPT} \\
    --feature-set {feature_set} \\
    --chunk-start {chunk_start} \\
    --chunk-end   {chunk_end} \\
    --jobs {N_WORKERS}
""")
    script.chmod(0o755)
    return script


def submit_all(feature_sets: list[str], dry_run: bool) -> dict:
    files = get_audio_files(INPUT_DIR)
    n = len(files)
    chunks = list(range(0, n, CHUNK_SIZE))
    print(f"Total files   : {n}")
    print(f"Chunk size    : {CHUNK_SIZE}")
    print(f"Chunks/set    : {len(chunks)}")
    print(f"Feature sets  : {feature_sets}")
    print(f"Total jobs    : {len(chunks) * len(feature_sets)}")
    print()

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = DISTRIBUTE_DIR / ts
    submitted: list[dict] = []

    for fs in feature_sets:
        for start in chunks:
            end = min(start + CHUNK_SIZE, n)
            job_dir = run_dir / fs / f"{start:05d}_{end:05d}"
            script  = make_job_script(job_dir, fs, start, end)

            if dry_run:
                print(f"[dry-run] would submit: {script}")
                submitted.append({"feature_set": fs, "start": start, "end": end, "job_id": None})
                continue

            result = subprocess.run(
                ["ssh", "genie60", f"qsub {script}"],
                capture_output=True, text=True,
            )
            job_id = result.stdout.strip()
            if result.returncode != 0:
                print(f"[ERROR] {fs} [{start}-{end}]: {result.stderr.strip()}")
            else:
                print(f"Submitted {fs} [{start:5d}-{end:5d}]  → {job_id}")
            submitted.append({"feature_set": fs, "start": start, "end": end, "job_id": job_id})

    if not dry_run:
        manifest = run_dir / "submitted_jobs.json"
        manifest.write_text(json.dumps(submitted, indent=2))
        print(f"\nManifest → {manifest}")
        print(f"\nTo merge chunks after jobs finish:")
        for fs in feature_sets:
            print(f"  python {SCRIPT} --feature-set {fs} --merge")

    return {"run_dir": str(run_dir), "jobs": submitted}


def merge_all(feature_sets: list[str]) -> None:
    for fs in feature_sets:
        print(f"\n─── Merging {fs} ───")
        subprocess.run(
            [PYTHON, str(SCRIPT), "--feature-set", fs, "--merge"],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-sets", nargs="+", default=ALL_FEATURE_SETS,
                   choices=ALL_FEATURE_SETS, metavar="FS")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--merge",   action="store_true",
                   help="Merge existing chunks instead of submitting jobs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.merge:
        merge_all(args.feature_sets)
    else:
        submit_all(args.feature_sets, dry_run=args.dry_run)
