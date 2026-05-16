"""
Distribute classical feature extraction across the SGE cluster via Elysium.

One Elysium collection per feature set; each job processes one chunk of
CHUNK_SIZE recordings using $JOB_INDEX to compute its slice boundaries.

Usage
-----
  # Submit all 4 feature sets, block until done
  python distribute_classical_features.py

  # Submit a subset without blocking
  python distribute_classical_features.py --feature-sets praat egemaps --no-wait

  # Dry run — print plan, no SSH calls
  python distribute_classical_features.py --dry-run

  # Merge chunks after all jobs finish
  python distribute_classical_features.py --merge

  # Reconnect to a previous run
  python distribute_classical_features.py --resume distribute_classical/260510_120000
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root → elysium importable
from elysium import ClusterParams, Elysium, JobStatus

# ─────────────────────────── constants ───────────────────────────────────── #

PYTHON = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
SCRIPT = Path(__file__).parent / "extract_classical_features.py"
INPUT_DIR = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    "Oct25_voice_full_length/Raw_voice"
)
DISTRIBUTE_DIR = Path(__file__).parent / "distribute_classical"

ALL_FEATURE_SETS = ["praat", "egemaps", "compare2016", "emobase"]
AUDIO_EXTS       = {".flac", ".wav", ".mp3"}

CHUNK_SIZE = 500
N_WORKERS  = 4
PARTITION  = "himem8.q"
MEM        = "32G"   # 4 slots × 8 G

# ─────────────────────────── helpers ─────────────────────────────────────── #

def _count_audio_files(input_dir: Path) -> int:
    return sum(1 for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def _n_chunks(total: int, chunk_size: int) -> int:
    return math.ceil(total / chunk_size)


def _merge_all(feature_sets: list[str]) -> None:
    for fs in feature_sets:
        print(f"\n─── Merging {fs} ───")
        subprocess.run(
            [PYTHON, str(SCRIPT), "--feature-set", fs, "--merge"],
            check=True,
        )


def _report_failures(collections: dict[str, object]) -> None:
    any_failed = False
    for fs, col in collections.items():
        failed = col.get_jobs(JobStatus.FAILED)
        if failed:
            any_failed = True
            print(f"\n[FAILED] {fs}: {len(failed)} job(s)")
            for job in failed:
                print(f"  {job.name}  exit={job.exit_code}  stderr → {job.stderr_file}")
    if not any_failed:
        print("\nAll jobs completed successfully.")


# ─────────────────────────── submit ──────────────────────────────────────── #

def submit(
    feature_sets: list[str],
    chunk_size: int,
    wait: bool,
    dry_run: bool,
) -> None:
    total = _count_audio_files(INPUT_DIR)
    n_chunks = _n_chunks(total, chunk_size)

    print(f"Total audio files : {total}")
    print(f"Chunk size        : {chunk_size}")
    print(f"Chunks per set    : {n_chunks}")
    print(f"Feature sets      : {feature_sets}")
    print(f"Total jobs        : {n_chunks * len(feature_sets)}")
    print()

    if dry_run:
        for fs in feature_sets:
            cmd = (
                f"{PYTHON} {SCRIPT} --feature-set {fs} "
                f"--chunk-start $((JOB_INDEX * {chunk_size})) "
                f"--chunk-end $(( (JOB_INDEX + 1) * {chunk_size} )) "
                f"--jobs {N_WORKERS}"
            )
            print(f"[dry-run] {fs}  ({n_chunks} jobs)")
            print(f"  command: {cmd}")
        return

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = DISTRIBUTE_DIR / ts

    collections: dict[str, object] = {}
    manifest_data: dict[str, dict] = {}

    for fs in feature_sets:
        # Each feature set gets its own remote_dir so collection IDs never collide
        fs_remote_dir = run_dir / fs
        cluster_params = ClusterParams(host_name="mcluster02", remote_dir=str(fs_remote_dir))
        elysium = Elysium(cluster_params)
        elysium.start()

        cmd = (
            f"{PYTHON} {SCRIPT} --feature-set {fs} "
            f"--chunk-start $((JOB_INDEX * {chunk_size})) "
            f"--chunk-end $(( (JOB_INDEX + 1) * {chunk_size} )) "
            f"--jobs {N_WORKERS}"
        )
        col = elysium.run(
            command=cmd,
            total_jobs=n_chunks,
            cpu=N_WORKERS,
            mem=MEM,
            partition=PARTITION,
            gpu=0,
            cwd=str(SCRIPT.parent),
        )
        collections[fs] = col
        manifest_data[fs] = {"remote_dir": str(fs_remote_dir), "collection_id": col.id}
        print(f"Submitted {fs}  ({n_chunks} jobs)  collection_id={col.id}")

    manifest = run_dir / "collections.json"
    manifest.write_text(json.dumps(manifest_data, indent=2))
    print(f"\nManifest → {manifest}")
    print(f"Run dir  → {run_dir}")

    if not wait:
        print("\nSubmitted. To reconnect:")
        print(f"  python {Path(__file__).name} --resume {run_dir}")
        return

    for col in collections.values():
        col.wait()

    _report_failures(collections)
    print(f"\nTo merge chunks:")
    print(f"  python {Path(__file__).name} --merge --feature-sets {' '.join(feature_sets)}")


# ─────────────────────────── resume ──────────────────────────────────────── #

def resume(run_dir: Path) -> None:
    manifest = run_dir / "collections.json"
    if not manifest.exists():
        raise SystemExit(f"No collections.json found in {run_dir}")

    manifest_data: dict[str, dict] = json.loads(manifest.read_text())

    collections: dict[str, object] = {}
    for fs, entry in manifest_data.items():
        cluster_params = ClusterParams(host_name="mcluster02", remote_dir=entry["remote_dir"])
        elysium = Elysium(cluster_params)
        elysium.start()
        col = elysium.resume(entry["collection_id"])
        collections[fs] = col

    for col in collections.values():
        col.wait()

    _report_failures(collections)


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-sets", nargs="+", default=ALL_FEATURE_SETS,
                   choices=ALL_FEATURE_SETS, metavar="FS")
    p.add_argument("--chunk-size",   type=int, default=CHUNK_SIZE)
    p.add_argument("--no-wait",      action="store_true",
                   help="Submit and exit; don't block for completion")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--merge",        action="store_true",
                   help="Merge existing chunk parquets and exit")
    p.add_argument("--resume",       type=Path, metavar="RUN_DIR",
                   help="Reconnect to a previous run and wait for completion")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        resume(args.resume)
    elif args.merge:
        _merge_all(args.feature_sets)
    else:
        submit(
            feature_sets=args.feature_sets,
            chunk_size=args.chunk_size,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
