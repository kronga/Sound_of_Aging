"""
Distribute voice-conditioned holdout analysis across mcluster02.

80 jobs total: 8 modalities × 10 seeds.
Each job runs baseline + conditioned for one (modality, seed) pair (both sexes).

Usage
-----
    python distribute_voice_conditioned_holdout.py               # submit all 80 jobs
    python distribute_voice_conditioned_holdout.py --resume <id>
    python distribute_voice_conditioned_holdout.py --summarise   # print results table + plots
    python distribute_voice_conditioned_holdout.py --partition himem7.q,himem8.q
    python distribute_voice_conditioned_holdout.py --throttle 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # DeepVoice root
from elysium import Elysium, ClusterParams

from run_voice_conditioned_holdout import (
    MODALITIES, SEEDS, TOTAL_JOBS, OUTPUT_BASE, OOF_OUTPUT_BASE,
    FULLPOOL_OUTPUT_BASE, FULLPOOL_OOF_OUTPUT_BASE, FULLPOOL_OOF_CAL_OUTPUT_BASE,
    _get_output_base, aggregate_and_save, plot_comparison,
)

WORKER_PATH = Path(__file__).with_name(
    "run_voice_conditioned_holdout.py"
).resolve()
WORKER_BASE = f'python "{WORKER_PATH}" --job-index $JOB_INDEX'
ACTIVATE = "source /net/mraid20/export/jasmine/david/anaconda3/bin/activate"


def _next_elysium_dir(base: str) -> str:
    existing = sorted(Path(base).glob("elysium*"))
    next_idx = max(
        (int(p.name.replace("elysium", "")) for p in existing
         if p.name.replace("elysium", "").isdigit()),
        default=0,
    ) + 1
    return f"{base}/elysium{next_idx}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",           type=str, default=None)
    parser.add_argument("--summarise",        action="store_true")
    parser.add_argument("--oof-train",        action="store_true",
                        help="Use inner K-fold OOF voice predictions for training subjects")
    parser.add_argument("--full-pool-voice",  action="store_true",
                        help="Train voice model for test on full pool minus test subjects")
    parser.add_argument("--calibrate-voice",  action="store_true",
                        help="Linearly calibrate voice predictions using train subjects")
    parser.add_argument("--force",            action="store_true",
                        help="Overwrite existing metrics instead of skipping completed jobs")
    parser.add_argument("--no-wait",          action="store_true",
                        help="Submit jobs and print the collection id without waiting")
    parser.add_argument("--host",             type=str, default="mcluster02")
    parser.add_argument("--partition",        type=str, default="himem7.q,himem8.q")
    parser.add_argument("--cpu",              type=int, default=10)
    parser.add_argument("--mem",              type=str, default="20G")
    parser.add_argument("--throttle",         type=int, default=None)
    parser.add_argument("--elysium-dir",      type=str, default=None)
    args = parser.parse_args()

    oof_train        = args.oof_train
    full_pool_voice  = args.full_pool_voice
    calibrate_voice  = args.calibrate_voice
    elysium_base     = _get_output_base(oof_train, full_pool_voice, calibrate_voice)
    worker_cmd       = (WORKER_BASE
                        + (" --oof-train"        if oof_train        else "")
                        + (" --full-pool-voice"  if full_pool_voice  else "")
                        + (" --calibrate-voice"  if calibrate_voice  else "")
                        + (" --force"            if args.force       else ""))

    if args.summarise:
        aggregate_and_save(oof_train, full_pool_voice, calibrate_voice)
        plot_comparison()
        return

    os.makedirs(elysium_base, exist_ok=True)
    remote_dir = args.elysium_dir or _next_elysium_dir(elysium_base)
    cluster_params = ClusterParams(host_name=args.host, remote_dir=remote_dir)
    print(f"oof_train={oof_train}  remote_dir={remote_dir}")

    elysium = Elysium(cluster_params)
    if args.no_wait:
        # Avoid racing the background daemon against the one explicit submit cycle below.
        elysium._daemon_started = True
    else:
        elysium.start()

    if args.resume:
        col = elysium.resume(args.resume)
        print(f"Resumed collection: {col.id}")
    else:
        run_kwargs = dict(
            command=worker_cmd,
            total_jobs=TOTAL_JOBS,
            cpu=args.cpu,
            mem=args.mem,
            partition=args.partition,
            gpu=0,
            cwd="/home/davidkro/PycharmProjects/DeepVoice",
            activate_cmd=ACTIVATE,
        )
        if args.throttle is not None:
            run_kwargs["throttle"] = args.throttle
        col = elysium.run(**run_kwargs)
        print(f"Submitted {TOTAL_JOBS} jobs. Collection ID: {col.id}")
        print("Run with --resume <id> to reconnect after disconnect.")

    if args.no_wait:
        elysium._daemon_handle(col.id)
        print("Not waiting because --no-wait was passed.")
        return

    col.wait()

    print("\nAll jobs done. Generating summary and plots...")
    aggregate_and_save(oof_train)
    plot_comparison()


if __name__ == "__main__":
    main()
