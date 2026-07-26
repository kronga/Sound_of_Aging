#!/usr/bin/env python3
"""Run the corrected sex-specific learning curves through Elysium."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
sys.path.insert(0, str(ROOT))

from elysium import ClusterParams, Elysium  # noqa: E402

PYTHON = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
WORKER = (
    Path(__file__).resolve().parents[1]
    / "6_visualization"
    / "power_analysis_age_prediction.py"
)
RUN_ROOT = (
    ROOT
    / "analysis_outputs"
    / "repeated_analysis"
    / "elysium_power_analysis"
)
ACTIVATE = (
    "source /net/mraid20/export/jasmine/david/anaconda3/bin/activate"
    " && export MPLCONFIGDIR=/tmp/matplotlib-davidkro"
    " && export OMP_NUM_THREADS=2"
    " && export OPENBLAS_NUM_THREADS=2"
    " && export MKL_NUM_THREADS=2"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("%y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    runner = Elysium(
        ClusterParams(host_name="mcluster02", remote_dir=str(run_dir))
    )
    runner.start()
    total_jobs = 20
    collection = runner.run(
        command=(
            f"{PYTHON} {WORKER} --job-index $JOB_INDEX"
        ),
        total_jobs=total_jobs,
        cpu=2,
        mem="32G",
        partition="himem8.q",
        gpu=0,
        throttle=10,
        cwd=str(ROOT),
        activate_cmd=ACTIVATE,
    )
    manifest = {
        "run_name": args.run_name,
        "collection_id": collection.id,
        "jobs": total_jobs,
        "cpu_per_job": 2,
        "throttle": 10,
        "cohort": "one latest QC-passed recording per participant",
        "design": (
            "ten shuffled participant-level five-fold outer partitions; "
            "fold-specific inner holdout; one sex-partition seed per job"
        ),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    collection.wait()
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["final"] = {
        status.value: count
        for status, count in collection.get_stats().items()
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
