#!/usr/bin/env python3
"""Run all engineered Ridge benchmarks with LSQR and an extended grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/davidkro/PycharmProjects/DeepVoice")
sys.path.insert(0, str(ROOT))

from elysium import ClusterParams, Elysium  # noqa: E402


PYTHON = "/net/mraid20/export/jasmine/david/anaconda3/bin/python"
WORKER = Path(__file__).with_name("run_repeated_analysis.py").resolve()
RUN_ROOT = (
    ROOT
    / "analysis_outputs"
    / "repeated_analysis"
    / "elysium_jobs"
)
ACTIVATE = (
    "source /net/mraid20/export/jasmine/david/anaconda3/bin/activate"
    " && export MPLCONFIGDIR=/tmp/matplotlib-davidkro"
    " && export OMP_NUM_THREADS=4"
    " && export MKL_NUM_THREADS=4"
    " && export OPENBLAS_NUM_THREADS=4"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("%y%m%d_%H%M%S_engineered_ridge"),
    )
    args = parser.parse_args()
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    runner = Elysium(
        ClusterParams(host_name="mcluster02", remote_dir=str(run_dir))
    )
    runner.start()
    collection = runner.run(
        command=(
            f"{PYTHON} {WORKER} --kind ridge "
            '--job-index "$((JOB_INDEX + 20))"'
        ),
        total_jobs=80,
        cpu=4,
        mem="48G",
        partition="himem8.q",
        gpu=0,
        throttle=10,
        cwd=str(ROOT),
        activate_cmd=ACTIVATE,
    )
    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(),
        "collection_id": collection.id,
        "jobs": 80,
        "ridge_task_indices": [20, 99],
        "purpose": "All engineered Ridge models, stable LSQR and extended grid",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    collection.wait()
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["final"] = {
        status.value: count for status, count in collection.get_stats().items()
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    time.sleep(1)


if __name__ == "__main__":
    main()
