#!/usr/bin/env python3
"""Submit and monitor the repeated CPU analyses with Elysium on mcluster02."""

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

from run_repeated_analysis import lgbm_tasks, ridge_tasks  # noqa: E402


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
        ClusterParams(
            host_name="mcluster02",
            remote_dir=str(run_dir),
        )
    )
    runner.start()

    ridge_collection = runner.run(
        command=f"{PYTHON} {WORKER} --kind ridge --job-index $JOB_INDEX",
        total_jobs=len(ridge_tasks()),
        cpu=2,
        mem="24G",
        partition="himem8.q",
        gpu=0,
        throttle=10,
        cwd=str(ROOT),
        activate_cmd=ACTIVATE,
    )
    # Elysium collection identifiers have one-second resolution.
    time.sleep(1.1)
    lgbm_collection = runner.run(
        command=(
            "export ANALYSIS_LGBM_THREADS=10"
            f" && {PYTHON} {WORKER} --kind lgbm --job-index $JOB_INDEX"
        ),
        total_jobs=len(lgbm_tasks()),
        cpu=10,
        mem="64G",
        partition="himem8.q",
        gpu=0,
        throttle=12,
        cwd=str(ROOT),
        activate_cmd=ACTIVATE,
    )

    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(),
        "ridge": {
            "collection_id": ridge_collection.id,
            "jobs": len(ridge_tasks()),
            "cpu_per_job": 2,
            "throttle": 10,
        },
        "lgbm": {
            "collection_id": lgbm_collection.id,
            "jobs": len(lgbm_tasks()),
            "cpu_per_job": 10,
            "throttle": 12,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

    # The daemon handles both collections while these waits provide a durable
    # nohup process and continuous status output.
    ridge_collection.wait()
    lgbm_collection.wait()

    manifest["completed_at"] = datetime.now().isoformat()
    manifest["ridge_final"] = {
        status.value: count
        for status, count in ridge_collection.get_stats().items()
    }
    manifest["lgbm_final"] = {
        status.value: count
        for status, count in lgbm_collection.get_stats().items()
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    time.sleep(1)


if __name__ == "__main__":
    main()
