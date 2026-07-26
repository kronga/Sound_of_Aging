"""Distribute the legacy fold-alignment sensitivity analysis.

This launcher is not used for the final Figure 3 results. Use
``distribute_voice_conditioned_holdout.py`` for the leakage-free,
intersection-cohort analysis reported in the manuscript.

80 jobs total: 8 modalities × 10 seeds.
Each job runs baseline + conditioned CV for one (modality, seed) pair.

Usage
-----
    python distribute_voice_conditioned_hpo.py                        # submit all 80 jobs
    python distribute_voice_conditioned_hpo.py --partition himem7.q,himem8.q   # different queues
    python distribute_voice_conditioned_hpo.py --cpu 8 --mem 16G      # different resources
    python distribute_voice_conditioned_hpo.py --throttle 20          # limit parallel jobs
    python distribute_voice_conditioned_hpo.py --resume <collection_id>
    python distribute_voice_conditioned_hpo.py --summarise            # print results table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))  # DeepVoice root
from elysium import Elysium, ClusterParams

from run_voice_conditioned_hpo_worker import MODALITIES, SEEDS, OUTPUT_BASE, TOTAL_JOBS

WORKER_PATH = Path(__file__).with_name(
    "run_voice_conditioned_hpo_worker.py"
).resolve()
WORKER_BASE = f'python "{WORKER_PATH}" --job-index $JOB_INDEX'
ACTIVATE = "source /net/mraid20/export/jasmine/david/anaconda3/bin/activate"
ELYSIUM_BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step4_voice_conditioned_hpo"


# ── Summarise results ──────────────────────────────────────────────────────────

def summarise(output_base: str = OUTPUT_BASE) -> None:
    rows = []
    for name, _, _ in MODALITIES:
        for seed in SEEDS:
            for sex in ("female", "male"):
                out_dir = Path(output_base) / name / f"seed_{seed}" / f"gender_{sex}"
                b_path = out_dir / "baseline_metrics.json"
                c_path = out_dir / "conditioned_metrics.json"
                v_path = out_dir / "voice_metrics.json"
                if not (b_path.exists() and c_path.exists()):
                    continue
                m_b = json.loads(b_path.read_text())
                m_c = json.loads(c_path.read_text())
                m_v = json.loads(v_path.read_text()) if v_path.exists() else {"oof_R2": np.nan}
                cond_r2 = m_c["oof_R2"]
                rows.append({
                    "modality":        name,
                    "seed":            seed,
                    "sex":             sex,
                    "voice_R2":        m_v["oof_R2"],
                    "baseline_R2":     m_b["oof_R2"],
                    "conditioned_R2":  cond_r2,
                    "delta_R2":        cond_r2 - m_b["oof_R2"],
                    "beats_baseline":  cond_r2 > m_b["oof_R2"],
                    "beats_voice":     cond_r2 > m_v["oof_R2"],
                    "beats_both":      (cond_r2 > m_b["oof_R2"]) and (cond_r2 > m_v["oof_R2"]),
                })

    if not rows:
        print("No results yet.")
        return

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["modality", "sex"])
        .agg(
            voice_R2_mean      =("voice_R2",       "mean"),
            baseline_R2_mean   =("baseline_R2",    "mean"),
            baseline_R2_std    =("baseline_R2",    "std"),
            conditioned_R2_mean=("conditioned_R2", "mean"),
            conditioned_R2_std =("conditioned_R2", "std"),
            delta_R2_mean      =("delta_R2",       "mean"),
            delta_R2_std       =("delta_R2",       "std"),
            beats_baseline_pct =("beats_baseline", "mean"),
            beats_voice_pct    =("beats_voice",    "mean"),
            beats_both_pct     =("beats_both",     "mean"),
            n_seeds            =("seed",           "count"),
        )
        .reset_index()
        .sort_values(["sex", "conditioned_R2_mean"], ascending=[True, False])
    )

    out_csv = Path(output_base) / "voice_conditioned_hpo_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--host",      type=str,   default="mcluster02")
    parser.add_argument("--partition", type=str,   default="himem7.q,himem8.q")
    parser.add_argument("--cpu",       type=int,   default=10)
    parser.add_argument("--mem",       type=str,   default="20G")
    parser.add_argument("--throttle",  type=int,   default=None)
    parser.add_argument("--elysium-dir", type=str, default=None,
                        help="Override elysium remote dir (default: auto-increments)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override results output dir (default: OUTPUT_BASE in worker)")
    args = parser.parse_args()

    output_base = args.output_dir or OUTPUT_BASE

    if args.summarise:
        summarise(output_base)
        return

    if args.elysium_dir:
        remote_dir = args.elysium_dir
    else:
        elysium_root = args.output_dir or ELYSIUM_BASE
        existing = sorted(Path(elysium_root).glob("elysium*"))
        next_idx = max((int(p.name.replace("elysium", "")) for p in existing
                        if p.name.replace("elysium", "").isdigit()), default=0) + 1
        remote_dir = f"{elysium_root}/elysium{next_idx}"

    cluster_params = ClusterParams(host_name=args.host, remote_dir=remote_dir)
    print(f"Using remote dir: {remote_dir}")
    if args.output_dir:
        print(f"Results output dir: {output_base}")

    worker_cmd = WORKER_BASE
    if args.output_dir:
        worker_cmd += f" --output-dir {args.output_dir}"

    elysium = Elysium(cluster_params)
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
        print("Run with --resume to reconnect after disconnect.")

    col.wait()

    print("\nAll jobs done. Generating summary...")
    summarise(output_base)


if __name__ == "__main__":
    main()
