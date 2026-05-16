#!/usr/bin/env python3
"""
Build the downstream phenotype table from Nastya's per-visit covariate tables.

This reconstructs the file currently consumed by volcano_visualization.py:
    /net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/combined_risk_factors.csv

The output preserves the expected join columns:
    RegistrationCode, research_stage, subject_id
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd


COVARIATES_DIR = Path("/net/mraid20/export/genie/LabData/Analyses/nastya/covariates")
DEFAULT_OUTPUT_PATH = Path(
    "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/combined_risk_factors.csv"
)

VISIT_FILES = {
    "baseline": "10k_cov_mean_updated_baseline.csv",
    "02_00_visit": "10k_cov_mean_updated_02_00_visit.csv",
    "04_00_visit": "10k_cov_mean_updated_04_00_visit.csv",
    "06_00_visit": "10k_cov_mean_updated_06_00_visit.csv",
}


def extract_subject_id(registration_code: pd.Series) -> pd.Series:
    parts = registration_code.astype(str).str.extract(r"^[^_]+_(\d+)")
    if parts[0].isna().any():
        bad_examples = registration_code[parts[0].isna()].head(5).tolist()
        raise ValueError(f"Failed parsing subject_id from RegistrationCode values: {bad_examples}")
    return parts[0].astype(int)


def load_visit_table(research_stage: str, csv_name: str) -> pd.DataFrame:
    path = COVARIATES_DIR / csv_name
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    if "RegistrationCode" not in df.columns:
        raise ValueError(f"{path} is missing RegistrationCode")

    df["research_stage"] = research_stage
    df["subject_id"] = extract_subject_id(df["RegistrationCode"])
    return df


def resolve_visit_files(modified_on: date) -> dict[str, str]:
    selected: dict[str, str] = {}
    for stage, csv_name in VISIT_FILES.items():
        path = COVARIATES_DIR / csv_name
        modified_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        if modified_date == modified_on:
            selected[stage] = csv_name

    if not selected:
        raise ValueError(
            f"No visit tables in {COVARIATES_DIR} were modified on {modified_on.isoformat()}"
        )
    return selected


def build_combined_table(modified_on: date) -> pd.DataFrame:
    visit_files = resolve_visit_files(modified_on)
    frames = [load_visit_table(stage, csv_name) for stage, csv_name in visit_files.items()]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["subject_id", "research_stage"], keep="first").copy()

    key_cols = ["RegistrationCode", "research_stage", "subject_id"]
    other_cols = [c for c in combined.columns if c not in key_cols]
    combined = combined[key_cols + other_cols]
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--modified-on",
        type=date.fromisoformat,
        default=date.today(),
        help="Only include visit tables whose filesystem modified date matches this YYYY-MM-DD value. "
        f"Default: {date.today().isoformat()}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = build_combined_table(args.modified_on)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(
        f"Saved {len(combined):,} rows x {len(combined.columns)} columns to {args.output} "
        f"using tables modified on {args.modified_on.isoformat()}"
    )


if __name__ == "__main__":
    main()
