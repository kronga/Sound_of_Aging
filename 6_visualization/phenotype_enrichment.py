#!/usr/bin/env python3
"""Add visit-matched DXA VAT area to the downstream phenotype table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DXA_TOTAL_BODY_SCAN_PATH = Path(
    "/net/mraid20/export/genie/LabData/Data/10K/warehouse/"
    "dxa_total_body_core_scan.csv"
)
VAT_AREA_COLUMN = "VAT area (DXA)"
INVALID_VAT_COLUMNS = ("Scanned VAT mass (DXA)", "vat/fat")


def load_vat_area_by_visit() -> pd.DataFrame:
    """Return one robust VAT-area value for each participant and study visit.

    The source occasionally contains duplicate exports of the same scan, with
    one value rounded and the other retaining full precision. Their median is
    therefore used. Unlike the derived VAT-mass column, VAT area is not passed
    through the threshold-based gram-to-kilogram conversion.
    """
    vat = pd.read_csv(
        DXA_TOTAL_BODY_SCAN_PATH,
        usecols=["participant_id", "research_stage", "vat_area"],
    )
    vat = vat.dropna(
        subset=["participant_id", "research_stage", "vat_area"]
    ).copy()
    vat["participant_id"] = vat["participant_id"].astype("int64")
    vat["research_stage"] = vat["research_stage"].replace(
        {"00_00_visit": "baseline"}
    )
    vat = (
        vat.groupby(["participant_id", "research_stage"], as_index=False)[
            "vat_area"
        ]
        .median()
        .rename(
            columns={
                "participant_id": "subject_id",
                "vat_area": VAT_AREA_COLUMN,
            }
        )
    )
    return vat


def add_vat_area(
    risk_factors: pd.DataFrame,
    *,
    drop_invalid_vat_columns: bool = True,
) -> pd.DataFrame:
    """Merge VAT area by participant and visit into a risk-factor table."""
    rf = risk_factors.copy()
    if "subject_id" not in rf.columns:
        if "subject_number" in rf.columns:
            rf["subject_id"] = rf["subject_number"]
        else:
            raise ValueError(
                "Risk-factor table requires subject_id or subject_number"
            )

    rf["subject_id"] = pd.to_numeric(rf["subject_id"], errors="raise").astype(
        "int64"
    )
    rf = rf.drop(columns=[VAT_AREA_COLUMN], errors="ignore")
    rf = rf.merge(
        load_vat_area_by_visit(),
        on=["subject_id", "research_stage"],
        how="left",
        validate="many_to_one",
    )

    if drop_invalid_vat_columns:
        rf = rf.drop(columns=list(INVALID_VAT_COLUMNS), errors="ignore")
    return rf
