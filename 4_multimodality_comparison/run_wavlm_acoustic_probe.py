"""
WavLM → acoustic feature probe (expanded, gender-split).

For each validated acoustic aging feature, train a ridge regression model
predicting the feature from WavLM-Large embeddings using 5-fold GroupKFold CV
(groups = subjects). Runs three times: all subjects, females only, males only.

Features cover the full modern voice-aging landscape:
  Perturbation      : jitter (local, RAP, PPQ5, eGeMAPSv02), shimmer (local, APQ5, APQ11, local_dB)
  Cepstral          : CPP, MFCC 1–4 (overall & voiced-frame)
  Glottal quality   : H1-H2 (breathiness), H1-A3 (open quotient proxy)
  Spectral / tilt   : alpha ratio (voiced & unvoiced), Hammarberg index (voiced & unvoiced),
                      spectral slope (0-500, 500-1500 Hz, voiced & unvoiced), spectral flux
  Formants          : F1/F2/F3 frequency, bandwidth & amplitude
  Harmonicity       : HNR (Praat, eGeMAPSv02), HNR variability
  F0 dynamics       : F0 mean, SD, CV, percentile range, rising/falling slope
  Voicing / breaks  : voiced ratio (Praat), voiced segments/sec, mean voiced segment length,
                      voiced segment SD, mean unvoiced segment length
  Loudness          : loudness mean, SD, loudness peaks/sec

Outputs
-------
  analysis_outputs/step_p5_wavlm_probe/probe_results_full.csv
  analysis_outputs/step_p5_wavlm_probe/probe_r2_gender_split.pdf/.png

Usage
-----
  python run_wavlm_acoustic_probe.py
  python run_wavlm_acoustic_probe.py --smoke   # 300 recordings, no plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── paths ───────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
WAVLM_CSV           = BASE / "WavLM_features.csv"
PRAAT_PARQUET       = BASE / "features_praat"   / "all_features.parquet"
EGEMAPS_PARQUET     = BASE / "features_egemaps" / "all_features.parquet"
SUBJECT_DETAILS_CSV = BASE / "subject_details_df_Oct25.csv"
OUTPUT_DIR = Path(__file__).parents[2] / "analysis_outputs" / "step_p5_wavlm_probe"

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_SPLITS = 5

# ── Feature registry ──────────────────────────────────────────────────────── #
# (column_name, display_label, source_parquet)
PROBE_FEATURES: list[tuple[str, str, str]] = [
    # Perturbation (Praat)
    ("praat_jitter_local",              "Jitter (local, Praat)",         "praat"),
    ("praat_jitter_rap",                "Jitter (RAP)",                  "praat"),
    ("praat_jitter_ppq5",               "Jitter (PPQ5)",                 "praat"),
    ("praat_shimmer_local",             "Shimmer (local, Praat)",        "praat"),
    ("praat_shimmer_local_db",          "Shimmer (local, dB)",           "praat"),
    ("praat_shimmer_apq5",              "Shimmer (APQ5)",                "praat"),
    ("praat_shimmer_apq11",             "Shimmer (APQ11)",               "praat"),
    # Perturbation (eGeMAPSv02)
    ("jitterLocal_sma3nz_amean",        "Jitter (local, eGeMAPSv02)",   "egemaps"),
    ("shimmerLocaldB_sma3nz_amean",     "Shimmer (local dB, eGeMAPSv02)","egemaps"),
    # Cepstral
    ("praat_cpps_db",                   "CPP",                           "praat"),
    ("mfcc1_sma3_amean",                "MFCC 1",                        "egemaps"),
    ("mfcc2_sma3_amean",                "MFCC 2",                        "egemaps"),
    ("mfcc3_sma3_amean",                "MFCC 3",                        "egemaps"),
    ("mfcc4_sma3_amean",                "MFCC 4",                        "egemaps"),
    ("mfcc1V_sma3nz_amean",             "MFCC 1 (voiced)",               "egemaps"),
    ("mfcc2V_sma3nz_amean",             "MFCC 2 (voiced)",               "egemaps"),
    ("mfcc3V_sma3nz_amean",             "MFCC 3 (voiced)",               "egemaps"),
    ("mfcc4V_sma3nz_amean",             "MFCC 4 (voiced)",               "egemaps"),
    # Glottal / voice quality
    ("logRelF0-H1-H2_sma3nz_amean",     "H1–H2 (breathiness)",           "egemaps"),
    ("logRelF0-H1-A3_sma3nz_amean",     "H1–A3 (open quotient proxy)",   "egemaps"),
    # Spectral tilt / energy balance
    ("alphaRatioV_sma3nz_amean",        "Alpha ratio (voiced)",          "egemaps"),
    ("alphaRatioUV_sma3nz_amean",       "Alpha ratio (unvoiced)",        "egemaps"),
    ("hammarbergIndexV_sma3nz_amean",   "Hammarberg index (voiced)",     "egemaps"),
    ("hammarbergIndexUV_sma3nz_amean",  "Hammarberg index (unvoiced)",   "egemaps"),
    ("slopeV0-500_sma3nz_amean",        "Spectral slope 0–500 Hz (V)",   "egemaps"),
    ("slopeV500-1500_sma3nz_amean",     "Spectral slope 500–1500 Hz (V)","egemaps"),
    ("slopeUV0-500_sma3nz_amean",       "Spectral slope 0–500 Hz (UV)",  "egemaps"),
    ("slopeUV500-1500_sma3nz_amean",    "Spectral slope 500–1500 Hz (UV)","egemaps"),
    ("spectralFluxV_sma3nz_amean",      "Spectral flux (voiced)",        "egemaps"),
    ("spectralFluxUV_sma3nz_amean",     "Spectral flux (unvoiced)",      "egemaps"),
    # Formants
    ("F1frequency_sma3nz_amean",        "F1 frequency",                  "egemaps"),
    ("F1bandwidth_sma3nz_amean",        "F1 bandwidth",                  "egemaps"),
    ("F1amplitudeLogRelF0_sma3nz_amean","F1 amplitude (rel. F0)",        "egemaps"),
    ("F2frequency_sma3nz_amean",        "F2 frequency",                  "egemaps"),
    ("F2bandwidth_sma3nz_amean",        "F2 bandwidth",                  "egemaps"),
    ("F2amplitudeLogRelF0_sma3nz_amean","F2 amplitude (rel. F0)",        "egemaps"),
    ("F3frequency_sma3nz_amean",        "F3 frequency",                  "egemaps"),
    ("F3bandwidth_sma3nz_amean",        "F3 bandwidth",                  "egemaps"),
    ("F3amplitudeLogRelF0_sma3nz_amean","F3 amplitude (rel. F0)",        "egemaps"),
    # Harmonicity
    ("praat_hnr_mean",                  "HNR mean (Praat)",              "praat"),
    ("praat_hnr_std",                   "HNR SD (Praat)",                "praat"),
    ("HNRdBACF_sma3nz_amean",           "HNR (eGeMAPSv02)",              "egemaps"),
    ("HNRdBACF_sma3nz_stddevNorm",      "HNR variability (eGeMAPSv02)",  "egemaps"),
    # F0 dynamics
    ("praat_f0_mean",                   "F0 mean",                       "praat"),
    ("praat_f0_std",                    "F0 SD",                         "praat"),
    ("praat_f0_cv",                     "F0 CV",                         "praat"),
    ("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", "F0 range (semitones)","egemaps"),
    ("F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope", "F0 rising slope",  "egemaps"),
    ("F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0 falling slope",  "egemaps"),
    # Voicing continuity / breaks
    ("praat_voiced_ratio",              "Voiced ratio (Praat)",          "praat"),
    ("VoicedSegmentsPerSec",            "Voiced segments/sec",           "egemaps"),
    ("MeanVoicedSegmentLengthSec",      "Mean voiced seg. length",       "egemaps"),
    ("StddevVoicedSegmentLengthSec",    "SD voiced seg. length",         "egemaps"),
    ("MeanUnvoicedSegmentLength",       "Mean unvoiced seg. length",     "egemaps"),
    # Loudness
    ("loudness_sma3_amean",             "Loudness mean",                 "egemaps"),
    ("loudness_sma3_stddevNorm",        "Loudness SD",                   "egemaps"),
    ("loudnessPeaksPerSec",             "Loudness peaks/sec",            "egemaps"),
    ("equivalentSoundLevel_dBp",        "Equivalent sound level",        "egemaps"),
]

CATEGORY_ORDER = [
    "Perturbation", "Cepstral", "Glottal quality", "Spectral / tilt",
    "Formants", "Harmonicity", "F0 dynamics", "Voicing / breaks", "Loudness",
]

FEATURE_CATEGORY: dict[str, str] = {
    "praat_jitter_local": "Perturbation",
    "praat_jitter_rap": "Perturbation",
    "praat_jitter_ppq5": "Perturbation",
    "praat_shimmer_local": "Perturbation",
    "praat_shimmer_local_db": "Perturbation",
    "praat_shimmer_apq5": "Perturbation",
    "praat_shimmer_apq11": "Perturbation",
    "jitterLocal_sma3nz_amean": "Perturbation",
    "shimmerLocaldB_sma3nz_amean": "Perturbation",
    "praat_cpps_db": "Cepstral",
    "mfcc1_sma3_amean": "Cepstral",
    "mfcc2_sma3_amean": "Cepstral",
    "mfcc3_sma3_amean": "Cepstral",
    "mfcc4_sma3_amean": "Cepstral",
    "mfcc1V_sma3nz_amean": "Cepstral",
    "mfcc2V_sma3nz_amean": "Cepstral",
    "mfcc3V_sma3nz_amean": "Cepstral",
    "mfcc4V_sma3nz_amean": "Cepstral",
    "logRelF0-H1-H2_sma3nz_amean": "Glottal quality",
    "logRelF0-H1-A3_sma3nz_amean": "Glottal quality",
    "alphaRatioV_sma3nz_amean": "Spectral / tilt",
    "alphaRatioUV_sma3nz_amean": "Spectral / tilt",
    "hammarbergIndexV_sma3nz_amean": "Spectral / tilt",
    "hammarbergIndexUV_sma3nz_amean": "Spectral / tilt",
    "slopeV0-500_sma3nz_amean": "Spectral / tilt",
    "slopeV500-1500_sma3nz_amean": "Spectral / tilt",
    "slopeUV0-500_sma3nz_amean": "Spectral / tilt",
    "slopeUV500-1500_sma3nz_amean": "Spectral / tilt",
    "spectralFluxV_sma3nz_amean": "Spectral / tilt",
    "spectralFluxUV_sma3nz_amean": "Spectral / tilt",
    "F1frequency_sma3nz_amean": "Formants",
    "F1bandwidth_sma3nz_amean": "Formants",
    "F1amplitudeLogRelF0_sma3nz_amean": "Formants",
    "F2frequency_sma3nz_amean": "Formants",
    "F2bandwidth_sma3nz_amean": "Formants",
    "F2amplitudeLogRelF0_sma3nz_amean": "Formants",
    "F3frequency_sma3nz_amean": "Formants",
    "F3bandwidth_sma3nz_amean": "Formants",
    "F3amplitudeLogRelF0_sma3nz_amean": "Formants",
    "praat_hnr_mean": "Harmonicity",
    "praat_hnr_std": "Harmonicity",
    "HNRdBACF_sma3nz_amean": "Harmonicity",
    "HNRdBACF_sma3nz_stddevNorm": "Harmonicity",
    "praat_f0_mean": "F0 dynamics",
    "praat_f0_std": "F0 dynamics",
    "praat_f0_cv": "F0 dynamics",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "F0 dynamics",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": "F0 dynamics",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": "F0 dynamics",
    "praat_voiced_ratio": "Voicing / breaks",
    "VoicedSegmentsPerSec": "Voicing / breaks",
    "MeanVoicedSegmentLengthSec": "Voicing / breaks",
    "StddevVoicedSegmentLengthSec": "Voicing / breaks",
    "MeanUnvoicedSegmentLength": "Voicing / breaks",
    "loudness_sma3_amean": "Loudness",
    "loudness_sma3_stddevNorm": "Loudness",
    "loudnessPeaksPerSec": "Loudness",
    "equivalentSoundLevel_dBp": "Loudness",
}

# ─────────────────────────── core probe ──────────────────────────────────── #

def probe_one_feature(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """5-fold GroupKFold ridge with nested alpha selection. Returns R², r, p, n."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        train_groups = groups[train_idx]
        unique_g = np.unique(train_groups)
        n_val = max(1, int(len(unique_g) * 0.2))
        val_groups = unique_g[:n_val]
        inner_mask = np.isin(train_groups, val_groups)
        X_itr, X_ival = X_tr[~inner_mask], X_tr[inner_mask]
        y_itr, y_ival = y_tr[~inner_mask], y_tr[inner_mask]

        scaler = StandardScaler()
        X_itr_s  = scaler.fit_transform(X_itr)
        X_ival_s = scaler.transform(X_ival)

        best_alpha, best_r2 = ALPHA_CANDIDATES[0], -np.inf
        for a in ALPHA_CANDIDATES:
            ridge = Ridge(alpha=a)
            ridge.fit(X_itr_s, y_itr)
            pred = ridge.predict(X_ival_s)
            ss_res = np.sum((y_ival - pred) ** 2)
            ss_tot = np.sum((y_ival - y_ival.mean()) ** 2) + 1e-10
            r2_val = 1.0 - ss_res / ss_tot
            if r2_val > best_r2:
                best_r2, best_alpha = r2_val, a

        scaler_f = StandardScaler()
        ridge_f  = Ridge(alpha=best_alpha)
        ridge_f.fit(scaler_f.fit_transform(X_tr), y_tr)
        oof_pred[test_idx] = ridge_f.predict(scaler_f.transform(X_te))

    valid = np.isfinite(oof_pred) & np.isfinite(y)
    y_v, p_v = y[valid], oof_pred[valid]
    ss_res = np.sum((y_v - p_v) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2) + 1e-10
    r2  = float(1.0 - ss_res / ss_tot)
    r, p = pearsonr(y_v, p_v)
    return {"r2": r2, "r": float(r), "p": float(p), "n": int(valid.sum())}


def _probe_subset(
    X_full: np.ndarray,
    df: pd.DataFrame,
    mask: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> dict[str, dict]:
    """Run all probe features on a subset defined by boolean mask."""
    X_sub  = X_full[mask]
    grp_sub = groups[mask]
    results = {}
    for col, display, _ in PROBE_FEATURES:
        if col not in df.columns:
            continue
        y_all = df[col].to_numpy().astype(float)
        y_sub = y_all[mask]
        valid = np.isfinite(y_sub)
        if valid.sum() < 50:
            print(f"  [SKIP-{label}] {col}: only {valid.sum()} valid rows")
            continue
        m = probe_one_feature(X_sub[valid], y_sub[valid], grp_sub[valid])
        results[col] = m
        print(f"  [{label}] {display:<35} R²={m['r2']:+.3f}  r={m['r']:+.3f}  p={m['p']:.2e}  n={m['n']}")
    return results


# ─────────────────────────── main run ────────────────────────────────────── #

def run(smoke: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading WavLM embeddings …")
    wavlm = pd.read_csv(WAVLM_CSV, index_col=0)
    wavlm = wavlm.astype("float32")
    wavlm.index.name = "filename"

    print("Loading Praat features …")
    praat = pd.read_parquet(PRAAT_PARQUET)
    praat.index.name = "filename"

    print("Loading eGeMAPSv02 features …")
    egemaps = pd.read_parquet(EGEMAPS_PARQUET)
    egemaps.index.name = "filename"

    print("Loading subject details …")
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, index_col="filename",
                     usecols=["filename", "gender", "subject_number"])

    df = (wavlm
          .join(praat,   how="inner")
          .join(egemaps, how="inner")
          .join(sd,      how="inner")
          .copy())
    print(f"Matched recordings: {len(df)}")

    if smoke:
        df = df.iloc[:300].copy()

    df["subject"] = df["subject_number"].astype(str)
    groups = df["subject"].to_numpy()

    embed_cols = [c for c in df.columns if c.startswith("feature_")]
    X = SimpleImputer(strategy="median").fit_transform(df[embed_cols].to_numpy())

    mask_all    = np.ones(len(df), dtype=bool)
    mask_female = (df["gender"].to_numpy() == 0)
    mask_male   = (df["gender"].to_numpy() == 1)

    print(f"\n{'='*60}")
    print(f"All subjects (n={mask_all.sum()})")
    print(f"{'='*60}")
    res_all = _probe_subset(X, df, mask_all, groups, "ALL")

    print(f"\n{'='*60}")
    print(f"Female (n={mask_female.sum()})")
    print(f"{'='*60}")
    res_f = _probe_subset(X, df, mask_female, groups, "F")

    print(f"\n{'='*60}")
    print(f"Male (n={mask_male.sum()})")
    print(f"{'='*60}")
    res_m = _probe_subset(X, df, mask_male, groups, "M")

    # ── Build combined results table ──────────────────────────────────────── #
    rows = []
    for col, display, _ in PROBE_FEATURES:
        if col not in res_all:
            continue
        row = {
            "feature":   col,
            "label":     display,
            "category":  FEATURE_CATEGORY.get(col, "Other"),
            "r2_all":    res_all[col]["r2"],
            "r_all":     res_all[col]["r"],
            "p_all":     res_all[col]["p"],
            "n_all":     res_all[col]["n"],
        }
        for key, res in (("female", res_f), ("male", res_m)):
            if col in res:
                row[f"r2_{key}"] = res[col]["r2"]
                row[f"r_{key}"]  = res[col]["r"]
                row[f"p_{key}"]  = res[col]["p"]
                row[f"n_{key}"]  = res[col]["n"]
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("r2_all", ascending=False)
    out_csv = OUTPUT_DIR / "probe_results_full.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")
    print(results_df[["label", "r2_all", "r2_female", "r2_male"]].to_string(index=False))

    if not smoke:
        _plot(results_df)

    return results_df


# ─────────────────────────── plot ────────────────────────────────────────── #

def _plot(df: pd.DataFrame) -> None:
    # Sort by category order then by r2_all within category
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df = df.copy()
    df["cat_rank"] = df["category"].map(cat_rank).fillna(99)
    df = df.sort_values(["cat_rank", "r2_all"], ascending=[True, True])

    labels   = df["label"].tolist()
    r2_all   = df["r2_all"].tolist()
    r2_f     = df.get("r2_female", pd.Series([np.nan]*len(df))).tolist()
    r2_m     = df.get("r2_male",   pd.Series([np.nan]*len(df))).tolist()

    n = len(labels)
    y_pos = np.arange(n)
    bar_h = 0.26

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.38)))

    # Category colors
    cat_colors = {
        "Perturbation": "#4878d0",
        "Cepstral":     "#ee854a",
        "Spectral / tilt": "#6acc65",
        "Formants":     "#d65f5f",
        "Harmonicity":  "#956cb4",
        "F0 dynamics":  "#8c613c",
        "Voicing":      "#dc7ec0",
    }
    bar_colors = [cat_colors.get(df["category"].iloc[i], "#aaaaaa") for i in range(n)]

    ax.barh(y_pos + bar_h,   r2_all, bar_h, color=bar_colors,  alpha=0.9, label="All")
    ax.barh(y_pos,           r2_f,   bar_h, color=bar_colors,  alpha=0.55, hatch="//", label="Female")
    ax.barh(y_pos - bar_h,   r2_m,   bar_h, color=bar_colors,  alpha=0.35, hatch="..", label="Male")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Out-of-fold R²", fontsize=9)
    ax.set_title(
        "WavLM-Large acoustic feature probe\n"
        "(5-fold GroupKFold ridge, 1024-dim → scalar)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")

    # Category separators
    prev_cat = None
    for i in range(n):
        cat = df["category"].iloc[i]
        if cat != prev_cat and i > 0:
            ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle="--")
        prev_cat = cat

    # Annotate all-subjects R² values
    for i, r2 in enumerate(r2_all):
        if np.isfinite(r2):
            ax.text(max(r2 + 0.01, 0.01), y_pos[i] + bar_h,
                    f"{r2:.3f}", va="center", fontsize=6.5)

    plt.tight_layout()
    out = OUTPUT_DIR / "probe_r2_gender_split.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Plot → {out}")


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Quick test on 300 recordings")
    args = p.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
