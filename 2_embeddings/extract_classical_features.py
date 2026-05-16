"""
Extract classical acoustic feature sets from raw voice recordings.

Feature sets
------------
  praat        Praat Voice Report via parselmouth:
               jitter (local/rap/ppq5/ddp), shimmer (local/dB/apq3/apq5/apq11/dda),
               HNR (mean/std), F0 (mean/std/cv), voiced_ratio, CPPS
  egemaps      openSMILE eGeMAPSv02  (88 functionals)
  compare2016  openSMILE ComParE_2016 (6373 functionals)
  emobase      openSMILE emobase      (988 functionals)

Each job writes one chunk parquet.  Run with --merge after all chunks finish.

Usage
-----
  # extract chunk 3 of 21 (0-indexed)
  python extract_classical_features.py \\
      --feature-set egemaps \\
      --chunk-start 1500 --chunk-end 2000 \\
      --jobs 4

  # merge chunks into all_features.parquet
  python extract_classical_features.py --feature-set egemaps --merge
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── paths & constants ───────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
INPUT_DIR = BASE / "Raw_voice"
AUDIO_EXTS = {".flac", ".wav", ".mp3"}

OUTPUT_DIRS: dict[str, Path] = {
    "praat":       BASE / "features_praat",
    "egemaps":     BASE / "features_egemaps",
    "compare2016": BASE / "features_compare2016",
    "emobase":     BASE / "features_emobase",
}

_SMILE_FEATURE_SET = {
    "egemaps":     "eGeMAPSv02",
    "compare2016": "ComParE_2016",
    "emobase":     "emobase",
}

PITCH_FLOOR  = 75.0
PITCH_CEIL   = 600.0
PERIOD_FLOOR = 0.0001
PERIOD_CEIL  = 0.02
MAX_PERIOD   = 1.3
MAX_AMP      = 1.6

TARGET_SR    = 16_000
TRIM_TOP_DB  = 30.0

# ─────────────────────────── shared preprocessing ────────────────────────── #

def _preprocess(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Resample → trim silence → peak-normalize. Mirrors the WavLM pipeline."""
    import librosa
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if y.size == 0:
        raise ValueError("empty after silence trim")
    y = librosa.util.normalize(y)
    return y, sr


# ─────────────────────────── Praat extraction ────────────────────────────── #

def _safe(fn) -> float:
    try:
        v = fn()
        return float(v) if v is not None and np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_praat(path_str: str) -> tuple[str, dict | None, str]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_praat")
    import librosa
    import parselmouth
    from parselmouth import praat

    path = Path(path_str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path_str, sr=None, mono=True)
        if y.size == 0:
            raise ValueError("empty audio")
        y, sr = _preprocess(y, sr)

        snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
        pp  = praat.call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEIL)

        pitch  = snd.to_pitch_ac(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEIL)
        f0     = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]

        harm    = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=PITCH_FLOOR, periods_per_window=4.5)
        # Praat marks unvoiced frames with -200 dB; keep only plausible voiced-frame values
        hnr_vals = harm.values[0]
        hnr_fin  = hnr_vals[np.isfinite(hnr_vals) & (hnr_vals > -200)]

        row: dict[str, float] = {
            "praat_f0_mean":      float(np.mean(voiced)) if len(voiced) else np.nan,
            "praat_f0_std":       float(np.std(voiced))  if len(voiced) > 1 else np.nan,
            "praat_f0_cv":        float(np.std(voiced) / np.mean(voiced))
                                  if len(voiced) > 1 and np.mean(voiced) > 0 else np.nan,
            "praat_voiced_ratio": float(len(voiced) / f0.size) if f0.size else np.nan,
            "praat_hnr_mean":     float(np.mean(hnr_fin)) if len(hnr_fin) else np.nan,
            "praat_hnr_std":      float(np.std(hnr_fin))  if len(hnr_fin) > 1 else np.nan,
            "praat_jitter_local":     _safe(lambda: praat.call(pp, "Get jitter (local)", 0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD)),
            "praat_jitter_rap":       _safe(lambda: praat.call(pp, "Get jitter (rap)",   0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD)),
            "praat_jitter_ppq5":      _safe(lambda: praat.call(pp, "Get jitter (ppq5)", 0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD)),
            "praat_jitter_ddp":       _safe(lambda: praat.call(pp, "Get jitter (ddp)",  0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD)),
            "praat_shimmer_local":    _safe(lambda: praat.call([snd, pp], "Get shimmer (local)",    0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
            "praat_shimmer_local_db": _safe(lambda: praat.call([snd, pp], "Get shimmer (local_dB)", 0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
            "praat_shimmer_apq3":     _safe(lambda: praat.call([snd, pp], "Get shimmer (apq3)",     0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
            "praat_shimmer_apq5":     _safe(lambda: praat.call([snd, pp], "Get shimmer (apq5)",     0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
            "praat_shimmer_apq11":    _safe(lambda: praat.call([snd, pp], "Get shimmer (apq11)",    0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
            "praat_shimmer_dda":      _safe(lambda: praat.call([snd, pp], "Get shimmer (dda)",      0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD, MAX_AMP)),
        }

        def _cpps():
            pc = praat.call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
            return praat.call(pc, "Get CPPS", True, 0.02, 0.0001, 60.0, 333.3, 0.05,
                              "Parabolic", 0.001, 0.0, "Exponential decay", "Robust")
        row["praat_cpps_db"] = _safe(_cpps)

        return path.stem, row, ""
    except Exception as exc:
        return path.stem, None, str(exc)


# ─────────────────────────── openSMILE extraction ────────────────────────── #

_smile_obj = None


def _init_smile(feature_set: str) -> None:
    global _smile_obj
    import opensmile
    _smile_obj = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, _SMILE_FEATURE_SET[feature_set]),
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=1,
        verbose=False,
    )


def _extract_smile(path_str: str) -> tuple[str, dict | None, str]:
    import librosa
    path = Path(path_str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path_str, sr=None, mono=True)
        y, sr = _preprocess(y, sr)
        feat = _smile_obj.process_signal(y.astype(np.float32), sr)
        return path.stem, feat.iloc[0].to_dict(), ""
    except Exception as exc:
        return path.stem, None, str(exc)


# ─────────────────────────── chunk processing ────────────────────────────── #

def discover_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    return files


def process_chunk(
    paths: list[Path],
    feature_set: str,
    output_dir: Path,
    chunk_tag: str,
    n_jobs: int,
) -> None:
    path_strs = [str(p) for p in paths]
    rows: list[dict] = []
    stems: list[str] = []
    errors: list[str] = []

    if feature_set == "praat":
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(extract_praat, p): p for p in path_strs}
            for future in as_completed(futures):
                stem, row, err = future.result()
                stems.append(stem)
                rows.append(row or {})
                if err:
                    errors.append(f"{stem}: {err}")
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_smile,
            initargs=(feature_set,),
        ) as pool:
            futures = {pool.submit(_extract_smile, p): p for p in path_strs}
            for future in as_completed(futures):
                stem, row, err = future.result()
                stems.append(stem)
                rows.append(row or {})
                if err:
                    errors.append(f"{stem}: {err}")

    if errors:
        err_path = output_dir / f"errors_{chunk_tag}.txt"
        err_path.write_text("\n".join(errors))
        print(f"[WARN] {len(errors)} failures → {err_path}")

    df = pd.DataFrame(rows, index=pd.Index(stems, name="audio_stem"))
    out = output_dir / f"chunk_{chunk_tag}.parquet"
    df.to_parquet(out, compression="snappy")
    print(f"Saved {len(df)} rows → {out}")


def merge_chunks(output_dir: Path) -> None:
    chunks = sorted(output_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise SystemExit(f"No chunk parquets found in {output_dir}")
    dfs = [pd.read_parquet(c) for c in chunks]
    merged = pd.concat(dfs)
    merged.index.name = "audio_stem"
    out = output_dir / "all_features.parquet"
    merged.to_parquet(out, compression="snappy")
    print(f"Merged {len(chunks)} chunks → {out}  ({len(merged)} rows, {len(merged.columns)} cols)")


# ─────────────────────────── CLI ─────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", required=True, choices=list(OUTPUT_DIRS))
    p.add_argument("--input-dir",  type=Path, default=INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override default output dir")
    p.add_argument("--chunk-start", type=int, default=None,
                   help="First file index (0-based, inclusive)")
    p.add_argument("--chunk-end",   type=int, default=None,
                   help="Last file index (0-based, exclusive)")
    p.add_argument("--jobs", type=int, default=4,
                   help="Parallel worker processes")
    p.add_argument("--merge", action="store_true",
                   help="Merge chunk parquets into all_features.parquet and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or OUTPUT_DIRS[args.feature_set]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_chunks(output_dir)
        return

    files = discover_files(args.input_dir)
    start = args.chunk_start if args.chunk_start is not None else 0
    end   = args.chunk_end   if args.chunk_end   is not None else len(files)
    chunk = files[start:end]

    if not chunk:
        raise SystemExit(f"No files in range [{start}, {end})")

    chunk_tag = f"{start:05d}_{end:05d}"
    out_path  = output_dir / f"chunk_{chunk_tag}.parquet"
    if out_path.exists():
        print(f"Already exists, skipping: {out_path}")
        return

    print(f"Feature set : {args.feature_set}")
    print(f"Files       : {len(chunk)} ({start}–{end-1})")
    print(f"Workers     : {args.jobs}")
    print(f"Output      : {out_path}")

    process_chunk(chunk, args.feature_set, output_dir, chunk_tag, args.jobs)


if __name__ == "__main__":
    main()
