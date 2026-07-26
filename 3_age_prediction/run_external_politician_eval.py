"""
External-validation evaluation of the WavLM-Large age-prediction model on the
Hebrew "politician" dataset.

- Trains per-gender Ridge regression on the manuscript's clean, quality-filtered
  WavLM features (ages 40-70), choosing alpha by GroupKFold inner CV — same
  pipeline as run_age_prediction_filtered.py / ridge_regression.py.
- Refits each per-gender model on the FULL filtered training data.
- Applies the trained models to the politician WavLM-Large mean embeddings,
  per gender, and joins predictions with the manifest ages.
- Reports MAE / R^2 / Pearson on all segments, per-gender segments, per-speaker
  aggregates, and (filtered) speakers inside the 40-70 training age range.
- Saves predictions + a scatter plot.

Run:
    python run_external_politician_eval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

# ─────────────────────────── paths ────────────────────────────────────────── #

TRAIN_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/WavLM_features_filtered_with_RF.csv"

PRED_DIR = Path("/net/mraid20/export/genie/LabData/Analyses/AudioLejepa/data/predictions")
POLITICIAN_PARQUET = PRED_DIR / "wavlm_large_politician" / "mean_embeddings.parquet"
POLITICIAN_MANIFEST = PRED_DIR / "mean_manifest_politician.csv"

OUTPUT_DIR = Path("/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/external_politician_wavlm_ridge")

# ─────────────────────────── config ───────────────────────────────────────── #

MIN_AGE, MAX_AGE = 40, 70
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────── helpers ──────────────────────────────────────── #

def load_training() -> pd.DataFrame:
    print(f"Loading training features: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV, index_col=0)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    print(f"  {len(df)} recordings, {df['subject_number'].nunique()} subjects, "
          f"{len(feat_cols)} WavLM features")
    print(f"  female={(df['gender']==0).sum()}  male={(df['gender']==1).sum()}")
    return df, feat_cols


def pick_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """GroupKFold inner CV; pick alpha minimising mean MAE."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    best_alpha, best_mae = None, np.inf
    for alpha in ALPHA_CANDIDATES:
        maes = []
        for tr, va in gkf.split(X, y, groups):
            imp = SimpleImputer(strategy="median").fit(X[tr])
            Xtr, Xva = imp.transform(X[tr]), imp.transform(X[va])
            mdl = Ridge(alpha=alpha, random_state=RANDOM_STATE).fit(Xtr, y[tr])
            maes.append(mean_absolute_error(y[va], mdl.predict(Xva)))
        mae = float(np.mean(maes))
        print(f"    alpha={alpha:<7} MAE={mae:.3f}")
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    return best_alpha


def fit_final(X: np.ndarray, y: np.ndarray, alpha: float):
    """Fit median-imputer + Ridge on the FULL per-gender training set."""
    imp = SimpleImputer(strategy="median").fit(X)
    mdl = Ridge(alpha=alpha, random_state=RANDOM_STATE).fit(imp.transform(X), y)
    return imp, mdl


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    alpha: float) -> np.ndarray:
    """GroupKFold out-of-fold predictions on the training set — used to estimate
    the regression-to-the-mean bias (predictions are compressed toward the
    training mean)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    for tr, va in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        mdl = Ridge(alpha=alpha, random_state=RANDOM_STATE).fit(imp.transform(X[tr]), y[tr])
        oof[va] = mdl.predict(imp.transform(X[va]))
    return oof


def fit_bias_correction(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit y_pred = a + b * y_true (OLS) on training OOF predictions.
    Returns (a, b); calibrated predictions on new data are (y_pred - a) / b.
    This is the standard 'regression to the mean' / brain-age style correction
    (Beheshti et al., de Lange & Cole)."""
    b, a = np.polyfit(y_true, y_pred, 1)
    return float(a), float(b)


def apply_bias_correction(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return (y_pred - a) / b


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2 or np.std(y_pred) == 0:
        return {"n": int(len(y_true)),
                "MAE": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else float("nan"),
                "R2": float("nan"),
                "Pearson_r": float("nan"),
                "Pearson_p": float("nan")}
    r, p = pearsonr(y_true, y_pred)
    return {"n": int(len(y_true)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Pearson_r": float(r),
            "Pearson_p": float(p)}


def block(label: str, m: dict) -> str:
    return (f"  {label:<32} n={m['n']:>5}  MAE={m['MAE']:.3f}  "
            f"R2={m['R2']:.3f}  r={m['Pearson_r']:.3f}")


# ─────────────────────────── main ─────────────────────────────────────────── #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, feat_cols = load_training()

    # 1. fit one ridge per gender on the manuscript's clean filtered data,
    #    AND fit a bias-correction on its OOF predictions
    fitted = {}
    chosen_alpha = {}
    bias = {}    # gender → (a, b) such that calibrated = (pred - a) / b
    for g_val, g_lab in [(0, "female"), (1, "male")]:
        sub = train_df[train_df["gender"] == g_val]
        Xg = sub[feat_cols].to_numpy(dtype=float)
        yg = sub["age"].to_numpy(dtype=float)
        grp = sub["subject_number"].to_numpy()
        print(f"\n--- Picking alpha for {g_lab} (n={len(sub)}) ---")
        alpha = pick_alpha(Xg, yg, grp)
        print(f"  → chosen alpha = {alpha}")
        chosen_alpha[g_lab] = alpha
        fitted[g_lab] = fit_final(Xg, yg, alpha)

        oof = oof_predictions(Xg, yg, grp, alpha)
        a, b = fit_bias_correction(yg, oof)
        bias[g_lab] = (a, b)
        oof_corr = apply_bias_correction(oof, a, b)
        print(f"  bias-correction fit on OOF preds: y_pred = {a:.3f} + {b:.4f} * y_true")
        print(f"    uncorrected OOF — MAE={mean_absolute_error(yg, oof):.3f}  "
              f"R2={r2_score(yg, oof):.3f}  r={pearsonr(yg, oof)[0]:.3f}")
        print(f"    corrected   OOF — MAE={mean_absolute_error(yg, oof_corr):.3f}  "
              f"R2={r2_score(yg, oof_corr):.3f}  r={pearsonr(yg, oof_corr)[0]:.3f}")

    # 2. politician embeddings + manifest
    print(f"\nLoading politician embeddings + manifest …")
    emb = pd.read_parquet(POLITICIAN_PARQUET)        # filename, embedding, speaker_id, target, n_segments, age_raw
    man = pd.read_csv(POLITICIAN_MANIFEST, low_memory=False)
    df = emb.merge(
        man[["filename", "age", "gender", "type", "speaker_id"]],
        on=["filename", "speaker_id"], how="inner",
    )
    print(f"  segments after merge: {len(df)} (manifest={len(man)}, emb={len(emb)})")

    # expand embedding column to numeric matrix
    X = np.vstack(df["embedding"].values).astype(float)
    print(f"  embedding matrix: {X.shape}")

    # 3. predict per gender (raw + bias-corrected)
    preds = np.full(len(df), np.nan)
    preds_corr = np.full(len(df), np.nan)
    for g_val, g_lab in [(0, "female"), (1, "male")]:
        mask = df["gender"].to_numpy() == g_val
        if mask.sum() == 0:
            continue
        imp, mdl = fitted[g_lab]
        raw = mdl.predict(imp.transform(X[mask]))
        preds[mask] = raw
        a, b = bias[g_lab]
        preds_corr[mask] = apply_bias_correction(raw, a, b)
        print(f"  {g_lab}: predicted {mask.sum()} segments "
              f"(alpha={chosen_alpha[g_lab]}, bias a={a:.3f}, b={b:.4f})")

    df["pred_age"] = preds
    df["pred_age_corrected"] = preds_corr

    # 4. per-segment + per-speaker tables (raw + corrected)
    seg_table = df[["speaker_id", "filename", "type", "gender", "age",
                    "pred_age", "pred_age_corrected"]].copy()
    seg_table.to_csv(OUTPUT_DIR / "segment_predictions.csv", index=False)

    spk_table = (
        seg_table.groupby("speaker_id")
                 .agg(age=("age", "first"),
                      gender=("gender", "first"),
                      type_majority=("type", lambda s: s.mode().iloc[0]),
                      pred_age_mean=("pred_age", "mean"),
                      pred_age_corrected_mean=("pred_age_corrected", "mean"),
                      n_segments=("filename", "count"))
                 .reset_index()
    )
    spk_table.to_csv(OUTPUT_DIR / "speaker_predictions.csv", index=False)

    # 5. metrics (raw + bias-corrected, per segment + per speaker)
    print("\n" + "=" * 72)
    print("External evaluation — WavLM-Large Ridge, trained on clean filtered data")
    print("=" * 72)

    report = {
        "chosen_alpha": chosen_alpha,
        "bias_correction": {g: {"a": bias[g][0], "b": bias[g][1]} for g in bias},
        "by_segment": {"raw": {}, "corrected": {}},
        "by_speaker": {"raw": {}, "corrected": {}},
    }

    def fill(frame, ycol, level, store):
        store["all"] = metrics(frame["age"].values, frame[ycol].values)
        print(block(f"[{level}] ALL", store["all"]))
        for g_val, g_lab in [(0, "female"), (1, "male")]:
            sub = frame[frame["gender"] == g_val]
            store[g_lab] = metrics(sub["age"].values, sub[ycol].values)
            print(block(f"[{level}] {g_lab}", store[g_lab]))
        in_rng = frame[(frame["age"] >= MIN_AGE) & (frame["age"] <= MAX_AGE)]
        key = f"all_age_{MIN_AGE}_{MAX_AGE}"
        store[key] = metrics(in_rng["age"].values, in_rng[ycol].values)
        print(block(f"[{level}] ALL (age {MIN_AGE}-{MAX_AGE})", store[key]))

    print("\n[Per segment — RAW]")
    fill(seg_table, "pred_age", "seg/raw", report["by_segment"]["raw"])
    print("\n[Per segment — CORRECTED]")
    fill(seg_table, "pred_age_corrected", "seg/cor", report["by_segment"]["corrected"])

    print("\n[Per speaker — RAW]")
    fill(spk_table, "pred_age_mean", "spk/raw", report["by_speaker"]["raw"])
    print("\n[Per speaker — CORRECTED]")
    fill(spk_table, "pred_age_corrected_mean", "spk/cor", report["by_speaker"]["corrected"])

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # 6. scatter plot — 2×2: rows = segment / speaker, cols = raw / corrected
    fig, ax = plt.subplots(2, 2, figsize=(13, 11), sharex=False)
    panels = [
        (seg_table, "pred_age",                   "Per segment — raw",        ax[0, 0]),
        (seg_table, "pred_age_corrected",         "Per segment — corrected",  ax[0, 1]),
        (spk_table, "pred_age_mean",              "Per speaker — raw",        ax[1, 0]),
        (spk_table, "pred_age_corrected_mean",    "Per speaker — corrected",  ax[1, 1]),
    ]
    for frame, ycol, title, axi in panels:
        for g_val, g_lab, c in [(0, "female", "tab:red"), (1, "male", "tab:blue")]:
            sub = frame[frame["gender"] == g_val]
            axi.scatter(sub["age"], sub[ycol], s=8, alpha=0.35,
                        c=c, label=f"{g_lab} (n={len(sub)})")
        lo = min(frame["age"].min(), frame[ycol].min())
        hi = max(frame["age"].max(), frame[ycol].max())
        axi.plot([lo, hi], [lo, hi], "k--", lw=1)
        axi.axvspan(MIN_AGE, MAX_AGE, color="grey", alpha=0.08, label="train range")
        axi.set_xlabel("True age (years)")
        axi.set_ylabel("Predicted age (years)")
        axi.set_title(title)
        axi.legend(fontsize=8)
    fig.suptitle("External validation — Hebrew politicians (WavLM-Large + Ridge), "
                 "raw vs. regression-to-the-mean corrected")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_true_vs_pred.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved everything → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
