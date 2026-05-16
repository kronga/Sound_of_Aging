"""
Power analysis for WavLM voice age prediction.

For each sample size and gender, subsamples the given number of subjects
from a fixed training pool (80% of subjects), trains a ridge regression model
(with nested alpha selection on a 20% inner holdout), and evaluates on a
fixed held-out test set (20% of subjects). Repeats with N_SEEDS random seeds
per sample size to estimate 95% confidence intervals.

A final "full pool" point is added where n = entire training pool; to get CI
at full capacity, the 80/20 train/test split itself is varied across N_SEEDS seeds
(rather than subsampling within a fixed split).

Outputs
-------
  paper_revision_outputs/power_analysis/power_analysis_results.csv
  paper_revision_outputs/power_analysis/power_analysis_learning_curve.pdf/.png

Usage
-----
  python power_analysis_age_prediction.py
  python power_analysis_age_prediction.py --smoke   # fast run, fewer seeds
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── config ──────────────────────────────────────── #

BASE = Path("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length")
# QC-filtered embeddings (RF classifier removed low-quality recordings)
WAVLM_CSV = BASE / "WavLM_features_filtered_with_RF.csv"
OUTPUT_DIR = Path(__file__).parents[2] / "paper_revision_outputs" / "power_analysis"

MIN_AGE, MAX_AGE = 40, 70
TEST_FRAC        = 0.20   # fixed holdout fraction of subjects
N_SEEDS          = 30     # seeds per sample size
HOLDOUT_SEED     = 0      # seed for train/test split (fixed for subsample runs)
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
INNER_VAL_FRAC   = 0.20   # fraction of subsample used for alpha selection

# Sample sizes to test (subjects, not recordings)
SAMPLE_SIZES = [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500,
                2000, 2500, 3000, 3500]

# ─────────────────────────── data loading ────────────────────────────────── #

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_female, y_female, X_male, y_male) using QC-filtered embeddings."""
    print("Loading QC-filtered WavLM embeddings …")
    df = pd.read_csv(WAVLM_CSV, index_col=0)

    df = df.dropna(subset=["age", "subject_number"])
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    df = df[~df["subject_number"].duplicated(keep="first")]

    print(f"Subjects after QC + age filter + dedup: {len(df)}  "
          f"(female={(df['gender']==0).sum()}, male={(df['gender']==1).sum()})")

    embed_cols = [c for c in df.columns if c.startswith("feature_")]
    X_all = SimpleImputer(strategy="median").fit_transform(df[embed_cols].to_numpy())
    y_all = df["age"].to_numpy()
    g_all = df["gender"].to_numpy()

    return (X_all[g_all == 0], y_all[g_all == 0],
            X_all[g_all == 1], y_all[g_all == 1])


# ─────────────────────────── model ───────────────────────────────────────── #

def _fit_eval(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Train ridge with inner alpha selection; evaluate on held-out test set."""
    n_inner_val = max(5, int(len(y_tr) * INNER_VAL_FRAC))
    val_idx = rng.choice(len(y_tr), n_inner_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    scaler_inner = StandardScaler()
    X_itr = scaler_inner.fit_transform(X_tr[tr_idx])
    X_ival = scaler_inner.transform(X_tr[val_idx])

    best_alpha, best_r2 = ALPHA_CANDIDATES[0], -np.inf
    for a in ALPHA_CANDIDATES:
        ridge = Ridge(alpha=a)
        ridge.fit(X_itr, y_tr[tr_idx])
        pred = ridge.predict(X_ival)
        ss_res = np.sum((y_tr[val_idx] - pred) ** 2)
        ss_tot = np.sum((y_tr[val_idx] - y_tr[val_idx].mean()) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_alpha = r2, a

    scaler = StandardScaler()
    ridge_f = Ridge(alpha=best_alpha)
    ridge_f.fit(scaler.fit_transform(X_tr), y_tr)
    pred_te = ridge_f.predict(scaler.transform(X_te))

    ss_res = np.sum((y_te - pred_te) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2) + 1e-10
    r2_te = float(1.0 - ss_res / ss_tot)
    r_te, _ = pearsonr(y_te, pred_te)
    mae_te = float(np.mean(np.abs(y_te - pred_te)))

    return {"r2": r2_te, "r": float(r_te), "mae": mae_te, "alpha": best_alpha}


def _summarise(seed_metrics: list[dict], gender_label: str, n: int,
               is_full: bool = False) -> dict:
    """Aggregate per-seed metrics into a single row with 95% CI."""
    r2s  = [m["r2"]  for m in seed_metrics]
    rs   = [m["r"]   for m in seed_metrics]
    maes = [m["mae"] for m in seed_metrics]

    def _ci95(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals)
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)

    r2_lo, r2_hi   = _ci95(r2s)
    r_lo,  r_hi    = _ci95(rs)
    mae_lo, mae_hi = _ci95(maes)

    row = {
        "gender": gender_label,
        "n": n,
        "is_full": is_full,
        "r2_mean":   float(np.mean(r2s)),
        "r2_std":    float(np.std(r2s, ddof=1)),
        "r2_ci_lo":  r2_lo, "r2_ci_hi":  r2_hi,
        "r_mean":    float(np.mean(rs)),
        "r_std":     float(np.std(rs, ddof=1)),
        "r_ci_lo":   r_lo,  "r_ci_hi":   r_hi,
        "mae_mean":  float(np.mean(maes)),
        "mae_std":   float(np.std(maes, ddof=1)),
        "mae_ci_lo": mae_lo, "mae_ci_hi": mae_hi,
    }
    tag = " [FULL]" if is_full else ""
    print(f"  n={n:>5}{tag}: R²={row['r2_mean']:.3f} [{r2_lo:.3f}, {r2_hi:.3f}]  "
          f"r={row['r_mean']:.3f} [{r_lo:.3f}, {r_hi:.3f}]  "
          f"MAE={row['mae_mean']:.2f} [{mae_lo:.2f}, {mae_hi:.2f}]")
    return row


# ─────────────────────────── power analysis ──────────────────────────────── #

def run_power_analysis(
    X: np.ndarray, y: np.ndarray,
    gender_label: str,
    sample_sizes: list[int],
    n_seeds: int,
) -> pd.DataFrame:
    n_subjects = len(y)
    n_test = max(50, int(n_subjects * TEST_FRAC))

    # Fixed train/test split for subsample runs
    rng_split = np.random.default_rng(HOLDOUT_SEED)
    all_idx   = np.arange(n_subjects)
    test_idx  = rng_split.choice(all_idx, n_test, replace=False)
    train_pool = np.setdiff1d(all_idx, test_idx)

    X_te, y_te     = X[test_idx],  y[test_idx]
    X_pool, y_pool = X[train_pool], y[train_pool]

    print(f"\n{gender_label.upper()}: {n_subjects} subjects  "
          f"→ train pool={len(train_pool)}, test={n_test}")

    rows = []

    # ── subsample runs ────────────────────────────────────────────────────── #
    for n in sample_sizes:
        if n > len(train_pool):
            print(f"  n={n}: skipped (larger than train pool {len(train_pool)})")
            continue

        seed_metrics: list[dict] = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 1000)
            sub_idx = rng.choice(len(train_pool), n, replace=False)
            m = _fit_eval(X_pool[sub_idx], y_pool[sub_idx], X_te, y_te, rng)
            seed_metrics.append(m)

        rows.append(_summarise(seed_metrics, gender_label, n))

    # ── full-pool point: vary the holdout split for CI ────────────────────── #
    # Each seed gets its own random 80/20 split of ALL subjects, training on the
    # full 80%. This gives uncertainty in performance at maximum training capacity.
    full_metrics: list[dict] = []
    n_train_full = n_subjects - n_test
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 2000)  # offset avoids overlap with above
        te_idx = rng.choice(n_subjects, n_test, replace=False)
        tr_idx = np.setdiff1d(all_idx, te_idx)
        m = _fit_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], rng)
        full_metrics.append(m)

    rows.append(_summarise(full_metrics, gender_label, n_train_full, is_full=True))

    return pd.DataFrame(rows)


# ─────────────────────────── plot ────────────────────────────────────────── #

def plot(df: pd.DataFrame) -> None:
    FONT = 8
    plt.rcParams.update({"font.size": FONT})

    colors = {"female": "#d62728", "male": "#1f77b4"}
    metrics = [
        ("r2_mean",  "r2_ci_lo",  "r2_ci_hi",  "R²",          (0.0, 1.0)),
        ("mae_mean", "mae_ci_lo", "mae_ci_hi", "MAE (years)",  None),
    ]

    # A4 width = 8.27 in; height chosen to give roughly square panels
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 3.5))

    sub_df = df[~df["is_full"]]

    for ax, (col, lo, hi, ylabel, ylim) in zip(axes, metrics):
        for gender, gdf in sub_df.groupby("gender"):
            c = colors[gender]
            gdf = gdf.sort_values("n")

            yerr = np.array([
                gdf[col].values - gdf[lo].values,
                gdf[hi].values  - gdf[col].values,
            ])
            ax.errorbar(
                gdf["n"], gdf[col],
                yerr=yerr,
                color=c, marker="o", ms=3,
                linewidth=1.2, elinewidth=0.8, capsize=2, capthick=0.8,
                label=gender,
            )
            ax.fill_between(gdf["n"], gdf[lo], gdf[hi], color=c, alpha=0.12)

        ax.set_xlabel("Training sample size (subjects)", fontsize=FONT)
        ax.set_ylabel(ylabel, fontsize=FONT)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.tick_params(labelsize=FONT)

    axes[0].legend(fontsize=FONT, title="Gender", title_fontsize=FONT,
                   handlelength=1.5, handletextpad=0.4)

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "power_analysis_learning_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"Plot → {out}")


# ─────────────────────────── main ────────────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Fast run: 5 seeds, fewer sample sizes")
    p.add_argument("--plot-only", action="store_true",
                   help="Re-plot from existing CSV without re-running analysis")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Ill-conditioned")

    out_csv = OUTPUT_DIR / "power_analysis_results.csv"

    if args.plot_only:
        df_all = pd.read_csv(out_csv)
        plot(df_all)
        return

    X_f, y_f, X_m, y_m = load_data()

    sample_sizes = [100, 300, 750, 2000, 3000] if args.smoke else SAMPLE_SIZES
    n_seeds = 5 if args.smoke else N_SEEDS

    results = []
    for X, y, label in [(X_f, y_f, "female"), (X_m, y_m, "male")]:
        df_g = run_power_analysis(X, y, label, sample_sizes, n_seeds)
        results.append(df_g)

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    if not args.smoke:
        plot(df_all)
    else:
        print("(plot skipped in smoke mode)")


if __name__ == "__main__":
    main()
