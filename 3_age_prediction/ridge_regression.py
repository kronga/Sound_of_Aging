"""
Ridge regression pipeline for age prediction from voice embeddings.

Core functions extracted from sensitivity_analysis.ipynb.

Usage
-----
    from ridge_regression import ridge_groupcv_with_exports, run_multi_seed_ridge

    # Single run
    metrics = ridge_groupcv_with_exports(
        df=wavlm_df,
        target_col="age",
        group_col="subject_number",
        output_dir="output/ridge_age_female",
        columns=wavlm_feature_cols,
        n_splits=5,
        alpha=0.5,
        optimize_alpha=True,
        alpha_candidates=[0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
        split_gender_post_train=True,
    )

    # Multi-seed bagging
    metrics = run_multi_seed_ridge(
        df=wavlm_df,
        target_col="age",
        group_col="subject_number",
        output_dir="output/multi_seed_Age_female",
        seeds=[42, 1, 2, 3, 4, 17, 99, 123, 256, 512],
        columns=wavlm_feature_cols,
        n_splits=5,
        alpha=0.5,
        optimize_alpha=True,
        split_gender_post_train=True,
    )
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_groupcv_with_exports(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    output_dir: str,
    *,
    columns: list | None = None,
    handle_nans: str = "impute",
    impute_strategy: str = "median",
    n_splits: int = 5,
    random_state: int = 42,
    alpha: float = 1.0,
    standardize: bool = True,
    shap_sample: int = 2000,
    save_plots: bool = True,
    split_gender: bool = False,
    split_gender_post_train: bool = False,
    optimize_alpha: bool = False,
    alpha_candidates: list | None = None,
    validation_fraction: float = 0.2,
    regress_out_confounds: bool = False,
    confound_columns: list | None = None,
    confound_alpha: float = 1.0,
) -> dict:
    """
    Ridge regression with GroupKFold CV and optional scaling.

    Exports predictions, metrics (JSON + CSV), coefficients, and SHAP summary
    plot to ``output_dir``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, target and group columns.
    target_col : str
        Name of the column to predict (e.g. "age").
    group_col : str
        Column used as the grouping key for GroupKFold (e.g. "subject_number").
    output_dir : str
        Directory where all outputs are written. Created if missing.
    columns : list | None
        Explicit feature column list. If None, all numeric columns except
        ``target_col`` are used.
    handle_nans : {"impute", "drop"}
        How to deal with NaN values in the feature matrix.
    impute_strategy : str
        Strategy passed to ``SimpleImputer`` (default "median").
    n_splits : int
        Number of folds for GroupKFold cross-validation (default 5).
    random_state : int
        Random seed for reproducibility.
    alpha : float
        Ridge regularisation strength (ignored when ``optimize_alpha=True``).
    standardize : bool
        Whether to z-score features before fitting.
    shap_sample : int
        Number of background samples for the SHAP LinearExplainer.
    save_plots : bool
        Whether to save the SHAP summary PNG.
    split_gender : bool
        If True, run separate analyses for gender==1 (male) and gender==0
        (female) by recursively calling this function.
    split_gender_post_train : bool
        If True, train on the combined dataset but report per-gender OOF
        metrics. Requires a "gender" column.
    optimize_alpha : bool
        If True, select alpha from ``alpha_candidates`` on a held-out
        validation slice of the training groups in each fold.
    alpha_candidates : list | None
        Candidate regularisation strengths to sweep (used when
        ``optimize_alpha=True``).
    validation_fraction : float
        Fraction of training groups set aside for alpha optimisation.
    regress_out_confounds : bool
        If True, regress out ``confound_columns`` from the features before
        fitting.
    confound_columns : list | None
        Column names to treat as confounders (required when
        ``regress_out_confounds=True``).
    confound_alpha : float
        Ridge alpha used for the confounder regression step.

    Returns
    -------
    dict
        OOF metrics including Pearson r, R², MAE, RMSE, per-fold R² list.
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found")

    if regress_out_confounds:
        if not confound_columns:
            raise ValueError("confound_columns must be provided when regress_out_confounds=True")
        missing = [c for c in confound_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Confound columns not found in df: {missing}")

    if split_gender_post_train:
        if "gender" not in df.columns:
            raise ValueError("split_gender_post_train=True requires a 'gender' column")
        if split_gender:
            raise ValueError("split_gender and split_gender_post_train cannot both be True")

    # ── gender-split wrapper ──────────────────────────────────────────────────
    if split_gender and "gender" in df.columns:
        metrics_rows, results_by_gender = [], {}
        for gender_value, label in [(1, "male"), (0, "female")]:
            sub_df = df[df["gender"] == gender_value].copy()
            if sub_df.empty:
                print(f"No rows for gender={gender_value}, skipping")
                continue
            sub_out = os.path.join(output_dir, f"gender_{label}")
            m = ridge_groupcv_with_exports(
                sub_df, target_col, group_col, sub_out,
                columns=columns, handle_nans=handle_nans,
                impute_strategy=impute_strategy, n_splits=n_splits,
                random_state=random_state, alpha=alpha,
                standardize=standardize, shap_sample=shap_sample,
                save_plots=save_plots, split_gender=False,
                split_gender_post_train=False,
                optimize_alpha=optimize_alpha,
                alpha_candidates=alpha_candidates,
                validation_fraction=validation_fraction,
                regress_out_confounds=regress_out_confounds,
                confound_columns=confound_columns,
                confound_alpha=confound_alpha,
            )
            metrics_rows.append({"gender": label, **m})
            results_by_gender[label] = m
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(
                os.path.join(output_dir, "metrics_by_gender.csv"), index=False
            )
        return results_by_gender

    # ── feature selection ────────────────────────────────────────────────────
    X_all = df.drop(columns=[target_col])
    if columns is None:
        feature_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [c for c in columns if c not in X_all.columns]
        if missing:
            raise ValueError(f"Requested columns not in df: {missing}")
        feature_cols = columns
    if not feature_cols:
        raise ValueError("No numeric feature columns found or selected")

    used_cols = list(feature_cols) + [target_col, group_col]
    if regress_out_confounds:
        for c in confound_columns:
            if c not in used_cols:
                used_cols.append(c)
    if split_gender_post_train and "gender" not in used_cols:
        used_cols.append("gender")

    df_used = df[used_cols].copy().dropna(subset=[target_col, group_col])
    X = df_used[feature_cols]
    y = df_used[target_col]
    groups = df_used[group_col]
    gender_series = df_used["gender"] if split_gender_post_train else None
    confounds = df_used[confound_columns] if regress_out_confounds else None

    if handle_nans not in {"impute", "drop"}:
        raise ValueError("handle_nans must be 'impute' or 'drop'")

    if handle_nans == "drop":
        mask = ~X.isna().any(axis=1)
        if confounds is not None:
            mask &= ~confounds.isna().any(axis=1)
        if gender_series is not None:
            mask &= ~gender_series.isna()
        X, y, groups = X.loc[mask], y.loc[mask], groups.loc[mask]
        if confounds is not None:
            confounds = confounds.loc[mask]
        if gender_series is not None:
            gender_series = gender_series.loc[mask]

    # ── optional: regress out confounders ────────────────────────────────────
    if regress_out_confounds:
        print(f"Regressing out confounds: {confound_columns}")
        conf_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ])
        feat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_strategy)),
        ])
        C_scaled = conf_pipe.fit_transform(confounds)
        X_arr = feat_pipe.fit_transform(X)
        X_res = np.zeros_like(X_arr)
        r_ridge = Ridge(alpha=confound_alpha, random_state=random_state)
        for i in range(X_arr.shape[1]):
            r_ridge.fit(C_scaled, X_arr[:, i])
            X_res[:, i] = X_arr[:, i] - r_ridge.predict(C_scaled)
        X = pd.DataFrame(X_res, index=X.index, columns=feature_cols)
        with open(os.path.join(output_dir, "confound_regression_info.json"), "w") as f:
            json.dump({"confound_columns": confound_columns,
                       "confound_alpha": confound_alpha}, f, indent=2)

    # ── build base pipeline ───────────────────────────────────────────────────
    def _make_pipe(a):
        steps = []
        if not regress_out_confounds:
            if handle_nans == "impute":
                steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
            if standardize:
                steps.append(("scaler", StandardScaler()))
        steps.append(("ridge", Ridge(alpha=a, random_state=random_state)))
        return Pipeline(steps)

    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    gkf = GroupKFold(n_splits=n_splits)
    oof = pd.Series(index=y.index, dtype=float)
    fold_r2, fold_best_alphas = [], []
    fold_metrics_by_gender: dict[str, list] = {"male": [], "female": []}

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if optimize_alpha:
            g_tr = groups.iloc[tr_idx]
            uniq = g_tr.unique()
            n_val = max(1, int(len(uniq) * validation_fraction))
            rng = np.random.RandomState(random_state + fold_idx)
            val_g = rng.choice(uniq, n_val, replace=False)
            m_val = g_tr.isin(val_g)
            best_a, best_v = alpha, -np.inf
            for a_cand in alpha_candidates:
                p = _make_pipe(a_cand)
                p.fit(X_tr.loc[~m_val], y_tr.loc[~m_val])
                v = r2_score(y_tr.loc[m_val], p.predict(X_tr.loc[m_val]))
                if v > best_v:
                    best_v, best_a = v, a_cand
            fold_best_alphas.append(best_a)
            cur_alpha = best_a
            print(f"Fold {fold_idx}: alpha={best_a:.4f} (val R²={best_v:.4f})")
        else:
            cur_alpha = alpha

        pipe_fold = _make_pipe(cur_alpha)
        pipe_fold.fit(X_tr, y_tr)
        y_pred = pipe_fold.predict(X_va)
        oof.iloc[va_idx] = y_pred
        fold_r2.append(r2_score(y_va, y_pred))

        if split_gender_post_train:
            g_va = gender_series.iloc[va_idx]
            for gv, gl in [(1, "male"), (0, "female")]:
                gm = g_va == gv
                if gm.sum() > 0:
                    fold_metrics_by_gender[gl].append({
                        "fold": fold_idx,
                        "n_samples": int(gm.sum()),
                        "R2": float(r2_score(y_va[gm], y_pred[gm])),
                        "Pearson_r": float(pearsonr(y_va[gm], y_pred[gm])[0])
                        if np.std(y_pred[gm]) > 0 else float("nan"),
                        "MAE": float(mean_absolute_error(y_va[gm], y_pred[gm])),
                        "RMSE": float(mean_squared_error(y_va[gm], y_pred[gm],
                                                          squared=False)),
                    })

    oof_r2 = r2_score(y, oof)
    oof_r = pearsonr(y, oof)[0] if np.std(oof) > 0 else float("nan")
    oof_mae = mean_absolute_error(y, oof)
    oof_rmse = mean_squared_error(y, oof, squared=False)

    final_alpha = float(np.mean(fold_best_alphas)) if optimize_alpha and fold_best_alphas else alpha
    pipe_final = _make_pipe(final_alpha)
    pipe_final.fit(X, y)
    ridge_step = pipe_final.named_steps["ridge"]

    # ── save coefficients ────────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": ridge_step.coef_.ravel(),
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df.sort_values("abs_coef", ascending=False).to_csv(
        os.path.join(output_dir, "coefficients.csv"), index=False
    )

    # ── save predictions ─────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "index": oof.index.astype(str),
        "group": groups.astype(str),
        "true_values": y.values,
        "predictions": oof.values,
    })
    fold_marks = pd.Series(index=y.index, dtype="Int64")
    for fi, (_, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        fold_marks.iloc[va_idx] = fi
    pred_df["fold"] = fold_marks.values
    if split_gender_post_train:
        pred_df["gender"] = gender_series.values
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # ── per-gender OOF metrics ───────────────────────────────────────────────
    if split_gender_post_train:
        gender_overall: dict = {}
        for gv, gl in [(1, "male"), (0, "female")]:
            gm = gender_series == gv
            if gm.sum() > 0:
                gender_overall[gl] = {
                    "n_samples": int(gm.sum()),
                    "oof_Pearson_r": float(pearsonr(y[gm], oof[gm])[0])
                    if np.std(oof[gm]) > 0 else float("nan"),
                    "oof_R2": float(r2_score(y[gm], oof[gm])),
                    "oof_MAE": float(mean_absolute_error(y[gm], oof[gm])),
                    "oof_RMSE": float(mean_squared_error(y[gm], oof[gm], squared=False)),
                    "per_fold_metrics": fold_metrics_by_gender[gl],
                }
        with open(os.path.join(output_dir, "metrics_by_gender.json"), "w") as f:
            json.dump(gender_overall, f, indent=2)
        rows = [{"gender": gl, **{k: v for k, v in m.items() if k != "per_fold_metrics"}}
                for gl, m in gender_overall.items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "metrics_by_gender_summary.csv"), index=False
        )

    # ── overall metrics ───────────────────────────────────────────────────────
    metrics = {
        "oof_Pearson_r": float(oof_r),
        "oof_R2": float(oof_r2),
        "oof_MAE": float(oof_mae),
        "oof_RMSE": float(oof_rmse),
        "per_fold_R2": [float(x) for x in fold_r2],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "alpha": float(final_alpha),
        "alpha_optimized": bool(optimize_alpha),
        "fold_best_alphas": [float(x) for x in fold_best_alphas] if optimize_alpha else None,
        "standardize": bool(standardize),
        "handle_nans": handle_nans,
        "split_gender_post_train": bool(split_gender_post_train),
        "regress_out_confounds": bool(regress_out_confounds),
        "confound_columns": confound_columns if regress_out_confounds else None,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    try:
        import shap
        preproc = Pipeline(pipe_final.steps[:-1])
        X_full = preproc.transform(X)
        if shap_sample and X.shape[0] > shap_sample:
            bg_idx = np.random.RandomState(123).choice(X.index, shap_sample, replace=False)
            X_bg = preproc.transform(X.loc[bg_idx])
        else:
            X_bg = X_full
        explainer = shap.LinearExplainer(ridge_step, X_bg, feature_names=feature_cols)
        shap_vals = explainer(X_full)
        if save_plots:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals.values, features=X_full,
                              feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200)
            plt.close()
    except Exception as e:
        with open(os.path.join(output_dir, "shap_error.txt"), "w") as f:
            f.write(str(e))

    print(f"Saved outputs → {output_dir}  (OOF R²={oof_r2:.4f}, r={oof_r:.4f})")
    return metrics


def run_multi_seed_ridge(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    output_dir: str,
    seeds: list[int] = (42, 1, 2, 3, 4),
    **kwargs,
) -> dict:
    """
    Run ``ridge_groupcv_with_exports`` across multiple random seeds and
    aggregate predictions by averaging (bagging) to reduce variance.

    Per-seed outputs are written to ``<output_dir>/seed_<n>/``.
    Averaged predictions, metrics and coefficient statistics are written to
    ``output_dir`` directly.

    Parameters
    ----------
    df, target_col, group_col, output_dir
        Forwarded to ``ridge_groupcv_with_exports``.
    seeds : list[int]
        Random seeds to iterate over. Default: (42, 1, 2, 3, 4).
    **kwargs
        All remaining keyword arguments are forwarded unchanged to
        ``ridge_groupcv_with_exports`` (e.g. ``columns``, ``alpha``,
        ``split_gender_post_train``, ``optimize_alpha``, …).

    Returns
    -------
    dict
        Metrics computed on the averaged predictions, including per-seed
        mean/std statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_pred_series: list[pd.Series] = []
    all_coef_dfs: list[pd.DataFrame] = []
    base_df: pd.DataFrame | None = None

    print(f"Starting multi-seed Ridge run  seeds={list(seeds)}")
    print("=" * 60)

    for seed in seeds:
        print(f"--> seed {seed}")
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        ridge_groupcv_with_exports(
            df, target_col, group_col, seed_dir,
            random_state=seed, **kwargs
        )
        pred_path = os.path.join(seed_dir, "predictions.csv")
        if os.path.exists(pred_path):
            spdf = pd.read_csv(pred_path)
            spdf["index"] = spdf["index"].astype(str)
            spdf = spdf.set_index("index")
            if base_df is None:
                base_cols = [c for c in spdf.columns if c not in ("predictions", "fold")]
                base_df = spdf[base_cols].copy()
            all_pred_series.append(spdf["predictions"].rename(f"pred_seed_{seed}"))
        coef_path = os.path.join(seed_dir, "coefficients.csv")
        if os.path.exists(coef_path):
            cdf = pd.read_csv(coef_path)
            cdf["seed"] = seed
            all_coef_dfs.append(cdf)

    print("=" * 60)
    print("Aggregating …")

    preds = pd.concat(all_pred_series, axis=1)
    base_df["mean_predictions"] = preds.mean(axis=1)
    base_df["pred_std"] = preds.std(axis=1)
    final_pred_df = pd.concat([base_df, preds], axis=1)
    final_pred_df.reset_index().to_csv(
        os.path.join(output_dir, "predictions_averaged.csv"), index=False
    )

    y_true = final_pred_df["true_values"]
    y_pred = final_pred_df["mean_predictions"]
    mask = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[mask], y_pred[mask]

    metrics = {
        "n_seeds": len(seeds),
        "seeds_used": list(seeds),
        "averaged_Pearson_r": float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan"),
        "averaged_R2": float(r2_score(y_true, y_pred)),
        "averaged_MAE": float(mean_absolute_error(y_true, y_pred)),
        "averaged_RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
    }

    # gender breakdown when split_gender_post_train was used
    if kwargs.get("split_gender_post_train") and "gender" in final_pred_df.columns:
        gm_dict: dict = {}
        for gv in final_pred_df["gender"].unique():
            lbl = "male" if str(gv) in ("1", "1.0") else "female"
            sub = final_pred_df[final_pred_df["gender"] == gv]
            yt, yp = sub["true_values"], sub["mean_predictions"]
            gm_dict[lbl] = {
                "R2": float(r2_score(yt, yp)),
                "Pearson_r": float(pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
                "MAE": float(mean_absolute_error(yt, yp)),
                "RMSE": float(mean_squared_error(yt, yp, squared=False)),
                "n_samples": int(len(sub)),
            }
        metrics["gender_metrics"] = gm_dict
        pd.DataFrame(gm_dict).T.to_csv(
            os.path.join(output_dir, "metrics_averaged_by_gender.csv")
        )

    with open(os.path.join(output_dir, "metrics_averaged.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # aggregate coefficients
    if all_coef_dfs:
        all_coefs = pd.concat(all_coef_dfs)
        coef_summary = (
            all_coefs.groupby("feature")["coef"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        coef_summary["abs_mean_coef"] = coef_summary["mean"].abs()
        coef_summary.sort_values("abs_mean_coef", ascending=False).to_csv(
            os.path.join(output_dir, "coefficients_averaged.csv"), index=False
        )

    print(f"Done. Averaged R²={metrics['averaged_R2']:.4f}")
    return metrics
