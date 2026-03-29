"""
LightGBM regression pipeline for age prediction across biological modalities.

Core functions extracted from predict_age_notebook.ipynb.

The pipeline mirrors ridge_regression.py but uses LightGBM instead of Ridge,
with optional hyperparameter optimisation via RandomizedSearchCV.

Supported modalities (match the directory naming convention):
    voice, sleep, blood_test, DEXA, NMR, metabolomics, retina, diet,
    microbiome, lifestyle

Usage
-----
    from lightgbm_regression import (
        lightgbm_groupcv_with_exports,
        run_multi_seed_lightgbm,
    )

    # Single run for one modality
    metrics = lightgbm_groupcv_with_exports(
        df=sleep_full,
        target_col="age",
        group_col="subject_number",
        output_dir="output/lgbm_sleep_age",
        columns=sleep_feature_cols,
        n_splits=5,
        split_gender=True,
    )

    # Multi-seed bagging across all modalities
    datasets = {
        "sleep":  (sleep_full,  sleep_feature_cols),
        "blood_test": (bt_full, bt_feature_cols),
        ...
    }
    for name, (df, cols) in datasets.items():
        run_multi_seed_lightgbm(
            df=df,
            target_col="age",
            group_col="subject_number",
            output_dir=f"output/multi_seed_lgbm_{name}",
            seeds=[42, 1, 2, 3, 4, 17, 99, 123, 256, 512],
            columns=cols,
            n_splits=5,
            split_gender=True,
        )
"""

from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


def lightgbm_groupcv_with_exports(
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
    lgbm_params: dict | None = None,
    num_boost_round: int = 100,
    early_stopping_rounds: int | None = None,
    shap_sample: int = 2000,
    save_plots: bool = True,
    split_gender: bool = False,
    split_gender_post_train: bool = False,
    regress_out_confounds: bool = False,
    confound_columns: list | None = None,
    confound_alpha: float = 1.0,
    optimize_hyperparams: bool = False,
    n_iter_search: int = 30,
    validation_fraction: float = 0.2,
) -> dict:
    """
    LightGBM regression with GroupKFold CV.

    Exports predictions, metrics, feature importance, and SHAP summary to
    ``output_dir``.  Feature column names are sanitised to remove characters
    that LightGBM does not accept; the original→sanitised mapping is saved as
    ``feature_name_mapping.csv``.

    Parameters
    ----------
    df : pd.DataFrame
    target_col, group_col, output_dir
        Same semantics as in ``ridge_regression.ridge_groupcv_with_exports``.
    lgbm_params : dict | None
        LightGBM training parameters.  Ignored when
        ``optimize_hyperparams=True``.
    num_boost_round : int
        Number of boosting rounds.
    early_stopping_rounds : int | None
        Triggers early stopping on the validation fold when set.
    optimize_hyperparams : bool
        If True, run ``RandomizedSearchCV`` on a held-out validation slice in
        each fold.
    n_iter_search : int
        Number of parameter settings to sample during hyperparameter search.
    validation_fraction : float
        Fraction of training groups used as validation for HPO.
    (remaining params)
        Same as ``ridge_regression.ridge_groupcv_with_exports``.

    Returns
    -------
    dict
        OOF metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found")

    if regress_out_confounds:
        if not confound_columns:
            raise ValueError("confound_columns required when regress_out_confounds=True")
        missing = [c for c in confound_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Confound columns not found: {missing}")

    if split_gender_post_train:
        if "gender" not in df.columns:
            raise ValueError("split_gender_post_train=True requires a 'gender' column")
        if split_gender:
            raise ValueError("split_gender and split_gender_post_train cannot both be True")

    # ── gender-split wrapper ──────────────────────────────────────────────────
    if split_gender and "gender" in df.columns:
        metrics_rows, results_by_gender = [], {}
        for gv, gl in [(1, "male"), (0, "female")]:
            sub = df[df["gender"] == gv].copy()
            if sub.empty:
                continue
            sub_out = os.path.join(output_dir, f"gender_{gl}")
            m = lightgbm_groupcv_with_exports(
                sub, target_col, group_col, sub_out,
                columns=columns, handle_nans=handle_nans,
                impute_strategy=impute_strategy, n_splits=n_splits,
                random_state=random_state, lgbm_params=lgbm_params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                shap_sample=shap_sample, save_plots=save_plots,
                split_gender=False, split_gender_post_train=False,
                regress_out_confounds=regress_out_confounds,
                confound_columns=confound_columns, confound_alpha=confound_alpha,
                optimize_hyperparams=optimize_hyperparams,
                n_iter_search=n_iter_search,
                validation_fraction=validation_fraction,
            )
            metrics_rows.append({"gender": gl, **m})
            results_by_gender[gl] = m
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(
                os.path.join(output_dir, "metrics_by_gender.csv"), index=False
            )
            with open(os.path.join(output_dir, "metrics_by_gender.json"), "w") as f:
                json.dump(results_by_gender, f, indent=2)
        return results_by_gender

    # ── feature selection & column sanitisation ───────────────────────────────
    X_all = df.drop(columns=[target_col])
    if columns is None:
        orig_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [c for c in columns if c not in X_all.columns]
        if missing:
            raise ValueError(f"Requested columns not in df: {missing}")
        orig_cols = list(columns)

    san_cols = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in orig_cols]
    # deduplicate sanitised names
    seen: dict[str, int] = {}
    final_san: list[str] = []
    for s in san_cols:
        if s in seen:
            seen[s] += 1
            final_san.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            final_san.append(s)
    san_cols = final_san
    name_map = dict(zip(orig_cols, san_cols))
    rev_map = dict(zip(san_cols, orig_cols))

    # ── build working dataframe ───────────────────────────────────────────────
    used_cols = list(orig_cols) + [target_col, group_col]
    has_rs = "research_stage" in df.columns
    if has_rs and "research_stage" not in used_cols:
        used_cols.append("research_stage")
    if regress_out_confounds:
        for c in confound_columns:
            if c not in used_cols:
                used_cols.append(c)
    if split_gender_post_train and "gender" not in used_cols:
        used_cols.append("gender")

    df_used = df[used_cols].copy().dropna(subset=[target_col, group_col])
    X = df_used[orig_cols].copy()
    X.columns = san_cols
    y = df_used[target_col]
    groups = df_used[group_col]
    rs_series = df_used["research_stage"] if has_rs else None
    gender_series = df_used["gender"] if split_gender_post_train else None
    confounds = df_used[confound_columns] if regress_out_confounds else None

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
        if rs_series is not None:
            rs_series = rs_series.loc[mask]
    elif handle_nans == "impute":
        imp = SimpleImputer(strategy=impute_strategy)
        X = pd.DataFrame(imp.fit_transform(X), index=X.index, columns=san_cols)
        if confounds is not None:
            confounds = pd.DataFrame(
                imp.fit_transform(confounds), index=confounds.index, columns=confound_columns
            )

    # ── optional confounder regression ───────────────────────────────────────
    if regress_out_confounds:
        print(f"Regressing out confounds: {confound_columns}")
        cs = StandardScaler().fit_transform(confounds)
        xs = StandardScaler().fit_transform(X)
        xr = np.zeros_like(xs)
        rc = Ridge(alpha=confound_alpha, random_state=random_state)
        for i in range(xs.shape[1]):
            rc.fit(cs, xs[:, i])
            xr[:, i] = xs[:, i] - rc.predict(cs)
        X = pd.DataFrame(xr, index=X.index, columns=san_cols)
        with open(os.path.join(output_dir, "confound_regression_info.json"), "w") as f:
            json.dump({"confound_columns": confound_columns,
                       "confound_alpha": confound_alpha}, f, indent=2)

    # ── default LightGBM params ───────────────────────────────────────────────
    if lgbm_params is None and not optimize_hyperparams:
        lgbm_params = {
            "objective": "regression", "metric": "rmse",
            "boosting_type": "gbdt", "verbosity": -1,
            "seed": random_state,
        }

    gkf = GroupKFold(n_splits=n_splits)
    oof = pd.Series(index=y.index, dtype=float)
    fold_r2: list[float] = []
    importance_list: list[pd.DataFrame] = []
    fold_best_params: list[dict] = []
    fold_metrics_by_gender: dict[str, list] = {"male": [], "female": []}

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if optimize_hyperparams:
            g_tr = groups.iloc[tr_idx]
            uniq = g_tr.unique()
            n_val = max(1, int(len(uniq) * validation_fraction))
            rng = np.random.RandomState(random_state + fold_idx)
            val_g = rng.choice(uniq, n_val, replace=False)
            m_val = g_tr.isin(val_g)
            param_dists = {
                "num_leaves": randint(20, 150), "max_depth": randint(3, 15),
                "learning_rate": uniform(0.01, 0.2), "n_estimators": randint(100, 1000),
                "min_child_samples": randint(10, 100),
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                "reg_alpha": uniform(0, 1), "reg_lambda": uniform(0, 1),
            }
            est = lgb.LGBMRegressor(objective="regression", metric="rmse",
                                    boosting_type="gbdt", verbosity=-1,
                                    random_state=random_state)
            rs = RandomizedSearchCV(est, param_dists, n_iter=n_iter_search,
                                    scoring="r2", cv=3,
                                    random_state=random_state + fold_idx, verbose=0)
            rs.fit(X_tr.loc[~m_val], y_tr.loc[~m_val])
            bp = rs.best_params_
            bp.update({"objective": "regression", "metric": "rmse",
                       "boosting_type": "gbdt", "verbosity": -1, "seed": random_state})
            fold_best_params.append(bp)
            cur_params = bp
            print(f"Fold {fold_idx}: best val R²={rs.best_score_:.4f}")
        else:
            cur_params = lgbm_params.copy()

        td = lgb.Dataset(X_tr, label=y_tr)
        if early_stopping_rounds:
            vd = lgb.Dataset(X_va, label=y_va, reference=td)
            model = lgb.train(cur_params, td, num_boost_round=num_boost_round,
                              valid_sets=[vd],
                              callbacks=[lgb.early_stopping(early_stopping_rounds,
                                                            verbose=False)])
        else:
            model = lgb.train(cur_params, td, num_boost_round=num_boost_round)

        y_pred = model.predict(X_va)
        oof.iloc[va_idx] = y_pred
        fold_r2.append(r2_score(y_va, y_pred))
        importance_list.append(pd.DataFrame({
            "feature": san_cols,
            "importance": model.feature_importance(importance_type="gain"),
            "fold": fold_idx,
        }))

        if split_gender_post_train:
            g_va = gender_series.iloc[va_idx]
            for gv, gl in [(1, "male"), (0, "female")]:
                gm = g_va == gv
                if gm.sum() > 0:
                    fold_metrics_by_gender[gl].append({
                        "fold": fold_idx, "n_samples": int(gm.sum()),
                        "R2": float(r2_score(y_va[gm], y_pred[gm])),
                        "Pearson_r": float(pearsonr(y_va[gm], y_pred[gm])[0])
                        if np.std(y_pred[gm]) > 0 else float("nan"),
                        "MAE": float(mean_absolute_error(y_va[gm], y_pred[gm])),
                        "RMSE": float(mean_squared_error(y_va[gm], y_pred[gm], squared=False)),
                    })

    oof_r2 = r2_score(y, oof)
    oof_r = pearsonr(y, oof)[0] if np.std(oof) > 0 else float("nan")
    oof_mae = mean_absolute_error(y, oof)
    oof_rmse = mean_squared_error(y, oof, squared=False)

    # final model
    if optimize_hyperparams and fold_best_params:
        fp: dict = {"objective": "regression", "metric": "rmse",
                    "boosting_type": "gbdt", "verbosity": -1, "seed": random_state}
        for k in fold_best_params[0]:
            vals = [p[k] for p in fold_best_params]
            fp[k] = int(np.mean(vals)) if isinstance(vals[0], (int, np.integer)) else float(np.mean(vals))
    else:
        fp = lgbm_params

    final_model = lgb.train(fp, lgb.Dataset(X, label=y), num_boost_round=num_boost_round)
    final_model.save_model(os.path.join(output_dir, "model.txt"))

    # feature name mapping
    pd.DataFrame({"original_name": orig_cols, "sanitized_name": san_cols}).to_csv(
        os.path.join(output_dir, "feature_name_mapping.csv"), index=False
    )

    # feature importance
    imp_df = pd.concat(importance_list)
    imp_summary = (
        imp_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    imp_summary["original_feature"] = imp_summary["feature"].map(rev_map)
    imp_summary = imp_summary.sort_values("mean", ascending=False)
    imp_summary.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # predictions
    pred_df = pd.DataFrame({
        "index": oof.index.astype(str), "group": groups.astype(str),
        "true_values": y.values, "predictions": oof.values,
    })
    fold_marks = pd.Series(index=y.index, dtype="Int64")
    for fi, (_, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        fold_marks.iloc[va_idx] = fi
    pred_df["fold"] = fold_marks.values
    if has_rs:
        pred_df["research_stage"] = rs_series.values
    if split_gender_post_train:
        pred_df["gender"] = gender_series.values
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # per-gender metrics
    if split_gender_post_train:
        gm_overall: dict = {}
        for gv, gl in [(1, "male"), (0, "female")]:
            gm = gender_series == gv
            if gm.sum() > 0:
                gm_overall[gl] = {
                    "n_samples": int(gm.sum()),
                    "oof_Pearson_r": float(pearsonr(y[gm], oof[gm])[0])
                    if np.std(oof[gm]) > 0 else float("nan"),
                    "oof_R2": float(r2_score(y[gm], oof[gm])),
                    "oof_MAE": float(mean_absolute_error(y[gm], oof[gm])),
                    "oof_RMSE": float(mean_squared_error(y[gm], oof[gm], squared=False)),
                    "per_fold_metrics": fold_metrics_by_gender[gl],
                }
        with open(os.path.join(output_dir, "metrics_by_gender.json"), "w") as f:
            json.dump(gm_overall, f, indent=2)

    metrics = {
        "oof_Pearson_r": float(oof_r), "oof_R2": float(oof_r2),
        "oof_MAE": float(oof_mae), "oof_RMSE": float(oof_rmse),
        "per_fold_R2": [float(x) for x in fold_r2],
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "lgbm_params": fp, "num_boost_round": num_boost_round,
        "handle_nans": handle_nans, "split_gender_post_train": bool(split_gender_post_train),
        "regress_out_confounds": bool(regress_out_confounds),
        "optimize_hyperparams": bool(optimize_hyperparams),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP
    try:
        import shap
        X_bg = X.sample(min(shap_sample, len(X)), random_state=123)
        X_bg_disp = X_bg.copy()
        X_bg_disp.columns = [rev_map.get(c, c) for c in X_bg.columns]
        explainer = shap.TreeExplainer(final_model)
        sv = explainer(X_bg)
        if save_plots:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(sv.values, features=X_bg_disp,
                              feature_names=list(X_bg_disp.columns), show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200)
            plt.close()
    except Exception as e:
        with open(os.path.join(output_dir, "shap_error.txt"), "w") as f:
            f.write(str(e))

    print(f"Saved outputs → {output_dir}  (OOF R²={oof_r2:.4f}, r={oof_r:.4f})")
    return metrics


def run_multi_seed_lightgbm(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    output_dir: str,
    seeds: list[int] = (42, 1, 2, 3, 4),
    **kwargs,
) -> dict:
    """
    Run ``lightgbm_groupcv_with_exports`` across multiple seeds and average
    predictions (bagging).

    When ``split_gender=True``, per-gender averaged outputs are saved to
    ``<output_dir>/gender_male/`` and ``<output_dir>/gender_female/``.

    Parameters
    ----------
    seeds : list[int]
        Random seeds.  Default: (42, 1, 2, 3, 4).
    **kwargs
        Forwarded to ``lightgbm_groupcv_with_exports``.

    Returns
    -------
    dict
        Metrics on the averaged predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_gender = kwargs.get("split_gender", False)

    print(f"Starting multi-seed LightGBM run  seeds={list(seeds)}")
    print("=" * 60)

    def _aggregate_gender(gender_label: str) -> dict:
        pred_series, importance_dfs, seed_metrics = [], [], []
        base_df: pd.DataFrame | None = None

        for seed in seeds:
            seed_dir = os.path.join(output_dir, f"seed_{seed}")
            run_kw = {**kwargs, "random_state": seed}
            lightgbm_groupcv_with_exports(df, target_col, group_col, seed_dir, **run_kw)

            gpath = os.path.join(seed_dir, f"gender_{gender_label}", "predictions.csv")
            if os.path.exists(gpath):
                spdf = pd.read_csv(gpath).assign(index=lambda d: d["index"].astype(str)).set_index("index")
                if base_df is None:
                    base_df = spdf[[c for c in spdf.columns if c not in ("predictions", "fold")]].copy()
                pred_series.append(spdf["predictions"].rename(f"pred_seed_{seed}"))
                yt, yp = spdf["true_values"], spdf["predictions"]
                mask = yt.notna() & yp.notna()
                seed_metrics.append({
                    "seed": seed,
                    "R2": float(r2_score(yt[mask], yp[mask])),
                    "Pearson_r": float(pearsonr(yt[mask], yp[mask])[0])
                    if np.std(yp[mask]) > 0 else float("nan"),
                    "MAE": float(mean_absolute_error(yt[mask], yp[mask])),
                    "RMSE": float(mean_squared_error(yt[mask], yp[mask], squared=False)),
                })
            imp_path = os.path.join(seed_dir, f"gender_{gender_label}", "feature_importance.csv")
            if os.path.exists(imp_path):
                imp = pd.read_csv(imp_path); imp["seed"] = seed
                importance_dfs.append(imp)

        if not pred_series:
            return {}

        preds = pd.concat(pred_series, axis=1)
        base_df["mean_predictions"] = preds.mean(axis=1)
        base_df["pred_std"] = preds.std(axis=1)
        final_pred = pd.concat([base_df, preds], axis=1)

        gout = os.path.join(output_dir, f"gender_{gender_label}")
        os.makedirs(gout, exist_ok=True)
        final_pred.reset_index().to_csv(os.path.join(gout, "predictions_averaged.csv"), index=False)

        yt, yp = final_pred["true_values"], final_pred["mean_predictions"]
        mask = yt.notna() & yp.notna()
        yt, yp = yt[mask], yp[mask]
        sdf = pd.DataFrame(seed_metrics)

        gm = {
            "gender": gender_label, "n_seeds": len(seeds), "seeds_used": list(seeds),
            "averaged_R2": float(r2_score(yt, yp)),
            "averaged_Pearson_r": float(pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
            "averaged_MAE": float(mean_absolute_error(yt, yp)),
            "averaged_RMSE": float(mean_squared_error(yt, yp, squared=False)),
            "per_seed_R2_mean": float(sdf["R2"].mean()),
            "per_seed_R2_std": float(sdf["R2"].std()),
            "n_samples": int(len(yt)),
        }
        with open(os.path.join(gout, "metrics_averaged.json"), "w") as f:
            json.dump(gm, f, indent=2)
        sdf.to_csv(os.path.join(gout, "metrics_per_seed.csv"), index=False)

        if importance_dfs:
            all_imp = pd.concat(importance_dfs)
            if "original_feature" in all_imp.columns:
                grp_col = "original_feature"
                val_col = "mean"
            else:
                grp_col = "feature"; val_col = "importance"
            is_agg = (
                all_imp.groupby(grp_col)[val_col]
                .agg(["mean", "std", "count"]).reset_index()
            )
            is_agg.columns = ["feature", "importance_mean", "importance_std", "count"]
            is_agg.sort_values("importance_mean", ascending=False).to_csv(
                os.path.join(gout, "feature_importance_averaged.csv"), index=False
            )
        print(f"  {gender_label}: averaged R²={gm['averaged_R2']:.4f}")
        return gm

    if split_gender:
        results = {}
        for seed in seeds:
            seed_dir = os.path.join(output_dir, f"seed_{seed}")
            run_kw = {**kwargs, "random_state": seed}
            lightgbm_groupcv_with_exports(df, target_col, group_col, seed_dir, **run_kw)

        for gender_label in ("male", "female"):
            pred_series, importance_dfs, seed_metrics = [], [], []
            base_df = None

            for seed in seeds:
                seed_dir = os.path.join(output_dir, f"seed_{seed}")
                gpath = os.path.join(seed_dir, f"gender_{gender_label}", "predictions.csv")
                if os.path.exists(gpath):
                    spdf = (pd.read_csv(gpath)
                            .assign(index=lambda d: d["index"].astype(str))
                            .set_index("index"))
                    if base_df is None:
                        base_df = spdf[[c for c in spdf.columns
                                        if c not in ("predictions", "fold")]].copy()
                    pred_series.append(spdf["predictions"].rename(f"pred_seed_{seed}"))
                    yt, yp = spdf["true_values"], spdf["predictions"]
                    m = yt.notna() & yp.notna()
                    seed_metrics.append({
                        "seed": seed,
                        "R2": float(r2_score(yt[m], yp[m])),
                        "Pearson_r": float(pearsonr(yt[m], yp[m])[0])
                        if np.std(yp[m]) > 0 else float("nan"),
                        "MAE": float(mean_absolute_error(yt[m], yp[m])),
                        "RMSE": float(mean_squared_error(yt[m], yp[m], squared=False)),
                    })
                imp_path = os.path.join(seed_dir, f"gender_{gender_label}",
                                        "feature_importance.csv")
                if os.path.exists(imp_path):
                    imp = pd.read_csv(imp_path); imp["seed"] = seed
                    importance_dfs.append(imp)

            if not pred_series:
                continue

            preds = pd.concat(pred_series, axis=1)
            base_df["mean_predictions"] = preds.mean(axis=1)
            base_df["pred_std"] = preds.std(axis=1)
            final_pred = pd.concat([base_df, preds], axis=1)

            gout = os.path.join(output_dir, f"gender_{gender_label}")
            os.makedirs(gout, exist_ok=True)
            final_pred.reset_index().to_csv(
                os.path.join(gout, "predictions_averaged.csv"), index=False
            )

            yt, yp = final_pred["true_values"], final_pred["mean_predictions"]
            ms = yt.notna() & yp.notna()
            yt, yp = yt[ms], yp[ms]
            sdf = pd.DataFrame(seed_metrics)

            gm = {
                "gender": gender_label, "n_seeds": len(seeds),
                "averaged_R2": float(r2_score(yt, yp)),
                "averaged_Pearson_r": float(pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
                "averaged_MAE": float(mean_absolute_error(yt, yp)),
                "averaged_RMSE": float(mean_squared_error(yt, yp, squared=False)),
                "per_seed_R2_mean": float(sdf["R2"].mean()),
                "per_seed_R2_std": float(sdf["R2"].std()),
                "n_samples": int(len(yt)),
            }
            with open(os.path.join(gout, "metrics_averaged.json"), "w") as f:
                json.dump(gm, f, indent=2)
            sdf.to_csv(os.path.join(gout, "metrics_per_seed.csv"), index=False)

            results[gender_label] = gm
            print(f"  {gender_label}: averaged R²={gm['averaged_R2']:.4f}")

        if results:
            with open(os.path.join(output_dir, "metrics_averaged_by_gender.json"), "w") as f:
                json.dump(results, f, indent=2)
        return results

    # ── non-split-gender path ────────────────────────────────────────────────
    pred_series, importance_dfs, seed_metrics = [], [], []
    base_df = None

    for seed in seeds:
        print(f"--> seed {seed}")
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        run_kw = {**kwargs, "random_state": seed}
        lightgbm_groupcv_with_exports(df, target_col, group_col, seed_dir, **run_kw)

        pp = os.path.join(seed_dir, "predictions.csv")
        if os.path.exists(pp):
            spdf = (pd.read_csv(pp)
                    .assign(index=lambda d: d["index"].astype(str))
                    .set_index("index"))
            if base_df is None:
                base_df = spdf[[c for c in spdf.columns
                                if c not in ("predictions", "fold")]].copy()
            pred_series.append(spdf["predictions"].rename(f"pred_seed_{seed}"))
            yt, yp = spdf["true_values"], spdf["predictions"]
            m = yt.notna() & yp.notna()
            seed_metrics.append({
                "seed": seed,
                "R2": float(r2_score(yt[m], yp[m])),
                "Pearson_r": float(pearsonr(yt[m], yp[m])[0]) if np.std(yp[m]) > 0 else float("nan"),
                "MAE": float(mean_absolute_error(yt[m], yp[m])),
                "RMSE": float(mean_squared_error(yt[m], yp[m], squared=False)),
            })
        ip = os.path.join(seed_dir, "feature_importance.csv")
        if os.path.exists(ip):
            imp = pd.read_csv(ip); imp["seed"] = seed
            importance_dfs.append(imp)

    preds = pd.concat(pred_series, axis=1)
    base_df["mean_predictions"] = preds.mean(axis=1)
    base_df["pred_std"] = preds.std(axis=1)
    final_pred = pd.concat([base_df, preds], axis=1)
    final_pred.reset_index().to_csv(
        os.path.join(output_dir, "predictions_averaged.csv"), index=False
    )

    yt, yp = final_pred["true_values"], final_pred["mean_predictions"]
    ms = yt.notna() & yp.notna()
    yt, yp = yt[ms], yp[ms]
    sdf = pd.DataFrame(seed_metrics)

    metrics = {
        "n_seeds": len(seeds), "seeds_used": list(seeds),
        "averaged_R2": float(r2_score(yt, yp)),
        "averaged_Pearson_r": float(pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
        "averaged_MAE": float(mean_absolute_error(yt, yp)),
        "averaged_RMSE": float(mean_squared_error(yt, yp, squared=False)),
        "per_seed_R2_mean": float(sdf["R2"].mean()),
        "per_seed_R2_std": float(sdf["R2"].std()),
    }

    if kwargs.get("split_gender_post_train") and "gender" in final_pred.columns:
        gm_dict: dict = {}
        for gv in final_pred["gender"].unique():
            lbl = "male" if str(gv) in ("1", "1.0") else "female"
            sub = final_pred[final_pred["gender"] == gv]
            yt2, yp2 = sub["true_values"], sub["mean_predictions"]
            gm_dict[lbl] = {
                "R2": float(r2_score(yt2, yp2)),
                "Pearson_r": float(pearsonr(yt2, yp2)[0]) if np.std(yp2) > 0 else float("nan"),
                "MAE": float(mean_absolute_error(yt2, yp2)),
                "RMSE": float(mean_squared_error(yt2, yp2, squared=False)),
                "n_samples": int(len(sub)),
            }
        metrics["gender_metrics"] = gm_dict
        pd.DataFrame(gm_dict).T.to_csv(
            os.path.join(output_dir, "metrics_averaged_by_gender.csv")
        )

    with open(os.path.join(output_dir, "metrics_averaged.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    sdf.to_csv(os.path.join(output_dir, "metrics_per_seed.csv"), index=False)

    if importance_dfs:
        all_imp = pd.concat(importance_dfs)
        grp = "original_feature" if "original_feature" in all_imp.columns else "feature"
        val = "mean" if grp == "original_feature" else "importance"
        is_agg = (all_imp.groupby(grp)[val]
                  .agg(["mean", "std", "count"]).reset_index())
        is_agg.columns = ["feature", "importance_mean", "importance_std", "count"]
        is_agg.sort_values("importance_mean", ascending=False).to_csv(
            os.path.join(output_dir, "feature_importance_averaged.csv"), index=False
        )

    print(f"Done. Averaged R²={metrics['averaged_R2']:.4f}")
    return metrics
