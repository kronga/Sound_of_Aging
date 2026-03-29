from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _find_prediction_files(root_dir: str | Path, filename: str = "predictions.csv") -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir not found: {root}")
    return sorted(root.rglob(filename))


def _infer_modality_and_gender(path: Path) -> Tuple[str, str]:
    """
    Expected layout:
      <root>/<modality_dir>/<gender_dir>/predictions.csv

    Example:
      ridge_blood_test_age_prediction/gender_male/predictions.csv
    """
    gender_dir = path.parent.name
    modality_dir = path.parent.parent.name
    return modality_dir, gender_dir


def _load_one_predictions_csv(
    csv_path: Path,
    group_col: str,
    pred_col: str = "predictions",
    keep_cols: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if group_col not in df.columns:
        raise ValueError(f"{csv_path}: group_col '{group_col}' not found. Columns: {list(df.columns)[:30]}")

    if pred_col not in df.columns:
        raise ValueError(f"{csv_path}: pred_col '{pred_col}' not found. Columns: {list(df.columns)[:30]}")

    # Keep first row per group (optionally after sorting)
    if sort_by is not None:
        if sort_by not in df.columns:
            raise ValueError(f"{csv_path}: sort_by '{sort_by}' not found.")
        df = df.sort_values(sort_by, ascending=True)

    df_first = df.drop_duplicates(subset=[group_col], keep="first").copy()

    cols = [group_col, pred_col]
    if keep_cols:
        for c in keep_cols:
            if c in df_first.columns and c not in cols:
                cols.append(c)

    return df_first.loc[:, cols]


def correlate_modalities_predictions(
    ridge_dir: str | Path,
    lgbm_dir: str | Path,
    group_col: str,
    pred_col: str = "predictions",
    genders: Tuple[str, ...] = ("gender_female", "gender_male"),
    filename: str = "predictions.csv",
    sort_by: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    For each gender separately:
      1) Look at metrics.json in each modality directory
      2) Select best model (ridge or lgbm) based on OOF_R2
      3) Use predictions from best model for each modality
      4) Inner-join predictions across modalities
      5) Compute correlation matrix of prediction columns
    
    Note: "lifestyle" is only available in lgbm, so it's taken from there.

    Returns:
      results[gender]["wide"] : merged table (one column per modality)
      results[gender]["corr"] : correlation matrix across modalities
      results[gender]["modality_order"] : ordered list of modality names used

    If output_dir is given, saves:
      <output_dir>/<gender>__merged.csv
      <output_dir>/<gender>__corr_pearson.csv
    """
    ridge_root = Path(ridge_dir)
    lgbm_root = Path(lgbm_dir)

    if not ridge_root.exists():
        raise FileNotFoundError(f"ridge_dir not found: {ridge_root}")
    if not lgbm_root.exists():
        raise FileNotFoundError(f"lgbm_dir not found: {lgbm_root}")

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for gender in genders:
        frames = []
        modality_order = []  # Track the order of modalities
        
        # Collect all unique modality directories from both ridge and lgbm
        ridge_modalities = {d.name for d in ridge_root.iterdir() if d.is_dir()}
        lgbm_modalities = {d.name for d in lgbm_root.iterdir() if d.is_dir()}
        all_modality_dirs = sorted(ridge_modalities | lgbm_modalities)
        
        # Map cleaned modality names to directory names
        modality_map = {}  # cleaned_name -> list of dir_names
        for dir_name in all_modality_dirs:
            # Extract modality name from directory name (e.g., "ridge_DEXA_x_age_prediction_y" -> "DEXA")
            modality_name = dir_name
            if modality_name.startswith("ridge_"):
                modality_name = modality_name[6:]  # Remove "ridge_" prefix
            elif modality_name.startswith("lgbm_"):
                modality_name = modality_name[5:]  # Remove "lgbm_" prefix
            
            # Remove "_age_prediction" and everything that follows it using regex
            modality_name = re.sub(r"_age_prediction.*", "", modality_name)
            
            if modality_name not in modality_map:
                modality_map[modality_name] = []
            modality_map[modality_name].append(dir_name)
        
        for modality_name, dir_names in sorted(modality_map.items()):
            best_model = None
            best_oof_r2 = -np.inf
            best_gender_path = None
            
            # Check both ridge and lgbm for this modality group
            for dir_name in dir_names:
                # Check ridge model
                ridge_modality_path = ridge_root / dir_name / gender
                if ridge_modality_path.exists():
                    metrics_file = ridge_modality_path / "metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            # If metrics is a dict with gender keys, extract the gender's metrics
                            if isinstance(metrics, dict) and gender in metrics:
                                metrics = metrics[gender]
                            
                            oof_r2 = metrics.get("oof_R2")
                            if oof_r2 is not None and oof_r2 > best_oof_r2:
                                best_oof_r2 = oof_r2
                                best_model = "ridge"
                                best_gender_path = ridge_modality_path
                        except Exception as e:
                            print(f"Warning: Error reading metrics from {metrics_file}: {e}")
                
                # Check lgbm model
                lgbm_modality_path = lgbm_root / dir_name / gender
                if lgbm_modality_path.exists():
                    metrics_file = lgbm_modality_path / "metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            # If metrics is a dict with gender keys, extract the gender's metrics
                            if isinstance(metrics, dict) and gender in metrics:
                                metrics = metrics[gender]
                            
                            oof_r2 = metrics.get("oof_R2")
                            if oof_r2 is not None and oof_r2 > best_oof_r2:
                                best_oof_r2 = oof_r2
                                best_model = "lgbm"
                                best_gender_path = lgbm_modality_path
                        except Exception as e:
                            print(f"Warning: Error reading metrics from {metrics_file}: {e}")
            
            if best_model is None:
                print(f"Warning: No metrics found for {modality_name} ({gender}), skipping")
                continue
            
            print(f"  {modality_name:50s} -> {best_model.upper():5s} (R² = {best_oof_r2:.4f})")
            
            # Load predictions.csv from the best model
            pred_file = best_gender_path / filename
            if not pred_file.exists():
                print(f"Warning: {filename} not found in {best_gender_path}, skipping {modality_name}")
                continue
            
            dfm = _load_one_predictions_csv(
                csv_path=pred_file,
                group_col=group_col,
                pred_col=pred_col,
                sort_by=sort_by,
            ).rename(columns={pred_col: modality_name})
            
            frames.append(dfm)
            modality_order.append(modality_name)
        
        if not frames:
            print(f"Warning: No data found for gender {gender}")
            continue
        
        if not frames:
            print(f"Warning: No data found for gender {gender}")
            continue
        
        # Inner join across all modalities on group_col
        wide = frames[0]
        for dfm in frames[1:]:
            wide = wide.merge(dfm, on=group_col, how="inner")
        
        # Correlation across modalities (exclude group_col)
        corr = wide.drop(columns=[group_col]).corr(method="pearson")
        
        results[gender] = {"wide": wide, "corr": corr, "modality_order": modality_order}
        
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            wide.to_csv(out / f"{gender}__merged.csv", index=False)
            corr.to_csv(out / f"{gender}__corr_pearson.csv")
    
    return results


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str | Path,
    title: str = "Modalities Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "rocket_r",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """
    Plot correlation matrix as a heatmap (lower diagonal only) and save to file.

    Args:
        corr_matrix: Correlation matrix (DataFrame)
        output_path: Path to save the figure
        title: Title for the heatmap
        figsize: Figure size (width, height)
        cmap: Colormap to use
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    # Create mask for upper triangle and diagonal (keep lower diagonal only)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar=False,
        annot=True,
        fmt=".2f",
        ax=ax,
    )

    
    # Remove first vertical label (y-axis) and last horizontal label (x-axis)
    # since diagonals are masked
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    
    if yticklabels:
        yticklabels[0] = ""
    if xticklabels:
        xticklabels[-1] = ""
    
    ax.set_yticklabels(yticklabels, rotation=0)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.tick_params(left=False, bottom=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def plot_voice_row(
    corr_matrix: pd.DataFrame,
    voice_row_name: str,
    output_path: str | Path,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "rocket_r",
    vmin: float = 0,
    vmax: float = 1,
    column_order: Optional[List[str]] = None,
) -> None:
    """
    Plot the voice row from correlation matrix as a heatmap and save to file.

    Args:
        corr_matrix: Correlation matrix (DataFrame)
        voice_row_name: Name of the voice row (index name in corr_matrix)
        output_path: Path to save the figure
        title: Title for the plot (auto-generated if None)
        figsize: Figure size (width, height)
        cmap: Colormap to use
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        column_order: Optional list specifying the order of columns
    """
    if voice_row_name not in corr_matrix.index:
        raise ValueError(f"'{voice_row_name}' not found in correlation matrix index.")

    voice_row = corr_matrix.loc[voice_row_name].to_frame().T
    
    # Define the desired order of modalities (keep voice in the display)
    desired_order = ["Metabo", "voice", "lifestyle", "sleep", "DEXA", "diet", "NMR", "retina", "microbiome", "blood_test"]
    
    # Reorder columns according to desired order
    ordered_cols = [col for col in desired_order if col in voice_row.columns]
    # Add any remaining columns not in the desired order
    remaining_cols = [col for col in voice_row.columns if col not in desired_order]
    final_order = ordered_cols + remaining_cols
    voice_row = voice_row[final_order]

    # Adjust figsize to show squares (width should be relative to number of columns)
    n_cols = len(voice_row.columns)
    figsize = (max(n_cols * 0.8, 8), 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        voice_row,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar=False,
        annot=True,
        fmt=".2f",
        ax=ax,
        square=True,
    )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved voice row heatmap to: {output_path}")


if __name__ == "__main__":
    # Example usage
    ridge_root = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/Ridge_clean"
    lgbm_root = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/LGBM_stuff"
    out = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/Ridge_clean"

    # Set group_col to your ID column, for example: "subject_number" or "RegistrationCode"
    results = correlate_modalities_predictions(
        ridge_dir=ridge_root,
        lgbm_dir=lgbm_root,
        group_col="group",
        pred_col="predictions",
        output_dir=out,
    )

    for gender, d in results.items():
        print(gender, "merged shape:", d["wide"].shape)
        print(gender, "corr shape:", d["corr"].shape)
        print(d["corr"].round(3))
        print()

        # Plot full correlation matrix heatmap (lower diagonal)
        heatmap_path = Path(out) / f"{gender}__corr_heatmap_full.png"
        plot_correlation_heatmap(
            d["corr"],
            output_path=heatmap_path,
            title=f"Modalities Correlation Matrix - {gender}",
        )

        # Plot voice row if it exists in the correlation matrix
        voice_identifiers = [col for col in d["corr"].columns if "voice" in col.lower()]
        if voice_identifiers:
            for voice_col in voice_identifiers:
                voice_row_path = Path(out) / f"{gender}__voice_row_heatmap_{voice_col}.png"
                plot_voice_row(
                    d["corr"],
                    voice_row_name=voice_col,
                    output_path=voice_row_path,
                    title=f"Correlations with {voice_col} - {gender}",
                )

