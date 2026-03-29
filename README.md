# Voice-Age: Predicting Biological Age from Voice

Computational pipeline accompanying the manuscript:

> **Voice-Age: Predicting Biological Age from Voice Using Deep Speech Embeddings**
> David Krongauz, Yanir Marmor et al.

---

## Overview

This pipeline predicts chronological age from voice recordings using WavLM Large
embeddings and Ridge regression, then benchmarks voice-age accuracy against
predictions derived from other biological modalities (blood tests, sleep, DXA,
retinal imaging, NMR, metabolomics, gut microbiome, diet).

Downstream analyses characterise subjects whose predicted voice age diverges most
from their true age, using volcano plots and SHAP-based feature attribution.

---

## Repository Structure

```
.
├── 1_preprocessing/            # Audio normalisation, silence trimming, segmentation, QC
│   ├── preprocess_voices.py        # Peak-normalise (max RMS = 1) and trim silence
│   ├── segment_audio.py            # Segment recordings into 5-second chunks
│   └── quality_control/
│       ├── extract_features_for_classifier.py   # Acoustic features for the QC model
│       └── train_classifier.py                  # Random Forest audio-fault classifier
│
├── 2_embeddings/               # WavLM Large embedding extraction
│   ├── audio_embedding_pipeline.py  # Batch-friendly extractor (all model families)
│   └── embeddings.py                # Individual embedder classes + mean-pooling
│
├── 3_age_prediction/           # Voice-age prediction via Ridge regression
│   ├── ridge_regression.py          # ridge_groupcv_with_exports / run_multi_seed_ridge
│   ├── run_age_prediction.py        # Entry-point script (edit CONFIG then run)
│   └── sensitivity_analysis.ipynb  # Interactive notebook: Ridge, SHAP, downstream
│
├── 4_multimodality_comparison/ # LightGBM age prediction for other biological modalities
│   ├── lightgbm_regression.py       # lightgbm_groupcv_with_exports / run_multi_seed_lightgbm
│   ├── run_multimodality_comparison.py  # Entry-point script for all modalities
│   └── predict_age_notebook.ipynb  # Interactive notebook for modality comparison
│
├── 5_downstream_analysis/      # Volcano plots: top vs bottom voice-age percentile groups
│   └── volcano_visualization.py     # Bin by true age, compare high/low predicted-age groups
│
└── 6_visualization/            # Cross-modality correlations and paper figures
    ├── modalities_correlations.py   # Correlation heatmap of predicted ages across modalities
    └── plots_for_paper.ipynb        # All paper-ready figures (demographics, box-plots, …)
```

---

## Pipeline Summary

```
Raw Audio (HPP-Voice cohort, ages 40–70)
         │
         ▼
1. Preprocessing & QC
   • Peak-normalise (max RMS = 1) and trim leading/trailing silence
   • Random Forest QC classifier trained on 488 manually labelled recordings
     (5-fold CV, mean AUC = 0.95); exclude recordings with fault probability > 50%
   • Segment each recording into 5-second chunks
         │
         ▼
2. Embedding Extraction
   • WavLM Large (317 M parameters, pre-trained on LVG 94k hr)
   • Frame-level outputs → mean pooling → single 1×d vector per recording
   • Only the first 5-second segment is used for regression
         │
         ▼
3. Voice-Age Prediction
   • Gender-stratified (male / female subsets)
   • Ridge regression with GroupKFold CV (5 folds, group = subject)
   • Alpha selection on held-out validation groups within each fold
   • 10 random seeds → averaged predictions (bagging) to reduce variance
   • SHAP LinearExplainer for feature attribution
         │
         ▼
4. Multimodality Comparison
   • Same multi-seed LightGBM regression applied to:
     sleep · blood tests · DXA · NMR · metabolomics
     retina · gut microbiome · diet · lifestyle
   • Best model (Ridge vs LightGBM) selected per modality by OOF R²
   • Cross-modality correlation of predicted ages (Pearson)
         │
         ▼
5. Downstream Analysis (Volcano Plots)
   • Within each 2-year age bin, identify top-25% and bottom-25%
     predicted-age subjects
   • Mann–Whitney U test per feature + Benjamini–Hochberg FDR correction
   • Volcano plots: Δ SD vs –log₁₀(p), coloured by significance
         │
         ▼
6. Visualisation
   • Correlation heatmap across all biological modalities
   • Age-distribution figures, box-plots (plots_for_paper.ipynb)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess audio

```bash
python 1_preprocessing/preprocess_voices.py --input_dir /path/to/raw_audio \
                                             --output_dir /path/to/processed_audio

python 1_preprocessing/segment_audio.py     --input_dir /path/to/processed_audio \
                                             --output_dir /path/to/segments
```

### 3. Extract WavLM embeddings

```bash
python 2_embeddings/audio_embedding_pipeline.py \
       --input_dir  /path/to/segments \
       --output_dir /path/to/embeddings \
       --model      wavlm_large
```

### 4. Predict voice age

Edit the `CONFIG` section of `3_age_prediction/run_age_prediction.py`, then:

```bash
python 3_age_prediction/run_age_prediction.py
```

### 5. Compare with other modalities

Edit the `CONFIG` section of `4_multimodality_comparison/run_multimodality_comparison.py`, then:

```bash
python 4_multimodality_comparison/run_multimodality_comparison.py
```

### 6. Volcano plots

```bash
python 5_downstream_analysis/volcano_visualization.py
```

### 7. Modality correlation heatmap

```bash
python 6_visualization/modalities_correlations.py
```

---

## Data Availability

The HPP-Voice dataset and associated clinical measurements are not publicly
available due to participant privacy regulations.  To request access, contact
the corresponding author.

---

## Key Methods

| Step | Method | Key Parameters |
|------|--------|----------------|
| Audio QC | Random Forest on 488 labelled recordings | 5-fold CV, AUC = 0.95 |
| Embeddings | WavLM Large | Mean pooling, first 5-sec segment |
| Age prediction | Ridge regression | GroupKFold (k=5), alpha optimised per fold, 10 seeds |
| Modality comparison | LightGBM | GroupKFold (k=5), 10 seeds, split by gender |
| Volcano analysis | Mann–Whitney U + BH-FDR | Top/bottom 25% within 2-year age bins |

---

## Citation

```bibtex
@article{krongauz2025voiceage,
  title   = {Voice-Age: Predicting Biological Age from Voice Using Deep Speech Embeddings},
  author  = {Krongauz, David et al.},
  year    = {2025},
}
```
