# Sound of Aging: Predicting Biological Age from Voice

Official computational pipeline accompanying the manuscript:

> **Sound of Aging: Large-Scale Evidence for a Voice-Based Biological Clock**
> David Krongauz, Yanir Marmor, Arad Zulti, Anastasia Godneva, Adina Weinberger, Eran Segal.

---

## Overview

This repository contains the full code base used to derive **Voice Age (VA)** — a
functional aging biomarker estimated from 30-second voice recordings — for 6,979
adults aged 40–70 in the Human Phenotype Project (HPP) cohort. Sex-stratified
ridge regression on WavLM-Large embeddings achieves Pearson *r* = 0.734 (female)
and 0.663 (male) with chronological age, exceeding the perceptual benchmark
established in voice science.

The pipeline covers six stages: (1) audio preprocessing and quality control,
(2) WavLM-Large and classical acoustic feature extraction (eGeMAPS, emobase,
ComParE 2016, Praat perturbation), (3) Voice Age prediction with nested
5-fold cross-validation, (4) head-to-head benchmarking against eight non-acoustic
aging clocks plus a voice-conditioned complementarity analysis, (5) phenome-wide
associations of age-residualized Voice Age acceleration (ΔVA), and (6) all
manuscript figures.

---

## Repository structure

```
.
├── 1_preprocessing/                Audio normalisation, silence trimming, QC
│   ├── preprocess_voices.py        Peak-normalise + trim silence
│   ├── segment_audio.py            Optional segmentation utility
│   └── quality_control/            Methods §2.3 / Supplementary S2
│       ├── extract_features_for_classifier.py   MFCC + spectral features
│       └── train_classifier.py                  Random Forest QC (AUROC 0.95)
│
├── 2_embeddings/                   Methods §2.4 / Supplementary S3
│   ├── audio_embedding_pipeline.py     Batch WavLM-Large extractor (GPU)
│   ├── embeddings.py                   Per-model embedder classes
│   ├── extract_classical_features.py   eGeMAPS / emobase / ComParE 2016 / Praat
│   ├── distribute_classical_features.py    HPC distribution (Elysium / SGE)
│   └── submit_classical_features.py        Cluster job submission helper
│
├── 3_age_prediction/               Methods §2.5 / Supplementary S6, S7, S7b
│   ├── ridge_regression.py             Nested-CV ridge pipeline (10 seeds)
│   ├── run_age_prediction.py           WavLM-Large ridge entry-point
│   ├── run_age_prediction_filtered.py  QC-filtered WavLM
│   ├── run_age_prediction_one_per_subject.py   Final analytic cohort (n=6,979)
│   ├── run_classical_age_prediction.py     Ridge on all 4 classical sets
│   ├── run_classical_boosting_age.py       LightGBM / XGBoost + Optuna HPO
│   ├── run_classical_rf_age.py             Random Forest baseline
│   ├── run_wavlm_rf_age.py                 RF on WavLM (control)
│   ├── run_egemaps_goodquality_ridge.py    eGeMAPS QC-filtered ridge
│   ├── run_emobase_mfcc_ridge.py           emobase MFCC subset ridge
│   ├── test_egemaps_lgbm_optuna.py         eGeMAPS LightGBM smoke test
│   ├── run_external_politician_eval.py     Out-of-cohort Hebrew validation
│   └── sensitivity_analysis.ipynb          Exploratory notebook
│
├── 4_multimodality_comparison/     Methods §2.6 / Supplementary S4, S6b, S9, S10
│   ├── lightgbm_regression.py             Nested-CV LightGBM pipeline
│   ├── run_multimodality_comparison.py    8-clock LightGBM (legacy entrypoint)
│   ├── run_multimodality_comparison_ridge.py    Ridge variant
│   ├── run_multimodality_comparison_lgbm.py     Final LightGBM + RandomizedSearch
│   ├── run_classical_lgbm_age.py          Classical feature sets with LightGBM HPO
│   ├── run_voice_conditioned_age.py       Voice-conditioned complementarity
│   ├── run_voice_conditioned_hpo_worker.py    Distributed nested-HPO worker
│   ├── distribute_voice_conditioned_hpo.py    Elysium job distributor (8×10=80 jobs)
│   ├── run_wavlm_acoustic_probe.py        WavLM→56-feature ridge probe (S9)
│   ├── run_wavlm_acoustic_probe_qc.py     QC-filtered probe variant
│   └── predict_age_notebook.ipynb         Exploratory notebook
│
├── 5_downstream_analysis/          Methods §2.7 / Supplementary S7
│   ├── volcano_visualization.py           OLS ΔVA residualization core + volcano engine
│   ├── run_all_volcano.py                 Orchestrator across all modalities
│   └── build_combined_risk_factors.py     Phenotype-table assembly
│
└── 6_visualization/                Figure-generation scripts
    ├── run_predicted_vs_chronological_age.py            Fig. 1b,c
    ├── run_predicted_vs_chronological_age_gradient.py   Fig. 1b,c (gradient variant)
    ├── make_combined_boxplot_fig.py                     Fig. 2c–h
    ├── run_lollipop_voice_residualized.py               Fig. 2a,b (ΔVA lollipop)
    ├── run_boxplots_voice_residualized.py               Fig. 2 supporting boxplots
    ├── gen_fig3_clocks_comparison.py                    Fig. 3a–c (R² bar + heatmaps)
    ├── modalities_correlations.py                       Fig. 3b,c source
    ├── gen_fig_wavlm_probe_top.py                       Fig. 4 (top-2 per category)
    ├── generate_participant_flowchart.py                Supp Fig. S4
    ├── generate_supp_s1_s5.py                           Supp Fig. S1 + S5
    ├── age_bias_check.py                                Supp Fig. S6
    ├── power_analysis_age_prediction.py                 Supp Fig. S7 (learning curve)
    ├── gen_supp_s10_wavlm_probe_full.py                 Supp Fig. S9 + S10
    ├── gen_supp_s11_complementarity.py                  Supp Fig. S11 (dumbbell)
    └── plots_for_paper.ipynb                            Aggregated figure notebook
```

---

## Analysis-to-figure mapping (manuscript ↔ code)

| Manuscript element | Methods § | Driver script |
|---|---|---|
| Voice QC random-forest classifier | Suppl. Methods S2 | `1_preprocessing/quality_control/train_classifier.py` |
| WavLM-Large mean-pooled embeddings | Suppl. Methods S3 | `2_embeddings/audio_embedding_pipeline.py` |
| Classical features (eGeMAPS / emobase / ComParE 2016 / Praat) | Methods §3 | `2_embeddings/extract_classical_features.py` |
| Sex-stratified ridge with nested-CV α selection | Methods §4, Suppl. S6 | `3_age_prediction/run_age_prediction_one_per_subject.py` |
| Classical-feature comparison (Ridge / RF / LightGBM + Optuna) | Results, Suppl. S1 | `3_age_prediction/run_classical_*.py`, `test_egemaps_lgbm_optuna.py` |
| Sample-size learning curve (n = 50–2,500, 30 seeds) | Suppl. Methods S7b | `6_visualization/power_analysis_age_prediction.py` |
| Seed-stability analysis (10 seeds × 5-fold) | Suppl. S5 | `6_visualization/generate_supp_s1_s5.py` (S5 panel) |
| Eight aging-clock comparison + per-modality HPO | Methods §6, Suppl. S6b | `4_multimodality_comparison/run_multimodality_comparison_lgbm.py` (+ `_ridge.py`) |
| Voice-conditioned complementarity (8 × 10 × 2 = 160 runs) | Suppl. Methods S10 | `4_multimodality_comparison/run_voice_conditioned_hpo_worker.py` (driver: `distribute_voice_conditioned_hpo.py`) |
| WavLM → 56 acoustic-feature ridge probe | Suppl. Methods S9 | `4_multimodality_comparison/run_wavlm_acoustic_probe.py` |
| ΔVA OLS residualization | Suppl. Methods S7 | `5_downstream_analysis/volcano_visualization.py` (`compute_residualized_delta`) |
| Phenome-wide quartile MWU + BH-FDR | Methods §7 | `5_downstream_analysis/run_all_volcano.py` + `6_visualization/run_lollipop_voice_residualized.py` |
| Fig. 1 scatter (predicted vs chronological age) | — | `6_visualization/run_predicted_vs_chronological_age*.py` |
| Fig. 2 lollipop + boxplots | — | `6_visualization/run_lollipop_voice_residualized.py`, `make_combined_boxplot_fig.py` |
| Fig. 3 nine-clock R² bar + correlation heatmaps | — | `6_visualization/gen_fig3_clocks_comparison.py` |
| Fig. 3d / Suppl. Fig. S11 complementarity dumbbell | — | `6_visualization/gen_supp_s11_complementarity.py` |
| Fig. 4 / Suppl. Fig. S9–S10 acoustic probe | — | `6_visualization/gen_fig_wavlm_probe_top.py`, `gen_supp_s10_wavlm_probe_full.py` |
| Suppl. Fig. S4 participant flowchart | — | `6_visualization/generate_participant_flowchart.py` |
| Suppl. Fig. S6 age-bias check | — | `6_visualization/age_bias_check.py` |
| Suppl. Fig. S7 sample-size curve | — | `6_visualization/power_analysis_age_prediction.py` |

---

## Pipeline summary

```
Raw audio (HPP cohort, ages 40–70, 30-s counting task)
        │
        ▼
1. Preprocessing
   • Peak-normalise, trim leading/trailing silence (energy VAD)
   • Random Forest QC classifier on MFCC + spectral features
     (488 labelled recordings, 5-fold CV, AUROC = 0.95 ± 0.04)
   • Retain one recording per participant (most recent visit) → n = 6,979
        │
        ▼
2. Feature extraction
   • WavLM-Large layer-24, mean-pooled → 1,024-d vector per recording
   • Classical sets via openSMILE (eGeMAPS, emobase, ComParE 2016)
     and Parselmouth/Praat (jitter, shimmer, HNR, F0, CPP)
        │
        ▼
3. Voice-Age model (sex-stratified)
   • Ridge regression, 5-fold GroupKFold outer CV
   • Inner-holdout α grid {0.01,0.05,0.1,0.2,0.5,1,5,10}
   • 10 random seeds; OOF predictions pooled across outer folds
        │
        ▼
4. Multimodal benchmarking
   • 8 non-voice clocks (MS metabolomics, NMR, DEXA, retinal, sleep,
     diet, lifestyle, gut microbiome): LightGBM + Optuna (50 trials)
     vs Ridge with α grid; per-modality model selected by OOF R²
   • Voice-conditioned analysis: append OOF Voice Age as extra
     feature, retrain with identical nested HPO → ΔR² complementarity
        │
        ▼
5. Downstream phenome-wide analysis
   • ΔVA = VA − OLS(VA ~ CA) within each sex (age-residualization)
   • Stratify by 2-year age bins, compare top vs bottom quartile
     with Mann–Whitney U + Benjamini–Hochberg FDR (q < 0.1)
        │
        ▼
6. Figures (manuscript main + supplementary)
   • Scripts in 6_visualization/ produce the exact PDFs used in the paper
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Classical features additionally require `opensmile` (Linux/Mac binaries bundled
with the wheel) and `praat-parselmouth`.

### 2. Preprocess audio + train QC classifier

```bash
python 1_preprocessing/preprocess_voices.py --input_dir <raw_audio> --output_dir <processed>
python 1_preprocessing/quality_control/extract_features_for_classifier.py \
       --input_dir <processed> --labels <annotations.csv> --output <qc_features.pkl>
python 1_preprocessing/quality_control/train_classifier.py --features <qc_features.pkl>
```

### 3. Extract embeddings + classical features

```bash
python 2_embeddings/audio_embedding_pipeline.py \
       --input_dir <processed> --output_dir <embeddings> --model wavlm_large
python 2_embeddings/extract_classical_features.py \
       --input_dir <processed> --output_dir <classical_features>
```

### 4. Train the Voice-Age model

Edit the `CONFIG` block at the top of the script, then:

```bash
python 3_age_prediction/run_age_prediction_one_per_subject.py
```

This writes per-seed metrics, predictions, and SHAP values to
`paper_revision_outputs/step3_voice_age_ridge/`.

### 5. Multimodal benchmark + complementarity

```bash
python 4_multimodality_comparison/run_multimodality_comparison_lgbm.py
python 4_multimodality_comparison/distribute_voice_conditioned_hpo.py   # cluster
```

### 6. Phenome-wide analysis

```bash
python 5_downstream_analysis/build_combined_risk_factors.py
python 5_downstream_analysis/run_all_volcano.py
python 6_visualization/run_lollipop_voice_residualized.py
```

### 7. Reproduce manuscript figures

```bash
python 6_visualization/generate_participant_flowchart.py        # Supp Fig. S4
python 6_visualization/generate_supp_s1_s5.py                   # Supp Figs. S1, S5
python 6_visualization/power_analysis_age_prediction.py         # Supp Fig. S7
python 6_visualization/age_bias_check.py                        # Supp Fig. S6
python 6_visualization/gen_fig3_clocks_comparison.py            # Fig. 3a–c
python 6_visualization/gen_supp_s11_complementarity.py          # Fig. 3d / Supp S11
python 6_visualization/gen_fig_wavlm_probe_top.py               # Fig. 4
python 6_visualization/gen_supp_s10_wavlm_probe_full.py         # Supp Figs. S9, S10
python 6_visualization/make_combined_boxplot_fig.py             # Fig. 2c–h
python 6_visualization/run_predicted_vs_chronological_age.py    # Fig. 1b,c
```

---

## Data availability

The Human Phenotype Project data are available to qualified researchers via a
formal application process administered by the HPP data access committee
(<https://humanphenotypeproject.org>). All paths in the entry-point scripts
reference internal lab storage and must be edited to match the user's
environment.

---

## Citation

```bibtex
@article{krongauz2026voiceage,
  title   = {Sound of Aging: Large-Scale Evidence for a Voice-Based Biological Clock},
  author  = {Krongauz, David and Marmor, Yanir and Zulti, Arad and Godneva, Anastasia and Weinberger, Adina and Segal, Eran},
  year    = {2026},
}
```

---

## License

GNU General Public License v3.0 — see `LICENSE`.
