# Sound of Aging: Large-Scale Evidence for a Voice-Based Biological Clock

Computational pipeline accompanying the manuscript:

> **Sound of Aging: Large-Scale Evidence for a Voice-Based Biological Clock**
> David Krongauz, Yanir Marmor, Arad Zulti, Anastasia Godneva, Adina
> Weinberger, and Eran Segal.

## Overview

This repository contains the analysis code used to derive Voice Age, a
functional aging biomarker estimated from a standardized 30-second voice
recording. The final analytic cohort contains 6,979 Hebrew-speaking Israeli
adults aged 40–70 years, with one latest quality-controlled recording per
participant (3,631 females and 3,348 males).

Across ten independently shuffled participant-level five-fold outer
partitions, WavLM-Large ridge models achieved:

| Sex | R², mean ± SD | MAE, mean ± SD |
|---|---:|---:|
| Female | 53.9% ± 0.8% | 3.95 ± 0.02 years |
| Male | 44.0% ± 0.4% | 4.41 ± 0.02 years |

The repository covers audio preprocessing and quality control, WavLM and
engineered acoustic-feature extraction, repeated outer-CV age prediction with
fold-specific inner tuning holdouts,
comparison with eight non-acoustic aging clocks, leakage-free
voice-conditioned multimodal analysis, age-residualized phenome-wide
associations, participant-level bootstrap uncertainty, and generation of all
main and supplementary figures.

## Repository structure

```text
.
├── 1_preprocessing/
│   ├── preprocess_voices.py
│   ├── segment_audio.py
│   └── quality_control/
│       ├── extract_features_for_classifier.py
│       └── train_classifier.py
├── 2_embeddings/
│   ├── audio_embedding_pipeline.py
│   ├── embeddings.py
│   ├── extract_classical_features.py
│   └── distribute_classical_features.py
├── 3_age_prediction/
│   ├── ridge_regression.py
│   ├── run_age_prediction_one_per_subject.py
│   ├── run_classical_age_prediction.py
│   ├── run_classical_boosting_age.py
│   └── run_age_prediction_calibration_comparison.py
├── 4_multimodality_comparison/
│   ├── run_multimodality_comparison_lgbm.py
│   ├── run_multimodality_comparison_ridge.py
│   ├── run_voice_conditioned_holdout.py
│   ├── distribute_voice_conditioned_holdout.py
│   ├── run_wavlm_acoustic_probe.py
│   └── run_wavlm_acoustic_probe_qc.py
├── 5_downstream_analysis/
│   ├── build_combined_risk_factors.py
│   ├── run_all_volcano.py
│   └── volcano_visualization.py
├── 6_visualization/
│   ├── run_lollipop_voice_residualized.py
│   ├── run_boxplots_voice_residualized.py
│   ├── run_boxplots_panel.py
│   ├── phenotype_enrichment.py
│   ├── assemble_figure2.py
│   ├── power_analysis_age_prediction.py
│   ├── generate_participant_flowchart.py
│   ├── gen_fig3_clocks_comparison.py
│   ├── gen_fig_wavlm_probe_top.py
│   └── gen_supp_s10_wavlm_probe_full.py
└── publication_analysis/
    ├── run_repeated_analysis.py
    ├── distribute_repeated_analysis.py
    ├── distribute_engineered_ridge_extended.py
    ├── distribute_power_analysis.py
    ├── aggregate_and_generate_figures.py
    └── generate_residualization_figure.py
```

## Authoritative analysis workflows

### Age-prediction and benchmark analyses

`publication_analysis/run_repeated_analysis.py` is the authoritative worker for
the repeated-partition results. It implements:

- the exact 6,979-participant latest-recording cohort;
- sex-stratified analysis;
- ten shuffled participant-level five-fold outer partitions;
- a participant-level inner tuning holdout inside every outer training fold;
- median imputation fitted only within training data;
- the final ridge regularization grids;
- LightGBM selection from 30 reproducible randomized configurations;
- one complete out-of-fold prediction vector per sex and partition.

The engineered ridge benchmark uses the LSQR solver and its extended
regularization grid. `aggregate_and_generate_figures.py` combines
the task outputs and generates the synchronized primary-performance,
engineered-feature, partition-stability, comparison-clock, and multimodal
figures.

### Voice-conditioned multimodal analysis

The final complementarity analysis is implemented in
`4_multimodality_comparison/run_voice_conditioned_holdout.py`. Each modality,
sex, and partition uses:

- an intersection cohort containing both voice and modality data;
- a stratified participant-level 80/20 holdout;
- voice-model training that excludes all outer test participants;
- out-of-fold voice predictions for training participants;
- training-only linear calibration of voice predictions;
- identical test participants for the modality-only, voice-only, and combined
  models;
- gain calculated relative to the stronger single-modality input within each
  partition.

Run with `--full-pool-voice --oof-train --calibrate-voice` to reproduce the
final protocol. The older `run_voice_conditioned_hpo_worker.py` is retained
only as a legacy sensitivity analysis and is not the Figure 3 workflow.

### Phenome-wide ΔVA analysis

The final phenome workflow:

1. Averages the ten partition-specific out-of-fold Voice Age predictions.
2. Residualizes Voice Age against chronological age separately within sex.
3. Defines global top and bottom ΔVA quartiles.
4. Compares quartiles using two-sided Mann–Whitney tests with
   Benjamini–Hochberg FDR control.
5. Calculates standardized mean differences.
6. Estimates 95% intervals using 10,000 participant-level bootstrap
   resamples.
7. Adds visit-matched DXA visceral adipose tissue area and excludes the
   superseded VAT-mass variables.

The central scripts are `run_lollipop_voice_residualized.py`,
`run_boxplots_panel.py`, `phenotype_enrichment.py`, and
`assemble_figure2.py`.

### Sample-size learning curve

`6_visualization/power_analysis_age_prediction.py` mirrors the primary model:
the same cohort, ten shuffled participant-level outer partitions, fold-specific
inner tuning holdouts, imputation, ridge grid, and no WavLM standardization.
Training participants are subsampled independently inside each outer training
fold. The full-capacity endpoints reproduce the primary Figure 1 estimates.

## Current figure mapping

| Manuscript item | Main script |
|---|---|
| Figure 1 | `publication_analysis/aggregate_and_generate_figures.py` |
| Figure 2 | `run_lollipop_voice_residualized.py`, `run_boxplots_panel.py`, `assemble_figure2.py` |
| Figure 3 | `run_voice_conditioned_holdout.py`, `aggregate_and_generate_figures.py` |
| Figure 4 | `gen_fig_wavlm_probe_top.py` |
| Supplementary Figure 1, partition stability | `aggregate_and_generate_figures.py` |
| Supplementary Figure 2, learning curve | `power_analysis_age_prediction.py` |
| Supplementary Figure 3, engineered-feature comparison | `aggregate_and_generate_figures.py` |
| Supplementary Figure 4, age residualization | `generate_residualization_figure.py` |
| Supplementary Figure 5, full acoustic probe | `gen_supp_s10_wavlm_probe_full.py` |
| Supplementary Figure 6, participant flowchart | `generate_participant_flowchart.py` |
| Supplementary Figure 7, comparison-clock model families | `aggregate_and_generate_figures.py` |

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

The HPP data are controlled-access and are not included. Configure the data
and output path constants in the entry-point scripts for your environment
before running the complete analyses.

### Inspect and smoke-test the repeated-analysis tasks

```bash
python publication_analysis/run_repeated_analysis.py --kind ridge --list-tasks
python publication_analysis/run_repeated_analysis.py --kind lgbm --list-tasks
python publication_analysis/run_repeated_analysis.py \
    --kind ridge --job-index 0 --smoke
```

### Run the leakage-free voice-conditioned analysis

```bash
python 4_multimodality_comparison/run_voice_conditioned_holdout.py \
    --full-pool-voice --oof-train --calibrate-voice
```

### Run or combine the learning-curve tasks

```bash
python 6_visualization/power_analysis_age_prediction.py --list-tasks
python 6_visualization/power_analysis_age_prediction.py --job-index 0 --smoke
python 6_visualization/power_analysis_age_prediction.py --combine
```

### Aggregate results and regenerate figures

```bash
python publication_analysis/aggregate_and_generate_figures.py
python publication_analysis/generate_residualization_figure.py
python 6_visualization/run_lollipop_voice_residualized.py
python 6_visualization/run_boxplots_panel.py
python 6_visualization/assemble_figure2.py
python 6_visualization/gen_supp_s10_wavlm_probe_full.py
python 6_visualization/generate_participant_flowchart.py
```

The `distribute_*.py` launchers reproduce the laboratory Elysium/SGE
distribution used for the full CPU runs. They are optional and require the
local Elysium helper and cluster configuration.

## Data availability

Human Phenotype Project data are available to qualified researchers through
the HPP data-access process: <https://humanphenotypeproject.org>. This
repository contains analysis code only and does not include participant-level
voice, clinical, imaging, or multi-omic data.

## Citation

```bibtex
@article{krongauz2026sound,
  title  = {Sound of Aging: Large-Scale Evidence for a Voice-Based Biological Clock},
  author = {Krongauz, David and Marmor, Yanir and Zulti, Arad and Godneva, Anastasia and Weinberger, Adina and Segal, Eran},
  year   = {2026}
}
```

## License

GNU General Public License v3.0. See `LICENSE`.
