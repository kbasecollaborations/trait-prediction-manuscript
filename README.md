# Trait prediction manuscript

Manuscript and analysis code for the carbon-utilization (trait) prediction
project. This README helps readers and reviewers locate the code behind each
figure and table.

## Compiling the manuscript

```bash
latexmk -pdf -f main.tex
```

## Repository layout

- `main.tex` — manuscript source
- `data/` — input data and generated outputs (not tracked in git)
- `figures/` — generated figures included in the manuscript
- `scripts/` — all analysis and figure/table-generation code (details below)

## Running scripts

Every script is run as a module from the repository root (no `.py`
extension):

```bash
uv run python -m scripts.figure3.figure3_plot
```

Within each figure folder the convention is:

- `*_data.py` — compute the underlying results and write them to `data/`
- `*_plot.py` — read those results and render the figure
- `figureN_plot.py` — assemble the full multi-panel figure

So a figure is reproduced by running its `*_data.py` scripts first, then the
corresponding `*_plot.py`. Each script's module docstring states what it
produces.

## Main figures → code

| Figure | Folder | Content |
| --- | --- | --- |
| Figure 1 | `scripts/figure1/` | Dataset characteristics: phenotype class distributions (1B, `figure1b_data.py` + `figure1b_plot.py`) and the analysis-variant performance ladder (1C, `figure1c_plot.py`). Panel 1A is a BioRender illustration (no script); `figure1_combine.py` stacks the three panel PDFs. |
| Figure 2 | `scripts/figure2/` | Baselines: GapMind (2A) and phylogeny-based nearest-neighbor (2B). |
| Figure 3 | `scripts/figure3/` | Model performance across train/test split strategies (random, dataset, phylogenetic), compared with GapMind. |
| Figure 4 | `scripts/figure4/` | Feature importance: SHAP-based identification of predictive gene functions (`figure4c_*`), with quadrant/summary panels. |
| Figure 5 | `scripts/figure5/` | GapMind concordance: performance on concordant samples, plus concordant-vs-discordant and full-test comparisons (5A–5D). |
| Figure 6 | `scripts/figure6/` | Training-set filtering diagnostics: confidence/concordance filtering, problematic-sample removal, and ML vs GapMind. |
| Figure 7 | `scripts/figure7/` | Selective prediction / deployment: risk–coverage, calibration, and active learning (`applicability.py`, `figure7d_active_learning.py`). |

Supporting diagnostics for Figure 5A underperformers live in
`scripts/alternate/figure5_diagnostic/`.

## Supplementary figures → code

| Figure | Folder | Content |
| --- | --- | --- |
| Figure S1 | (no script) | Pruned GTDB reference tree (`figure_s1.png`), rendered outside this repository. |
| Figure S2 | `scripts/figureS2/` | Phylogenetic distance distributions across evaluation split types. |
| Figure S3 | `scripts/figureS3/` | In-clade vs out-of-clade balanced accuracy. |
| Figure S4 | `scripts/pangenome_completeness/` | Pangenome-based completeness audit. |
| Figure S5 | `scripts/figureS5/` | KOFAM features on concordant samples. |
| Figure S6 | `scripts/figureS6/figureS6_multiphenotype_plot.py` | Per-phenotype performance versus training-set size, all 15 shared phenotypes (source data `data/outputs/learning_curves/`). The histidine-only grid from `figureS6_data.py`/`figureS6_plot.py` is now an alternate, with source data in `data/outputs/learning_curves_histidine/`. |
| Figure S7 | `scripts/bacdive/`, `scripts/figureS7/` | BacDive data-volume comparison and feature stability (asset `figure_s7.pdf`). |

Unselected, superseded, and exploratory code is kept under `scripts/alternate/`
(including `figureS7_legacy/`, `figureS8/`, `figureS8_legacy/`, `figureS9_legacy/`,
and `figure5_diagnostic/`). Supplementary Figures S7 and S8 were removed from the
manuscript; the BacDive figure was renumbered S9 -> S7.

## Tables → code

All in `scripts/tables/`:

- `main_table1.py` — main-text Table 1 (concordance vs full feature
  comparison)
- `table1.py` — Supplementary Table S4 (stable features under full-data and
  concordant training)
- `table2.py` — compatibility entry point for the merged Supplementary Table S4
- `kegg_module_coverage.py` — KEGG-module pathway coverage

p-values reported in the text are produced by
`scripts/stats/manuscript_pvalues.py`. Supplementary Table S9 (cross-dataset
label conflicts, Supplementary Text S12) is produced by
`scripts/measurement_reliability/label_conflict.py`.

## Shared pipeline and utilities

Data preparation and modeling code shared across figures lives at the top of
`scripts/`:

- Data prep: `create_data_splits.py`, `combine_features.py`,
  `create_gapmind_features.py`, `feature_filtering.py`,
  `feature_clustering.py`, `refresh_kegg_data.py`
- Modeling: `ml.py` (classifier factory and CV/train-test helpers),
  `ml_splits.py`, `run_concordant_models.py`, `minority_filter.py`
- Baseline classifiers: `scripts/classifiers/` (Bernoulli, nearest-neighbor,
  identity)
- Helpers: `io.py`, `visualization.py`, `collections.py`, `distances.py`,
  `splitter.py`

Additional analyses not tied to a single figure are under
`scripts/checkm2/`, `scripts/pangenome_completeness/`, and `scripts/misc/`.

## Conventions

- Features: KOFAM annotations by default (correlation filter 0.95, variance
  filter 0.01).
- Model: CatBoost via `make_classifier` in `scripts/ml.py`.
- Plots: `scienceplots` style; `scripts/figure4/figure4c_plot.py` is the
  styling reference.
