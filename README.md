# Trait prediction manuscript

Manuscript source and analysis code for *"Data coherence over data volume drives
generalisable genome-based prediction of microbial carbon utilisation"*
([bioRxiv 2026.08.06.743333](https://doi.org/10.64898/2026.08.06.743333)). This
README helps readers and reviewers locate the code behind each figure and table.

The reusable machine-learning library this code builds on lives in a separate
repository: <https://github.com/kbasecollaborations/trait-prediction>.

## Data

The input data, figure and table source data, trained models and predicted
proteomes are deposited on Zenodo (<https://doi.org/10.5281/zenodo.21223428>).
Unpack both archives at the repository root:

```bash
unzip trait-prediction-data.zip -d .
unzip proteomes_all_seqs.zip -d .
```

`trait-prediction-data.zip` recreates the `data/` tree the scripts read, including
the source data each plotting script draws from; `proteomes_all_seqs.zip` adds the
predicted proteomes under `data/raw/all_seqs/`. The record's own README describes
every file.

## Compiling the manuscript

```bash
just main     # build/main.pdf: main text with the Supplementary Information appended
just submit   # build/main.pdf and build/supplement.pdf as separate documents
```

`just main` runs `latexmk -pdf -f main.tex`. After `just submit`, run `just clean`
before `just main`, or latexmk keeps the split `main.pdf`.

## Repository layout

- `main.tex` — root manuscript file; the text, figure legends and Supplementary
  Information are in `sections/`
- `supplement.tex` — standalone Supplementary Information, built by `just submit`
- `references.bib` — bibliography
- `data/` — input data and generated outputs (not tracked in git; see Data)
- `figures/` — generated figures included in the manuscript
- `scripts/` — all analysis and figure/table-generation code (details below)

## Running scripts

Python scripts are run as modules from the repository root (no `.py`
extension):

```bash
uv run python -m scripts.figure3.figure3_plot
```

Within each figure folder the convention is:

- `*_data.py` — compute the underlying results and write them to `data/`
- `*_plot.py` — read those results and render the figure
- `figureN_plot.py` — assemble the full multi-panel figure

The Zenodo archive includes every figure's source data, so each `*_plot.py` runs
directly after unpacking. To regenerate the source data, run the data scripts
first, in this order:

1. `scripts.figure2.figure2a_data`, which writes the GapMind calls that define
   concordance for the later scripts.
2. Each figure's data scripts: the `*_data.py` files plus the other producers
   named in the tables below. Also run
   `scripts.figure5.figure5a_data --features gapmind_raw` for the Figure 5A
   reference lines.
3. `scripts.feature_clustering`, then `scripts.figure4.figure4c_data` and
   `scripts.figure5.figure5b_data` again, so that Figures 4C and 5B and Tables 1
   and S3 count redundancy clusters.
4. `scripts.stats.manuscript_pvalues`, which writes the McNemar tables that
   Figure 6C reads.

Each script's module docstring states its inputs and outputs. Two steps use tools
outside the uv environment: `bash scripts/pangenome_completeness/run_local.sh`
(MMseqs2, Figure S4) and `sbatch scripts/checkm2/run_checkm2_nersc.sh` (CheckM2 on
a SLURM cluster).

## Main figures → code

| Figure | Folder | Content |
| --- | --- | --- |
| Figure 1 | `scripts/figure1/` | Dataset characteristics: phenotype class distributions (1B, `figure1b_data.py` + `figure1b_plot.py`) and the analysis-variant performance ladder (1C, `figure1c_plot.py`, which reads the Figure 3, 5 and 6 outputs). Panel 1A is a BioRender illustration (no script); `figure1_combine.py` stacks the three panel PDFs. |
| Figure 2 | `scripts/figure2/` | Baselines: GapMind (2A) and phylogeny-based nearest-neighbor (2B). |
| Figure 3 | `scripts/figure3/` | Model performance across train/test split strategies (random, dataset, phylogenetic), compared with GapMind (`figure3a_gapmind_random_split.py`, `figure3b_gapmind_dataset_split.py`). |
| Figure 4 | `scripts/figure4/` | GapMind–experiment discordance and cross-dataset feature stability: concordance quadrants (4A, `figure4a_quadrant.py`), GapMind outcome counts by phenotype and dataset (4B, `figure4b_plot.py`), and stable SHAP feature clusters, shared versus dataset-specific (4C, `figure4c_data.py` + `figure4c_plot.py`). `figure4_plot.py` assembles the figure. |
| Figure 5 | `scripts/figure5/` | Concordant training: balanced accuracy on concordant held-out samples (5A, `figure5a_data.py`), cross-dataset stability of SHAP redundancy clusters (5B, `figure5b_data.py`), and ML versus GapMind on the full held-out test set and its concordant and discordant subsets (5C–5D, `figure5cd_data.py`, `figure5d_full_test_data.py`). `figure5_plot.py` draws the figure. |
| Figure 6 | `scripts/figure6/` | Confidence weighting and feature choice: problematic genomes (6A, `figure6a_plot.py`), the soft-label weight sweep with its reference conditions (6B, `figure6b_weight_sweep.py`, `figure6c_data.py`), per-phenotype recall change against GapMind (6C, `figure6c_mechfree_per_sample.py`), and combined versus phenotype-filtered GapMind features (6D, `figure6d_data.py`). `figure6b_aggregate_plot.py` and `figure6d_plot.py` draw the panels; `figure6_plot.py` assembles the figure. |
| Figure 7 | `scripts/figure7/` | Selective prediction and experiment prioritisation: risk–coverage and calibration (7A–B, `figure7_data.py`) and label-selection strategies (7C–D, `figure7d_active_learning.py`). `figure7_plot.py` draws the figure; `applicability.py` holds shared helpers. |

Supporting diagnostics for Figure 5A underperformers live in
`scripts/figure5_diagnostic/`.

## Supplementary figures → code

| Figure | Folder | Content |
| --- | --- | --- |
| Figure S1 | `scripts/figureS1/` | Circular phylogeny of the 629 GTDB-placed genomes, drawn from the pruned GTDB reference tree with pyCirclize. |
| Figure S2 | `scripts/figureS2/` | Phylogenetic distance distributions across evaluation split types. |
| Figure S3 | `scripts/figureS3/` | In-clade vs out-of-clade balanced accuracy. |
| Figure S4 | `scripts/pangenome_completeness/` | Pangenome-based completeness audit. |
| Figure S5 | `scripts/figureS5/` | KOFAM features on concordant samples. |
| Figure S6 | `scripts/figureS6/figureS6_plot.py` | Per-phenotype performance versus training-set size, all 15 shared phenotypes. Source data `data/outputs/learning_curves/` from `figureS6_data.py`; drawing helpers in `figureS6_grid.py`. |
| Figure S7 | `scripts/bacdive/`, `scripts/figureS7/` | BacDive data-volume comparison and feature stability (asset `figure_s7.pdf`). |

## Tables → code

- `scripts/tables/main_table1.py` — main-text Table 1 (concordance vs full
  feature comparison), written to `sections/table_main_feature_comparison.tex`
- `scripts/tables/histidine_feature_table.py` — Supplementary Table S3, the
  histidine worked example (stable features under full-data and concordant
  training), written to `sections/table_feature_comparison.tex`
- `scripts/tables/kegg_module_coverage.py` — phenotype-to-KEGG pathway and module
  assignments used by both generators; Table S3 adds a module-coverage line to
  each phenotype
- Supplementary Table S1 (dataset overview) is written directly in
  `sections/supplementary.tex`; Supplementary Table S2 (model families)
  summarises the `scripts/ml_comparison/` runs

`scripts/stats/manuscript_pvalues.py` recomputes the Benjamini–Hochberg-adjusted
primary tests described in Methods and the related comparisons, and writes them to
`data/outputs/stats/`. The p-values annotated on Figures 5, 6D and S7 are computed
by their plotting scripts. The cross-dataset label-conflict analysis behind
Supplementary Text S12 is produced by
`scripts/measurement_reliability/label_conflict.py`.

## Shared pipeline and utilities

Data preparation and modeling code shared across figures lives at the top of
`scripts/`:

- Phenotype harmonisation: `build_substrate_identity.py` binds each phenotype
  column to one source substrate (Supplementary Text S1), then
  `harmonize_phenotypes.py` writes `data/processed/phenotypes/` from
  `data/raw/phenotypes/`
- Features: `refresh_kegg_data.py` (KEGG mapping files),
  `create_gapmind_features.py` (GapMind step scores), `feature_filtering.py`
  (variance and correlation filtering), then `combine_features.py`
- Splits: `create_data_splits.py` (random, leave-one-dataset-out and phylogeny
  splits)
- Redundancy clusters: `feature_clustering.py` groups the stable KOs from the
  Figure 4C and 5B SHAP runs (Supplementary Text S7; see Running scripts for the
  order)
- Modeling: `ml.py` (classifier factory and CV/train-test helpers),
  `ml_splits.py`, `minority_filter.py`, `run_concordant_models.py` (the 75
  per-fold concordant models), and `train_full_data_models.py` and
  `train_concordant_full_models.py` (the 30 per-phenotype deployment models in
  the Zenodo record)
- Zenodo record: `build_biolog_provenance.py` writes
  `biolog_assay_provenance.tsv`, and `build_zenodo_deposit.py` assembles the
  record under `data/zenodo/`
- Baseline classifiers: `scripts/classifiers/` (Bernoulli, nearest-neighbor,
  identity)
- Helpers: `io.py` (readers), `visualization.py` (shared colour palette and
  matplotlib style), `splitter.py`

## Supporting analyses

Analyses that back supplementary text rather than a single figure:

- `scripts/ml_comparison/` — alternative model families and tuning
  (Supplementary Table S2, Supplementary Text S3); the LightGBM runs need
  `uv run --with lightgbm python -m scripts.ml_comparison.runner`
- `scripts/leakage_check_feature_dup.py` — duplicate feature vectors across
  random-holdout splits (Supplementary Text S3)
- `scripts/figure5/figure5b_random_control.py` — size- and class-matched
  random-subset control (Supplementary Text S5)
- `scripts/figure5_diagnostic/` — why individual phenotypes underperform on
  Figure 5A, and what GapMind false negatives contribute
- `scripts/figure5/fp_only_filter.py` and
  `scripts/agreement_analysis/agreement_analysis.py` — false-positive-only
  filtering and ML–GapMind agreement at prediction time (Supplementary Text S10)
- `scripts/bacdive/` — the data-volume comparison behind Supplementary Text S11
- `scripts/measurement_reliability/` — cross-dataset label conflicts
  (Supplementary Text S12)
- `scripts/checkm2/` — CheckM2 runs and MIMAG quality tiers (Supplementary
  Text S2)
- `scripts/pangenome_completeness/` — pangenome completeness (Supplementary
  Text S2, Figure S4)

## Conventions

- Features: KOFAM annotations by default (correlation filter 0.95, variance
  filter 0.01).
- Model: CatBoost via `make_classifier` in `scripts/ml.py`.
- Plots: `scienceplots` style; `scripts/figure4/figure4c_plot.py` is the
  styling reference.

## Citation

> Kishore D, Ranjan P, Neely C, Cashman M, Sakkaff Z, Riehl WJ, Joachimiak MP,
> Edirisinghe JN, Faria JP, Weisenhorn P, Cohen MB, Pelletier DA, Cottingham RW,
> Henry CS, Doktycz MJ, Arkin AP, Dehal PS. *Data coherence over data volume
> drives generalisable genome-based prediction of microbial carbon utilisation.*
> bioRxiv 2026.08.06.743333 (2026). <https://doi.org/10.64898/2026.08.06.743333>

## License

See [LICENSE](LICENSE).
