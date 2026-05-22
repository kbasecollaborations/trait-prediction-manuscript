# Figure 8 redesign — design spec

**Date:** 2026-05-22
**Topic:** Redesign Figure 8 of the trait-prediction manuscript from an incoherent
"diagnostic/applicability-domain toolkit" into a coherent capstone:
*reliability of genome-based predictions, its limits, and which experiments to run next.*

## Motivation

The current Figure 8 mixes unrelated questions and contains two methodological flaws
surfaced during review:

1. **Panel C (shortcut-gap vs concordance-benefit) is artifactual.** Both axes subtract
   `cross_ba_full`, so they share a term with the same sign; the reported Spearman
   ρ = 0.27 (p = 0.32, already non-significant) is largely manufactured by that shared
   term. `random_ba` is near-constant across phenotypes, so the shortcut gap is
   essentially `constant − cross_ba_full`.
2. **Panel B (GapMind–ML agreement) is conceptually circular for the concordant model.**
   The concordant model is *defined* by training on GapMind-concordant samples, so
   checking its agreement with GapMind measures "did it absorb GapMind," not independent
   reliability. (A perfect specialist on the concordant domain trivially agrees with
   GapMind there.)

The figure also conflated per-genome trust (A, B) with a per-phenotype model-selection
question (C), and only Panel C touched the full-data model while A/B/D used only the
concordant model.

## Decisions (from brainstorming)

- **Message / spine:** experiment prioritization. Figure 8 = *"from reliability to
  action."* Establish per-genome reliability and its limits, then use the model's own
  signals to choose which genomes and which phenotypes to measure next. Pairs with
  Figure 7 (Fig 7 = "how many samples"; Fig 8 = "which samples and which phenotypes").
- **Confidence signal = the model's own probabilities** (`max(p, 1−p)`), not GapMind
  agreement. GapMind-agreement panels are retired.
- **Reframe the phenotype level as prioritization, not confidence prediction.** Research
  showed there is *no* working label-free per-phenotype confidence signal (see below);
  we state that honestly and pivot to "which phenotypes need more experiments."
- **Two-model comparison** runs through A and B (reliability + limits of each model);
  C and D concern improving the *deployed concordant specialist* and are single-model.
  This is a deliberate, flagged relaxation of "every panel compares both models."

## Empirical findings the design rests on

Computed on the leave-one-dataset-out held-out test sets (15 phenotypes, 4 datasets;
KOFAM matrix 822 genomes × 5544 features). Two models: `concordant` (deployed
specialist) and `full_data`.

- **Per-genome confidence works, modestly.** AUROC(confidence → correct) = 0.645
  (concordant), 0.62 (full-data).
- **Both models are overconfident.** ECE = 0.176 (concordant), 0.119 (full-data); in the
  top probability bin (mean pred 0.97) observed positive rate is only 0.80.
- **Per-genome feature-distance (OOD) is weaker** (AUROC 0.56–0.58) and does not beat
  confidence, even combined.
- **No label-free per-phenotype confidence signal** (n = 15): mean confidence ρ = +0.04
  (p = 0.90), fraction-high-confidence ρ = −0.10, mean OOD ρ = −0.17, fraction-in-domain
  ρ = +0.05 — all p > 0.37. Reason: genomes are shared across phenotypes, so genome-level
  properties are near-constant across phenotypes; phenotype difficulty is a property of
  the genotype→phenotype mapping, not genome novelty. (The only signal that ever
  correlated, GapMind-agreement rate ρ = 0.70, is the rejected circular one.)
- **Experiment prioritization works.** Selecting lowest-confidence held-out genomes to
  label improves cross-dataset BA more than random (+0.027 BA, Wilcoxon p = 0.0037,
  win-rate 0.56; K = 25, 6 phenotypes × 4 datasets × 3 seeds). High-OOD and diversity
  selection do **not** beat random. Per-phenotype gains concentrate on currently-weak
  phenotypes (Glucose, Cellobiose, Mannose); strong ones (Maltose, Histidine) barely move.

These numbers must be reproduced by the formal pipeline before they enter the manuscript.

## Data architecture

One concept: **two per-sample prediction tables** on the same leave-one-dataset-out
held-out genomes, plus one **prioritization-simulation** output.

- `data/outputs/figure8/figure8_per_sample.tsv` — concordant model (exists). Columns:
  `phenotype, held_out_dataset, genome, y_true, y_pred, proba, confidence, gapmind_pred`.
- `data/outputs/figure8/figure8_per_sample_fulldata.tsv` — full-data model (generated
  during exploration; must be produced by a permanent pipeline step folded into
  `figure8_data.py`, training on all samples rather than the concordant subset).
- Prioritization simulation output (new): per (strategy, phenotype, held_out_dataset,
  seed, n_added) → cross-dataset BA gain. Adapt
  `scripts/figure8/figure8d_active_learning.py`, replacing GapMind-disagreement selection
  with label-free strategies {low_confidence, high_ood, diversity, random}.

`gapmind_pred` stays in the per-sample tables (harmless, used nowhere in the new figure).

## Panels

### Panel A — Per-genome reliability (risk–coverage), both models, three representatives
- **Layout:** small multiples — three mini-plots for m-Inositol (strong), Histidine
  (medium), Glucose (weak) cross-dataset generalisers. Each mini-plot shows two curves
  (full-data, concordant).
- **Construction:** class-stratified retention per phenotype per model (the fix for the
  one-sided-confidence degeneracy): at each coverage level retain the top fraction of
  most-confident class-1 predictions and most-confident class-0 predictions in parallel,
  then recompute balanced accuracy. Random 0.5 reference line.
- **Point:** each model's reliability and whether its own confidence supports abstention;
  abstention lifts the medium generaliser (Histidine) most, cannot save the weak one
  (Glucose), adds little to the strong one (m-Inositol).
- **Connects to message:** establishes confidence as a usable but imperfect per-genome
  reliability signal for both models.

### Panel B — Calibration, both models (pooled)
- **Layout:** reliability diagram (predicted confidence vs empirical accuracy), one curve
  per model, pooled across all phenotypes; diagonal = perfect calibration; annotate ECE
  per model.
- **Point:** the honest limitation — both models are overconfident (ECE 0.18 / 0.12);
  high confidence ≠ correct under cross-dataset shift. Motivates abstention (A) and
  gathering more data (C, D).
- **Connects to message:** why raw probabilities cannot be trusted as probabilities, only
  as a ranking — bridging reliability to the need for more experiments.

### Panel C — Which genomes to measure (experiment prioritization, pooled)
- **Layout:** cross-dataset BA gain vs number of added labels; one curve per selection
  strategy {low_confidence, high_ood, diversity, random}; error band over
  seeds/datasets. Pooled across the simulated phenotypes.
- **Point:** the deployed model's own uncertainty is the actionable acquisition signal —
  low-confidence selection beats random (+0.027 BA, p = 0.004); OOD and diversity do not.
- **Connects to message:** turns the imperfect confidence signal (A, B) into a concrete
  genome-selection rule for improving the model.

### Panel D — Which phenotypes to invest in (prioritization, all 15 phenotypes)
- **Layout:** per-phenotype BA gain per added sample vs current cross-dataset BA
  (scatter or ordered bars), all 15 phenotypes labelled.
- **Point:** gains concentrate on currently-weak phenotypes (Glucose, Cellobiose,
  Mannose); strong ones barely move. Extends Fig 7's "how many samples" to "which
  phenotypes."
- **Connects to message:** completes the prioritization wrap-up at the phenotype level.

**Throughline:** model confidence is the spine — it ranks predictions (A), is shown to be
overconfident (B), yet still drives the best experiment selection (C), targeted at the
phenotypes that need it most (D).

## Retired

- `compute_phenotype_diagnostic` and the shortcut-gap vs concordance-benefit scatter
  (artifactual).
- The GapMind–ML agreement panel and the standalone GapMind/ML baseline lines
  (circular for the concordant model).
- The per-phenotype single-model class-stratified Panel A built earlier (subsumed by the
  new two-model small-multiples A).
- The GapMind-disagreement-guided active-learning selection (replaced by label-free
  confidence/OOD/diversity strategies).

## Code changes (high level; detailed in implementation plan)

- `scripts/figure8/figure8_data.py`: add a permanent full-data per-sample generation step
  (parallel to the concordant one); keep the concordant per-sample and the class-stratified
  risk-coverage builder (now run per model, for both models, for the three Panel-A
  phenotypes); drop `compute_phenotype_diagnostic` and `build_agreement_table` entirely
  (neither is used by the new figure).
- `scripts/figure8/figure8d_active_learning.py`: replace GapMind-disagreement selection
  with {low_confidence, high_ood, diversity, random}; output per-strategy BA-gain curves
  and per-phenotype gains.
- `scripts/figure8/figure8_plot.py`: rewrite to render A (small multiples, both models),
  B (calibration, both models), C (prioritization curves), D (per-phenotype prioritization).
- Add a calibration/ECE helper and an OOD-distance helper (kNN Jaccard to per-split
  training genomes) for the simulation.

## Manuscript text changes (high level)

- Results subsection "Model confidence defines an applicability domain…": rewrite around
  reliability + its limits + experiment prioritization; state the negative result (no
  label-free per-phenotype confidence signal) honestly; remove shortcut-gap and
  GapMind-agreement claims.
- Figure 8 caption: rewrite for the four new panels; descriptive only (interpretation in
  Results), matching the Fig 6/7 caption style.
- Methods "Applicability-Domain and Shortcut-Gap Analyses": replace shortcut-gap and
  GapMind-agreement methods with calibration/ECE, class-stratified risk-coverage per
  model, OOD distance, and the label-free prioritization simulation.
- Discussion: update the practical-takeaway thread (confidence certifies predictions but
  is overconfident; low-confidence genomes and weak phenotypes are the priority
  experiments) and drop the GapMind-agreement and shortcut-gap sentences.

## Honest limitations to state in the paper

- Per-genome confidence is only a modest discriminator (AUROC ≈ 0.65) and the model is
  overconfident; probabilities are usable for ranking, not as calibrated probabilities.
- No label-free signal predicts per-phenotype reliability; phenotype difficulty reflects
  the genotype→phenotype mapping, not genome novelty.
- Uncertainty-based acquisition is theoretically fragile under distribution shift; here it
  empirically wins, but the gain is modest (+0.027 BA) and should be presented as such.

## Connection to Figures 5–7

- Fig 5: concordant training yields a bounded specialist that recovers mechanism.
- Fig 6: label-free filters/feature restriction partially recover the concordance gain.
- Fig 7: ~50–100 coherent samples saturate performance; remaining limits are
  representational, not statistical ("how many samples").
- Fig 8 (this redesign): given the deployed specialist, how reliable are its predictions,
  what are the limits, and which genomes/phenotypes to measure next ("which samples and
  which phenotypes") — the practitioner-facing capstone.

## Open questions / risks

- C/D depend on a prioritization simulation whose pilot result (+0.027 BA, p = 0.004)
  must be reproduced by the formal pipeline; if it does not replicate at full scale, C's
  message weakens and we revisit.
- Panel D's "gain per added sample" must be computed consistently with C's simulation
  (same strategy, same seeds) to avoid a mismatch between panels.
- Whether to show both models in C/D (currently single-model) — flagged; default is
  concordant-only.
