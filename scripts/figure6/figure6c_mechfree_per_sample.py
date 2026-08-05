#!/usr/bin/env python3
"""Per-genome predictions for the mechanism-free confidence filter (Figure 6C).

The concordant arm of Figure 6C is testable per phenotype because
``scripts/figure7/figure7_data.py`` emits per-genome predictions for it. The
mechanism-free arm had no equivalent, so its bars could not carry a significance
marker. This script produces the matching table.

Only the training-set filter differs from the concordant arm: samples are kept
when the soft label built from phylogenetic k-NN agreement and the experimental
label alone (``w_gapmind = 0``) falls outside the ambiguous band. The model,
random state, column alignment, early stopping and held-out test set are the
concordant arm's, reused directly from ``figure7_data``, so the two series are
compared on identical terms.

Run with ``uv run python -m scripts.figure6.figure6c_mechfree_per_sample``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.figure6.figure6b_parameter_exploration import build_inputs
from scripts.figure6.figure6b_weight_sweep import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    PHASE2_CONFIGS,
    compute_y_soft,
)
from scripts.figure7.figure7_data import (
    GAPMIND_FILE,
    KOFAM_FEATURE_FILE,
    PHENOTYPE_DIR,
    SPLITS_DIR,
    fit_concordant_model_and_predict_proba,
    parse_held_out_dataset,
)
from scripts.ml_splits import load_split_data

OUTPUT_FILE: Path = Path("data/outputs/figure6/figure6c_mechfree_per_sample.tsv")

MECHFREE_CONFIG_NAME: str = "free_balanced"
"""The $w_{gap} = 0$ arm of the Figure 6B sweep, i.e. the mechanism-free filter."""


def mechfree_retained_genomes() -> dict[str, set[str]]:
    """Genome IDs retained by the mechanism-free confidence filter, per phenotype.

    Returns
    -------
    dict[str, set[str]]
        Mapping phenotype name to the set of genomes whose soft label lies
        outside the ambiguous band and are therefore eligible for training.

    Raises
    ------
    ValueError
        If the mechanism-free configuration is absent from the sweep configs, or
        if it is not in fact mechanism-free.
    """
    config = next(
        (c for c in PHASE2_CONFIGS if c.name == MECHFREE_CONFIG_NAME), None
    )
    if config is None:
        raise ValueError(
            f"config {MECHFREE_CONFIG_NAME!r} not in PHASE2_CONFIGS; "
            "the mechanism-free arm of Figure 6B has been renamed"
        )
    if config.w_gapmind != 0.0:
        raise ValueError(
            f"config {config.name!r} has w_gapmind={config.w_gapmind}; "
            "it is not mechanism-free"
        )

    y_soft = compute_y_soft(config, build_inputs())
    retained: dict[str, set[str]] = {}
    for phenotype, soft in y_soft.items():
        confident = (soft < CONFIDENCE_THRESHOLD_LOW) | (
            soft > CONFIDENCE_THRESHOLD_HIGH
        )
        retained[phenotype] = set(soft.index[confident])
    return retained


def collect_per_sample_predictions() -> pd.DataFrame:
    """Fit the mechanism-free model per cross-dataset split and pool predictions.

    Returns
    -------
    pd.DataFrame
        One row per held-out test genome with columns ``phenotype``,
        ``held_out_dataset``, ``genome``, ``y_true``, ``y_pred``, ``proba`` and
        ``gapmind_pred``, matching ``figure7_per_sample.tsv``.
    """
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    retained = mechfree_retained_genomes()

    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=["dataset_split"],
        feature_file=KOFAM_FEATURE_FILE,
    )
    splits = split_data["dataset_split"]

    records: list[dict[str, object]] = []
    for key, split in tqdm(splits.items(), total=len(splits), desc="mech-free splits"):
        held_out = parse_held_out_dataset(key)
        if held_out is None:
            continue
        phenotype = key.split("_", 1)[0]
        eligible = retained.get(phenotype)
        if not eligible:
            continue

        predictions = fit_concordant_model_and_predict_proba(split, eligible)
        if predictions is None:
            continue

        y_test = split["y_test"]
        gm_col = (
            gapmind_predictions[phenotype]
            if phenotype in gapmind_predictions.columns
            else None
        )
        for genome in predictions.index:
            y_true = y_test.loc[genome]
            if pd.isna(y_true):
                continue
            gm_pred: float | int = np.nan
            if gm_col is not None and genome in gm_col.index:
                gm_val = gm_col.loc[genome]
                if not pd.isna(gm_val):
                    gm_pred = int(gm_val)
            records.append(
                {
                    "phenotype": phenotype,
                    "held_out_dataset": held_out,
                    "genome": genome,
                    "y_true": int(y_true),
                    "y_pred": int(predictions.loc[genome, "y_pred"]),
                    "proba": float(predictions.loc[genome, "proba"]),
                    "gapmind_pred": gm_pred,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    """Write the mechanism-free per-sample prediction table."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = collect_per_sample_predictions()
    table.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nWrote {len(table)} rows to {OUTPUT_FILE}")
    print(
        table.groupby("phenotype")
        .agg(n=("genome", "size"), n_pos=("y_true", "sum"))
        .to_string()
    )


if __name__ == "__main__":
    main()
