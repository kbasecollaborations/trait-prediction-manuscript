#!/usr/bin/env python3
"""Per-genome predictions for the mechanism-free confidence filter (Figure 6C).

Tree-placed training and validation samples receive weights equal to the
confidence assigned to their experimental label by phylogenetic k-NN agreement
and the experimental label alone (``w_gapmind = 0``). Model, random state,
column alignment, early stopping and held-out test set match the concordant arm.

Writes ``data/outputs/figure6/figure6c_mechfree_per_sample.tsv``.

Run with ``uv run python -m scripts.figure6.figure6c_mechfree_per_sample``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import Pool
from tqdm import tqdm

from scripts.figure5.figure5cd_data import load_gapmind_predictions
from scripts.figure6.figure6b_data import load_phylogenetic_data
from scripts.figure6.figure6b_weight_sweep import (
    DEFAULT_WEIGHTING_MODE,
    PHASE2_CONFIGS,
    WeightingMode,
    compute_split_sample_weights,
)
from scripts.figure7.figure7_data import (
    GAPMIND_FILE,
    KOFAM_FEATURE_FILE,
    SPLITS_DIR,
    parse_held_out_dataset,
)
from scripts.ml import make_classifier
from scripts.ml_splits import align_columns, load_split_data

OUTPUT_FILE: Path = Path("data/outputs/figure6/figure6c_mechfree_per_sample.tsv")

MECHFREE_CONFIG_NAME: str = "free_balanced"
"""The $w_{gap} = 0$ arm of the Figure 6B sweep, i.e. the mechanism-free filter."""


def mechfree_sample_weights(
    split: dict[str, Any],
    phenotype: str,
    distance_df: pd.DataFrame,
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> dict[str, pd.Series]:
    """Return split-local mechanism-free train and validation weights.

    Parameters
    ----------
    split : dict[str, Any]
        One train/validation/test split.
    phenotype : str
        Phenotype represented by the split.
    distance_df : pd.DataFrame
        Precomputed phylogenetic distance matrix.
    weighting_mode : {"label_confidence", "boundary_certainty"}, optional
        Mapping from composite probability to sample weight.

    Returns
    -------
    dict[str, pd.Series]
        Per-row weights for tree-placed train and validation genomes.

    Raises
    ------
    ValueError
        If the mechanism-free configuration is absent from the sweep configs, or
        if it is not in fact mechanism-free.
    """
    config = next((c for c in PHASE2_CONFIGS if c.name == MECHFREE_CONFIG_NAME), None)
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

    return compute_split_sample_weights(
        config,
        split,
        phenotype,
        None,
        distance_df,
        weighting_mode,
    )


def fit_mechfree_model_and_predict_proba(
    split: dict[str, Any],
    weights: dict[str, pd.Series],
) -> pd.DataFrame | None:
    """Fit one weighted mechanism-free model and predict its full test set.

    Parameters
    ----------
    split : dict[str, Any]
        One train/validation/test split.
    weights : dict[str, pd.Series]
        Train and validation weights indexed by eligible genome ID.

    Returns
    -------
    pd.DataFrame | None
        Test probabilities and binary predictions, or ``None`` if a weighted
        fitting subset lacks both classes.
    """
    train_idx = weights["train"].index
    val_idx = weights["val"].index
    y_train = split["y_train"].loc[train_idx]
    y_val = split["y_val"].loc[val_idx]
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        return None

    X_train = split["X_train"].loc[train_idx]
    X_val = align_columns(X_train, split["X_val"].loc[val_idx])
    X_test = align_columns(X_train, split["X_test"])
    model = make_classifier("cb", random_state=42)
    model.fit(
        Pool(X_train, y_train, weight=weights["train"].loc[train_idx]),
        eval_set=Pool(X_val, y_val, weight=weights["val"].loc[val_idx]),
        use_best_model=True,
        verbose=False,
    )
    proba = np.asarray(model.predict_proba(X_test))[:, 1]
    return pd.DataFrame(
        {"proba": proba, "y_pred": (proba >= 0.5).astype(int)},
        index=X_test.index,
    )


def collect_per_sample_predictions(
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> pd.DataFrame:
    """Fit the mechanism-free model per cross-dataset split and pool predictions.

    Returns
    -------
    pd.DataFrame
        One row per held-out test genome with columns ``phenotype``,
        ``held_out_dataset``, ``genome``, ``y_true``, ``y_pred``, ``proba`` and
        ``gapmind_pred``, matching ``figure7_per_sample.tsv``.
    """
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    _tree, distance_df = load_phylogenetic_data()

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
        weights = mechfree_sample_weights(
            split, phenotype, distance_df, weighting_mode
        )
        if any(weight.empty for weight in weights.values()):
            continue

        predictions = fit_mechfree_model_and_predict_proba(split, weights)
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


def main(
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> None:
    """Write the mechanism-free per-sample prediction table."""
    output_file = (
        OUTPUT_FILE
        if weighting_mode == DEFAULT_WEIGHTING_MODE
        else OUTPUT_FILE.with_name(
            f"{OUTPUT_FILE.stem}_{weighting_mode}{OUTPUT_FILE.suffix}"
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = collect_per_sample_predictions(weighting_mode)
    table.to_csv(output_file, sep="\t", index=False)
    print(f"\nWrote {len(table)} rows to {output_file}")
    print(
        table.groupby("phenotype")
        .agg(n=("genome", "size"), n_pos=("y_true", "sum"))
        .to_string()
    )


if __name__ == "__main__":
    main(
        cast(
            WeightingMode,
            os.environ.get("FIGURE6_WEIGHTING_MODE", DEFAULT_WEIGHTING_MODE),
        )
    )
