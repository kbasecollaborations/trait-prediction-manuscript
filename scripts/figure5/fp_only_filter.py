#!/usr/bin/env python3
"""Compare false-positive-only GapMind train-set filtering with symmetric filtering.

Three leave-one-dataset-out train-set filter modes per phenotype:

1. ``no_filter``  : all training samples.
2. ``concordant`` : keep only GapMind-concordant samples (drop FP and FN).
3. ``fp_only``    : drop only false positives (GapMind=1 & experiment=0),
   keeping concordant samples plus false negatives.

The held-out test set is left full and unfiltered in every mode, so all modes
are scored on identical evaluation data. Each mode is run with and without
CatBoost balanced class weights. Backs the false-negative-retention numbers in
Supplementary Text S10.

Writes data/outputs/figure5_fp_only/fp_only_comparison.csv.

Run with::

    uv run python -m scripts.figure5.fp_only_filter [options]
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml_splits import load_split_data, perform_split_ml

# Thread cap applied to every CatBoost fit.
THREADS: int = int(os.environ.get("EXPERIMENT_THREADS", "4"))

SPLITS_DIR: Path = Path("data/processed/train_test_splits")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
DEFAULT_OUT: Path = Path("data/outputs/figure5_fp_only/fp_only_comparison.csv")

RANDOM_STATE: int = 42
MIN_TRAIN_SAMPLES: int = 5
MIN_TEST_SAMPLES: int = 5

# Filter modes, in reporting order.
FilterMode = Literal["no_filter", "concordant", "fp_only"]
FILTER_MODES: tuple[FilterMode, ...] = ("no_filter", "concordant", "fp_only")

# Class-weight settings: label -> CatBoost auto_class_weights value (None = unset).
ClassWeight = Literal["none", "balanced"]
CLASS_WEIGHT_VALUES: dict[ClassWeight, str | None] = {
    "none": None,
    "balanced": "Balanced",
}

# Subset of the shared scorer's default metrics.
SCORING: list[str] = [
    "balanced_accuracy",
    "recall",
    "precision",
    "specificity",
]

HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")


def parse_held_out_dataset(key: str) -> str | None:
    """Extract the held-out dataset name from a ``dataset_split`` key.

    Parameters
    ----------
    key : str
        Split key of the form ``"<phenotype>_train(...),test(<dataset>)"``.

    Returns
    -------
    str | None
        Held-out dataset name (e.g. ``"pmi"``), or ``None`` if the key does not
        match the expected pattern.
    """
    match = HELD_OUT_RE.search(key)
    if match is None:
        return None
    return match.group(1)


def build_filter_masks(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> dict[str, set[str]]:
    """Partition genomes into concordant, false-positive and false-negative sets.

    A genome is counted only when it has both a non-NaN GapMind call and a
    non-NaN experimental label for ``phenotype``. False positive means
    GapMind=1 while experiment=0; false negative means GapMind=0 while
    experiment=1.

    Parameters
    ----------
    gapmind_predictions : pd.DataFrame
        Loose-mode GapMind predictions (genomes x phenotypes, 0/1).
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype table (genomes x phenotypes, 0/1).
    phenotype : str
        Phenotype column name.

    Returns
    -------
    dict[str, set[str]]
        Keys ``"concordant"``, ``"false_positive"``, ``"false_negative"`` and
        ``"keep_fp_only"`` mapping to sets of genome IDs. All sets are empty
        when the phenotype is absent from either table.
    """
    empty: dict[str, set[str]] = {
        "concordant": set(),
        "false_positive": set(),
        "false_negative": set(),
        "keep_fp_only": set(),
    }
    if phenotype not in gapmind_predictions.columns:
        return empty
    if phenotype not in experimental_phenotypes.columns:
        return empty

    common = gapmind_predictions.index.intersection(experimental_phenotypes.index)
    exp = experimental_phenotypes.loc[common, phenotype]
    valid = exp.dropna().index

    gm_vals = gapmind_predictions.loc[valid, phenotype]
    exp_vals = experimental_phenotypes.loc[valid, phenotype]
    valid = gm_vals.dropna().index
    gm_vals = gm_vals.loc[valid]
    exp_vals = exp_vals.loc[valid]

    concordant_mask = gm_vals == exp_vals
    fp_mask = (gm_vals == 1) & (exp_vals == 0)
    fn_mask = (gm_vals == 0) & (exp_vals == 1)

    concordant = set(valid[concordant_mask])
    false_positive = set(valid[fp_mask])
    false_negative = set(valid[fn_mask])

    return {
        "concordant": concordant,
        "false_positive": false_positive,
        "false_negative": false_negative,
        # Keep everything except false positives. Genomes with no GapMind or
        # experimental call are kept too, matching no_filter behaviour for
        # samples GapMind cannot speak to.
        "keep_fp_only": false_positive,
    }


def filter_train_val(
    split: Mapping[str, pd.DataFrame | pd.Series],
    mode: FilterMode,
    masks: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    """Apply a train-set filter mode to the train and validation sets.

    The held-out test set is not touched; the caller scores on the full
    unfiltered test set. Train and validation sets receive the same filter so
    early stopping sees the same population as training.

    Parameters
    ----------
    split : Mapping[str, pd.DataFrame | pd.Series]
        One split dict with keys ``X_train``, ``y_train``, ``X_val``,
        ``y_val``, ``X_test``, ``y_test``.
    mode : FilterMode
        One of ``"no_filter"``, ``"concordant"`` or ``"fp_only"``.
    masks : Mapping[str, set[str]]
        Output of :func:`build_filter_masks` for the phenotype.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None
        ``(X_train, y_train, X_val, y_val)`` after filtering, or ``None`` if
        the filtered train or validation set is too small or single-class.

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised filter mode.
    """
    x_train: pd.DataFrame = split["X_train"]  # type: ignore[assignment]
    y_train: pd.Series = split["y_train"]  # type: ignore[assignment]
    x_val: pd.DataFrame = split["X_val"]  # type: ignore[assignment]
    y_val: pd.Series = split["y_val"]  # type: ignore[assignment]

    if mode == "no_filter":
        train_idx = list(x_train.index)
        val_idx = list(x_val.index)
    elif mode == "concordant":
        concordant = masks["concordant"]
        train_idx = sorted(set(x_train.index) & concordant)
        val_idx = sorted(set(x_val.index) & concordant)
    elif mode == "fp_only":
        false_positive = masks["false_positive"]
        train_idx = sorted(set(x_train.index) - false_positive)
        val_idx = sorted(set(x_val.index) - false_positive)
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"Unknown filter mode: {mode}")

    if len(train_idx) < MIN_TRAIN_SAMPLES or len(val_idx) < MIN_TRAIN_SAMPLES:
        return None

    x_train_f = x_train.loc[train_idx]
    y_train_f = y_train.loc[train_idx]
    x_val_f = x_val.loc[val_idx]
    y_val_f = y_val.loc[val_idx]

    if y_train_f.nunique() != 2 or y_val_f.nunique() != 2:
        return None

    return x_train_f, y_train_f, x_val_f, y_val_f


def evaluate_combination(
    split: Mapping[str, pd.DataFrame | pd.Series],
    mode: FilterMode,
    masks: Mapping[str, set[str]],
    class_weight: ClassWeight,
    threads: int,
) -> dict[str, float | int | str] | None:
    """Train and score one (mode, class-weight) combination on a split.

    Parameters
    ----------
    split : Mapping[str, pd.DataFrame | pd.Series]
        One split dict (train/val/test).
    mode : FilterMode
        Train-set filter mode.
    masks : Mapping[str, set[str]]
        Output of :func:`build_filter_masks`.
    class_weight : ClassWeight
        ``"none"`` (CatBoost default) or ``"balanced"``
        (``auto_class_weights="Balanced"``).
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    dict[str, float | int | str] | None
        Metric record (balanced_accuracy, recall, precision, specificity,
        n_train, n_test), or ``None`` if the split could not be evaluated.
    """
    filtered = filter_train_val(split, mode, masks)
    if filtered is None:
        return None
    x_train, y_train, x_val, y_val = filtered

    x_test: pd.DataFrame = split["X_test"]  # type: ignore[assignment]
    y_test: pd.Series = split["y_test"]  # type: ignore[assignment]
    if len(x_test) < MIN_TEST_SAMPLES or y_test.nunique() != 2:
        return None

    model_kwargs: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "thread_count": threads,
    }
    auto_weights = CLASS_WEIGHT_VALUES[class_weight]
    if auto_weights is not None:
        model_kwargs["auto_class_weights"] = auto_weights

    scores = perform_split_ml(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        model_type="cb",
        scoring=SCORING,
        **model_kwargs,
    )

    record: dict[str, float | int | str] = {
        "balanced_accuracy": float(scores["balanced_accuracy"]),
        "recall": float(scores["recall"]),
        "precision": float(scores["precision"]),
        "specificity": float(scores["specificity"]),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    return record


def run_sweep(
    split_data: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotypes: Sequence[str] | None,
    datasets: Sequence[str] | None,
    class_weights: Sequence[ClassWeight],
    threads: int,
) -> pd.DataFrame:
    """Run the full (phenotype x held-out dataset x mode x class-weight) sweep.

    Parameters
    ----------
    split_data : Mapping
        Nested ``dataset_split`` dictionary from :func:`load_split_data`.
    gapmind_predictions : pd.DataFrame
        Loose-mode GapMind predictions.
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype table.
    phenotypes : Sequence[str] | None
        Restrict to these phenotypes; ``None`` runs all available.
    datasets : Sequence[str] | None
        Restrict to these held-out datasets; ``None`` runs all available.
    class_weights : Sequence[ClassWeight]
        Class-weight settings to evaluate.
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    pd.DataFrame
        Tidy long-format results, one row per evaluated combination.
    """
    dataset_splits = split_data["dataset_split"]
    pheno_set = set(phenotypes) if phenotypes is not None else None
    dataset_set = set(datasets) if datasets is not None else None

    # Cache masks per phenotype (they do not depend on the held-out dataset).
    mask_cache: dict[str, dict[str, set[str]]] = {}
    records: list[dict[str, float | int | str]] = []

    keys = sorted(dataset_splits)
    for key in keys:
        phenotype = key.split("_", 1)[0]
        if pheno_set is not None and phenotype not in pheno_set:
            continue
        held_out = parse_held_out_dataset(key)
        if held_out is None:
            continue
        if dataset_set is not None and held_out not in dataset_set:
            continue

        if phenotype not in mask_cache:
            mask_cache[phenotype] = build_filter_masks(
                gapmind_predictions, experimental_phenotypes, phenotype
            )
        masks = mask_cache[phenotype]

        split = dataset_splits[key]
        for mode in FILTER_MODES:
            for class_weight in class_weights:
                print(
                    f"  {phenotype} | held_out={held_out} | mode={mode} | "
                    f"class_weight={class_weight}",
                    flush=True,
                )
                record = evaluate_combination(split, mode, masks, class_weight, threads)
                if record is None:
                    print(
                        "    skipped (insufficient or single-class samples)", flush=True
                    )
                    continue
                record.update(
                    {
                        "phenotype": phenotype,
                        "held_out_dataset": held_out,
                        "mode": mode,
                        "class_weight": class_weight,
                    }
                )
                records.append(record)

    columns = [
        "phenotype",
        "held_out_dataset",
        "mode",
        "class_weight",
        "balanced_accuracy",
        "recall",
        "precision",
        "specificity",
        "n_train",
        "n_test",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


def print_pivot_summary(results: pd.DataFrame) -> None:
    """Print mean balanced accuracy and recall by mode x class-weight.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`run_sweep`.
    """
    if results.empty:
        print("\nNo results to summarize.")
        return

    print("\nMean balanced_accuracy by mode x class_weight:")
    ba_pivot = results.pivot_table(
        index="mode",
        columns="class_weight",
        values="balanced_accuracy",
        aggfunc="mean",
    ).reindex(list(FILTER_MODES))
    print(ba_pivot.round(3))

    print("\nMean recall by mode x class_weight:")
    recall_pivot = results.pivot_table(
        index="mode",
        columns="class_weight",
        values="recall",
        aggfunc="mean",
    ).reindex(list(FILTER_MODES))
    print(recall_pivot.round(3))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``threads``, ``class_weights``, ``phenotypes``,
        ``datasets``, ``out``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help="CatBoost thread_count (default: EXPERIMENT_THREADS env or 4).",
    )
    parser.add_argument(
        "--class-weights",
        choices=["none", "balanced", "both"],
        default="both",
        help="Which class-weight settings to evaluate (default: both).",
    )
    parser.add_argument(
        "--phenotypes",
        type=str,
        default=None,
        help="Optional comma-separated list of phenotypes (default: all).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated list of held-out datasets (default: all).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, run the FP-only filter sweep, and write the tidy CSV."""
    args = parse_args()

    class_weights: list[ClassWeight]
    if args.class_weights == "both":
        class_weights = ["none", "balanced"]
    else:
        class_weights = [args.class_weights]  # type: ignore[list-item]

    phenotypes = (
        [p.strip() for p in args.phenotypes.split(",") if p.strip()]
        if args.phenotypes
        else None
    )
    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets
        else None
    )

    print(f"Thread cap (CatBoost thread_count): {args.threads}")
    print(f"Class-weight settings: {class_weights}")
    print(f"Phenotypes: {phenotypes if phenotypes else 'all'}")
    print(f"Held-out datasets: {datasets if datasets else 'all'}")

    print("\nLoading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(
        f"  Loaded {len(gapmind_predictions)} genomes, "
        f"{len(gapmind_predictions.columns)} phenotypes"
    )

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(
        f"  Loaded {len(experimental_phenotypes)} genomes, "
        f"{len(experimental_phenotypes.columns)} phenotypes"
    )

    print("\nLoading dataset_split train-test splits (KOFAM features)...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=["dataset_split"],
        feature_file=KOFAM_FEATURE_FILE,
    )
    print(f"  Loaded {len(split_data['dataset_split'])} dataset splits")

    print("\nRunning sweep...")
    results = run_sweep(
        split_data=split_data,
        gapmind_predictions=gapmind_predictions,
        experimental_phenotypes=experimental_phenotypes,
        phenotypes=phenotypes,
        datasets=datasets,
        class_weights=class_weights,
        threads=args.threads,
    )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows to {out_path}")

    print_pivot_summary(results)


if __name__ == "__main__":
    main()
