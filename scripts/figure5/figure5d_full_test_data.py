#!/usr/bin/env python3
"""
Generate Figure 5D data: concordant-trained model performance on the **full**
held-out cross-dataset test set, decomposed by GapMind discordance categories.

This script addresses Reviewer 2's request to evaluate concordant-trained models
on the full held-out test set rather than only on the concordant subset (Fig 5A)
or only on the discordant subset (Fig 5C). For each phenotype × held-out dataset
combination, it reports:

    - Total balanced accuracy on the full test set (from the existing
      ``figure5c_concordant_train_different_test.csv`` artifact).
    - Counts of test samples in each GapMind discordance category
      (concordant, FP-discordant, FN-discordant).
    - Per-subset balanced accuracies obtained by re-fitting the same
      concordant-trained CatBoost model (deterministic with random_state=42 and
      sorted index ordering) and partitioning predictions by category.

Outputs ``data/outputs/figure5/figure5d_full_test.tsv``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    get_concordant_and_discordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml import make_classifier
from scripts.ml_splits import load_split_data
from trait_prediction.pipeline import align_columns


SPLITS_DIR: Path = Path("data/processed/train_test_splits")
OUTPUT_DIR: Path = Path("data/outputs/figure5")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
EXISTING_5C_FILE: Path = OUTPUT_DIR / "figure5c_concordant_train_different_test.csv"
S8_COUNTS_FILE: Path = Path("data/outputs/figureS8/concordance_counts.tsv")

RANDOM_STATE: int = 42
MIN_TRAIN_SAMPLES: int = 5
MIN_TEST_SAMPLES: int = 5

HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")


def parse_held_out_dataset(key: str) -> str | None:
    """
    Extract the held-out dataset name from a dataset_split key.

    Parameters
    ----------
    key : str
        Split key of the form ``"<phenotype>_train(...),test(<dataset>)"``.

    Returns
    -------
    str | None
        Held-out dataset name (e.g. ``"marine"``) or ``None`` if the key does
        not match the expected pattern.
    """
    match = HELD_OUT_RE.search(key)
    if match is None:
        return None
    return match.group(1)


def categorize_test_samples(
    test_index: pd.Index,
    test_labels: pd.Series,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> dict[str, list[str]]:
    """
    Partition test samples into concordant, FP-discordant, FN-discordant subsets.

    "Concordant" means the GapMind prediction matches the experimental
    phenotype. "FP-discordant" means GapMind=1 but experimental=0 (false
    positive of GapMind). "FN-discordant" means GapMind=0 but experimental=1
    (false negative of GapMind). Samples without a GapMind call for the
    phenotype are dropped.

    Parameters
    ----------
    test_index : pd.Index
        Genome IDs in the held-out test set.
    test_labels : pd.Series
        Experimental phenotype labels for the test samples (indexed by
        genome ID, values 0/1).
    gapmind_predictions : pd.DataFrame
        Loose-mode GapMind predictions (genomes × phenotypes, 0/1).
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype table (genomes × phenotypes, 0/1).
    phenotype : str
        Phenotype column name.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys ``"concordant"``, ``"fp_discordant"``,
        ``"fn_discordant"``, each mapping to a sorted list of genome IDs.
    """
    if phenotype not in gapmind_predictions.columns:
        return {"concordant": [], "fp_discordant": [], "fn_discordant": []}

    concordant: list[str] = []
    fp_discordant: list[str] = []
    fn_discordant: list[str] = []

    gm_col = gapmind_predictions[phenotype]
    for genome in test_index:
        if genome not in gm_col.index:
            continue
        gm_val = gm_col.loc[genome]
        if pd.isna(gm_val):
            continue
        exp_val = test_labels.loc[genome]
        if pd.isna(exp_val):
            continue
        gm_int = int(gm_val)
        exp_int = int(exp_val)
        if gm_int == exp_int:
            concordant.append(genome)
        elif gm_int == 1 and exp_int == 0:
            fp_discordant.append(genome)
        elif gm_int == 0 and exp_int == 1:
            fn_discordant.append(genome)

    return {
        "concordant": sorted(concordant),
        "fp_discordant": sorted(fp_discordant),
        "fn_discordant": sorted(fn_discordant),
    }


def fit_concordant_model_and_predict(
    split: Mapping[str, pd.DataFrame | pd.Series],
    concordant_genomes: set[str],
) -> pd.Series | None:
    """
    Fit a CatBoost classifier on concordant train/val and predict the full test.

    The implementation mirrors ``figure5cd_data.run_ml_on_concordant_train_with_different_test_sets``
    but with sorted index ordering to make CatBoost training deterministic
    across runs (``set`` iteration order otherwise leaks into model fitting).

    Parameters
    ----------
    split : Mapping[str, pd.DataFrame | pd.Series]
        Output of ``load_single_split_data`` for one split: keys
        ``X_train``, ``y_train``, ``X_val``, ``y_val``, ``X_test``, ``y_test``.
    concordant_genomes : set[str]
        Set of genome IDs that are GapMind-concordant for the phenotype.

    Returns
    -------
    pd.Series | None
        Predicted 0/1 labels indexed by full-test genome ID, or ``None`` if
        the concordant training subset is too small or class-imbalanced.
    """
    train_idx = sorted(set(split["X_train"].index) & concordant_genomes)
    val_idx = sorted(set(split["X_val"].index) & concordant_genomes)

    if len(train_idx) < MIN_TRAIN_SAMPLES or len(val_idx) < MIN_TRAIN_SAMPLES:
        return None

    X_train = split["X_train"].loc[train_idx]
    y_train = split["y_train"].loc[train_idx]
    X_val = split["X_val"].loc[val_idx]
    y_val = split["y_val"].loc[val_idx]

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        return None

    X_test = split["X_test"]
    if len(X_test) == 0:
        return None

    model = make_classifier("cb", random_state=RANDOM_STATE)
    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val_aligned, y_val),
        use_best_model=True,
        verbose=False,
    )

    preds = model.predict(X_test_aligned)
    if hasattr(preds, "ravel"):
        preds = preds.ravel()
    return pd.Series(preds.astype(int), index=X_test.index, name="prediction")


def safe_balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute balanced accuracy, falling back to plain accuracy when needed.

    GapMind FP-discordant test subsets contain only ``y_true == 0`` samples and
    FN-discordant subsets contain only ``y_true == 1`` samples by definition.
    Balanced accuracy is undefined on a single-class set, so for these subsets
    the function returns the per-class recall (which equals accuracy on the
    single-class subset). When both classes are present the standard scikit-learn
    balanced accuracy is returned.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels.
    y_pred : pd.Series
        Predicted labels (must align on index with ``y_true``).

    Returns
    -------
    float
        Balanced accuracy when ``y_true`` contains both classes; per-class
        recall (= accuracy) when ``y_true`` contains a single class; ``np.nan``
        when ``y_true`` is empty.
    """
    if len(y_true) == 0:
        return float("nan")
    if y_true.nunique() < 2:
        return float((y_true.values == y_pred.values).mean())
    return float(balanced_accuracy_score(y_true, y_pred))


def build_full_test_table(
    existing_5c: pd.DataFrame,
    split_data: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the per-(phenotype, held-out dataset) Figure 5D table.

    For each ``dataset_split`` × ``test_type=="full"`` row in the existing
    Figure 5C/D output, the function reuses ``balanced_accuracy`` and
    ``n_test`` directly. Discordance counts are computed from the GapMind and
    experimental loaders. Per-subset balanced accuracies are computed by
    re-fitting the concordant-trained model deterministically and partitioning
    predictions by discordance category.

    Parameters
    ----------
    existing_5c : pd.DataFrame
        Contents of ``figure5c_concordant_train_different_test.csv``.
    split_data : Mapping
        Nested split dictionary returned by ``load_split_data``.
    gapmind_predictions : pd.DataFrame
        Loose-mode GapMind predictions table.
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype table.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per (phenotype, held-out dataset).
    """
    full_rows = existing_5c[
        (existing_5c["split_type"] == "dataset_split")
        & (existing_5c["test_type"] == "full")
    ].copy()

    records: list[dict[str, object]] = []

    iterator = tqdm(
        full_rows.itertuples(index=False),
        total=len(full_rows),
        desc="Figure 5D rows",
    )
    for row in iterator:
        key = row.key
        phenotype = row.phenotype
        held_out = parse_held_out_dataset(key)
        if held_out is None:
            continue

        split = split_data["dataset_split"].get(key)
        if split is None:
            continue

        concordant_genomes, _ = get_concordant_and_discordant_samples(
            gapmind_predictions, experimental_phenotypes, phenotype
        )

        y_test = split["y_test"]
        categories = categorize_test_samples(
            test_index=split["X_test"].index,
            test_labels=y_test,
            gapmind_predictions=gapmind_predictions,
            experimental_phenotypes=experimental_phenotypes,
            phenotype=phenotype,
        )

        n_test_full = int(row.n_test)
        n_concordant = len(categories["concordant"])
        n_fp = len(categories["fp_discordant"])
        n_fn = len(categories["fn_discordant"])

        record: dict[str, object] = {
            "phenotype": phenotype,
            "held_out_dataset": held_out,
            "n_test_full": n_test_full,
            "n_test_concordant": n_concordant,
            "n_test_FP_discordant": n_fp,
            "n_test_FN_discordant": n_fn,
            "balanced_accuracy_full": float(row.balanced_accuracy),
            "balanced_accuracy_concordant_subset": float("nan"),
            "balanced_accuracy_FP_subset": float("nan"),
            "balanced_accuracy_FN_subset": float("nan"),
        }

        predictions = fit_concordant_model_and_predict(split, concordant_genomes)
        if predictions is not None:
            for subset_name, subset_key in [
                ("balanced_accuracy_concordant_subset", "concordant"),
                ("balanced_accuracy_FP_subset", "fp_discordant"),
                ("balanced_accuracy_FN_subset", "fn_discordant"),
            ]:
                subset_ids = [g for g in categories[subset_key] if g in predictions.index]
                if len(subset_ids) >= MIN_TEST_SAMPLES:
                    record[subset_name] = safe_balanced_accuracy(
                        y_test.loc[subset_ids], predictions.loc[subset_ids]
                    )

        records.append(record)

    return pd.DataFrame.from_records(records)


def main() -> None:
    """
    Build and persist the Figure 5D data table.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EXISTING_5C_FILE.exists():
        raise FileNotFoundError(
            f"Required input not found: {EXISTING_5C_FILE}. "
            "Run scripts.figure5.figure5cd_data first."
        )

    print(f"Reading existing 5C/D output from {EXISTING_5C_FILE}")
    existing_5c = pd.read_csv(EXISTING_5C_FILE)

    if S8_COUNTS_FILE.exists():
        print(
            f"Note: {S8_COUNTS_FILE} exists but Figure 5D recomputes "
            "discordance categories on the fly for self-consistency."
        )
    else:
        print(
            f"Wave-1C counts file not found at {S8_COUNTS_FILE}; "
            "recomputing discordance categories from raw loaders."
        )

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(
        f"  Loaded {len(gapmind_predictions)} genomes, "
        f"{len(gapmind_predictions.columns)} phenotypes"
    )

    print("Loading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(
        f"  Loaded {len(experimental_phenotypes)} genomes, "
        f"{len(experimental_phenotypes.columns)} phenotypes"
    )

    print("Loading dataset_split train-test splits...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=["dataset_split"],
        feature_file=KOFAM_FEATURE_FILE,
    )

    table = build_full_test_table(
        existing_5c=existing_5c,
        split_data=split_data,
        gapmind_predictions=gapmind_predictions,
        experimental_phenotypes=experimental_phenotypes,
    )

    table = table.sort_values(["phenotype", "held_out_dataset"]).reset_index(
        drop=True
    )

    # Annotate each row with its full-test minority-class count (Methods).
    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    table = annotate_minority_test(
        table,
        full_test_minority_counts(),
        test_dataset_column="held_out_dataset",
    )

    output_file = OUTPUT_DIR / "figure5d_full_test.tsv"
    table.to_csv(output_file, sep="\t", index=False)
    print(f"\nSaved Figure 5D data to {output_file}")
    print(f"  rows: {len(table)}")
    print(f"  phenotypes: {table['phenotype'].nunique()}")
    print(f"  held-out datasets: {sorted(table['held_out_dataset'].unique())}")

    median_full = float(np.nanmedian(table["balanced_accuracy_full"]))
    median_conc = float(
        np.nanmedian(table["balanced_accuracy_concordant_subset"])
    )
    median_fp = float(np.nanmedian(table["balanced_accuracy_FP_subset"]))
    median_fn = float(np.nanmedian(table["balanced_accuracy_FN_subset"]))
    print("\nMedian balanced accuracy:")
    print(f"  full held-out test set:   {median_full:.3f}")
    print(f"  concordant subset:        {median_conc:.3f}")
    print(f"  FP-discordant subset:     {median_fp:.3f}")
    print(f"  FN-discordant subset:     {median_fn:.3f}")


if __name__ == "__main__":
    main()
