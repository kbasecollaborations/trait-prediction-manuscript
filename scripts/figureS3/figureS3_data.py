#!/usr/bin/env python3
"""
Generate Figure S3 data: in-clade vs out-of-clade balanced accuracy across all
15 shared phenotypes and the 4 leave-one-dataset-out (LOO) splits.

This script reuses the canonical pattern from ``scripts/figure3/figure3c_data.py``
- it loads the dataset_split splits (4 LOO combinations per phenotype) and uses
agglomerative clustering on the phylogenetic distance matrix to partition the
held-out test set into two non-overlapping subsets:

- ``in_clade``: test samples that fall in clusters with at least two training
  samples (the same definition figure3c uses).
- ``out_of_clade``: test samples in clusters that do **not** contain enough
  training samples (the complement).

For each (phenotype, LOO combination, split flavour) tuple a CatBoost model is
trained via :func:`scripts.ml.make_classifier` with ``model_type="cb"`` and
``random_state=42`` on KOFAM features and evaluated on the corresponding test
subset. Combinations with fewer than 10 test samples or fewer than 10 minority-
class samples in the test subset are not silently dropped - they are recorded
with ``excluded=True`` and an exclusion reason for downstream auditing.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml

warnings.filterwarnings("ignore")


SPLITS_DIR: Path = Path("data/processed/train_test_splits")
DISTANCE_FILE: Path = Path("data/processed/phylogeny/distance_matrix.tsv")
OUTPUT_DIR: Path = Path("data/outputs/figureS3")
OUTPUT_FILE: Path = OUTPUT_DIR / "figureS3_data.tsv"

MIN_TEST_SAMPLES: int = 10
MIN_MINORITY_SAMPLES: int = 10


def partition_test_samples_by_clade(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Partition test samples into in-clade and out-of-clade subsets.

    Performs agglomerative clustering on the phylogenetic distance matrix
    restricted to the union of train and test samples, then assigns each test
    sample to either ``in_clade`` (cluster contains >= 2 training samples) or
    ``out_of_clade`` (otherwise). The clustering parameters mirror
    ``scripts/figure3/figure3c_data.py``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix indexed by genome ID.
    X_test : pd.DataFrame
        Test feature matrix indexed by genome ID.
    distance_df : pd.DataFrame
        Pairwise phylogenetic distance matrix indexed by genome ID.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(in_clade_samples, out_of_clade_samples)``. Test samples not present
        in ``distance_df`` are dropped from both lists.
    """
    samples = list(X_train.index.union(X_test.index).intersection(distance_df.index))
    distance_df_subset = distance_df.loc[samples, samples]
    n_clusters = int(4 * np.sqrt(len(samples) / 2))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    clustering.fit(distance_df_subset)
    labels = clustering.labels_

    train_index = set(X_train.index)
    test_index = set(X_test.index)

    in_clade: list[str] = []
    out_of_clade: list[str] = []

    for label in np.unique(labels):
        cluster_samples = [samples[i] for i in range(len(labels)) if labels[i] == label]
        cluster_train = [s for s in cluster_samples if s in train_index]
        cluster_test = [s for s in cluster_samples if s in test_index]

        if not cluster_test:
            continue

        if len(cluster_train) >= 2:
            in_clade.extend(cluster_test)
        else:
            out_of_clade.extend(cluster_test)

    return in_clade, out_of_clade


def parse_train_test_config(key: str) -> tuple[str, str]:
    """
    Parse a dataset-split key into train and test dataset descriptions.

    Parameters
    ----------
    key : str
        Key produced by the splits loader, formatted as
        ``"<phenotype>_train(<datasets>),test(<dataset>)"``.

    Returns
    -------
    tuple[str, str]
        ``(train_datasets, test_dataset)`` where ``train_datasets`` is the
        '+' joined list of training datasets and ``test_dataset`` is the held
        out dataset name.

    Raises
    ------
    ValueError
        If ``key`` does not contain the expected ``train(...)`` /
        ``test(...)`` substrings.
    """
    if "train(" not in key or "test(" not in key:
        raise ValueError(f"Unexpected dataset-split key format: {key}")
    train_datasets = key.split("train(")[1].split(")")[0]
    test_dataset = key.split("test(")[1].split(")")[0]
    return train_datasets, test_dataset


def evaluate_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test_subset: pd.DataFrame,
    y_test_subset: pd.Series,
    random_state: int = 42,
) -> float:
    """
    Train CatBoost on the LOO training data and score on a clade subset.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Full training features and labels for this LOO combination.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation features and labels (used for CatBoost early stopping).
    X_test_subset, y_test_subset : pd.DataFrame, pd.Series
        Test features and labels restricted to the clade subset of interest.
    random_state : int, optional
        Seed forwarded to ``make_classifier``, by default 42.

    Returns
    -------
    float
        Balanced accuracy on the clade-restricted test subset.
    """
    scores = perform_split_ml(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test_subset,
        y_test_subset,
        model_type="cb",
        scoring=["balanced_accuracy"],
        random_state=random_state,
    )
    return float(scores["balanced_accuracy"])


def build_results(
    split_data: dict[str, dict[str, pd.DataFrame | pd.Series]],
    distance_df: pd.DataFrame,
    phenotypes: Iterable[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Iterate over all (phenotype, LOO, clade) combinations and collect results.

    Parameters
    ----------
    split_data : dict[str, dict[str, pd.DataFrame | pd.Series]]
        Output of ``load_split_data(..., split_types=["dataset_split"])``
        restricted to the dataset_split entries.
    distance_df : pd.DataFrame
        Pairwise phylogenetic distance matrix.
    phenotypes : Iterable[str] | None, optional
        Subset of phenotypes to evaluate. If ``None``, evaluate every phenotype
        present in ``split_data``.
    random_state : int, optional
        Seed forwarded to CatBoost, by default 42.

    Returns
    -------
    pd.DataFrame
        Long-form results with one row per (phenotype, LOO combo, split flavour)
        combination. Excluded rows carry ``balanced_accuracy = NaN`` and the
        reason in ``exclusion_reason``.
    """
    keys = list(split_data.keys())
    if phenotypes is not None:
        wanted = set(phenotypes)
        keys = [k for k in keys if k.split("_", 1)[0] in wanted]

    rows: list[dict[str, object]] = []

    for key in tqdm(keys, desc="LOO splits"):
        phenotype = key.split("_", 1)[0]
        train_datasets, test_dataset = parse_train_test_config(key)

        split = split_data[key]
        X_train = split["X_train"]
        y_train = split["y_train"]
        X_val = split["X_val"]
        y_val = split["y_val"]
        X_test = split["X_test"]
        y_test = split["y_test"]

        # Sanity: training/validation must be binary for CatBoost classifier
        train_is_binary = len(y_train.unique()) == 2
        val_is_binary = len(y_val.unique()) == 2

        in_clade_samples, out_of_clade_samples = partition_test_samples_by_clade(
            X_train, X_test, distance_df
        )

        for split_type, subset in (
            ("in_clade", in_clade_samples),
            ("out_of_clade", out_of_clade_samples),
        ):
            X_test_subset = X_test.loc[subset]
            y_test_subset = y_test.loc[subset]
            n_test = int(len(X_test_subset))
            class_counts = y_test_subset.value_counts()
            n_minority_test = int(class_counts.min()) if not class_counts.empty else 0

            row: dict[str, object] = {
                "phenotype": phenotype,
                "train_datasets": train_datasets,
                "test_dataset": test_dataset,
                "split_type": split_type,
                "n_train": int(len(X_train)),
                "n_test": n_test,
                "n_minority_test": n_minority_test,
                "balanced_accuracy": float("nan"),
                "excluded": False,
                "exclusion_reason": "",
            }

            if not train_is_binary or not val_is_binary:
                row["excluded"] = True
                row["exclusion_reason"] = "train_or_val_not_binary"
                rows.append(row)
                continue

            if n_test < MIN_TEST_SAMPLES:
                row["excluded"] = True
                row["exclusion_reason"] = (
                    f"n_test<{MIN_TEST_SAMPLES} (got {n_test})"
                )
                rows.append(row)
                continue

            if len(class_counts) < 2:
                row["excluded"] = True
                row["exclusion_reason"] = "test_subset_single_class"
                rows.append(row)
                continue

            if n_minority_test < MIN_MINORITY_SAMPLES:
                row["excluded"] = True
                row["exclusion_reason"] = (
                    f"n_minority_test<{MIN_MINORITY_SAMPLES} (got {n_minority_test})"
                )
                rows.append(row)
                continue

            row["balanced_accuracy"] = evaluate_split(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test_subset,
                y_test_subset,
                random_state=random_state,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """
    Entry point: load splits + distance matrix, run all combinations, save TSV.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotype",
        type=str,
        default=None,
        help="If provided, restrict evaluation to this phenotype (smoke test).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading phylogenetic distance matrix from {DISTANCE_FILE}...")
    distance_df = pd.read_csv(DISTANCE_FILE, sep="\t", index_col=0)

    print(f"Loading dataset splits from {SPLITS_DIR}...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=["dataset_split"])
    dataset_splits = split_data["dataset_split"]
    print(f"Loaded {len(dataset_splits)} LOO splits")

    phenotypes = [args.phenotype] if args.phenotype else None
    results = build_results(
        dataset_splits,
        distance_df,
        phenotypes=phenotypes,
        random_state=42,
    )

    results.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved {len(results)} rows to {OUTPUT_FILE}")

    if not results.empty:
        retained = results[~results["excluded"]]
        excluded = results[results["excluded"]]
        print("\nRetained combinations per split type:")
        print(retained.groupby("split_type").size())
        if not excluded.empty:
            print("\nExclusion reasons:")
            print(excluded["exclusion_reason"].value_counts())
        print("\nMean balanced accuracy per split type:")
        print(retained.groupby("split_type")["balanced_accuracy"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
