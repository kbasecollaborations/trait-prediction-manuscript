#!/usr/bin/env python3
"""
Generate intra-dataset vs inter-dataset comparison data for Figure 3.

This script generates two types of analyses:
1. Intra-dataset performance: Cross-validation within each dataset
2. Inter-dataset performance: Train on one dataset, test on another

The script processes carbon utilization phenotypes across three datasets
(atleaf, marine, lit) using KOFAM features.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from trait_prediction.main import DataSet

from scripts.io import read_features, read_phenotypes
from scripts.ml import perform_cv, perform_train_test


def calculate_cv_results(
    dataset: DataSet, model_type: str = "cb_noeval", n_splits: int = 5
) -> pd.DataFrame:
    """
    Calculate cross-validation results for all phenotype-feature combinations.

    Parameters
    ----------
    dataset : DataSet
        Dataset containing features and phenotypes
    model_type : str, optional
        Type of model to use ('rf' or 'cb_noeval'), by default 'cb_noeval'

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results including balanced_accuracy, phenotype,
        dataset, feature_type, and representation columns
    """
    results = []

    # Calculate total iterations for progress bar
    total_iterations = len(dataset.feature_set.features) * len(
        dataset.phenotype_set.phenotypes
    )

    with tqdm(total=total_iterations, desc="Calculating CV results") as pbar:
        for feature in dataset.feature_set.features:
            for phenotype in dataset.phenotype_set.phenotypes:
                pindex = phenotype.pindex
                findex = feature.findex

                # Update progress bar
                pbar.update(1)

                # Skip if feature dataset doesn't match phenotype dataset
                dataset_name = findex.name.split("_")[0]
                if dataset_name != pindex.category:
                    continue

                # Get data
                phenotype_obj, feature_obj = dataset.get_data(pindex, findex)
                feature_data = feature_obj.feature_data
                phenotype_data = phenotype_obj.phenotype_data

                # Skip if training set doesn't have two classes
                class_counts = phenotype_data.value_counts()
                if len(class_counts) != 2:
                    continue

                # Skip if minority class has fewer than 10 samples
                if class_counts.min() < 10:
                    continue

                # Perform CV
                cv_results = perform_cv(
                    feature_data,
                    phenotype_data,
                    model_type,
                    n_splits=n_splits,
                    minority_class_min_samples=10,
                )

                if cv_results is None:
                    continue

                # Add metadata
                cv_results["model_type"] = model_type
                cv_results["phenotype"] = pindex.name
                cv_results["dataset"] = pindex.category
                cv_results["feature_type"] = findex.name.split("_")[1]
                cv_results["representation"] = "full"

                results.append(cv_results)

    return pd.concat(results, ignore_index=True)


def calculate_test_results(
    dataset: DataSet,
    model_type: str = "cb_noeval",
    test_size: int = 150,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Calculate train-test results for cross-dataset evaluation.

    For each phenotype and feature type, trains on one dataset and tests on another.

    Parameters
    ----------
    dataset : DataSet
        Dataset containing features and phenotypes
    model_type : str, optional
        Type of model to use ('rf' or 'cb_noeval'), by default 'cb_noeval'
    test_size : int, optional
        Number of samples to use for testing, by default 150
    n_repeats : int, optional
        Number of repeated random samples for testing, by default 5

    Returns
    -------
    pd.DataFrame
        DataFrame with test results including balanced_accuracy, phenotype,
        train_dataset, test_dataset, feature_type, and representation columns
    """
    results = []

    # Create a mapping of all dataset-phenotype-feature combinations
    data_map = {}
    for feature in dataset.feature_set.features:
        for phenotype in dataset.phenotype_set.phenotypes:
            pindex = phenotype.pindex
            findex = feature.findex

            # Skip if feature dataset doesn't match phenotype dataset
            dataset_name = findex.name.split("_")[0]
            if dataset_name != pindex.category:
                continue

            phenotype_obj, feature_obj = dataset.get_data(pindex, findex)
            key = (pindex.name, pindex.category, findex.name.split("_")[1])
            data_map[key] = (feature_obj.feature_data, phenotype_obj.phenotype_data)

    # For each combination, train and test across datasets
    keys = list(data_map.keys())
    total_iterations = len(keys) * len(keys)

    with tqdm(total=total_iterations, desc="Calculating train-test results") as pbar:
        for train_key in keys:
            phenotype_name, train_dataset, feature_type = train_key
            train_X, train_y = data_map[train_key]

            for test_key in keys:
                test_phenotype_name, test_dataset, test_feature_type = test_key

                # Update progress bar
                pbar.update(1)

                # Skip if same dataset (that's covered by CV)
                if train_dataset == test_dataset:
                    continue

                # Skip if different phenotypes
                if phenotype_name != test_phenotype_name:
                    continue

                # Skip if different feature types
                if feature_type != test_feature_type:
                    continue

                test_X, test_y = data_map[test_key]

                # Skip if training set doesn't have two classes
                train_class_counts = train_y.value_counts()
                if len(train_class_counts) != 2:
                    continue

                # Skip if minority class in training data has fewer than 10 samples
                if train_class_counts.min() < 10:
                    continue

                # Perform train-test
                test_results = perform_train_test(
                    train_X,
                    train_y,
                    test_X,
                    test_y,
                    model_type,
                    test_size=test_size,
                    n_repeats=n_repeats,
                )

                # Add metadata
                test_results["phenotype"] = phenotype_name
                test_results["model_type"] = model_type
                test_results["feature_type"] = feature_type
                test_results["train_dataset"] = train_dataset
                test_results["test_dataset"] = test_dataset
                test_results["representation"] = "full"

                results.append(test_results)

    return pd.concat(results, ignore_index=True)


def main() -> None:
    """
    Main function to generate Figure 3 data.
    """
    # Define paths
    FEATURE_DIR = Path("data/processed/features_reduced")
    PHENOTYPE_DIR = Path("data/processed/phenotypes")
    OUTPUT_DIR = Path("data/outputs/figure3/intra_vs_inter")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define datasets to include
    DATASETS_TO_INCLUDE = ["atleaf", "lit", "marine", "pmi"]

    # Load features
    feature_type = "kofam"
    print(f"Loading features for {feature_type}...")
    feature_files = list(FEATURE_DIR.glob(f"**/{feature_type}.tsv"))
    # Filter to only include specified datasets
    feature_files = [f for f in feature_files if f.parent.stem in DATASETS_TO_INCLUDE]
    feature_set = read_features(feature_files)

    # Load phenotypes and identify common phenotypes across datasets
    print("Loading phenotypes...")
    phenotype_files_all = list(PHENOTYPE_DIR.glob("**/*.tsv"))
    phenotype_files = []
    dataset_phenotype_name_map = defaultdict(set)

    for phenotype_file in phenotype_files_all:
        dataset_name = phenotype_file.parent.stem

        # Only include specified datasets
        if dataset_name not in DATASETS_TO_INCLUDE:
            continue

        phenotype_files.append(phenotype_file)
        dataset_phenotype_name_map[dataset_name].add(phenotype_file.stem)

    # Find common phenotypes across all datasets
    COMMON_PHENOTYPES = sorted(set.intersection(*dataset_phenotype_name_map.values()))
    print(f"Found {len(COMMON_PHENOTYPES)} common phenotypes across datasets")

    # Filter to only common phenotypes
    phenotype_files = [p for p in phenotype_files if p.stem in COMMON_PHENOTYPES]
    phenotype_set = read_phenotypes(phenotype_files)

    # Create dataset
    print("Creating dataset...")
    dataset = DataSet(phenotype_set, feature_set)

    # Generate CV results
    print("\nGenerating cross-validation results...")
    cv_results_file = OUTPUT_DIR / "cv_results.csv"
    if not cv_results_file.exists():
        cv_results = calculate_cv_results(dataset, model_type="cb_noeval", n_splits=5)
        cv_results.to_csv(cv_results_file, index=False)
        print(f"Saved CV results to {cv_results_file}")
    else:
        print(f"CV results already exist at {cv_results_file}, skipping...")

    # Generate test results
    print("\nGenerating train-test results...")
    test_results_file = OUTPUT_DIR / "test_results.csv"
    if not test_results_file.exists():
        test_results = calculate_test_results(
            dataset, model_type="cb_noeval", test_size=150, n_repeats=5
        )
        test_results.to_csv(test_results_file, index=False)
        print(f"Saved test results to {test_results_file}")
    else:
        print(f"Test results already exist at {test_results_file}, skipping...")

    print("\nDone!")


if __name__ == "__main__":
    main()
