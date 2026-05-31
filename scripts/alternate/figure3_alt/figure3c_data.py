#!/usr/bin/env python3
"""Generate phylogeny-independent inter-dataset data for Figure 3C: full testing vs
in-clade testing (test samples phylogenetically matched to training)."""

import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from trait_prediction.main import DataSet

from scripts.io import read_features, read_phenotypes
from scripts.ml import perform_cv, perform_train_test

warnings.filterwarnings("ignore")


def _get_samples(cluster_samples: list[str], X: pd.DataFrame) -> list[str]:
    """Filter cluster samples to those present in the dataset.

    Parameters
    ----------
    cluster_samples : list[str]
        List of sample IDs in the cluster
    X : pd.DataFrame
        Feature matrix with sample IDs as index

    Returns
    -------
    list[str]
        List of sample IDs present in both cluster and dataset
    """
    return [sample for sample in cluster_samples if sample in X.index]


def get_test_samples(
    X_train: pd.DataFrame, X_test: pd.DataFrame, distance_df: pd.DataFrame
) -> list[str]:
    """Select phylogenetically matched test samples using agglomerative clustering.

    Clusters the combined train+test samples, then keeps test samples only from
    clusters that contain sufficient training samples.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Test feature matrix
    distance_df : pd.DataFrame
        Phylogenetic distance matrix

    Returns
    -------
    list[str]
        List of test sample IDs that have phylogenetic matches in training data
    """
    samples = list(X_train.index.union(X_test.index).intersection(distance_df.index))
    distance_df_subset = distance_df.loc[samples, samples]
    n_clusters = int(4 * np.sqrt(len(samples) / 2))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    clustering.fit(distance_df_subset)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    test_samples = []
    for label in unique_labels:
        cluster_samples = [samples[i] for i in range(len(labels)) if labels[i] == label]
        curr_train_samples = _get_samples(cluster_samples, X_train)
        curr_test_samples = _get_samples(cluster_samples, X_test)

        if len(curr_train_samples) < 2 or len(curr_test_samples) == 0:
            continue

        test_samples.extend(curr_test_samples)

    return test_samples


def calculate_distance(
    train_samples: pd.Index, test_samples: pd.Index, distance_df: pd.DataFrame
) -> tuple[float, float]:
    """Calculate mean of per-test-sample average and minimum distances to training.

    Parameters
    ----------
    train_samples : pd.Index
        Training sample IDs
    test_samples : pd.Index
        Test sample IDs
    distance_df : pd.DataFrame
        Phylogenetic distance matrix

    Returns
    -------
    tuple[float, float]
        (mean of average distances, mean of minimum distances)
    """
    avg_distances = []
    min_distances = []
    row_inds = distance_df.index.intersection(train_samples)
    col_inds = distance_df.columns.intersection(test_samples)

    for col_ind in col_inds:
        distances = []
        for row_ind in row_inds:
            distance = distance_df.loc[row_ind, col_ind]
            distances.append(distance)
        avg_distances.append(np.mean(distances))
        min_distances.append(np.min(distances))

    return np.mean(avg_distances), np.mean(min_distances)


def calculate_test_results(
    dataset: DataSet,
    distance_df: pd.DataFrame,
    model_type: str = "cb_noeval",
    test_size: int = 50,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Calculate inter-dataset test results with phylogenetic controls.

    Tests all train/test dataset combinations per phenotype, comparing full vs
    phylogenetically matched (in-clade) testing.

    Parameters
    ----------
    dataset : DataSet
        Dataset containing features and phenotypes
    distance_df : pd.DataFrame
        Phylogenetic distance matrix
    model_type : str, optional
        Type of model to use, by default "cb_noeval"
    test_size : int, optional
        Number of test samples per repeat, by default 50
    n_repeats : int, optional
        Number of repeated random samples, by default 5

    Returns
    -------
    pd.DataFrame
        Test results with columns: repeat, accuracy, balanced_accuracy,
        matthews_corrcoef, phenotype, feature, train_dataset, test_dataset,
        test_type, avg_dist, min_dist
    """
    # Map every dataset-phenotype-feature combination to its data (PMI excluded)
    data_map = {}
    for feature in dataset.feature_set.features:
        for phenotype in dataset.phenotype_set.phenotypes:
            pindex = phenotype.pindex
            findex = feature.findex

            dataset_name = findex.name.split("_")[0]
            if dataset_name != pindex.category:
                continue
            if dataset_name == "pmi":
                continue

            phenotype_obj, feature_obj = dataset.get_data(pindex, findex)
            key = (pindex.name, pindex.category, findex.name)
            data_map[key] = (feature_obj.feature_data, phenotype_obj.phenotype_data)

    results = []
    combinations_list = list(product(data_map.keys(), repeat=2))
    total_combinations = len(combinations_list)

    with tqdm(
        total=total_combinations, desc="Calculating phylo-independent test results"
    ) as pbar:
        for train_key, test_key in combinations_list:
            pbar.update(1)

            train_phenotype, train_dataset, train_feature = train_key
            test_phenotype, test_dataset, test_feature = test_key

            # Same-dataset pairs are covered by CV; require matching phenotype and feature type
            if train_dataset == test_dataset:
                continue
            if train_phenotype != test_phenotype:
                continue
            if train_feature.split("_")[-1] != test_feature.split("_")[-1]:
                continue

            X_train, y_train = data_map[train_key]
            X_test, y_test = data_map[test_key]

            # Require both classes with minority class >= 10 samples in training
            train_class_counts = y_train.value_counts()
            if len(train_class_counts) != 2:
                continue
            if train_class_counts.min() < 10:
                continue

            test_samples_inclade = get_test_samples(X_train, X_test, distance_df)

            for test_type in ["full", "in-clade"]:
                if test_type == "full":
                    X_test_subset = X_test
                    y_test_subset = y_test
                else:
                    X_test_subset = X_test.loc[test_samples_inclade]
                    y_test_subset = y_test.loc[test_samples_inclade]

                if len(y_test_subset) < 10:
                    continue

                if test_size >= len(y_test_subset):
                    updated_test_size = int(len(y_test_subset) * 0.8)
                else:
                    updated_test_size = test_size

                if updated_test_size < 10:
                    continue

                test_results = perform_train_test(
                    X_train,
                    y_train,
                    X_test_subset,
                    y_test_subset,
                    model_type,
                    test_size=updated_test_size,
                    n_repeats=n_repeats,
                )

                avg_dist, min_dist = calculate_distance(
                    X_train.index, X_test_subset.index, distance_df
                )

                test_results["phenotype"] = train_phenotype
                test_results["feature"] = train_feature
                test_results["train_dataset"] = train_dataset
                test_results["test_dataset"] = test_dataset
                test_results["test_type"] = test_type
                test_results["avg_dist"] = avg_dist
                test_results["min_dist"] = min_dist

                results.append(test_results)

    return pd.concat(results, ignore_index=True)


def calculate_cv_results(
    dataset: DataSet, model_type: str = "cb_noeval", n_splits: int = 5
) -> pd.DataFrame:
    """Calculate cross-validation results for intra-dataset performance.

    Parameters
    ----------
    dataset : DataSet
        Dataset containing features and phenotypes
    model_type : str, optional
        Type of model to use, by default "cb_noeval"
    n_splits : int, optional
        Number of CV folds, by default 5

    Returns
    -------
    pd.DataFrame
        CV results with columns: accuracy, balanced_accuracy, matthews_corrcoef,
        fold, features, model_type, phenotype, dataset, feature_type, representation
    """
    results = []

    total_iterations = len(list(dataset.feature_set.features)) * len(
        list(dataset.phenotype_set.phenotypes)
    )

    with tqdm(total=total_iterations, desc="Calculating CV results") as pbar:
        for feature in dataset.feature_set.features:
            for phenotype in dataset.phenotype_set.phenotypes:
                pindex = phenotype.pindex
                findex = feature.findex

                pbar.update(1)

                # Feature dataset must match phenotype dataset (PMI excluded)
                dataset_name = findex.name.split("_")[0]
                if dataset_name != pindex.category:
                    continue
                if dataset_name == "pmi":
                    continue

                phenotype_obj, feature_obj = dataset.get_data(pindex, findex)
                feature_data = feature_obj.feature_data
                phenotype_data = phenotype_obj.phenotype_data

                # Require both classes with minority class >= 10 samples
                class_counts = phenotype_data.value_counts()
                if len(class_counts) != 2:
                    continue
                if class_counts.min() < 10:
                    continue

                cv_results = perform_cv(
                    feature_data,
                    phenotype_data,
                    model_type,
                    n_splits=n_splits,
                    minority_class_min_samples=10,
                )

                if cv_results is None:
                    continue

                cv_results["model_type"] = model_type
                cv_results["phenotype"] = pindex.name
                cv_results["dataset"] = pindex.category
                cv_results["feature_type"] = findex.name.split("_")[1]
                cv_results["representation"] = "full"

                results.append(cv_results)

    return pd.concat(results, ignore_index=True)


def main() -> None:
    """Generate Figure 3C phylogeny-independent data."""
    FEATURE_DIR = Path("data/processed/features_reduced")
    PHENOTYPE_DIR = Path("data/processed/phenotypes")
    DISTANCE_FILE = Path("data/processed/phylogeny/distance_matrix.tsv")
    OUTPUT_DIR = Path("data/outputs/figure3_alt/phylo_indep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATASETS_TO_INCLUDE = ["atleaf", "lit", "marine"]

    print("Loading phylogenetic distance matrix...")
    distance_df = pd.read_csv(DISTANCE_FILE, sep="\t", index_col=0)

    feature_type = "kofam"
    print(f"Loading features for {feature_type}...")
    feature_files = list(FEATURE_DIR.glob(f"**/{feature_type}.tsv"))
    feature_files = [f for f in feature_files if f.parent.stem in DATASETS_TO_INCLUDE]
    feature_set = read_features(feature_files)

    print("Loading phenotypes...")
    phenotype_files_all = list(PHENOTYPE_DIR.glob("**/*.tsv"))
    phenotype_files = []
    dataset_phenotype_name_map = defaultdict(set)

    for phenotype_file in phenotype_files_all:
        dataset_name = phenotype_file.parent.stem

        if dataset_name not in DATASETS_TO_INCLUDE:
            continue

        phenotype_files.append(phenotype_file)
        dataset_phenotype_name_map[dataset_name].add(phenotype_file.stem)

    COMMON_PHENOTYPES = sorted(set.intersection(*dataset_phenotype_name_map.values()))
    print(f"Found {len(COMMON_PHENOTYPES)} common phenotypes across datasets")

    phenotype_files = [p for p in phenotype_files if p.stem in COMMON_PHENOTYPES]
    phenotype_set = read_phenotypes(phenotype_files)

    print("Creating dataset...")
    dataset = DataSet(phenotype_set, feature_set)

    print("\nGenerating cross-validation results...")
    cv_results_file = OUTPUT_DIR / "cv_results.csv"
    if not cv_results_file.exists():
        cv_results = calculate_cv_results(dataset, model_type="cb_noeval", n_splits=5)
        cv_results.to_csv(cv_results_file, index=False)
        print(f"Saved CV results to {cv_results_file}")
    else:
        print(f"CV results already exist at {cv_results_file}, skipping...")

    print("\nGenerating phylogeny-independent test results...")
    test_results_file = OUTPUT_DIR / "test_results.tsv"
    if not test_results_file.exists():
        test_results = calculate_test_results(
            dataset,
            distance_df,
            model_type="cb_noeval",
            test_size=50,
            n_repeats=5,
        )
        test_results.to_csv(test_results_file, sep="\t", index=False)
        print(f"Saved test results to {test_results_file}")
    else:
        print(f"Test results already exist at {test_results_file}, skipping...")

    print("\nDone!")


if __name__ == "__main__":
    main()
