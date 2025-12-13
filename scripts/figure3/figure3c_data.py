#!/usr/bin/env python3
"""
Generate phylogeny-independent dataset analysis data for Figure 3C.

This script processes pre-generated dataset splits and applies phylogenetic
filtering to create two test conditions:
1. Full inter-dataset testing (all test samples)
2. In-clade testing (phylogenetically matched test samples only)

The clustering approach ensures test samples are phylogenetically similar to
training samples, controlling for phylogenetic distance effects.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml

warnings.filterwarnings("ignore")


def _get_samples(cluster_samples: list[str], X: pd.DataFrame) -> list[str]:
    """
    Filter cluster samples to only those present in the dataset.

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
    """
    Get phylogenetically matched test samples using agglomerative clustering.

    Performs clustering on the combined train+test samples, then selects only
    test samples from clusters that contain sufficient training samples.

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

        # Only include clusters with sufficient training samples
        if len(curr_train_samples) < 2 or len(curr_test_samples) == 0:
            continue

        test_samples.extend(curr_test_samples)

    return test_samples


def calculate_distance(
    train_samples: pd.Index, test_samples: pd.Index, distance_df: pd.DataFrame
) -> tuple[float, float]:
    """
    Calculate average and minimum phylogenetic distances.

    Computes the average distance from each test sample to all training samples,
    and the minimum distance from each test sample to any training sample.

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


def run_ml_on_dataset_splits_with_phylo_filter(
    split_data: dict[str, dict[str, pd.DataFrame | pd.Series]],
    distance_df: pd.DataFrame,
    model_type: str = "cb_noeval",
    random_state: int = 42,
    min_test_samples: int = 10,
) -> pd.DataFrame:
    """
    Run ML on dataset splits with phylogenetic filtering.

    For each split, performs both full testing and in-clade testing,
    comparing performance with and without phylogenetic filtering.

    Parameters
    ----------
    split_data : dict
        Nested dictionary from load_split_data() containing dataset splits
    distance_df : pd.DataFrame
        Phylogenetic distance matrix
    model_type : str, optional
        Model type to use ('cb', 'cb_noeval', 'rf', etc.), by default "cb_noeval"
    random_state : int, optional
        Random state for reproducibility, by default 42
    min_test_samples : int, optional
        Minimum number of test samples required to run ML, by default 10

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns for all scoring metrics plus:
        - test_type: 'full' or 'in-clade'
        - avg_dist: Average phylogenetic distance
        - min_dist: Minimum phylogenetic distance
        - phenotype: Phenotype name
        - train_test_config: Train/test dataset configuration
        - n_train, n_val, n_test: Number of samples in each set
    """
    results = []
    scoring = [
        "accuracy",
        "balanced_accuracy",
        "matthews_corrcoef",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "roc_auc",
    ]

    total_splits = len(split_data)

    with tqdm(
        total=total_splits * 2, desc="Running ML with phylo filtering"
    ) as pbar:
        for key in split_data:
            split = split_data[key]
            X_train = split["X_train"]
            y_train = split["y_train"]
            X_val = split["X_val"]
            y_val = split["y_val"]
            X_test = split["X_test"]
            y_test = split["y_test"]

            # Extract phenotype and train/test config from key
            # Key format: "Phenotype_train(datasets),test(dataset)"
            parts = key.split("_", 1)
            phenotype = parts[0]
            train_test_config = parts[1] if len(parts) > 1 else "unknown"

            # Skip if training or validation sets don't have both classes
            if len(y_train.unique()) != 2 or len(y_val.unique()) != 2:
                pbar.update(2)
                continue

            # Get phylogenetically matched test samples
            test_samples_inclade = get_test_samples(X_train, X_test, distance_df)

            # Perform both full and in-clade testing
            for test_type in ["full", "in-clade"]:
                pbar.set_postfix_str(f"{key} ({test_type})")

                if test_type == "full":
                    X_test_subset = X_test
                    y_test_subset = y_test
                else:
                    X_test_subset = X_test.loc[test_samples_inclade]
                    y_test_subset = y_test.loc[test_samples_inclade]

                # Skip if test set is too small
                if len(X_test_subset) < min_test_samples:
                    pbar.update(1)
                    continue

                # Skip if test set doesn't have both classes
                if len(y_test_subset.unique()) != 2:
                    pbar.update(1)
                    continue

                # Run ML
                result = perform_split_ml(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test_subset,
                    y_test_subset,
                    model_type=model_type,
                    scoring=scoring,
                    random_state=random_state,
                )

                # Calculate distances
                avg_dist, min_dist = calculate_distance(
                    X_train.index, X_test_subset.index, distance_df
                )

                # Add metadata
                result["test_type"] = test_type
                result["avg_dist"] = avg_dist
                result["min_dist"] = min_dist
                result["phenotype"] = phenotype
                result["train_test_config"] = train_test_config
                result["model_type"] = model_type
                result["n_train"] = len(X_train)
                result["n_val"] = len(X_val)
                result["n_test"] = len(X_test_subset)

                results.append(result)
                pbar.update(1)

    return pd.DataFrame(results)


def main() -> None:
    """
    Main function to generate Figure 3C data.
    """
    # Define paths
    SPLITS_DIR = Path("data/processed/train_test_splits")
    DISTANCE_FILE = Path("data/processed/phylogeny/distance_matrix.tsv")
    OUTPUT_DIR = Path("data/outputs/figure3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load distance matrix
    print("Loading phylogenetic distance matrix...")
    distance_df = pd.read_csv(DISTANCE_FILE, sep="\t", index_col=0)

    # Load dataset splits only
    print("\nLoading dataset splits...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=["dataset_split"])

    # Print summary of loaded data
    print(f"\nLoaded {len(split_data['dataset_split'])} dataset splits")

    # Run ML with phylogenetic filtering
    print("\nRunning ML with phylogenetic filtering...")
    results = run_ml_on_dataset_splits_with_phylo_filter(
        split_data["dataset_split"],
        distance_df,
        model_type="cb_noeval",
        random_state=42,
        min_test_samples=10,
    )

    # Save results
    results_file = OUTPUT_DIR / "figure3c_results.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    # Print summary statistics
    print("\nResults summary:")
    print(f"  Total experiments: {len(results)}")
    print("\nBy test type (mean balanced accuracy):")
    test_type_summary = (
        results.groupby("test_type")[["balanced_accuracy", "avg_dist", "min_dist"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(test_type_summary)

    print("\nBy phenotype (mean balanced accuracy):")
    phenotype_summary = (
        results.groupby(["phenotype", "test_type"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(phenotype_summary)

    print("\n\nPhylogenetic distance comparison:")
    dist_comparison = (
        results.groupby("test_type")[["avg_dist", "min_dist"]].mean().round(4)
    )
    print(dist_comparison)

    print("\nDone!")


if __name__ == "__main__":
    main()
