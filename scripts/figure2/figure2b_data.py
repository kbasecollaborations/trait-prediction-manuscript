#!/usr/bin/env python3
"""
Generate baseline predictions using nearest neighbor and null models.

This script generates baseline predictions for phenotype prediction using:
- Identity classifier (always predicts the majority class)
- Bernoulli classifier (random predictions based on class frequencies)
- Nearest neighbor classifier (phylogeny-based predictions)

The script processes three types of data splits:
- Random splits (5 repeats per phenotype)
- Dataset splits (cross-dataset validation)
- Out-of-clade splits (phylogeny-based cross-validation)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ete3 import Tree
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from scripts.classifiers import (
    BernoulliClassifier,
    IdentityClassifier,
    NearestNeighborClassifier,
)


def load_data_from_folder(
    folder_path: Path,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Load train, validation, and test data from a folder.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing the split data files.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.Series]]
        Dictionary with keys 'train', 'val', 'test' containing tuples of
        (features, labels) for each split.
    """
    data = {}
    X_train_path = folder_path / "X_train.tsv"
    y_train_path = folder_path / "y_train.tsv"
    X_val_path = folder_path / "X_val.tsv"
    y_val_path = folder_path / "y_val.tsv"
    X_test_path = folder_path / "X_test.tsv"
    y_test_path = folder_path / "y_test.tsv"

    X_train = pd.read_csv(X_train_path, index_col=0, sep="\t", dtype={"genomeID": str})
    X_val = pd.read_csv(X_val_path, index_col=0, sep="\t", dtype={"genomeID": str})
    X_test = pd.read_csv(X_test_path, index_col=0, sep="\t", dtype={"genomeID": str})

    y_train = pd.read_csv(
        y_train_path, index_col=0, sep="\t", dtype={"genomeID": str}
    ).iloc[:, 0]
    y_val = pd.read_csv(
        y_val_path, index_col=0, sep="\t", dtype={"genomeID": str}
    ).iloc[:, 0]
    y_test = pd.read_csv(
        y_test_path, index_col=0, sep="\t", dtype={"genomeID": str}
    ).iloc[:, 0]

    data["train"] = (X_train, y_train)
    data["val"] = (X_val, y_val)
    data["test"] = (X_test, y_test)

    return data


def get_scores(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float | str]:
    """
    Calculate classification metrics for a model.

    Parameters
    ----------
    model : Any
        Trained classifier with a predict method.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels.

    Returns
    -------
    dict[str, float | str]
        Dictionary containing accuracy, balanced_accuracy, matthews_corrcoef,
        precision, recall, and f1 score.
    """
    y_pred = model.predict(X)
    result = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }
    return result


def perform_ml(
    data: dict[str, dict[str, tuple[pd.DataFrame, pd.Series]]],
    tree: Tree,
    tree_leaves: list[str],
    distance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform machine learning evaluation for all datasets.

    Parameters
    ----------
    data : dict[str, dict[str, tuple[pd.DataFrame, pd.Series]]]
        Dictionary mapping dataset keys to train/val/test splits.
    tree : Tree
        Phylogenetic tree.
    tree_leaves : list[str]
        List of leaf names in the tree.
    distance_df : pd.DataFrame
        Pairwise phylogenetic distance matrix.

    Returns
    -------
    pd.DataFrame
        Results dataframe with metrics for all models and datasets.
    """
    results = []
    for key in tqdm(data):
        train_X, train_y = data[key]["train"]
        common_train_index = train_X.index.intersection(tree_leaves)
        train_X = train_X.loc[common_train_index]
        train_y = train_y.loc[common_train_index]

        test_X, test_y = data[key]["test"]
        common_test_index = test_X.index.intersection(tree_leaves)
        test_X = test_X.loc[common_test_index]
        test_y = test_y.loc[common_test_index]

        # Initialize classifiers
        bernoulli_classifier = BernoulliClassifier(
            random_state=42, categorical_feature_names=[]
        )
        nearest_neighbor_classifier = NearestNeighborClassifier(
            random_state=42,
            categorical_feature_names=[],
            tree=tree,
            distances=distance_df,
            k=3,
        )
        identity_classifier = IdentityClassifier(
            random_state=42, categorical_feature_names=[]
        )

        # Fit classifiers
        bernoulli_classifier.fit(train_X, train_y)
        nearest_neighbor_classifier.fit(train_X, train_y)
        identity_classifier.fit(train_X, train_y)

        # Get scores
        bernoulli_scores = get_scores(bernoulli_classifier, test_X, test_y)
        bernoulli_scores["model"] = "bernoulli"
        bernoulli_scores["phenotype"] = key.split("_")[0]

        nearest_neighbor_scores = get_scores(
            nearest_neighbor_classifier, test_X, test_y
        )
        nearest_neighbor_scores["model"] = "nearest_neighbor"
        nearest_neighbor_scores["phenotype"] = key.split("_")[0]

        identity_scores = get_scores(identity_classifier, test_X, test_y)
        identity_scores["model"] = "identity"
        identity_scores["phenotype"] = key.split("_")[0]

        results.append(bernoulli_scores)
        results.append(nearest_neighbor_scores)
        results.append(identity_scores)

    return pd.DataFrame(results)


def main() -> None:
    """
    Main function to generate baseline predictions.
    """
    # Define paths
    RANDOM_SPLIT_DIR = Path("data/processed/train_test_splits/random_split")
    DATASET_SPLIT_DIR = Path("data/processed/train_test_splits/dataset_split")
    PHYLO_SPLIT_DIR = Path("data/processed/train_test_splits/phylogeny_split")
    OUTPUT_DIR = Path("data/outputs/figure2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tree and distance matrix
    tree_file = Path("data/processed/phylogeny/gtdb-pruned.nwk")
    tree = Tree(str(tree_file), format=1)

    distance_file = Path("data/processed/phylogeny/distance_matrix.tsv")
    distance_df = pd.read_csv(distance_file, sep="\t", index_col=0)
    # Make the diagonal np.inf to avoid selecting the same genome
    distance_df.values[np.arange(len(distance_df)), np.arange(len(distance_df))] = (
        np.inf
    )

    tree_leaves = [leaf.name for leaf in tree.get_leaves()]
    assert len(tree_leaves) == len(distance_df)

    # Load random split data
    print("Loading random split data...")
    random_split_data = {}
    for random_dir in tqdm(list(RANDOM_SPLIT_DIR.iterdir())):
        if not random_dir.is_dir():
            continue
        phenotype_name = random_dir.name
        for repeat_dir in random_dir.iterdir():
            if not repeat_dir.is_dir():
                continue
            repeat_name = repeat_dir.name
            key = f"{phenotype_name}_{repeat_name}"
            data = load_data_from_folder(repeat_dir)
            random_split_data[key] = data

    # Load dataset split data
    print("Loading dataset split data...")
    dataset_split_data = {}
    for dataset_dir in tqdm(list(DATASET_SPLIT_DIR.iterdir())):
        if not dataset_dir.is_dir():
            continue
        phenotype_name = dataset_dir.name
        for split_dir in dataset_dir.iterdir():
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            key = f"{phenotype_name}_{split_name}"
            data = load_data_from_folder(split_dir)
            dataset_split_data[key] = data

    # Load out-of-clade split data
    print("Loading out-of-clade split data...")
    out_of_clade_split_data = {}
    for phenotype_dir in tqdm(list(PHYLO_SPLIT_DIR.iterdir())):
        if not phenotype_dir.is_dir():
            continue
        phenotype_name = phenotype_dir.name
        dataset_dir = phenotype_dir / "out-of-clade"
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue
        for split_dir in dataset_dir.iterdir():
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            key = f"{phenotype_name}_ooc_{split_name}"
            data = load_data_from_folder(split_dir)
            out_of_clade_split_data[key] = data

    # Perform ML and save results
    print("Performing ML for random split...")
    random_results_df = perform_ml(random_split_data, tree, tree_leaves, distance_df)
    random_results_df.to_csv(OUTPUT_DIR / "random_split_baselines.tsv", sep="\t")

    print("Performing ML for dataset split...")
    dataset_results_df = perform_ml(dataset_split_data, tree, tree_leaves, distance_df)
    dataset_results_df.to_csv(OUTPUT_DIR / "dataset_split_baselines.tsv", sep="\t")

    print("Performing ML for out-of-clade split...")
    out_of_clade_results_df = perform_ml(
        out_of_clade_split_data, tree, tree_leaves, distance_df
    )
    out_of_clade_results_df.to_csv(
        OUTPUT_DIR / "out_of_clade_split_baselines.tsv", sep="\t"
    )

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
