#!/usr/bin/env python3
"""Generate baseline predictions (identity, Bernoulli, nearest neighbor) across split types.

Covers random, dataset, and out-of-clade splits.
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
from scripts.ml_splits import load_single_split_data


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
    data: dict[str, dict[str, pd.DataFrame | pd.Series]],
    tree: Tree,
    tree_leaves: list[str],
    distance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform machine learning evaluation for all datasets.

    Parameters
    ----------
    data : dict[str, dict[str, pd.DataFrame | pd.Series]]
        Dictionary mapping dataset keys to data splits. Each split contains
        X_train, y_train, X_val, y_val, X_test, y_test.
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
        train_X = data[key]["X_train"]
        train_y = data[key]["y_train"]
        common_train_index = train_X.index.intersection(tree_leaves)
        train_X = train_X.loc[common_train_index]
        train_y = train_y.loc[common_train_index]

        test_X = data[key]["X_test"]
        test_y = data[key]["y_test"]
        common_test_index = test_X.index.intersection(tree_leaves)
        test_X = test_X.loc[common_test_index]
        test_y = test_y.loc[common_test_index]

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

        bernoulli_classifier.fit(train_X, train_y)
        nearest_neighbor_classifier.fit(train_X, train_y)
        identity_classifier.fit(train_X, train_y)

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
    """Main function to generate baseline predictions."""
    RANDOM_SPLIT_DIR = Path("data/processed/train_test_splits/random_split")
    DATASET_SPLIT_DIR = Path("data/processed/train_test_splits/dataset_split")
    PHYLO_SPLIT_DIR = Path("data/processed/train_test_splits/phylogeny_split")
    FEATURE_FILE = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
    OUTPUT_DIR = Path("data/outputs/figure2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading feature data from {FEATURE_FILE}")
    feature_data = pd.read_csv(
        FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"Feature data shape: {feature_data.shape}")

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
            data = load_single_split_data(repeat_dir, feature_data)
            random_split_data[key] = data

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
            data = load_single_split_data(split_dir, feature_data)
            dataset_split_data[key] = data

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
            data = load_single_split_data(split_dir, feature_data)
            out_of_clade_split_data[key] = data

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
