#!/usr/bin/env python3
"""Script to generate training data splits for the ML pipeline.

This script creates train/validation/test splits using three different strategies:
1. Random split (cluster-based)
2. Dataset split (leave-one-dataset-out)
3. Phylogeny split (in-clade and out-of-clade)

The splits are created for all common phenotypes across all 4 datasets using kofam features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from sklearn.model_selection import train_test_split

from scripts.splitter import InCladeSplitter, LargeTreeTraverseOOCSplitter

# Constants
RANDOM_STATE = 42
OUTPUT_DIR = Path("data/processed/train_test_splits")
PHENOTYPE_DIR = Path("data/processed/phenotypes")
FEATURE_DIR = Path("data/processed/features_reduced")
PHYLOGENY_DIR = Path("data/processed/phylogeny")

# All four datasets
DATASET_SUBSET = ["atleaf", "lit", "marine", "pmi"]

# All common phenotypes across all 4 datasets (15 total)
COMMON_PHENOTYPES = [
    "Alanine",
    "Arginine",
    "Cellobiose",
    "Fructose",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Glycerol",
    "Histidine",
    "Maltose",
    "Mannitol",
    "Mannose",
    "Serine",
    "Sucrose",
    "m-Inositol",
]


def create_sample_map() -> dict[str, str]:
    """Create mapping from sample IDs to dataset names.

    Returns
    -------
    dict[str, str]
        Dictionary mapping sample IDs to their dataset names.
    """
    sample_map = {}
    for dataset_name in DATASET_SUBSET:
        curr_dataset_dir = FEATURE_DIR / dataset_name
        feature_file = curr_dataset_dir / "kofam.tsv"
        feature_data = pd.read_csv(
            feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
        )
        samples = feature_data.index.tolist()
        for sample in samples:
            sample_map[sample] = dataset_name
    return sample_map


def load_combined_features() -> pd.DataFrame:
    """Load combined kofam features from the combined_datasets folder.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with samples as rows and features as columns.
    """
    feature_file = FEATURE_DIR / "combined_datasets" / "kofam.tsv"
    feature_data = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    return feature_data


def load_phenotype_data() -> dict[str, pd.DataFrame]:
    """Load phenotype data for all common phenotypes.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping phenotype names to their data frames.
    """
    phenotype_data_dict = {}
    for phenotype_name in COMMON_PHENOTYPES:
        # Combine phenotype data from all datasets
        phenotype_dfs = []
        for dataset_name in DATASET_SUBSET:
            phenotype_file = PHENOTYPE_DIR / dataset_name / f"{phenotype_name}.tsv"
            if phenotype_file.exists():
                pheno_data = pd.read_csv(
                    phenotype_file, sep="\t", index_col=0, dtype={"genomeID": str}
                )
                phenotype_dfs.append(pheno_data)

        if phenotype_dfs:
            phenotype_data_dict[phenotype_name] = pd.concat(phenotype_dfs, axis=0)

    return phenotype_data_dict


def create_xy_data(
    feature_data: pd.DataFrame, phenotype_data_dict: dict[str, pd.DataFrame]
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create X, y pairs for each phenotype.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Combined feature matrix.
    phenotype_data_dict : dict[str, pd.DataFrame]
        Dictionary of phenotype data frames.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Dictionary mapping phenotype names to (X, y) tuples.
    """
    xy_data = {}
    for phenotype_name in COMMON_PHENOTYPES:
        if phenotype_name not in phenotype_data_dict:
            print(f"Warning: {phenotype_name} not found in phenotype data")
            continue

        phenotype_data = phenotype_data_dict[phenotype_name]
        common_index = feature_data.index.intersection(phenotype_data.index)
        X = feature_data.loc[common_index, :]
        y = phenotype_data.loc[common_index, :]

        # Drop rows where phenotype data is missing
        mask = ~y.isna().any(axis=1)
        X = X.loc[mask, :]
        y = y.loc[mask, :]

        # Convert phenotype data to int
        y = y.astype(int)

        print(f"Phenotype: {phenotype_name}")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        xy_data[phenotype_name] = (X, y)

    return xy_data


def cluster_based_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split_ratio: dict[str, float],
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Create cluster-based random splits.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Target matrix.
    split_ratio : dict[str, float]
        Dictionary with 'train', 'val', 'test' ratios.
    n_clusters : int | None, optional
        Number of clusters. If None, uses sqrt(n_samples).
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with 'train', 'val', 'test' sample indices.
    """
    from sklearn.cluster import AgglomerativeClustering

    np.random.seed(random_state)
    if n_clusters is None:
        n_clusters = int(np.sqrt(len(X)))

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    cluster_labels = clustering.fit_predict(X)

    train_indices, val_indices, test_indices = [], [], []
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = X.index[cluster_mask]
        cluster_samples = np.random.permutation(cluster_samples)
        n_cluster_samples = len(cluster_samples)
        n_train = int(split_ratio["train"] * n_cluster_samples)
        n_val = int(split_ratio["val"] * n_cluster_samples)
        train_indices.extend(cluster_samples[:n_train])
        val_indices.extend(cluster_samples[n_train : n_train + n_val])
        test_indices.extend(cluster_samples[n_train + n_val :])

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def save_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split_dict: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Save train/val/test splits to files.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Target matrix.
    split_dict : dict[str, list[str]]
        Dictionary with 'train', 'val', 'test' sample indices.
    output_dir : Path
        Output directory for saving splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = X.loc[split_dict["train"], :]
    X_val = X.loc[split_dict["val"], :]
    X_test = X.loc[split_dict["test"], :]
    y_train = y.loc[split_dict["train"]].astype(int)
    y_val = y.loc[split_dict["val"]].astype(int)
    y_test = y.loc[split_dict["test"]].astype(int)

    X_train.to_csv(output_dir / "X_train.tsv", sep="\t", index=True)
    X_val.to_csv(output_dir / "X_val.tsv", sep="\t", index=True)
    X_test.to_csv(output_dir / "X_test.tsv", sep="\t", index=True)
    y_train.to_csv(output_dir / "y_train.tsv", sep="\t", index=True)
    y_val.to_csv(output_dir / "y_val.tsv", sep="\t", index=True)
    y_test.to_csv(output_dir / "y_test.tsv", sep="\t", index=True)


def create_random_splits(xy_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    """Create cluster-based random splits for all phenotypes.

    Parameters
    ----------
    xy_data : dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Dictionary mapping phenotype names to (X, y) tuples.
    """
    print("\n=== Creating random splits ===")
    for phenotype_name in xy_data:
        print(f"Processing {phenotype_name}...")
        output_dir = OUTPUT_DIR / f"random_split/{phenotype_name}"
        X, y = xy_data[phenotype_name]
        for i in range(5):
            curr_output_dir = output_dir / f"{i}"
            split_dict = cluster_based_split(
                X,
                y,
                split_ratio={"train": 0.7, "val": 0.15, "test": 0.15},
                random_state=RANDOM_STATE + i,
            )
            save_split(X, y, split_dict, curr_output_dir)


def create_dataset_splits(
    xy_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]], sample_map: dict[str, str]
) -> None:
    """Create leave-one-dataset-out splits for all phenotypes.

    Parameters
    ----------
    xy_data : dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Dictionary mapping phenotype names to (X, y) tuples.
    sample_map : dict[str, str]
        Mapping from sample IDs to dataset names.
    """
    print("\n=== Creating dataset splits ===")
    for phenotype_name in xy_data:
        print(f"Processing {phenotype_name}...")
        output_dir = OUTPUT_DIR / f"dataset_split/{phenotype_name}"
        X, y = xy_data[phenotype_name]
        for dataset_name in DATASET_SUBSET:
            training_datasets = [d for d in DATASET_SUBSET if d != dataset_name]
            folder_name = f"train({'+'.join(training_datasets)}),test({dataset_name})"
            curr_output_dir = output_dir / folder_name

            test_indices = [ind for ind in X.index if sample_map[ind] == dataset_name]
            train_val_indices = [ind for ind in X.index if ind not in test_indices]

            # Skip if test set is empty
            if len(test_indices) == 0:
                print(f"  Skipping {dataset_name} (no test samples)")
                continue

            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=0.15, random_state=RANDOM_STATE
            )

            split_dict = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            save_split(X, y, split_dict, curr_output_dir)


def load_phylogeny() -> tuple[Tree, pd.DataFrame]:
    """Load phylogenetic tree and compute distance matrix.

    Returns
    -------
    tuple[Tree, pd.DataFrame]
        Tuple of (tree, distance_matrix).
    """
    tree_file = PHYLOGENY_DIR / "gtdb-pruned.nwk"
    tree = Tree(str(tree_file), format=1)

    distance_file = PHYLOGENY_DIR / "distance_matrix.tsv"
    if distance_file.exists():
        distance_df = pd.read_csv(distance_file, sep="\t", index_col=0)
    else:
        # Calculate distance matrix
        print("Calculating distance matrix...")
        leaves = list(tree.get_leaf_names())
        n_leaves = len(leaves)
        distance_matrix = np.zeros((n_leaves, n_leaves), dtype=np.float64)

        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                distance = tree.get_distance(leaves[i], leaves[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        distance_df = pd.DataFrame(
            distance_matrix,
            index=pd.Index(leaves, name="genome"),
            columns=pd.Index(leaves, name="genome"),
        )
        distance_df.to_csv(distance_file, sep="\t")

    return tree, distance_df


def create_phylogeny_splits(
    xy_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Create phylogeny-based splits (in-clade and out-of-clade) for all phenotypes.

    Parameters
    ----------
    xy_data : dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Dictionary mapping phenotype names to (X, y) tuples.
    """
    print("\n=== Creating phylogeny splits ===")

    # Load tree and distance matrix
    tree, distance_df = load_phylogeny()

    for phenotype_name in xy_data:
        print(f"Processing {phenotype_name}...")
        output_dir = OUTPUT_DIR / f"phylogeny_split/{phenotype_name}"
        X, y = xy_data[phenotype_name]

        # Get samples that are in the tree
        samples = [leaf for leaf in tree.iter_leaf_names() if leaf in X.index]
        print(f"  {len(samples)} samples found in tree (out of {len(X)})")

        for split_type in ["in-clade", "out-of-clade"]:
            print(f"  Creating {split_type} splits...")
            if split_type == "in-clade":
                splitter = InCladeSplitter(tree, distance_df, test_set_ratio=0.15)
            elif split_type == "out-of-clade":
                splitter = LargeTreeTraverseOOCSplitter(
                    tree,
                    test_set_range=(0.12, 0.18),
                    single_clades=None,
                    n_max_clade=2,
                    prefer_small_clade=False,
                    growth_data=None,
                    min_zeros=0,
                    min_ones=0,
                    time_out_iter=None,
                )
            else:
                raise ValueError(f"Unknown split type: {split_type}")

            for i in range(5):
                curr_output_dir = output_dir / f"{split_type}/{i}"
                test_samples = list(splitter.split(samples))
                train_val_samples = list(set(samples) - set(test_samples))

                train_indices, val_indices = train_test_split(
                    train_val_samples, test_size=0.15, random_state=RANDOM_STATE + i
                )

                split_dict = {
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_samples,
                }
                save_split(X, y, split_dict, curr_output_dir)


def main() -> None:
    """Main function to create all data splits."""
    print("Creating train/test splits for ML pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of phenotypes: {len(COMMON_PHENOTYPES)}")
    print(f"Number of datasets: {len(DATASET_SUBSET)}")

    # Create sample map
    print("\n=== Creating sample map ===")
    sample_map = create_sample_map()
    print(f"Total samples: {len(sample_map)}")

    # Load features
    print("\n=== Loading features ===")
    feature_data = load_combined_features()
    print(f"Feature matrix shape: {feature_data.shape}")

    # Load phenotypes
    print("\n=== Loading phenotypes ===")
    phenotype_data_dict = load_phenotype_data()
    print(f"Loaded {len(phenotype_data_dict)} phenotypes")

    # Create X, y data
    print("\n=== Creating X, y pairs ===")
    xy_data = create_xy_data(feature_data, phenotype_data_dict)

    # Create all splits
    create_random_splits(xy_data)
    create_dataset_splits(xy_data, sample_map)
    create_phylogeny_splits(xy_data)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
