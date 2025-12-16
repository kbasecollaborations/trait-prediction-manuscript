#!/usr/bin/env python3
"""
Generate data for Figure 6B: ML performance on confident samples only.

This script:
1. Loads train-test splits (random_split and dataset_split)
2. Calculates confidence scores (y_soft) for each sample using:
   - Phylogenetic k-NN agreement
   - GapMind mechanistic predictions
   - Experimental data (y_hard)
3. Filters to keep only confident samples (y_soft < 0.4 OR y_soft > 0.6)
4. Runs ML on the filtered confident samples
5. Saves results for comparison with full-data performance
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from tqdm import tqdm

from scripts.classifiers.nearest_neighbor import NearestNeighborClassifier
from scripts.ml_splits import load_split_data, perform_split_ml


# Confidence parameters (from notebook)
K_NEIGHBORS = 3
W_PHYLO = 0.2
W_GAPMIND = 0.3
W_EXP = 0.5

# Filtering threshold
CONFIDENCE_THRESHOLD_LOW = 0.4
CONFIDENCE_THRESHOLD_HIGH = 0.6


def load_gapmind_data() -> pd.DataFrame:
    """
    Load and process GapMind predictions.

    Returns
    -------
    pd.DataFrame
        GapMind predictions as binary (0/1) for each phenotype.
    """
    # Phenotype mapping
    phenotype_dict = {
        "alanine": "Alanine",
        "arginine": "Arginine",
        "histidine": "Histidine",
        "serine": "Serine",
        "fructose": "Fructose",
        "galactose": "Galactose",
        "glucose": "Glucose",
        "maltose": "Maltose",
        "mannose": "Mannose",
        "sucrose": "Sucrose",
        "myoinositol": "m-Inositol",
        "mannitol": "Mannitol",
        "glycerol": "Glycerol",
        "galacturonate": "Galacturonic-Acid",
        "cellobiose": "Cellobiose",
    }

    # Load marine ID mapping
    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    with open(marine_ids_file, "r") as f:
        marine_ids_map = {v.rsplit("_", 2)[0]: k for k, v in json.load(f).items()}

    # Load GapMind data
    from scripts.io import index_format_func

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict.keys()]
    datasets = ["s__at-leaf-lit-pmi", "s__marine-seqs"]
    gapmind_data_list = [
        pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{dataset}_categories.csv")
        for dataset in datasets
    ]
    gapmind_data = pd.concat(gapmind_data_list, axis=0)
    gapmind_data["genomeId"] = (
        gapmind_data["genome_id"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    gapmind_data.index = gapmind_data["genomeId"]  # type: ignore
    gapmind_data.index = [marine_ids_map.get(ind, ind) for ind in gapmind_data.index]
    gapmind_data = gapmind_data.loc[:, gapmind_phenotype_subset]
    gapmind_data.columns = gapmind_data.columns.str.replace("Carbon__", "")
    gapmind_data.columns = gapmind_data.columns.map(phenotype_dict)  # type: ignore

    # Convert to binary
    replace_dict = {
        "complete": 1,
        "likely_complete": 1,
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }
    gapmind_data_binary = gapmind_data.replace(replace_dict).astype(np.uint8)

    return gapmind_data_binary


def load_gapmind_confidence() -> dict[str, pd.Series]:
    """
    Load GapMind data and create confidence scores.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping phenotype names to confidence scores (0-1).
    """
    # Phenotype mapping
    phenotype_dict = {
        "alanine": "Alanine",
        "arginine": "Arginine",
        "histidine": "Histidine",
        "serine": "Serine",
        "fructose": "Fructose",
        "galactose": "Galactose",
        "glucose": "Glucose",
        "maltose": "Maltose",
        "mannose": "Mannose",
        "sucrose": "Sucrose",
        "myoinositol": "m-Inositol",
        "mannitol": "Mannitol",
        "glycerol": "Glycerol",
        "galacturonate": "Galacturonic-Acid",
        "cellobiose": "Cellobiose",
    }

    # Load marine ID mapping
    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    with open(marine_ids_file, "r") as f:
        marine_ids_map = {v.rsplit("_", 2)[0]: k for k, v in json.load(f).items()}

    # Load GapMind data (categorical, not binary)
    from scripts.io import index_format_func

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict.keys()]
    datasets = ["s__at-leaf-lit-pmi", "s__marine-seqs"]
    gapmind_data_list = [
        pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{dataset}_categories.csv")
        for dataset in datasets
    ]
    gapmind_data = pd.concat(gapmind_data_list, axis=0)
    gapmind_data["genomeId"] = (
        gapmind_data["genome_id"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    gapmind_data.index = gapmind_data["genomeId"]  # type: ignore
    gapmind_data.index = [marine_ids_map.get(ind, ind) for ind in gapmind_data.index]
    gapmind_data = gapmind_data.loc[:, gapmind_phenotype_subset]
    gapmind_data.columns = gapmind_data.columns.str.replace("Carbon__", "")
    gapmind_data.columns = gapmind_data.columns.map(phenotype_dict)  # type: ignore

    # Map categories to confidence values
    gapmind_values = {
        "complete": 1.0,
        "likely_complete": 0.9,
        "steps_missing_medium": 0.6,
        "steps_missing_low": 0.3,
        "not_present": 0.1,
        "no_evidence": 0.0,
        "uncategorized": 0.0,
    }

    # Create confidence scores
    conf_mech = {}
    for phenotype_name in gapmind_data.columns:
        y_conf = gapmind_data[phenotype_name].map(gapmind_values)  # type: ignore
        conf_mech[phenotype_name] = y_conf

    return conf_mech


def load_phylogenetic_data() -> tuple[Tree, pd.DataFrame]:
    """
    Load phylogenetic tree and distance matrix.

    Returns
    -------
    tuple[Tree, pd.DataFrame]
        Tree and distance matrix (with diagonal set to inf).
    """
    tree_file = Path("data/processed/phylogenetic_tree/gtdb-pruned.nwk")
    tree = Tree(str(tree_file), format=1)

    distance_file = Path("data/processed/phylogenetic_tree/distance_matrix.tsv")
    distance_df = pd.read_csv(distance_file, sep="\t", index_col=0)
    # Set diagonal to inf (exclude self from nearest neighbors)
    distance_df.values[np.arange(len(distance_df)), np.arange(len(distance_df))] = (
        np.inf
    )

    return tree, distance_df


def calculate_phylo_confidence(
    y_exp: pd.Series, tree: Tree, distance_df: pd.DataFrame, k: int = 3
) -> pd.Series:
    """
    Calculate confidence based on phylogenetic k-NN agreement.

    Parameters
    ----------
    y_exp : pd.Series
        Experimental phenotype values (binary).
    tree : Tree
        Phylogenetic tree.
    distance_df : pd.DataFrame
        Phylogenetic distance matrix.
    k : int, optional
        Number of nearest neighbors to consider, by default 3.

    Returns
    -------
    pd.Series
        Confidence scores (0-1) based on k-NN agreement.
    """
    tree_leaves = [leaf.name for leaf in tree.get_leaves()]
    common_inds = y_exp.index.intersection(tree_leaves)
    y_exp_subset = y_exp.loc[common_inds]

    # Use NearestNeighborClassifier to get k-NN predictions
    classifier = NearestNeighborClassifier(
        random_state=42,
        categorical_feature_names=[],
        tree=tree,
        distances=distance_df,
        k=k,
    )
    # Fit on all data (using dummy X with same index as y)
    X_dummy = pd.DataFrame(index=y_exp_subset.index)
    classifier.fit(X_dummy, y_exp_subset)
    # Predict returns average of k neighbors (float between 0 and 1)
    y_conf = classifier.predict(X_dummy, round_to_int=False).apply(float)

    return y_conf


def load_phenotype_data() -> dict[str, pd.DataFrame]:
    """
    Load experimental phenotype data.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping phenotype names to DataFrames with phenotype values.
    """
    phenotype_data_dir = Path("data/processed/phenotypes/combined_phenotypes/")
    combined_phenotype_dict = {}
    for phenotype_file in phenotype_data_dir.glob("*.tsv"):
        phenotype_name = phenotype_file.stem
        phenotype_data = pd.read_csv(
            phenotype_file, sep="\t", index_col=0, dtype={"genomeID": str}
        )  # type: ignore
        combined_phenotype_dict[phenotype_name] = phenotype_data
    return combined_phenotype_dict


def calculate_y_soft_all_phenotypes(
    phenotype_data: dict[str, pd.DataFrame],
    tree: Tree,
    distance_df: pd.DataFrame,
    conf_mech: dict[str, pd.Series],
    k: int = 3,
    w_phylo: float = 0.2,
    w_gapmind: float = 0.3,
    w_exp: float = 0.5,
) -> dict[str, pd.Series]:
    """
    Calculate soft labels (y_soft) for all phenotypes.

    Parameters
    ----------
    phenotype_data : dict[str, pd.DataFrame]
        Experimental phenotype data.
    tree : Tree
        Phylogenetic tree.
    distance_df : pd.DataFrame
        Phylogenetic distance matrix.
    conf_mech : dict[str, pd.Series]
        GapMind confidence scores.
    k : int, optional
        Number of nearest neighbors for phylogenetic confidence, by default 3.
    w_phylo : float, optional
        Weight for phylogenetic confidence, by default 0.2.
    w_gapmind : float, optional
        Weight for GapMind confidence, by default 0.3.
    w_exp : float, optional
        Weight for experimental data, by default 0.5.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping phenotype names to y_soft values (0-1).
    """
    y_soft = {}

    for phenotype_name in tqdm(
        phenotype_data.keys(), desc="Calculating y_soft for phenotypes"
    ):
        y_exp = phenotype_data[phenotype_name].iloc[:, 0]

        # Calculate phylogenetic confidence
        conf_phylo = calculate_phylo_confidence(y_exp, tree, distance_df, k=k)

        # Get GapMind confidence
        if phenotype_name not in conf_mech:
            print(f"Warning: {phenotype_name} not in GapMind data, skipping")
            continue
        y_soft_mech = conf_mech[phenotype_name]

        # Find common indices
        common_inds = (
            conf_phylo.index.intersection(y_soft_mech.index)
            .intersection(y_exp.index)
        )

        y_exp_subset = y_exp.loc[common_inds]
        conf_phylo_subset = conf_phylo.loc[common_inds]
        y_soft_mech_subset = y_soft_mech.loc[common_inds]

        # Calculate weighted confidence
        y_conf = (
            conf_phylo_subset * w_phylo
            + y_soft_mech_subset * w_gapmind
            + y_exp_subset * w_exp
        )

        # Clip to valid probability range
        y_soft[phenotype_name] = np.clip(y_conf, 0.01, 1 - 0.01)

    return y_soft


def filter_confident_samples(
    split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    y_soft: dict[str, pd.Series],
    threshold_low: float = 0.4,
    threshold_high: float = 0.6,
) -> dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """
    Filter split data to keep only confident samples.

    Keeps samples where y_soft < threshold_low OR y_soft > threshold_high.

    Parameters
    ----------
    split_data : dict
        Nested dictionary from load_split_data() containing all splits.
    y_soft : dict[str, pd.Series]
        Dictionary mapping phenotype names to y_soft values.
    threshold_low : float, optional
        Lower confidence threshold, by default 0.4.
    threshold_high : float, optional
        Upper confidence threshold, by default 0.6.

    Returns
    -------
    dict
        Filtered split data with same structure as input.
    """
    filtered_data = {}

    for split_type in split_data:
        filtered_data[split_type] = {}
        for key in split_data[split_type]:
            phenotype_name = key.split("_")[0]

            if phenotype_name not in y_soft:
                print(f"Warning: {phenotype_name} not in y_soft, skipping {key}")
                continue

            split = split_data[split_type][key]
            y_soft_phenotype = y_soft[phenotype_name]

            # Filter each set (train, val, test)
            filtered_split = {}

            for set_name in ["train", "val", "test"]:
                X_key = f"X_{set_name}"
                y_key = f"y_{set_name}"

                X = split[X_key]
                y = split[y_key]

                # Get y_soft for this set
                common_inds = y.index.intersection(y_soft_phenotype.index)
                y_soft_set = y_soft_phenotype.loc[common_inds]

                # Filter to confident samples
                confident_mask = (y_soft_set < threshold_low) | (
                    y_soft_set > threshold_high
                )
                confident_inds = y_soft_set[confident_mask].index

                # Filter X and y
                filtered_split[X_key] = X.loc[confident_inds]
                filtered_split[y_key] = y.loc[confident_inds]

            filtered_data[split_type][key] = filtered_split

    return filtered_data


def run_ml_on_filtered_splits(
    filtered_split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    model_type: str = "cb",
    random_state: int = 42,
    min_test_samples: int = 10,
) -> pd.DataFrame:
    """
    Run machine learning on filtered (confident-only) splits.

    Parameters
    ----------
    filtered_split_data : dict
        Filtered split data from filter_confident_samples().
    model_type : str, optional
        Model type to use, by default "cb".
    random_state : int, optional
        Random state for reproducibility, by default 42.
    min_test_samples : int, optional
        Minimum test samples required, by default 10.

    Returns
    -------
    pd.DataFrame
        Results with same format as figure3ab_data.py.
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

    # Calculate total iterations for progress bar
    total_splits = sum(len(splits) for splits in filtered_split_data.values())

    with tqdm(total=total_splits, desc="Running ML on filtered splits") as pbar:
        for split_type in filtered_split_data:
            for key in filtered_split_data[split_type]:
                pbar.set_postfix_str(f"{split_type}/{key}")
                pbar.update(1)

                split = filtered_split_data[split_type][key]
                X_train = split["X_train"]
                y_train = split["y_train"]
                X_val = split["X_val"]
                y_val = split["y_val"]
                X_test = split["X_test"]
                y_test = split["y_test"]

                # Skip if test set is too small
                n_test_samples = len(X_test)
                if n_test_samples < min_test_samples:
                    print(
                        f"\nSkipping {split_type}/{key}: test set has only {n_test_samples} samples"
                    )
                    continue

                # Skip if training or validation sets don't have both classes
                if len(y_train.unique()) != 2 or len(y_val.unique()) != 2:
                    print(
                        f"\nSkipping {split_type}/{key}: training or validation set doesn't have 2 classes"
                    )
                    continue

                # Run ML
                result = perform_split_ml(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    model_type=model_type,
                    scoring=scoring,
                    random_state=random_state,
                )

                # Add metadata
                result["split_type"] = split_type
                result["key"] = key
                result["phenotype"] = key.split("_")[0]
                result["model_type"] = model_type
                result["n_train"] = len(X_train)
                result["n_val"] = len(X_val)
                result["n_test"] = len(X_test)

                results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    """
    Main function to generate Figure 6B data.
    """
    # Define paths
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure6")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only process random_split and dataset_split
    SPLIT_TYPES = ["random_split", "dataset_split"]

    # Load all splits
    print("Loading train-test splits...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=SPLIT_TYPES)

    # Print summary of loaded data
    print("\nLoaded splits summary:")
    for split_type in split_data:
        print(f"  {split_type}: {len(split_data[split_type])} splits")

    # Load phylogenetic data
    print("\nLoading phylogenetic data...")
    tree, distance_df = load_phylogenetic_data()
    print(f"  Tree has {len(tree.get_leaves())} leaves")
    print(f"  Distance matrix shape: {distance_df.shape}")

    # Load GapMind confidence
    print("\nLoading GapMind confidence scores...")
    conf_mech = load_gapmind_confidence()
    print(f"  GapMind confidence for {len(conf_mech)} phenotypes")

    # Load phenotype data
    print("\nLoading phenotype data...")
    phenotype_data = load_phenotype_data()
    print(f"  Loaded {len(phenotype_data)} phenotypes")

    # Calculate y_soft for all phenotypes
    print("\nCalculating y_soft for all phenotypes...")
    y_soft = calculate_y_soft_all_phenotypes(
        phenotype_data,
        tree,
        distance_df,
        conf_mech,
        k=K_NEIGHBORS,
        w_phylo=W_PHYLO,
        w_gapmind=W_GAPMIND,
        w_exp=W_EXP,
    )
    print(f"  Calculated y_soft for {len(y_soft)} phenotypes")

    # Save y_soft for reference
    y_soft_file = OUTPUT_DIR / "y_soft.pkl"
    import pickle

    with open(y_soft_file, "wb") as f:
        pickle.dump(y_soft, f)
    print(f"  Saved y_soft to {y_soft_file}")

    # Filter to confident samples only
    print("\nFiltering to confident samples only...")
    print(
        f"  Keeping samples with y_soft < {CONFIDENCE_THRESHOLD_LOW} OR y_soft > {CONFIDENCE_THRESHOLD_HIGH}"
    )
    filtered_split_data = filter_confident_samples(
        split_data,
        y_soft,
        threshold_low=CONFIDENCE_THRESHOLD_LOW,
        threshold_high=CONFIDENCE_THRESHOLD_HIGH,
    )

    # Print filtering summary
    print("\nFiltering summary:")
    for split_type in filtered_split_data:
        total_samples_original = sum(
            len(split_data[split_type][key]["X_test"])
            for key in split_data[split_type]
        )
        total_samples_filtered = sum(
            len(filtered_split_data[split_type][key]["X_test"])
            for key in filtered_split_data[split_type]
        )
        pct_retained = (
            100 * total_samples_filtered / total_samples_original
            if total_samples_original > 0
            else 0
        )
        print(
            f"  {split_type}: {total_samples_filtered}/{total_samples_original} "
            f"test samples retained ({pct_retained:.1f}%)"
        )

    # Run ML on filtered splits
    print("\nRunning machine learning on filtered splits...")
    results = run_ml_on_filtered_splits(
        filtered_split_data, model_type="cb", random_state=42, min_test_samples=10
    )

    # Save results
    results_file = OUTPUT_DIR / "figure6b_confident_ml_results.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    # Print summary statistics
    print("\nResults summary:")
    print(f"  Total experiments: {len(results)}")
    print("\nBy split type:")
    summary = (
        results.groupby("split_type")["balanced_accuracy"].describe().round(3)
    )
    print(summary)

    print("\nBy phenotype (mean balanced accuracy):")
    phenotype_summary = (
        results.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(phenotype_summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
