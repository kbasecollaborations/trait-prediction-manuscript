#!/usr/bin/env python3
"""
Generate SHAP-based feature importance data for Figure 4C.

This script performs two main analyses:
1. Trains ML models on combined train-test splits and identifies consistent top features
   using SHAP values across multiple random seeds.
2. Trains ML models on individual datasets (atleaf, lit, marine) and identifies consistent
   top features using SHAP values across multiple random seeds.
3. Compares features between combined and individual dataset models to find intersection
   and unique features.

The script uses CatBoost's native SHAP implementation for feature importance.
"""

import json
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from trait_prediction.main import DataSet

from scripts.io import read_features, read_phenotypes
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data

warnings.filterwarnings("ignore")


def get_shap_top_features(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
) -> list[str]:
    """
    Get top features using SHAP values from a trained CatBoost model.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained CatBoost model
    X : pd.DataFrame
        Feature matrix used for SHAP calculation
    y : pd.Series
        Target variable (needed for Pool creation)
    n_features : int, optional
        Number of top features to return, by default 10

    Returns
    -------
    list[str]
        List of top n_features feature names ranked by mean absolute SHAP value
    """
    # Create Pool for SHAP calculation
    pool = Pool(data=X, label=y)

    # Get SHAP values using CatBoost's native implementation
    # Returns array of shape (n_samples, n_features + 1) - last column is base value
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")

    # Remove last column (base value) and take mean absolute SHAP value per feature
    shap_values = shap_values[:, :-1]  # Remove base value column
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create Series and sort by importance
    feature_importance = pd.Series(mean_abs_shap, index=X.columns)
    feature_importance.sort_values(ascending=False, inplace=True)

    return feature_importance.head(n_features).index.tolist()


def train_and_get_top_features_split(
    split_data: dict[str, pd.DataFrame | pd.Series],
    random_state: int = 42,
    n_features: int = 10,
) -> list[str]:
    """
    Train model on combined train+val split and get top SHAP features.

    Parameters
    ----------
    split_data : dict
        Dictionary with keys X_train, y_train, X_val, y_val
    random_state : int, optional
        Random state for sampling, by default 42
    n_features : int, optional
        Number of top features to return, by default 10

    Returns
    -------
    list[str]
        List of top feature names
    """
    # Combine train and val sets
    X_combined = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
    y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)

    # Resample 80% with stratification
    X_train, _, y_train, _ = train_test_split(
        X_combined,
        y_combined,
        train_size=0.8,
        stratify=y_combined,
        random_state=random_state,
        shuffle=True,
    )

    # Train cb_noeval model
    model = make_classifier("cb_noeval", random_state=random_state)
    model.fit(X_train, y_train, verbose=False)

    # Get top features using SHAP
    top_features = get_shap_top_features(model, X_train, y_train, n_features=n_features)

    return top_features


def train_and_get_top_features_individual(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_features: int = 10,
) -> list[str]:
    """
    Train model on individual dataset and get top SHAP features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    random_state : int, optional
        Random state for sampling, by default 42
    n_features : int, optional
        Number of top features to return, by default 10

    Returns
    -------
    list[str]
        List of top feature names
    """
    # Resample 80% with stratification
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=random_state, shuffle=True
    )

    # Train cb_noeval model
    model = make_classifier("cb_noeval", random_state=random_state)
    model.fit(X_train, y_train, verbose=False)

    # Get top features using SHAP
    top_features = get_shap_top_features(model, X_train, y_train, n_features=n_features)

    return top_features


def get_consistent_features(
    feature_lists: list[list[str]], threshold: float = 0.7
) -> list[str]:
    """
    Get features that appear in at least threshold proportion of runs.

    Parameters
    ----------
    feature_lists : list[list[str]]
        List of feature lists from different runs
    threshold : float, optional
        Minimum proportion of runs a feature must appear in, by default 0.7

    Returns
    -------
    list[str]
        List of features appearing in >= threshold proportion of runs, sorted by frequency
    """
    # Count feature occurrences
    all_features = [feat for feat_list in feature_lists for feat in feat_list]
    feature_counts = Counter(all_features)

    # Filter by threshold
    min_count = int(np.ceil(len(feature_lists) * threshold))
    consistent_features = [
        feat for feat, count in feature_counts.items() if count >= min_count
    ]

    # Sort by frequency (descending)
    consistent_features.sort(key=lambda x: feature_counts[x], reverse=True)

    return consistent_features


def load_individual_dataset(
    dataset_name: str, phenotype: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load features and phenotype for an individual dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (atleaf, lit, marine)
    phenotype : str
        Name of phenotype

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target variable
    """
    features_path = Path("data/processed/features_reduced") / dataset_name / "kofam.tsv"
    phenotype_path = (
        Path("data/processed/phenotypes") / dataset_name / f"{phenotype}.tsv"
    )

    # Load features using the proper reading functions
    feature_set = read_features([features_path])

    # Load phenotype using the proper reading functions
    phenotype_set = read_phenotypes([phenotype_path])

    # Create dataset which handles NaN values and alignment
    dataset = DataSet(phenotype_set, feature_set)

    # Get the specific phenotype data
    for feature_object in dataset.feature_set.features:
        for phenotype_object in dataset.phenotype_set.phenotypes:
            pindex = phenotype_object.pindex
            findex = feature_object.findex
            phenotype_object_common, feature_object_common = dataset.get_data(
                pindex, findex
            )
            phenotype_df = phenotype_object_common.phenotype_data
            feature_df = feature_object_common.feature_data
            break
        break
    X = feature_df.copy()
    y = phenotype_df.copy()

    return X, y


def load_all_datasets_combined(
    datasets: Sequence[str], phenotype: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load features and phenotype from all datasets combined.

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names (atleaf, lit, marine, pmi)
    phenotype : str
        Name of phenotype

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target variable from all datasets combined
    """
    # Collect all feature and phenotype files
    feature_files = []
    phenotype_files = []

    for dataset_name in datasets:
        features_path = (
            Path("data/processed/features_reduced") / dataset_name / "kofam.tsv"
        )
        phenotype_path = (
            Path("data/processed/phenotypes") / dataset_name / f"{phenotype}.tsv"
        )

        if features_path.exists() and phenotype_path.exists():
            feature_files.append(features_path)
            phenotype_files.append(phenotype_path)

    # Load all features and phenotypes
    feature_set = read_features(feature_files)
    phenotype_set = read_phenotypes(phenotype_files)

    # Create dataset which handles NaN values and alignment
    dataset = DataSet(phenotype_set, feature_set)

    # Combine all phenotype data for this phenotype across all datasets
    all_X_list = []
    all_y_list = []

    for feature_object in dataset.feature_set.features:
        for phenotype_object in dataset.phenotype_set.phenotypes:
            pindex = phenotype_object.pindex
            findex = feature_object.findex
            phenotype_object_common, feature_object_common = dataset.get_data(
                pindex, findex
            )
            phenotype_df = phenotype_object_common.phenotype_data
            feature_df = feature_object_common.feature_data

            all_X_list.append(feature_df)
            all_y_list.append(phenotype_df)

    # Concatenate all datasets
    X_combined = pd.concat(all_X_list, axis=0)
    y_combined = pd.concat(all_y_list, axis=0)

    # Handle duplicate indices by keeping first occurrence
    X_combined = X_combined[~X_combined.index.duplicated(keep="first")]
    y_combined = y_combined[~y_combined.index.duplicated(keep="first")]

    # Align indices
    common_idx = X_combined.index.intersection(y_combined.index)
    X_combined = X_combined.loc[common_idx]
    y_combined = y_combined.loc[common_idx]

    return X_combined, y_combined


def get_test_dataset_from_key(key: str) -> str | None:
    """
    Extract the test dataset name from a dataset_split key.

    Parameters
    ----------
    key : str
        Key in format "Phenotype_train(datasets),test(dataset)"

    Returns
    -------
    str | None
        Test dataset name (atleaf, lit, marine, or pmi), or None if not a dataset_split
    """
    # Example key: "Alanine_train(atleaf+lit+marine),test(pmi)"
    if "test(" not in key:
        return None

    test_part = key.split("test(")[1]
    test_dataset = test_part.rstrip(")")

    return test_dataset


def analyze_combined_splits(
    splits_dir: Path,
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
) -> dict[str, Any]:
    """
    Analyze combined train-test splits and get consistent top features.

    Parameters
    ----------
    splits_dir : Path
        Path to train_test_splits directory
    n_seeds : int, optional
        Number of random seeds to run, by default 20
    threshold : float, optional
        Minimum proportion for consistent features, by default 0.7
    n_features : int, optional
        Number of top features per run, by default 10

    Returns
    -------
    dict[str, Any]
        Dictionary with keys as split identifiers and values as consistent feature lists
    """
    dataset_split_dir = splits_dir / "dataset_split"

    # Get all phenotypes
    phenotypes = [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]

    results = {}

    for phenotype in tqdm(phenotypes, desc="Analyzing combined splits"):
        phenotype_dir = dataset_split_dir / phenotype

        # Get all split types for this phenotype
        split_types = [d.name for d in phenotype_dir.iterdir() if d.is_dir()]

        for split_type in split_types:
            split_dir = phenotype_dir / split_type
            key = f"{phenotype}_{split_type}"

            # Load split data
            try:
                split_data = load_single_split_data(split_dir)
            except Exception as e:
                print(f"Error loading {key}: {e}")
                continue

            # Check if we have enough samples for both classes
            y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)
            if len(y_combined.unique()) != 2:
                continue

            # Run for multiple seeds
            feature_lists = []
            for seed in range(n_seeds):
                try:
                    top_features = train_and_get_top_features_split(
                        split_data, random_state=seed, n_features=n_features
                    )
                    feature_lists.append(top_features)
                except Exception as e:
                    print(f"Error in {key} with seed {seed}: {e}")
                    continue

            # Get consistent features
            if len(feature_lists) > 0:
                consistent_features = get_consistent_features(
                    feature_lists, threshold=threshold
                )
                results[key] = consistent_features

    return results


def analyze_individual_datasets(
    datasets: Sequence[str],
    phenotypes: Sequence[str],
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
) -> dict[str, dict[str, list[str]]]:
    """
    Analyze individual datasets and get consistent top features for each phenotype.

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names (atleaf, lit, marine)
    phenotypes : Sequence[str]
        List of phenotype names to analyze (only common phenotypes)
    n_seeds : int, optional
        Number of random seeds to run, by default 20
    threshold : float, optional
        Minimum proportion for consistent features, by default 0.7
    n_features : int, optional
        Number of top features per run, by default 10

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Nested dictionary: {dataset: {phenotype: [consistent_features]}}
    """
    results: dict[str, dict[str, list[str]]] = {}

    for dataset in tqdm(datasets, desc="Analyzing individual datasets"):
        dataset_results: dict[str, list[str]] = {}

        # Process only the specified phenotypes
        for phenotype in tqdm(phenotypes, desc=f"Processing {dataset}", leave=False):
            try:
                # Load data
                X, y = load_individual_dataset(dataset, phenotype)

                # Check if we have enough samples for both classes
                if len(y.unique()) != 2:
                    continue

                # Check minimum samples per class
                class_counts = y.value_counts()
                if class_counts.min() < 10:
                    continue

                # Run for multiple seeds
                feature_lists = []
                for seed in range(n_seeds):
                    try:
                        top_features = train_and_get_top_features_individual(
                            X, y, random_state=seed, n_features=n_features
                        )
                        feature_lists.append(top_features)
                    except Exception as e:
                        print(f"Error in {dataset}/{phenotype} with seed {seed}: {e}")
                        continue

                # Get consistent features
                if len(feature_lists) > 0:
                    consistent_features = get_consistent_features(
                        feature_lists, threshold=threshold
                    )
                    dataset_results[phenotype] = consistent_features

            except Exception as e:
                print(f"Error loading {dataset}/{phenotype}: {e}")
                continue

        results[dataset] = dataset_results

    return results


def analyze_all_datasets_combined(
    datasets: Sequence[str],
    phenotypes: Sequence[str],
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
) -> dict[str, list[str]]:
    """
    Analyze all datasets combined and get consistent top features for each phenotype.

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names (atleaf, lit, marine, pmi)
    phenotypes : Sequence[str]
        List of phenotype names to analyze (only common phenotypes)
    n_seeds : int, optional
        Number of random seeds to run, by default 20
    threshold : float, optional
        Minimum proportion for consistent features, by default 0.7
    n_features : int, optional
        Number of top features per run, by default 10

    Returns
    -------
    dict[str, list[str]]
        Dictionary: {phenotype: [consistent_features]}
    """
    results: dict[str, list[str]] = {}

    for phenotype in tqdm(phenotypes, desc="Analyzing all datasets combined"):
        try:
            # Load data from all datasets combined
            X, y = load_all_datasets_combined(datasets, phenotype)

            # Check if we have enough samples for both classes
            if len(y.unique()) != 2:
                continue

            # Check minimum samples per class
            class_counts = y.value_counts()
            if class_counts.min() < 10:
                continue

            # Run for multiple seeds
            feature_lists = []
            for seed in range(n_seeds):
                try:
                    top_features = train_and_get_top_features_individual(
                        X, y, random_state=seed, n_features=n_features
                    )
                    feature_lists.append(top_features)
                except Exception as e:
                    print(f"Error in all_combined/{phenotype} with seed {seed}: {e}")
                    continue

            # Get consistent features
            if len(feature_lists) > 0:
                consistent_features = get_consistent_features(
                    feature_lists, threshold=threshold
                )
                results[phenotype] = consistent_features

        except Exception as e:
            print(f"Error loading all_combined/{phenotype}: {e}")
            continue

    return results


def compare_features(
    combined_results: dict[str, list[str]],
    individual_results: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, Any]]:
    """
    Compare features between combined and individual dataset models.

    For each combination where the dataset was not used in training the combined model,
    find the intersection and unique features.

    Parameters
    ----------
    combined_results : dict[str, list[str]]
        Results from combined splits: {key: [features]}
    individual_results : dict[str, dict[str, list[str]]]
        Results from individual datasets: {dataset: {phenotype: [features]}}

    Returns
    -------
    dict[str, dict[str, Any]]
        Comparison results with structure:
        {comparison_key: {
            'phenotype': str,
            'test_dataset': str,
            'combined_features': list,
            'individual_features': list,
            'intersection': list,
            'unique_to_individual': list,
            'unique_to_combined': list
        }}
    """
    comparisons = {}

    for combined_key, combined_features in combined_results.items():
        # Extract phenotype and test dataset
        phenotype = combined_key.split("_")[0]
        test_dataset = get_test_dataset_from_key(combined_key)

        if test_dataset is None:
            continue

        # Check if this is one of our target datasets
        if test_dataset not in ["atleaf", "lit", "marine"]:
            continue

        # Check if we have individual results for this dataset/phenotype
        if (
            test_dataset in individual_results
            and phenotype in individual_results[test_dataset]
        ):
            individual_features = individual_results[test_dataset][phenotype]

            # Calculate intersection and unique features
            combined_set = set(combined_features)
            individual_set = set(individual_features)

            intersection = sorted(list(combined_set & individual_set))
            unique_to_individual = sorted(list(individual_set - combined_set))
            unique_to_combined = sorted(list(combined_set - individual_set))

            comparison_key = f"{phenotype}_{test_dataset}"
            comparisons[comparison_key] = {
                "phenotype": phenotype,
                "test_dataset": test_dataset,
                "combined_features": combined_features,
                "individual_features": individual_features,
                "intersection": intersection,
                "unique_to_individual": unique_to_individual,
                "unique_to_combined": unique_to_combined,
                "n_intersection": len(intersection),
                "n_unique_to_individual": len(unique_to_individual),
                "n_unique_to_combined": len(unique_to_combined),
            }

    return comparisons


def main() -> None:
    """
    Main function to generate Figure 4C data.
    """
    # Define paths
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure4")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters
    N_SEEDS = 20
    THRESHOLD = 0.7
    N_FEATURES = 10
    DATASETS = ["atleaf", "lit", "marine"]

    # Get common phenotypes from dataset_split directory
    dataset_split_dir = SPLITS_DIR / "dataset_split"
    COMMON_PHENOTYPES = sorted(
        [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]
    )

    print("=" * 80)
    print("Figure 4C: SHAP-based Feature Importance Analysis")
    print("=" * 80)
    print(f"\nCommon phenotypes to analyze: {len(COMMON_PHENOTYPES)}")
    print(f"  {', '.join(COMMON_PHENOTYPES[:5])}...")

    # Step 1 & 2: Analyze combined train-test splits
    combined_file = OUTPUT_DIR / "combined_splits_shap_features.json"

    if combined_file.exists():
        print("\nStep 1-2: Loading existing combined splits results...")
        with open(combined_file, "r") as f:
            combined_results_filtered = json.load(f)
        print(
            f"Loaded {len(combined_results_filtered)} combined splits from: {combined_file}"
        )
    else:
        print("\nStep 1-2: Analyzing combined train-test splits...")
        print(f"  - Running {N_SEEDS} random seeds per split")
        print(f"  - Extracting top {N_FEATURES} features per run")
        print(f"  - Keeping features appearing in ≥{THRESHOLD * 100}% of runs")

        combined_results = analyze_combined_splits(
            SPLITS_DIR, n_seeds=N_SEEDS, threshold=THRESHOLD, n_features=N_FEATURES
        )

        # Filter to only include common phenotypes
        combined_results_filtered = {
            k: v
            for k, v in combined_results.items()
            if k.split("_")[0] in COMMON_PHENOTYPES
        }

        print(
            f"\nCompleted analysis for {len(combined_results_filtered)} combined splits (filtered to common phenotypes)"
        )

        # Save combined results
        with open(combined_file, "w") as f:
            json.dump(combined_results_filtered, f, indent=2)
        print(f"Saved combined results to: {combined_file}")

    # Step 3 & 4: Analyze individual datasets
    individual_file = OUTPUT_DIR / "individual_datasets_shap_features.json"

    if individual_file.exists():
        print("\nStep 3-4: Loading existing individual datasets results...")
        with open(individual_file, "r") as f:
            individual_results = json.load(f)
        print(
            f"Loaded results for {len(individual_results)} datasets from: {individual_file}"
        )
        for dataset, phenotypes in individual_results.items():
            print(f"  - {dataset}: {len(phenotypes)} phenotypes")
    else:
        print(f"\nStep 3-4: Analyzing individual datasets: {DATASETS}")
        print(f"  - Only analyzing common phenotypes: {len(COMMON_PHENOTYPES)}")
        print(f"  - Running {N_SEEDS} random seeds per dataset/phenotype")
        print(f"  - Extracting top {N_FEATURES} features per run")
        print(f"  - Keeping features appearing in ≥{THRESHOLD * 100}% of runs")

        individual_results = analyze_individual_datasets(
            DATASETS,
            COMMON_PHENOTYPES,
            n_seeds=N_SEEDS,
            threshold=THRESHOLD,
            n_features=N_FEATURES,
        )

        print(f"\nCompleted analysis for individual datasets:")
        for dataset, phenotypes in individual_results.items():
            print(f"  - {dataset}: {len(phenotypes)} phenotypes")

        # Save individual results
        with open(individual_file, "w") as f:
            json.dump(individual_results, f, indent=2)
        print(f"Saved individual results to: {individual_file}")

    # Step 3.5: Analyze all datasets combined
    all_combined_file = OUTPUT_DIR / "all_datasets_combined_shap_features.json"
    ALL_DATASETS = ["atleaf", "lit", "marine", "pmi"]

    if all_combined_file.exists():
        print("\nStep 3.5: Loading existing all datasets combined results...")
        with open(all_combined_file, "r") as f:
            all_combined_results = json.load(f)
        print(
            f"Loaded results for {len(all_combined_results)} phenotypes from: {all_combined_file}"
        )
    else:
        print(f"\nStep 3.5: Analyzing all datasets combined: {ALL_DATASETS}")
        print(f"  - Only analyzing common phenotypes: {len(COMMON_PHENOTYPES)}")
        print(f"  - Running {N_SEEDS} random seeds per phenotype")
        print(f"  - Extracting top {N_FEATURES} features per run")
        print(f"  - Keeping features appearing in ≥{THRESHOLD * 100}% of runs")

        all_combined_results = analyze_all_datasets_combined(
            ALL_DATASETS,
            COMMON_PHENOTYPES,
            n_seeds=N_SEEDS,
            threshold=THRESHOLD,
            n_features=N_FEATURES,
        )

        print(
            f"\nCompleted analysis for all datasets combined: {len(all_combined_results)} phenotypes"
        )

        # Save all combined results
        with open(all_combined_file, "w") as f:
            json.dump(all_combined_results, f, indent=2)
        print(f"Saved all combined results to: {all_combined_file}")

    # Step 5: Compare features
    comparison_file = OUTPUT_DIR / "feature_comparison.json"
    summary_file = OUTPUT_DIR / "feature_comparison_summary.csv"

    if comparison_file.exists() and summary_file.exists():
        print("\nStep 5: Loading existing comparison results...")
        with open(comparison_file, "r") as f:
            comparisons = json.load(f)
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded {len(comparisons)} comparisons from: {comparison_file}")
        print(f"Loaded summary from: {summary_file}")
    else:
        print("\nStep 5: Comparing features between combined and individual models...")
        comparisons = compare_features(combined_results_filtered, individual_results)

        print(f"\nFound {len(comparisons)} valid comparisons")

        # Save comparison results
        with open(comparison_file, "w") as f:
            json.dump(comparisons, f, indent=2)
        print(f"Saved comparison results to: {comparison_file}")

        # Create summary DataFrame
        summary_data = []
        for comp_key, comp_data in comparisons.items():
            summary_data.append(
                {
                    "comparison": comp_key,
                    "phenotype": comp_data["phenotype"],
                    "test_dataset": comp_data["test_dataset"],
                    "n_combined_features": len(comp_data["combined_features"]),
                    "n_individual_features": len(comp_data["individual_features"]),
                    "n_intersection": comp_data["n_intersection"],
                    "n_unique_to_individual": comp_data["n_unique_to_individual"],
                    "n_unique_to_combined": comp_data["n_unique_to_combined"],
                    "intersection": ";".join(comp_data["intersection"]),
                    "unique_to_individual": ";".join(comp_data["unique_to_individual"]),
                    "unique_to_combined": ";".join(comp_data["unique_to_combined"]),
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to: {summary_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    print(f"\nCombined splits analyzed: {len(combined_results_filtered)}")
    print(
        f"Individual dataset/phenotype combinations: {sum(len(p) for p in individual_results.values())}"
    )
    print(f"All datasets combined phenotypes analyzed: {len(all_combined_results)}")
    print(f"Valid comparisons: {len(comparisons)}")

    if len(summary_df) > 0:
        print("\nFeature overlap statistics:")
        print(
            f"  Mean intersection size: {summary_df['n_intersection'].mean():.1f} ± {summary_df['n_intersection'].std():.1f}"
        )
        print(
            f"  Mean unique to individual: {summary_df['n_unique_to_individual'].mean():.1f} ± {summary_df['n_unique_to_individual'].std():.1f}"
        )
        print(
            f"  Mean unique to combined: {summary_df['n_unique_to_combined'].mean():.1f} ± {summary_df['n_unique_to_combined'].std():.1f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
