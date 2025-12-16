#!/usr/bin/env python3
"""
Generate SHAP-based feature importance data for Figure 5B (concordant samples only).

This script is similar to Figure 4C analysis but only uses GapMind-concordant samples.
It performs two main analyses:
1. Trains ML models on combined train-test splits (concordant samples only) and identifies
   consistent top features using SHAP values across multiple random seeds.
2. Trains ML models on individual datasets (concordant samples only) and identifies
   consistent top features using SHAP values across multiple random seeds.

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

warnings.filterwarnings("ignore")


def load_gapmind_predictions(gapmind_file: Path) -> pd.DataFrame:
    """
    Load GapMind predictions.

    Parameters
    ----------
    gapmind_file : Path
        Path to GapMind predictions TSV file

    Returns
    -------
    pd.DataFrame
        GapMind predictions with genomeID as index and phenotypes as columns
    """
    gapmind_df = pd.read_csv(
        gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    return gapmind_df


def load_experimental_phenotypes(phenotype_dir: Path) -> pd.DataFrame:
    """
    Load and combine experimental phenotype data from all datasets.

    Parameters
    ----------
    phenotype_dir : Path
        Path to the directory containing phenotype subdirectories

    Returns
    -------
    pd.DataFrame
        Combined phenotype data with genomeID as index and phenotypes as columns
    """
    # Dictionary to store data for each phenotype across all datasets
    phenotype_data: dict[str, list[pd.DataFrame]] = {}

    # Iterate through each dataset directory
    for dataset_dir in phenotype_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Load each phenotype file in the dataset
        for phenotype_file in dataset_dir.glob("*.tsv"):
            phenotype_name = phenotype_file.stem

            # Read the phenotype data
            df = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})

            # Skip if phenotype not already in dictionary
            if phenotype_name not in phenotype_data:
                phenotype_data[phenotype_name] = []

            # Add this dataset's data
            phenotype_data[phenotype_name].append(df)

    # Combine all datasets for each phenotype
    combined_phenotypes = {}
    for phenotype_name, df_list in phenotype_data.items():
        # Concatenate all datasets
        combined = pd.concat(df_list, ignore_index=True)

        # Remove duplicates, keeping the first occurrence
        combined = combined.drop_duplicates(subset=["genomeID"], keep="first")

        # Set genomeID as index
        combined = combined.set_index("genomeID")

        combined_phenotypes[phenotype_name] = combined[phenotype_name]

    # Create a single DataFrame with all phenotypes
    experimental_data = pd.DataFrame(combined_phenotypes)

    return experimental_data


def get_concordant_samples(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> set[str]:
    """
    Get set of genome IDs where GapMind predictions match experimental data.

    Parameters
    ----------
    gapmind_predictions : pd.DataFrame
        GapMind predictions
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype data
    phenotype : str
        Phenotype name to check concordance for

    Returns
    -------
    set[str]
        Set of genome IDs with concordant predictions
    """
    # Check if phenotype exists in both datasets
    if phenotype not in gapmind_predictions.columns:
        return set()
    if phenotype not in experimental_phenotypes.columns:
        return set()

    # Get common genomes
    common_genomes = gapmind_predictions.index.intersection(
        experimental_phenotypes.index
    )

    # Filter to genomes with non-NaN experimental data
    exp_data = experimental_phenotypes.loc[common_genomes, phenotype]
    valid_genomes = exp_data.dropna().index

    # Get predictions for valid genomes
    gapmind_vals = gapmind_predictions.loc[valid_genomes, phenotype]
    exp_vals = experimental_phenotypes.loc[valid_genomes, phenotype]

    # Find concordant samples (where predictions match experimental data)
    concordant_mask = gapmind_vals == exp_vals
    concordant_genomes = set(valid_genomes[concordant_mask])

    return concordant_genomes


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
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")

    # Remove last column (base value) and take mean absolute SHAP value per feature
    shap_values = shap_values[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create Series and sort by importance
    feature_importance = pd.Series(mean_abs_shap, index=X.columns)
    feature_importance.sort_values(ascending=False, inplace=True)

    return feature_importance.head(n_features).index.tolist()


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
        List of features appearing in >= threshold proportion of runs
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
    features_path = Path("data/processed/features_reduced") / dataset_name / "gapmind.tsv"
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
            Path("data/processed/features_reduced") / dataset_name / "gapmind.tsv"
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


def analyze_individual_datasets(
    datasets: Sequence[str],
    phenotypes: Sequence[str],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
) -> dict[str, dict[str, list[str]]]:
    """
    Analyze individual datasets and get consistent top features (concordant samples only).

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names (atleaf, lit, marine)
    phenotypes : Sequence[str]
        List of phenotype names to analyze
    gapmind_predictions : pd.DataFrame
        GapMind predictions
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype data
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

    for dataset in tqdm(datasets, desc="Analyzing individual datasets (concordant)"):
        dataset_results: dict[str, list[str]] = {}

        for phenotype in tqdm(phenotypes, desc=f"Processing {dataset}", leave=False):
            try:
                # Load data
                X, y = load_individual_dataset(dataset, phenotype)

                # Get concordant samples
                concordant_genomes = get_concordant_samples(
                    gapmind_predictions, experimental_phenotypes, phenotype
                )

                if len(concordant_genomes) == 0:
                    continue

                # Filter to concordant samples
                concordant_in_data = set(X.index) & concordant_genomes
                if len(concordant_in_data) < 20:  # Need minimum samples
                    continue

                X_concordant = X.loc[list(concordant_in_data)]
                y_concordant = y.loc[list(concordant_in_data)]

                # Check if we have enough samples for both classes
                if len(y_concordant.unique()) != 2:
                    continue

                # Check minimum samples per class
                class_counts = y_concordant.value_counts()
                if class_counts.min() < 10:
                    continue

                # Run for multiple seeds
                feature_lists = []
                for seed in range(n_seeds):
                    try:
                        top_features = train_and_get_top_features_individual(
                            X_concordant,
                            y_concordant,
                            random_state=seed,
                            n_features=n_features,
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
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
) -> dict[str, list[str]]:
    """
    Analyze all datasets combined and get consistent top features (concordant samples only).

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names (atleaf, lit, marine, pmi)
    phenotypes : Sequence[str]
        List of phenotype names to analyze
    gapmind_predictions : pd.DataFrame
        GapMind predictions
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype data
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

    for phenotype in tqdm(
        phenotypes, desc="Analyzing all datasets combined (concordant)"
    ):
        try:
            # Load data from all datasets combined
            X, y = load_all_datasets_combined(datasets, phenotype)

            # Get concordant samples
            concordant_genomes = get_concordant_samples(
                gapmind_predictions, experimental_phenotypes, phenotype
            )

            if len(concordant_genomes) == 0:
                continue

            # Filter to concordant samples
            concordant_in_data = set(X.index) & concordant_genomes
            if len(concordant_in_data) < 20:  # Need minimum samples
                continue

            X_concordant = X.loc[list(concordant_in_data)]
            y_concordant = y.loc[list(concordant_in_data)]

            # Check if we have enough samples for both classes
            if len(y_concordant.unique()) != 2:
                continue

            # Check minimum samples per class
            class_counts = y_concordant.value_counts()
            if class_counts.min() < 10:
                continue

            # Run for multiple seeds
            feature_lists = []
            for seed in range(n_seeds):
                try:
                    top_features = train_and_get_top_features_individual(
                        X_concordant,
                        y_concordant,
                        random_state=seed,
                        n_features=n_features,
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
) -> pd.DataFrame:
    """
    Compare features between combined and individual dataset models.

    Parameters
    ----------
    combined_results : dict[str, list[str]]
        Results from all datasets combined: {phenotype: [features]}
    individual_results : dict[str, dict[str, list[str]]]
        Results from individual datasets: {dataset: {phenotype: [features]}}

    Returns
    -------
    pd.DataFrame
        Comparison summary DataFrame
    """
    summary_data = []

    for dataset in individual_results:
        for phenotype in individual_results[dataset]:
            if phenotype not in combined_results:
                continue

            combined_features = combined_results[phenotype]
            individual_features = individual_results[dataset][phenotype]

            # Calculate intersection and unique features
            combined_set = set(combined_features)
            individual_set = set(individual_features)

            intersection = sorted(list(combined_set & individual_set))
            unique_to_individual = sorted(list(individual_set - combined_set))
            unique_to_combined = sorted(list(combined_set - individual_set))

            summary_data.append(
                {
                    "comparison": f"{phenotype}_{dataset}",
                    "phenotype": phenotype,
                    "test_dataset": dataset,
                    "n_combined_features": len(combined_features),
                    "n_individual_features": len(individual_features),
                    "n_intersection": len(intersection),
                    "n_unique_to_individual": len(unique_to_individual),
                    "n_unique_to_combined": len(unique_to_combined),
                    "intersection": ";".join(intersection),
                    "unique_to_individual": ";".join(unique_to_individual),
                    "unique_to_combined": ";".join(unique_to_combined),
                }
            )

    return pd.DataFrame(summary_data)


def main() -> None:
    """
    Main function to generate Figure 5B data (concordant samples only).
    """
    # Define paths
    OUTPUT_DIR = Path("data/outputs/figure5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    PHENOTYPE_DIR = Path("data/processed/phenotypes")

    # Parameters
    N_SEEDS = 5  # Reduced for faster execution
    THRESHOLD = 0.7
    N_FEATURES = 10
    DATASETS = ["atleaf", "lit", "marine"]
    ALL_DATASETS = ["atleaf", "lit", "marine", "pmi"]

    # Load GapMind predictions and experimental phenotypes
    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(f"  Loaded {len(gapmind_predictions)} genomes")

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"  Loaded {len(experimental_phenotypes)} genomes")

    # Get common phenotypes
    COMMON_PHENOTYPES = sorted(
        list(set(gapmind_predictions.columns) & set(experimental_phenotypes.columns))
    )
    print(f"\nCommon phenotypes to analyze: {len(COMMON_PHENOTYPES)}")

    print("=" * 80)
    print("Figure 5B: SHAP-based Feature Importance (Concordant Samples Only)")
    print("=" * 80)

    # Analyze individual datasets
    individual_file = OUTPUT_DIR / "figure5b_individual_datasets_shap_features.json"

    print(f"\nAnalyzing individual datasets: {DATASETS}")
    print(f"  - Concordant samples only")
    print(f"  - Running {N_SEEDS} random seeds per dataset/phenotype")
    print(f"  - Extracting top {N_FEATURES} features per run")
    print(f"  - Keeping features appearing in >={THRESHOLD * 100}% of runs")

    individual_results = analyze_individual_datasets(
        DATASETS,
        COMMON_PHENOTYPES,
        gapmind_predictions,
        experimental_phenotypes,
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

    # Analyze all datasets combined
    all_combined_file = OUTPUT_DIR / "figure5b_all_datasets_combined_shap_features.json"

    print(f"\nAnalyzing all datasets combined: {ALL_DATASETS}")
    print(f"  - Concordant samples only")
    print(f"  - Running {N_SEEDS} random seeds per phenotype")
    print(f"  - Extracting top {N_FEATURES} features per run")
    print(f"  - Keeping features appearing in >={THRESHOLD * 100}% of runs")

    all_combined_results = analyze_all_datasets_combined(
        ALL_DATASETS,
        COMMON_PHENOTYPES,
        gapmind_predictions,
        experimental_phenotypes,
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

    # Compare features
    comparison_file = OUTPUT_DIR / "figure5b_feature_comparison_summary.csv"

    print("\nComparing features between combined and individual models...")
    summary_df = compare_features(all_combined_results, individual_results)

    print(f"\nFound {len(summary_df)} valid comparisons")

    summary_df.to_csv(comparison_file, index=False)
    print(f"Saved comparison summary to: {comparison_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    print(
        f"\nIndividual dataset/phenotype combinations: {sum(len(p) for p in individual_results.values())}"
    )
    print(f"All datasets combined phenotypes analyzed: {len(all_combined_results)}")
    print(f"Valid comparisons: {len(summary_df)}")

    if len(summary_df) > 0:
        print("\nFeature overlap statistics:")
        print(
            f"  Mean intersection size: {summary_df['n_intersection'].mean():.1f} +/- {summary_df['n_intersection'].std():.1f}"
        )
        print(
            f"  Mean unique to individual: {summary_df['n_unique_to_individual'].mean():.1f} +/- {summary_df['n_unique_to_individual'].std():.1f}"
        )
        print(
            f"  Mean unique to combined: {summary_df['n_unique_to_combined'].mean():.1f} +/- {summary_df['n_unique_to_combined'].std():.1f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
