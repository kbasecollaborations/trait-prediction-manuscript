#!/usr/bin/env python3
"""SHAP-based feature importance data for Figure 5B (concordant samples only).

KOFAM annotations provide the feature space; GapMind is used only to stratify
samples. Each phenotype/split is screened to a broad CatBoost-important
candidate set before seeded SHAP stability analysis.
"""

import argparse
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

from scripts.io import cache_is_fresh, read_features, read_phenotypes
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data

# Thread budget per CatBoost fit / SHAP call. Default -1 (all cores) preserves
# the original Figure 5B behaviour; downstream drivers (e.g. the random-subset
# control) lower this so the many small-subset cells can be run in parallel
# without oversubscribing the machine.
_THREAD_COUNT = -1

warnings.filterwarnings("ignore")


def load_gapmind_predictions(gapmind_file: Path) -> pd.DataFrame:
    """Load GapMind predictions.

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
    """Load and combine experimental phenotype data from all datasets.

    Parameters
    ----------
    phenotype_dir : Path
        Path to the directory containing phenotype subdirectories

    Returns
    -------
    pd.DataFrame
        Combined phenotype data with genomeID as index and phenotypes as columns
    """
    phenotype_data: dict[str, list[pd.DataFrame]] = {}

    for dataset_dir in phenotype_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        for phenotype_file in dataset_dir.glob("*.tsv"):
            phenotype_name = phenotype_file.stem
            df = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})
            if phenotype_name not in phenotype_data:
                phenotype_data[phenotype_name] = []
            phenotype_data[phenotype_name].append(df)

    combined_phenotypes = {}
    for phenotype_name, df_list in phenotype_data.items():
        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.drop_duplicates(subset=["genomeID"], keep="first")
        combined = combined.set_index("genomeID")
        combined_phenotypes[phenotype_name] = combined[phenotype_name]

    experimental_data = pd.DataFrame(combined_phenotypes)

    return experimental_data


def get_concordant_samples(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> set[str]:
    """Get genome IDs where GapMind predictions match experimental data.

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
    if phenotype not in gapmind_predictions.columns:
        return set()
    if phenotype not in experimental_phenotypes.columns:
        return set()

    common_genomes = gapmind_predictions.index.intersection(
        experimental_phenotypes.index
    )

    # Restrict to genomes with non-NaN experimental data.
    exp_data = experimental_phenotypes.loc[common_genomes, phenotype]
    valid_genomes = exp_data.dropna().index

    gapmind_vals = gapmind_predictions.loc[valid_genomes, phenotype]
    exp_vals = experimental_phenotypes.loc[valid_genomes, phenotype]

    concordant_mask = gapmind_vals == exp_vals
    concordant_genomes = set(valid_genomes[concordant_mask])

    return concordant_genomes


def get_test_dataset_from_key(key: str) -> str | None:
    """Extract the test dataset name from a dataset_split key.

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


def get_shap_top_features(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
) -> list[str]:
    """Get top features by mean absolute SHAP value from a trained CatBoost model.

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
    pool = Pool(data=X, label=y)

    shap_values = model.get_feature_importance(
        data=pool,
        type="ShapValues",
        thread_count=_THREAD_COUNT,
    )

    # Drop the base-value column, then mean absolute SHAP per feature.
    shap_values = shap_values[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.Series(mean_abs_shap, index=X.columns)
    feature_importance.sort_values(ascending=False, inplace=True)

    return feature_importance.head(n_features).index.tolist()


def get_screened_feature_names(
    X: pd.DataFrame,
    y: pd.Series,
    n_candidate_features: int,
    random_state: int = 42,
) -> list[str]:
    """Select a broad KOFAM candidate set before SHAP calculation.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training labels.
    n_candidate_features : int
        Number of candidate features to retain.
    random_state : int, optional
        Random state for the screening model, by default 42.

    Returns
    -------
    list[str]
        Feature names ranked by CatBoost PredictionValuesChange importance.
    """
    model = make_classifier("cb_noeval", random_state=random_state, thread_count=_THREAD_COUNT)
    model.fit(X, y, verbose=False)

    importances = model.get_feature_importance(
        type="PredictionValuesChange",
        thread_count=_THREAD_COUNT,
    )
    feature_importance = pd.Series(importances, index=X.columns)
    feature_importance.sort_values(ascending=False, inplace=True)

    n_keep = min(n_candidate_features, len(feature_importance))
    return feature_importance.head(n_keep).index.tolist()


def get_screened_split_data(
    split_data: dict[str, pd.DataFrame | pd.Series],
    n_candidate_features: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Restrict a train/validation split to screened candidate features.

    Parameters
    ----------
    split_data : dict[str, pd.DataFrame | pd.Series]
        Split data with ``X_train``, ``X_val``, ``y_train``, and ``y_val``.
    n_candidate_features : int
        Number of candidate features to retain.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Copy of ``split_data`` with feature matrices restricted to candidates.
    """
    X_combined = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
    y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)
    candidate_features = get_screened_feature_names(
        X_combined,
        y_combined,
        n_candidate_features=n_candidate_features,
    )

    return {
        "X_train": split_data["X_train"].loc[:, candidate_features],
        "y_train": split_data["y_train"],
        "X_val": split_data["X_val"].loc[:, candidate_features],
        "y_val": split_data["y_val"],
    }


def get_consistent_features(
    feature_lists: list[list[str]], threshold: float = 0.7
) -> list[str]:
    """Get features appearing in at least the threshold proportion of runs.

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
    all_features = [feat for feat_list in feature_lists for feat in feat_list]
    feature_counts = Counter(all_features)

    min_count = int(np.ceil(len(feature_lists) * threshold))
    consistent_features = [
        feat for feat, count in feature_counts.items() if count >= min_count
    ]

    consistent_features.sort(key=lambda x: feature_counts[x], reverse=True)

    return consistent_features


def load_individual_dataset(
    dataset_name: str, phenotype: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and phenotype for an individual dataset.

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
    features_path = (
        Path("data/processed/features_reduced") / dataset_name / "kofam.tsv"
    )
    phenotype_path = (
        Path("data/processed/phenotypes") / dataset_name / f"{phenotype}.tsv"
    )

    feature_set = read_features([features_path], ftype="int")
    phenotype_set = read_phenotypes([phenotype_path])

    # DataSet handles NaN values and index alignment.
    dataset = DataSet(phenotype_set, feature_set)

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
    """Load features and phenotype from all datasets combined.

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

    feature_set = read_features(feature_files)
    phenotype_set = read_phenotypes(phenotype_files)

    # DataSet handles NaN values and index alignment.
    dataset = DataSet(phenotype_set, feature_set)

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

    X_combined = pd.concat(all_X_list, axis=0)
    y_combined = pd.concat(all_y_list, axis=0)

    X_combined = X_combined[~X_combined.index.duplicated(keep="first")]
    y_combined = y_combined[~y_combined.index.duplicated(keep="first")]

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
    """Train model on an individual dataset and get top SHAP features.

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
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=random_state, shuffle=True
    )

    model = make_classifier("cb_noeval", random_state=random_state, thread_count=_THREAD_COUNT)
    model.fit(X_train, y_train, verbose=False)

    top_features = get_shap_top_features(model, X_train, y_train, n_features=n_features)

    return top_features


def train_and_get_top_features_split(
    split_data: dict[str, pd.DataFrame | pd.Series],
    random_state: int = 42,
    n_features: int = 10,
) -> list[str]:
    """Train model on combined train+val split and get top SHAP features.

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
    X_combined = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
    y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)

    X_train, _, y_train, _ = train_test_split(
        X_combined,
        y_combined,
        train_size=0.8,
        stratify=y_combined,
        random_state=random_state,
        shuffle=True,
    )

    model = make_classifier("cb_noeval", random_state=random_state, thread_count=_THREAD_COUNT)
    model.fit(X_train, y_train, verbose=False)

    top_features = get_shap_top_features(model, X_train, y_train, n_features=n_features)

    return top_features


def analyze_combined_splits(
    splits_dir: Path,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
    n_candidate_features: int = 300,
) -> dict[str, list[str]]:
    """Analyze combined train-test splits for consistent top features (concordant only).

    Parameters
    ----------
    splits_dir : Path
        Path to train_test_splits directory
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
    n_candidate_features : int, optional
        Number of candidate KOFAM features retained before seeded SHAP analysis,
        by default 300.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys as split identifiers and values as consistent feature lists
    """
    dataset_split_dir = splits_dir / "dataset_split"

    phenotypes = [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]

    feature_file = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
    feature_data = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )

    results = {}

    for phenotype in tqdm(phenotypes, desc="Analyzing combined splits (concordant)"):
        phenotype_dir = dataset_split_dir / phenotype

        split_types = [d.name for d in phenotype_dir.iterdir() if d.is_dir()]

        for split_type in split_types:
            split_dir = phenotype_dir / split_type
            key = f"{phenotype}_{split_type}"

            try:
                split_data = load_single_split_data(split_dir, feature_data)
            except Exception as e:
                print(f"Error loading {key}: {e}")
                continue

            X_combined = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
            y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)

            concordant_genomes = get_concordant_samples(
                gapmind_predictions, experimental_phenotypes, phenotype
            )

            if len(concordant_genomes) == 0:
                continue

            # Mask, not list(set(...)): set order is per-process salted, so
            # listing it reorders rows and makes the fits irreproducible.
            concordant_mask = X_combined.index.isin(concordant_genomes)
            if concordant_mask.sum() < 20:
                continue

            X_concordant = X_combined.loc[concordant_mask]
            y_concordant = y_combined.loc[concordant_mask]

            if len(y_concordant.unique()) != 2:
                continue

            class_counts = y_concordant.value_counts()
            if class_counts.min() < 10:
                continue

            # Re-split the concordant samples into train/val (80/20).
            X_train_conc, X_val_conc, y_train_conc, y_val_conc = train_test_split(
                X_concordant,
                y_concordant,
                train_size=0.8,
                stratify=y_concordant,
                random_state=42,
                shuffle=True,
            )

            filtered_split_data = {
                "X_train": X_train_conc,
                "y_train": y_train_conc,
                "X_val": X_val_conc,
                "y_val": y_val_conc,
            }

            filtered_split_data = get_screened_split_data(
                filtered_split_data,
                n_candidate_features=n_candidate_features,
            )

            feature_lists = []
            for seed in range(n_seeds):
                try:
                    top_features = train_and_get_top_features_split(
                        filtered_split_data, random_state=seed, n_features=n_features
                    )
                    feature_lists.append(top_features)
                except Exception as e:
                    print(f"Error in {key} with seed {seed}: {e}")
                    continue

            if len(feature_lists) > 0:
                consistent_features = get_consistent_features(
                    feature_lists, threshold=threshold
                )
                results[key] = consistent_features

    return results


def analyze_individual_datasets(
    datasets: Sequence[str],
    phenotypes: Sequence[str],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    n_seeds: int = 20,
    threshold: float = 0.7,
    n_features: int = 10,
    n_candidate_features: int = 300,
) -> dict[str, dict[str, list[str]]]:
    """Analyze individual datasets for consistent top features (concordant only).

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
    n_candidate_features : int, optional
        Number of candidate KOFAM features retained before seeded SHAP analysis,
        by default 300.

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
                X, y = load_individual_dataset(dataset, phenotype)

                concordant_genomes = get_concordant_samples(
                    gapmind_predictions, experimental_phenotypes, phenotype
                )

                if len(concordant_genomes) == 0:
                    continue

                # Mask, not list(set(...)): see the note above.
                concordant_mask = X.index.isin(concordant_genomes)
                if concordant_mask.sum() < 20:
                    continue

                X_concordant = X.loc[concordant_mask]
                y_concordant = y.loc[concordant_mask]

                if len(y_concordant.unique()) != 2:
                    continue

                class_counts = y_concordant.value_counts()
                if class_counts.min() < 10:
                    continue

                candidate_features = get_screened_feature_names(
                    X_concordant,
                    y_concordant,
                    n_candidate_features=n_candidate_features,
                )
                X_concordant = X_concordant.loc[:, candidate_features]

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
    n_candidate_features: int = 300,
) -> dict[str, list[str]]:
    """Analyze all datasets combined for consistent top features (concordant only).

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
    n_candidate_features : int, optional
        Number of candidate KOFAM features retained before seeded SHAP analysis,
        by default 300.

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
            X, y = load_all_datasets_combined(datasets, phenotype)

            concordant_genomes = get_concordant_samples(
                gapmind_predictions, experimental_phenotypes, phenotype
            )

            if len(concordant_genomes) == 0:
                continue

            # Mask, not list(set(...)): see the note above.
            concordant_mask = X.index.isin(concordant_genomes)
            if concordant_mask.sum() < 20:
                continue

            X_concordant = X.loc[concordant_mask]
            y_concordant = y.loc[concordant_mask]

            if len(y_concordant.unique()) != 2:
                continue

            class_counts = y_concordant.value_counts()
            if class_counts.min() < 10:
                continue

            candidate_features = get_screened_feature_names(
                X_concordant,
                y_concordant,
                n_candidate_features=n_candidate_features,
            )
            X_concordant = X_concordant.loc[:, candidate_features]

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

            if len(feature_lists) > 0:
                consistent_features = get_consistent_features(
                    feature_lists, threshold=threshold
                )
                results[phenotype] = consistent_features

        except Exception as e:
            print(f"Error loading all_combined/{phenotype}: {e}")
            continue

    return results


def _kos_to_clusters(
    kos: list[str], ko_to_cluster: dict[str, int] | None
) -> set[int | str]:
    """Map a list of KOs to the set of redundancy clusters they belong to.

    Parameters
    ----------
    kos : list[str]
        KO identifiers.
    ko_to_cluster : dict[str, int] | None
        Per-phenotype mapping from KO to cluster ID. If ``None``, return an
        empty set.

    Returns
    -------
    set[int | str]
        Cluster identifiers represented in ``kos``. KOs absent from the
        mapping fall back to a singleton string identifier so they remain
        comparable across lists.
    """
    if ko_to_cluster is None:
        return set()
    out: set[int | str] = set()
    for ko in kos:
        if ko in ko_to_cluster:
            out.add(int(ko_to_cluster[ko]))
        else:
            out.add(f"singleton:{ko}")
    return out


def compare_features(
    combined_results: dict[str, list[str]],
    individual_results: dict[str, dict[str, list[str]]],
    ko_clusters_by_phenotype: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Compare features between combined train-test splits and individual dataset models.

    For each combination where the dataset was not used in training the combined model,
    find the intersection and unique features at both the KO level and (when a
    cluster mapping is supplied) at the redundancy-cluster level.

    Parameters
    ----------
    combined_results : dict[str, list[str]]
        Results from combined splits: {key: [features]}
    individual_results : dict[str, dict[str, list[str]]]
        Results from individual datasets: {dataset: {phenotype: [features]}}
    ko_clusters_by_phenotype : dict[str, dict[str, int]] | None, optional
        Per-phenotype mapping from KO identifier to integer cluster ID, as
        produced by ``scripts/feature_clustering.py``. When supplied, every
        comparison row gains cluster-level counts.

    Returns
    -------
    pd.DataFrame
        Comparison summary DataFrame with KO-level columns plus, when a
        cluster mapping is supplied, ``n_intersection_clusters``,
        ``n_unique_to_individual_clusters``, ``n_unique_to_combined_clusters``,
        ``n_combined_clusters``, ``n_individual_clusters``.
    """
    summary_data = []

    for combined_key, combined_features in combined_results.items():
        phenotype = combined_key.split("_")[0]
        test_dataset = get_test_dataset_from_key(combined_key)

        if test_dataset is None:
            continue

        if test_dataset not in ["atleaf", "lit", "marine"]:
            continue

        if (
            test_dataset in individual_results
            and phenotype in individual_results[test_dataset]
        ):
            individual_features = individual_results[test_dataset][phenotype]

            combined_set = set(combined_features)
            individual_set = set(individual_features)

            intersection = sorted(list(combined_set & individual_set))
            unique_to_individual = sorted(list(individual_set - combined_set))
            unique_to_combined = sorted(list(combined_set - individual_set))

            comparison_key = f"{phenotype}_{test_dataset}"
            row: dict[str, Any] = {
                "comparison": comparison_key,
                "phenotype": phenotype,
                "test_dataset": test_dataset,
                "n_combined_features": len(combined_features),
                "n_individual_features": len(individual_features),
                "n_intersection": len(intersection),
                "n_unique_to_individual": len(unique_to_individual),
                "n_unique_to_combined": len(unique_to_combined),
                "intersection": ";".join(intersection),
                "unique_to_individual": ";".join(unique_to_individual),
                "unique_to_combined": ";".join(unique_to_combined),
            }

            if ko_clusters_by_phenotype is not None:
                ko_to_cluster = ko_clusters_by_phenotype.get(phenotype)
                combined_clusters = _kos_to_clusters(combined_features, ko_to_cluster)
                individual_clusters = _kos_to_clusters(
                    individual_features, ko_to_cluster
                )
                cluster_intersection = combined_clusters & individual_clusters
                cluster_unique_individual = individual_clusters - combined_clusters
                cluster_unique_combined = combined_clusters - individual_clusters
                row.update(
                    {
                        "n_intersection_clusters": len(cluster_intersection),
                        "n_unique_to_individual_clusters": len(
                            cluster_unique_individual
                        ),
                        "n_unique_to_combined_clusters": len(cluster_unique_combined),
                        "n_combined_clusters": len(combined_clusters),
                        "n_individual_clusters": len(individual_clusters),
                    }
                )

            summary_data.append(row)

    return pd.DataFrame(summary_data)


def main() -> None:
    """Generate Figure 5B data (concordant samples only).

    SHAP results are cached as JSON. The cache is reused only when it post-dates the
    phenotype labels and splits it derives from; pass ``--fresh`` to ignore it.
    """
    parser = argparse.ArgumentParser(description="Figure 5B concordant SHAP analysis")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore cached SHAP JSONs and recompute from scratch")
    args = parser.parse_args()

    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    PHENOTYPE_DIR = Path("data/processed/phenotypes")

    N_SEEDS = 20
    THRESHOLD = 0.7
    N_FEATURES = 10
    N_CANDIDATE_FEATURES = 300
    DATASETS = ["atleaf", "lit", "marine"]

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(f"  Loaded {len(gapmind_predictions)} genomes")

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"  Loaded {len(experimental_phenotypes)} genomes")

    dataset_split_dir = SPLITS_DIR / "dataset_split"
    COMMON_PHENOTYPES = sorted(
        [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]
    )
    print(f"\nCommon phenotypes to analyze: {len(COMMON_PHENOTYPES)}")

    print("=" * 80)
    print("Figure 5B: SHAP-based Feature Importance (Concordant Samples Only)")
    print("=" * 80)

    combined_file = OUTPUT_DIR / "figure5b_combined_splits_shap_features.json"

    if not args.fresh and cache_is_fresh(combined_file, PHENOTYPE_DIR, SPLITS_DIR):
        print("\nStep 1: Loading existing combined splits results (concordant)...")
        with open(combined_file, "r") as f:
            combined_results_filtered = json.load(f)
        print(f"  Loaded {len(combined_results_filtered)} combined splits from: {combined_file}")
    else:
        print("\nStep 1: Analyzing combined train-test splits (concordant samples)...")
        print(f"  - Running {N_SEEDS} random seeds per split")
        print(f"  - Screening to top {N_CANDIDATE_FEATURES} KOFAM candidates per split")
        print(f"  - Extracting top {N_FEATURES} features per run")
        print(f"  - Keeping features appearing in >={THRESHOLD * 100}% of runs")

        combined_results = analyze_combined_splits(
            SPLITS_DIR,
            gapmind_predictions,
            experimental_phenotypes,
            n_seeds=N_SEEDS,
            threshold=THRESHOLD,
            n_features=N_FEATURES,
            n_candidate_features=N_CANDIDATE_FEATURES,
        )

        combined_results_filtered = {
            k: v
            for k, v in combined_results.items()
            if k.split("_")[0] in COMMON_PHENOTYPES
        }

        print(
            f"\nCompleted analysis for {len(combined_results_filtered)} combined splits (concordant samples)"
        )

        with open(combined_file, "w") as f:
            json.dump(combined_results_filtered, f, indent=2)
        print(f"Saved combined results to: {combined_file}")

    individual_file = OUTPUT_DIR / "figure5b_individual_datasets_shap_features.json"

    if not args.fresh and cache_is_fresh(individual_file, PHENOTYPE_DIR, SPLITS_DIR):
        print("\nStep 2: Loading existing individual datasets results (concordant)...")
        with open(individual_file, "r") as f:
            individual_results = json.load(f)
        print(f"  Loaded results for {len(individual_results)} datasets from: {individual_file}")
        for dataset, phenotypes in individual_results.items():
            print(f"  - {dataset}: {len(phenotypes)} phenotypes")
    else:
        print(f"\nStep 2: Analyzing individual datasets: {DATASETS}")
        print(f"  - Concordant samples only")
        print(f"  - Only analyzing common phenotypes: {len(COMMON_PHENOTYPES)}")
        print(f"  - Running {N_SEEDS} random seeds per dataset/phenotype")
        print(f"  - Screening to top {N_CANDIDATE_FEATURES} KOFAM candidates per dataset/phenotype")
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
            n_candidate_features=N_CANDIDATE_FEATURES,
        )

        print(f"\nCompleted analysis for individual datasets:")
        for dataset, phenotypes in individual_results.items():
            print(f"  - {dataset}: {len(phenotypes)} phenotypes")

        with open(individual_file, "w") as f:
            json.dump(individual_results, f, indent=2)
        print(f"Saved individual results to: {individual_file}")

    # Always re-run the comparison step (cheap) so cluster-mapping updates propagate.
    comparison_file = OUTPUT_DIR / "figure5b_feature_comparison_summary.csv"

    cluster_file = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")
    if cluster_file.exists():
        with open(cluster_file, "r") as f:
            ko_clusters_by_phenotype = json.load(f)
        print(
            f"\nStep 3: Loaded cluster mapping for {len(ko_clusters_by_phenotype)} phenotypes"
        )
    else:
        ko_clusters_by_phenotype = None
        print(f"\nStep 3: cluster mapping not found at {cluster_file}; skipping cluster columns")

    print("\nStep 3: Comparing features between combined splits and individual models...")
    summary_df = compare_features(
        combined_results_filtered,
        individual_results,
        ko_clusters_by_phenotype=ko_clusters_by_phenotype,
    )

    print(f"\nFound {len(summary_df)} valid comparisons")

    summary_df.to_csv(comparison_file, index=False)
    print(f"Saved comparison summary to: {comparison_file}")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    print(f"\nCombined splits analyzed (concordant): {len(combined_results_filtered)}")
    print(
        f"Individual dataset/phenotype combinations: {sum(len(p) for p in individual_results.values())}"
    )
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
