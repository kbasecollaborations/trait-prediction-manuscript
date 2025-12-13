"""
Machine learning utilities for train-test splits.

This script provides utilities to run machine learning pipelines on
pre-generated train-test split data.
"""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from scripts.ml import _get_scores, get_feature_importances, make_classifier

warnings.filterwarnings("ignore")


def load_single_split_data(folder_path: Path) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load train/val/test data from a single split folder.

    Parameters
    ----------
    folder_path : Path
        Path to folder containing split files (X_train.tsv, y_train.tsv, etc.)

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test.
        Each y array is a Series (single column from the DataFrame).
    """
    data: dict[str, pd.DataFrame | pd.Series] = {}

    # Load training data
    X_train = pd.read_csv(
        folder_path / "X_train.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    y_train = pd.read_csv(
        folder_path / "y_train.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    # Convert to Series (assume single column)
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    # Load validation data
    X_val = pd.read_csv(
        folder_path / "X_val.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    y_val = pd.read_csv(
        folder_path / "y_val.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
        y_val = y_val.iloc[:, 0]

    # Load test data
    X_test = pd.read_csv(
        folder_path / "X_test.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    y_test = pd.read_csv(
        folder_path / "y_test.tsv", sep="\t", index_col=0, dtype={"genomeID": str}
    )
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    data["X_train"] = X_train
    data["y_train"] = y_train
    data["X_val"] = X_val
    data["y_val"] = y_val
    data["X_test"] = X_test
    data["y_test"] = y_test

    return data


def load_split_data(
    base_dir: Path = Path("data/processed/train_test_splits"),
    split_types: list[str] | None = None,
) -> dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """
    Load all train/val/test splits from the base directory.

    This function loads splits from all subdirectories in the base directory,
    including random_split, dataset_split, and phylogeny_split (with in-clade
    and out-of-clade variants).

    Parameters
    ----------
    base_dir : Path, optional
        Base directory containing split folders, by default
        Path("data/processed/train_test_splits")
    split_types : list[str] | None, optional
        List of split types to load. Options are: "random_split", "dataset_split",
        "phylo_ooc" (out-of-clade), "phylo_ic" (in-clade). If None, loads all
        split types, by default None

    Returns
    -------
    dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]
        Nested dictionary with structure:
        {split_type: {phenotype_key: {X_train, y_train, X_val, y_val, X_test, y_test}}}

        Where:
        - split_type: One of "random_split", "dataset_split", "phylo_ooc", "phylo_ic"
        - phenotype_key: Unique key like "Alanine_0" or "Serine_ooc_2"
        - Inner dict contains the train/val/test data from load_single_split_data()

    Examples
    --------
    Load all splits:
    >>> data = load_split_data()
    >>> print(data.keys())
    dict_keys(['random_split', 'dataset_split', 'phylo_ooc', 'phylo_ic'])

    Load only random splits:
    >>> data = load_split_data(split_types=["random_split"])
    >>> print(data.keys())
    dict_keys(['random_split'])

    Access a specific split:
    >>> random_data = data["random_split"]["Alanine_0"]
    >>> X_train = random_data["X_train"]
    """
    # Default to loading all split types
    if split_types is None:
        split_types = ["random_split", "dataset_split", "phylo_ooc", "phylo_ic"]

    result: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]] = {}

    # Load random splits
    if "random_split" in split_types:
        random_split_dir = base_dir / "random_split"
        if random_split_dir.exists():
            random_split_data = {}
            for phenotype_dir in tqdm(
                list(random_split_dir.iterdir()), desc="Loading random splits"
            ):
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                for repeat_dir in phenotype_dir.iterdir():
                    if not repeat_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_{repeat_dir.name}"
                    data = load_single_split_data(repeat_dir)
                    random_split_data[key] = data
            result["random_split"] = random_split_data

    # Load dataset splits
    if "dataset_split" in split_types:
        dataset_split_dir = base_dir / "dataset_split"
        if dataset_split_dir.exists():
            dataset_split_data = {}
            for phenotype_dir in tqdm(
                list(dataset_split_dir.iterdir()), desc="Loading dataset splits"
            ):
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                for split_dir in phenotype_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_{split_dir.name}"
                    data = load_single_split_data(split_dir)
                    dataset_split_data[key] = data
            result["dataset_split"] = dataset_split_data

    # Load phylogenetic out-of-clade splits
    if "phylo_ooc" in split_types:
        phylo_split_dir = base_dir / "phylogeny_split"
        if phylo_split_dir.exists():
            phylo_ooc_data = {}
            for phenotype_dir in tqdm(
                list(phylo_split_dir.iterdir()), desc="Loading phylo OOC splits"
            ):
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                ooc_dir = phenotype_dir / "out-of-clade"
                if not ooc_dir.exists():
                    continue
                for split_dir in ooc_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_ooc_{split_dir.name}"
                    data = load_single_split_data(split_dir)
                    phylo_ooc_data[key] = data
            result["phylo_ooc"] = phylo_ooc_data

    # Load phylogenetic in-clade splits
    if "phylo_ic" in split_types:
        phylo_split_dir = base_dir / "phylogeny_split"
        if phylo_split_dir.exists():
            phylo_ic_data = {}
            for phenotype_dir in tqdm(
                list(phylo_split_dir.iterdir()), desc="Loading phylo IC splits"
            ):
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                ic_dir = phenotype_dir / "in-clade"
                if not ic_dir.exists():
                    continue
                for split_dir in ic_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_ic_{split_dir.name}"
                    data = load_single_split_data(split_dir)
                    phylo_ic_data[key] = data
            result["phylo_ic"] = phylo_ic_data

    return result


def perform_split_ml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "cb",
    scoring: list[str] = [
        "accuracy",
        "balanced_accuracy",
        "matthews_corrcoef",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "roc_auc",
    ],
    **model_kwargs,
) -> dict[str, Any]:
    """
    Perform machine learning on a single train/val/test split.

    This function trains a model on the training set with validation set for early
    stopping (if applicable), evaluates on the test set, and returns performance
    metrics along with feature importances.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target variable vector
    X_val : pd.DataFrame
        Validation feature matrix
    y_val : pd.Series
        Validation target variable vector
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target variable vector
    model_type : str, optional
        Type of classifier model to use ('cb', 'rf', 'dt', etc.), by default "cb"
    scoring : list[str], optional
        List of scoring metrics to evaluate, by default includes accuracy,
        balanced_accuracy, matthews_corrcoef, precision, recall, f1,
        sensitivity, specificity, and roc_auc
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    dict[str, Any]
        Dictionary containing test scores (one key per scoring metric) and
        'features' key with list of top feature names sorted by importance
    """
    random_state = model_kwargs.get("random_state", 42)
    model = make_classifier(model_type, **model_kwargs)

    # Align columns between train and test/val
    # Make sure X_test and X_val have the same columns as X_train
    X_val_aligned = X_val.copy()
    missing_cols_val = X_train.columns.difference(X_val_aligned.columns)
    if len(missing_cols_val) > 0:
        missing_df = pd.DataFrame(0, index=X_val_aligned.index, columns=missing_cols_val)
        X_val_aligned = pd.concat([X_val_aligned, missing_df], axis=1)
    X_val_aligned = X_val_aligned[X_train.columns]

    X_test_aligned = X_test.copy()
    missing_cols_test = X_train.columns.difference(X_test_aligned.columns)
    if len(missing_cols_test) > 0:
        missing_df = pd.DataFrame(0, index=X_test_aligned.index, columns=missing_cols_test)
        X_test_aligned = pd.concat([X_test_aligned, missing_df], axis=1)
    X_test_aligned = X_test_aligned[X_train.columns]

    # Train model with eval_set for CatBoost models
    if model_type == "cb":
        # CatBoost with early stopping using validation set
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val_aligned, y_val),
            use_best_model=True,
            verbose=False,
        )
    elif model_type == "cb_noeval":
        # CatBoost without early stopping
        model.fit(X_train, y_train, verbose=False)
    else:
        # Other models (RF, DT, etc.)
        model.fit(X_train, y_train)  # type: ignore

    # Get scores on test set
    scores = _get_scores(model, X_test_aligned, y_test, scoring)

    # Get feature importances (returns list of feature names sorted by importance)
    features = get_feature_importances(model, X_train).index.tolist()

    scores["features"] = features

    return scores
