"""
Machine learning utilities for train-test splits.

This module provides utilities to run machine learning pipelines on
pre-generated train-test split data. It wraps trait_prediction utilities
with manuscript-specific configurations.
"""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

# Import from trait_prediction
from trait_prediction.pipeline import (
    align_columns,
    get_feature_importances,
    get_scores,
    load_single_split as _load_single_split,
    load_splits as _load_splits,
)

# Import local utilities (for model type aliases)
from scripts.ml import make_classifier

warnings.filterwarnings("ignore")


def load_single_split_data(
    folder_path: Path, feature_data: pd.DataFrame
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load train/val/test data from a single split folder.

    This is a wrapper around trait_prediction.pipeline.load_single_split
    for backward compatibility.

    Parameters
    ----------
    folder_path : Path
        Path to folder containing split files (y_train.tsv, y_val.tsv, y_test.tsv)
    feature_data : pd.DataFrame
        Feature matrix with samples as rows and features as columns

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test.
    """
    return _load_single_split(folder_path, feature_data)


def load_split_data(
    base_dir: Path = Path("data/processed/train_test_splits"),
    split_types: list[str] | None = None,
    feature_file: Path = Path("data/processed/features_reduced/combined_datasets/kofam.tsv"),
) -> dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """
    Load all train/val/test splits from the base directory.

    This is a wrapper around trait_prediction.pipeline.load_splits
    for backward compatibility.

    Parameters
    ----------
    base_dir : Path, optional
        Base directory containing split folders
    split_types : list[str] | None, optional
        List of split types to load. Options: "random_split", "dataset_split",
        "phylo_ooc", "phylo_ic". If None, loads all.
    feature_file : Path, optional
        Path to combined feature file

    Returns
    -------
    dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]
        Nested dictionary: {split_type: {phenotype_key: {X_train, y_train, ...}}}
    """
    return _load_splits(
        base_dir=base_dir,
        feature_file=feature_file,
        split_types=split_types,
        verbose=True,
    )


def perform_split_ml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "cb",
    scoring: list[str] | None = None,
    **model_kwargs,
) -> dict[str, Any]:
    """
    Perform machine learning on a single train/val/test split.

    This function trains a model on the training set with validation set for
    early stopping (if applicable), evaluates on the test set, and returns
    performance metrics along with feature importances.

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
    scoring : list[str] | None, optional
        List of scoring metrics to evaluate. If None, uses default metrics.
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    dict[str, Any]
        Dictionary containing test scores (one key per scoring metric) and
        'features' key with list of top feature names sorted by importance
    """
    if scoring is None:
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

    model = make_classifier(model_type, **model_kwargs)

    # Align columns using trait_prediction utility
    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)

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
        model.fit(X_train, y_train)

    # Get scores on test set using trait_prediction utility
    scores = get_scores(model, X_test_aligned, y_test, scoring)

    # Get feature importances using trait_prediction utility
    features = get_feature_importances(model, X_train).index.tolist()
    scores["features"] = features

    return scores


def perform_split_ml_with_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "cb",
    scoring: list[str] | None = None,
    **model_kwargs,
) -> tuple[dict[str, Any], BaseEstimator]:
    """
    Perform machine learning on a single train/val/test split and return the fitted model.

    Same as perform_split_ml but also returns the fitted classifier for saving
    (e.g. CatBoost .cbm, metadata, feature importances).

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
    scoring : list[str] | None, optional
        List of scoring metrics to evaluate. If None, uses default metrics.
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    tuple[dict[str, Any], BaseEstimator]
        (scores_dict, fitted_model). scores_dict contains test scores and 'features' key.
    """
    if scoring is None:
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

    model = make_classifier(model_type, **model_kwargs)

    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)

    if model_type == "cb":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val_aligned, y_val),
            use_best_model=True,
            verbose=False,
        )
    elif model_type == "cb_noeval":
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    scores = get_scores(model, X_test_aligned, y_test, scoring)
    # Return all features (not just top 10) so callers can save full importances
    n_feats = X_train.shape[1]
    features = get_feature_importances(model, X_train, n_features=n_feats).index.tolist()
    scores["features"] = features

    return scores, model


# Re-export for convenience
__all__ = [
    "load_single_split_data",
    "load_split_data",
    "perform_split_ml",
    "perform_split_ml_with_model",
    "align_columns",
    "get_feature_importances",
    "get_scores",
]
