"""Classifier construction and evaluation helpers (manuscript model-type aliases)."""

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

from trait_prediction.classifiers import make_classifier as _make_classifier
from trait_prediction.pipeline import (
    align_columns,
    get_feature_importances,
    get_scores as _get_scores,
)

sklearn.set_config(enable_metadata_routing=True)  # type: ignore

# Manuscript uses "cb" but trait_prediction uses "catboost"
MODEL_TYPE_ALIASES = {
    "cb": "catboost",
    "cb_noeval": "catboost_noeval",
    "rfe_cb": "rfe_catboost",
}


def make_classifier(model_type: str, **kwargs) -> BaseEstimator:
    """
    Create a classifier with default parameters that can be overridden.

    Parameters
    ----------
    model_type : str
        Type of classifier to create. Must be one of:
        - 'rf': Random Forest classifier
        - 'cb': CatBoost classifier (with early stopping)
        - 'cb_noeval': CatBoost classifier (without early stopping)
        - 'dt': Decision Tree classifier
        - 'rfe_rf': Random Forest with Recursive Feature Elimination
        - 'rfe_cb': CatBoost with Recursive Feature Elimination
        - 'rfe_cv': Cross-validated RFE with Random Forest
    **kwargs : dict
        Additional keyword arguments to override default parameters

    Returns
    -------
    BaseEstimator
        Configured classifier instance

    Raises
    ------
    ValueError
        If model_type is not one of the supported values
    """
    mapped_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
    return _make_classifier(mapped_type, **kwargs)


def perform_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_splits: int = 5,
    minority_class_min_samples: int = 10,
    scoring: list[str] | None = None,
    **model_kwargs,
) -> pd.DataFrame | None:
    """
    Perform cross-validation on a classifier model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable vector
    model_type : str
        Type of classifier model to use ('rf', 'cb', 'dt', etc.)
    n_splits : int, optional
        Number of folds for cross-validation, by default 5
    minority_class_min_samples : int, optional
        Minimum samples required in minority class, by default 10
    scoring : list[str] | None, optional
        List of scoring metrics to evaluate. If None, uses default metrics.
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    pd.DataFrame | None
        DataFrame containing cross-validation results with columns for each
        scoring metric, fold number, and top features. Returns None if
        insufficient data.
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

    random_state = model_kwargs.get("random_state", 42)
    model = make_classifier(model_type, **model_kwargs)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Bail out if any fold lacks two classes or enough minority samples
    for train_idx, test_idx in kfold.split(X, y):
        y_train_fold = y.iloc[train_idx]
        train_class_counts = y_train_fold.value_counts()

        if len(train_class_counts) != 2:
            return None

        if train_class_counts.min() < minority_class_min_samples:
            return None

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=["accuracy"],
        return_estimator=True,
    )

    all_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        estimator = cv_results["estimator"][fold_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        fold_scores = _get_scores(estimator, X_test_fold, y_test_fold, scoring)
        fold_scores["fold"] = fold_idx
        fold_scores["features"] = get_feature_importances(estimator, X).index.tolist()
        all_scores.append(fold_scores)

    return pd.DataFrame(all_scores)


def perform_train_test(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    model_type: str,
    test_size: int | None = None,
    scoring: list[str] | None = None,
    n_repeats: int | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Perform train/test evaluation on a classifier model.

    Parameters
    ----------
    train_X : pd.DataFrame
        Training feature matrix
    train_y : pd.Series
        Training target variable vector
    test_X : pd.DataFrame
        Test feature matrix
    test_y : pd.Series
        Test target variable vector
    model_type : str
        Type of classifier model to use ('rf', 'cb', 'dt', etc.)
    test_size : int | None, optional
        Number of test samples to use. If None, uses all test samples.
    scoring : list[str] | None, optional
        List of scoring metrics. If None, uses default metrics.
    n_repeats : int | None, optional
        Number of repeated evaluations with different test subsamples.
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation results
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
    random_state = model_kwargs.get("random_state", 42)

    X_train = train_X.copy()
    y_train = train_y.copy()
    X_test = test_X.copy()
    y_test = test_y.copy()

    X_test = align_columns(X_train, X_test)

    model.fit(X_train, y_train)

    results = []
    if n_repeats is None:
        if test_size is None:
            test_indices = y_test.index
        else:
            test_indices = y_test.sample(
                n=test_size, replace=False, random_state=random_state
            ).index
        scores = _get_scores(
            model, X_test.loc[test_indices, :], y_test.loc[test_indices], scoring
        )
        results.append({"repeat": 0, **scores})
    else:
        for i in range(n_repeats):
            if test_size is None:
                test_indices = y_test.index
            else:
                test_indices = y_test.sample(
                    n=test_size, replace=False, random_state=(random_state + i)
                ).index
            X_test_sample = X_test.loc[test_indices, :]
            y_test_sample = y_test.loc[test_indices]
            scores = _get_scores(model, X_test_sample, y_test_sample, scoring)
            results.append({"repeat": i, **scores})

    return pd.DataFrame(results)


__all__ = [
    "make_classifier",
    "get_feature_importances",
    "perform_cv",
    "perform_train_test",
    "align_columns",
    "_get_scores",
]

_get_scores = _get_scores
