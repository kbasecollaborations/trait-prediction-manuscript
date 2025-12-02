"""
Machine learning utilities
"""

import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

sklearn.set_config(enable_metadata_routing=True)  # type: ignore


def make_classifier(model_type: str, **kwargs) -> BaseEstimator:
    """
    Create a classifier with default parameters that can be overridden.

    Parameters
    ----------
    model_type : str
        Type of classifier to create. Must be one of:
        - 'rf': Random Forest classifier
        - 'cb': CatBoost classifier
        - 'rfe_rf': Random Forest classifier with Recursive Feature Elimination
        - 'rfe_cb': CatBoost classifier with Recursive Feature Elimination
        - 'rfe_cv': Cross-validation of a classifier with Recursive Feature Elimination
        - 'dt': Decision Tree classifier
    **kwargs : dict
        Additional keyword arguments to override default parameters for the classifier

    Returns
    -------
    RandomForestClassifier | CatBoostClassifier | RFE
        Configured classifier instance

    Raises
    ------
    ValueError
        If model_type is not one of the supported values
    """
    if model_type == "rf":
        default_kwargs = {
            "n_estimators": 1000,
            "random_state": 42,
            "n_jobs": -1,
            "max_features": None,  # type: ignore
        }
        updated_kwargs = {**default_kwargs, **kwargs}
        return RandomForestClassifier(**updated_kwargs)
    elif model_type == "dt":
        default_kwargs = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        updated_kwargs = {**default_kwargs, **kwargs}
        return DecisionTreeClassifier(**updated_kwargs)
    elif model_type == "cb":
        default_kwargs = {
            "random_state": 42,
            "thread_count": -1,
            "eval_metric": "BalancedAccuracy",
            "od_type": "IncToDec",
            "od_wait": 20,
            "task_type": "CPU",
            "verbose": False,
            "allow_writing_files": False,
        }
        updated_kwargs = {**default_kwargs, **kwargs}
        return CatBoostClassifier(**updated_kwargs)  # type: ignore
    elif model_type == "rfe_rf":
        default_kwargs = {
            "step": 0.1,
            "n_features_to_select": 100,
            "verbose": 0,
        }
        random_state = kwargs.pop("random_state", 42)
        updated_kwargs = {**default_kwargs, **kwargs}
        return RFE(
            estimator=make_classifier("rf", random_state=random_state), **updated_kwargs
        )
    elif model_type == "rfe_cb":
        default_kwargs = {
            "step": 0.1,
            "n_features_to_select": 100,
            "verbose": 0,
        }
        random_state = kwargs.pop("random_state", 42)
        updated_kwargs = {**default_kwargs, **kwargs}
        return RFE(
            estimator=make_classifier("cb", random_state=random_state), **updated_kwargs
        )
    elif model_type == "rfe_cv":
        random_state = kwargs.pop("random_state", 42)
        n_splits = kwargs.pop("n_splits", 5)
        default_kwargs = {
            "step": 0.1,
            "min_features_to_select": 100,
            "scoring": "balanced_accuracy",
            "verbose": 0,
            "n_jobs": -1,
        }
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        updated_kwargs = {**default_kwargs, **kwargs}
        return RFECV(
            estimator=make_classifier("rf", random_state=random_state),
            cv=cv,
            **updated_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_importances(
    model: RandomForestClassifier | CatBoostClassifier | RFE,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    n_features: int = 10,
) -> pd.Series:
    """
    Get the most important features from a trained model.

    Parameters
    ----------
    model : RandomForestClassifier | CatBoostClassifier
        Trained model to extract feature importances from
    X : pd.DataFrame
        Feature matrix used to train the model
    feature_names : list[str] | None, optional
        List of feature names. If None, uses X.columns, by default None
    n_features : int, optional
        Number of top features to return, by default 10

    Returns
    -------
    list[str]
        List of the top `n_features` most important features
    """
    if isinstance(model, RFE):
        estimator: RandomForestClassifier | CatBoostClassifier = model.estimator_  # type: ignore
        if feature_names is not None:
            raise ValueError("feature_names must be None for RFE models")
        feature_names = list(model.get_feature_names_out())
    else:
        estimator = model
    importances = estimator.feature_importances_
    if feature_names is None:
        feature_names = list(X.columns)
    importance_df = pd.Series(importances, index=feature_names)
    importance_df.sort_values(ascending=False, inplace=True)
    return importance_df.head(n_features)


def perform_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_splits: int = 5,
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
) -> pd.DataFrame:
    """
    Perform cross-validation on a classifier model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable vector
    model_type : str
        Type of classifier model to use
        - 'rf': Random Forest classifier
        - 'cb': CatBoost classifier
        - 'rfe_rf': Random Forest classifier with Recursive Feature Elimination
        - 'rfe_cb': CatBoost classifier with Recursive Feature Elimination
        - 'dt': Decision Tree classifier
    n_splits : int, optional
        Number of folds for cross-validation, by default 5
    scoring : list[str], optional
        List of scoring metrics to evaluate, by default ["accuracy", "balanced_accuracy", "matthews_corrcoef"]
    **model_kwargs
        Additional keyword arguments passed to the classifier

    Returns
    -------
    pd.DataFrame
        DataFrame containing cross-validation results with columns for each scoring metric,
        fold number, and top features for each fold
    """
    random_state = model_kwargs.get("random_state", 42)
    model = make_classifier(model_type, **model_kwargs)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=["accuracy"],  # Use a simple scoring metric for cross_validate
        return_estimator=True,
    )
    all_scores = []
    top_features_list: list[list[str]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        estimator = cv_results["estimator"][fold_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        fold_scores = _get_scores(estimator, X_test_fold, y_test_fold, scoring)
        fold_scores["fold"] = fold_idx
        all_scores.append(fold_scores)
        top_features_list.append(get_feature_importances(estimator, X))
    results_df = pd.DataFrame(all_scores)
    results_df["features"] = top_features_list
    return results_df


def _get_scores(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, scoring: list[str]
) -> dict[str, float]:
    results = dict()
    for scoring_name in scoring:
        if scoring_name == "sensitivity":
            scorer = get_scorer("recall")
            kwargs = {"pos_label": 1}
        elif scoring_name == "specificity":
            scorer = get_scorer("recall")
            kwargs = {"pos_label": 0}
        else:
            scorer = get_scorer(scoring_name)
            kwargs = {}
        results[scoring_name] = scorer(model, X, y, **kwargs)
    return results


# TODO: Might need to be updated for clade analysis
def perform_train_test(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    model_type: str,
    test_size: int | None = None,
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
    n_repeats: int | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    model = make_classifier(model_type, **model_kwargs)
    random_state = model_kwargs.get("random_state", 42)
    X_train = train_X.copy()
    y_train = train_y.copy()
    X_test = test_X.copy()
    y_test = test_y.copy()
    # Make sure X_test has the same columns as X_train and fill missing columns with 0
    for col in X_train.columns.difference(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    model.fit(X_train, y_train)  # type: ignore
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
            results.append(
                {
                    "repeat": i,
                    **scores,
                }
            )
    return pd.DataFrame(results)
