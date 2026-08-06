"""Per-(model, phenotype, subset) hyperparameter tuning via Optuna.

Covers the tuned aliases CatBoost, Random Forest and Elastic-Net. Each
(model, phenotype, subset) is tuned once, on the random_split repeat-0
training partition, with ``N_CV_FOLDS``-fold stratified CV on balanced
accuracy; the best parameters are cached under
``data/processed/hpo_tuned_params`` and re-used for every (split_type,
repeat) of that pair.

Search spaces cover only the over/under-fitting hyperparameters; binning and
border parameters are omitted because binary features have a single split
point. Non-searched settings and the class-weight policy match each model's
manuscript default.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

CACHE_BASE = Path("data/processed/hpo_tuned_params")
# Each extra fold is a full refit on a 5.5k-feature matrix.
N_CV_FOLDS = 2
RANDOM_STATE = 42

# Tuned aliases handled here; each has a factory in models.py.
TUNED_MODELS = ("cb_tuned", "rf_tuned", "enet_tuned")

# Per-model Optuna trial budgets.
N_TRIALS = {
    "cb_tuned": 8,
    "rf_tuned": 12,
    "enet_tuned": 10,
}


def _suggest_cb(trial: optuna.Trial) -> dict[str, Any]:
    # depth, iterations and rsm drive fit time (~65 s per fit on the
    # 5.5k-feature matrix), so they stay near the manuscript default
    # (depth=4, rsm=0.1, iterations=500). learning_rate and l2_leaf_reg span
    # their full ranges.
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "depth": trial.suggest_int("depth", 4, 5),
        "rsm": trial.suggest_float("rsm", 0.05, 0.15),
        "iterations": trial.suggest_int("iterations", 200, 500, step=100),
    }


def _suggest_rf(trial: optuna.Trial) -> dict[str, Any]:
    # Fit cost scales with n_estimators x features-per-split x tree depth, so
    # n_estimators is capped at 800 and the upper feature fractions are dropped
    # (max_features=0.3 is ~1.6k features per split on this matrix).
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=100),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.05, 0.1]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
    }


def _suggest_enet(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
    }


_SUGGESTERS: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "cb_tuned": _suggest_cb,
    "rf_tuned": _suggest_rf,
    "enet_tuned": _suggest_enet,
}


def _cv_thread_override(model: str, cv_threads: int) -> dict[str, Any]:
    """Return the per-model threading override used during CV.

    CatBoost and Random Forest parallelize a single fit well and take
    ``cv_threads``; Elastic-Net (saga, binary) does not benefit from threads
    and stays single-threaded, relying on process-level parallelism.
    """
    if model == "cb_tuned":
        return {"thread_count": cv_threads}
    if model == "rf_tuned":
        return {"n_jobs": cv_threads}
    return {}


def _build_estimator(model: str, params: dict[str, Any], *, cv_threads: int):
    """Construct the tuned estimator via the shared ``models.py`` factory, so
    the tuning loop and the final evaluation build the same model."""
    from scripts.alternate.ml_comparison.models import make_alternate

    tuned = dict(params)
    tuned.update(_cv_thread_override(model, cv_threads))
    return make_alternate(model, random_state=RANDOM_STATE, tuned_params=tuned)


def _cv_score(
    model: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_threads: int = 1,
) -> float:
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []
    for train_idx, val_idx in skf.split(X, y):
        yt = y.iloc[train_idx]
        yv = y.iloc[val_idx]
        if yt.nunique() < 2 or yv.nunique() < 2:
            return 0.5
        clf = _build_estimator(model, params, cv_threads=cv_threads)
        clf.fit(X.iloc[train_idx], yt)
        pred = clf.predict(X.iloc[val_idx])
        fold_scores.append(balanced_accuracy_score(yv, np.asarray(pred).ravel()))
    return float(np.mean(fold_scores))


def tune(
    model: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int | None = None,
    cv_threads: int = 1,
) -> dict[str, Any]:
    """Run the Optuna TPE search for one model; return best params dict."""
    if model not in _SUGGESTERS:
        raise ValueError(f"no search space for model {model!r}")
    n_trials = n_trials if n_trials is not None else N_TRIALS[model]
    suggest = _SUGGESTERS[model]

    def _objective(trial: optuna.Trial) -> float:
        return _cv_score(model, suggest(trial), X, y, cv_threads=cv_threads)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def cache_path_for(model: str, phenotype: str, subset: str) -> Path:
    return CACHE_BASE / model / f"{phenotype}__{subset}.json"


def load_or_tune(
    model: str,
    phenotype: str,
    subset: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int | None = None,
    cv_threads: int = 1,
) -> dict[str, Any]:
    """Return cached best params if present, else tune and cache atomically."""
    path = cache_path_for(model, phenotype, subset)
    if path.exists():
        try:
            with path.open() as f:
                return json.load(f)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    params = tune(model, X, y, n_trials=n_trials, cv_threads=cv_threads)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp, path)
    return params
