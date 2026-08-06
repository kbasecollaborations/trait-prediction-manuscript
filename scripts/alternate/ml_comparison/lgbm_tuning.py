"""Per-(phenotype, subset) LightGBM hyperparameter tuning via Optuna.

Each (phenotype, subset) is tuned once on the random_split-repeat-0 training
partition using 3-fold stratified CV with balanced accuracy as the objective.
The best parameter set is cached under ``data/processed/lgbm_tuned_params`` and
re-used for every (split_type, repeat) within the same (phenotype, subset), so
tuning and evaluation partitions overlap.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Silence optuna/lightgbm logs during tuning.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

CACHE_DIR = Path("data/processed/lgbm_tuned_params")
N_TRIALS = 25
N_CV_FOLDS = 3
RANDOM_STATE = 42


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def _cv_score(params: dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for train_idx, val_idx in skf.split(X, y):
        Xt, Xv = X.iloc[train_idx], X.iloc[val_idx]
        yt, yv = y.iloc[train_idx], y.iloc[val_idx]
        if yt.nunique() < 2 or yv.nunique() < 2:
            return 0.5
        clf = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            **params,
        )
        clf.fit(Xt, yt)
        pred = clf.predict(Xv)
        fold_scores.append(balanced_accuracy_score(yv, pred))
    return float(np.mean(fold_scores))


def tune(X: pd.DataFrame, y: pd.Series, *, n_trials: int = N_TRIALS) -> dict[str, Any]:
    """Run Optuna search; return best params dict."""

    def _objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        return _cv_score(params, X, y)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def cache_path_for(phenotype: str, subset: str) -> Path:
    return CACHE_DIR / f"{phenotype}__{subset}.json"


def load_or_tune(
    phenotype: str,
    subset: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = N_TRIALS,
) -> dict[str, Any]:
    """Return cached params if present, else tune and cache."""
    path = cache_path_for(phenotype, subset)
    if path.exists():
        try:
            with path.open() as f:
                return json.load(f)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    params = tune(X, y, n_trials=n_trials)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp, path)
    return params


def make_tuned_classifier(params: dict[str, Any], random_state: int = RANDOM_STATE):
    """Instantiate LightGBM with the cached tuned parameters."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1,
        **params,
    )
