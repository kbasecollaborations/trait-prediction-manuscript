"""Model factory for the alternative-ML comparison study.

Each entry returns an sklearn-compatible classifier. Linear and PLS models are
wrapped in a Pipeline with sparse-safe scaling so they accept the same {0,1}
KOFAM feature matrices the tree models use, without caller-side preprocessing.
Hyperparameters are literature defaults; no tuning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model aliases owned by this module; dispatched via patch_make_classifier below.
ALTERNATE_MODELS = (
    "lasso",
    "enet",
    "glasso_kegg",
    "splsda",
    "lgbm",
    "lgbm_tuned",
    "xgb",
    "pclr",
    "phylo_glmm",
    "tabpfn",
    "cb_tuned",
    "cb_libdefault",
    "rf_tuned",
    "enet_tuned",
)


def importances_from_model(model: Any, feature_names: list[str]) -> pd.Series:
    """Extract feature importances, descending.

    Tries ``feature_importances_``, then ``|coef_|``, then ``x_weights_``
    (PLS), and returns all zeros when none apply.
    """
    est = _unwrap(model)
    p = len(feature_names)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float).ravel()
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim == 2:
            vals = np.abs(coef).sum(axis=0)
        else:
            vals = np.abs(coef)
    elif hasattr(est, "x_weights_"):
        vals = np.abs(np.asarray(est.x_weights_, dtype=float)).sum(axis=1)
    else:
        vals = np.zeros(p, dtype=float)
    if vals.shape[0] != p:
        vals = np.zeros(p, dtype=float)
    s = pd.Series(vals, index=feature_names)
    s.sort_values(ascending=False, inplace=True)
    return s


def _unwrap(model: Any) -> Any:
    """Peel off a Pipeline / GroupLassoLogisticClassifier wrapper."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    if hasattr(model, "estimator_"):
        return model.estimator_
    if hasattr(model, "inner_"):
        return model.inner_
    return model


def _lasso_lr(random_state: int = 42) -> BaseEstimator:
    """L1-penalized logistic regression with saga solver."""
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _enet_lr(random_state: int = 42) -> BaseEstimator:
    """Elastic-net logistic regression."""
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _phylo_glmm(random_state: int = 42) -> BaseEstimator:
    """Tree-kinship-aware EN-LR head; the runner appends the top-k kinship PCs
    to X before calling fit."""
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _pclr(random_state: int = 42, n_pcs: int = 20) -> BaseEstimator:
    """EN-LR head for the phylogeny-aware proxy.

    Features are residualized against the top-k phylogenetic PCs upstream in
    ``PhyloPCAdjuster`` (see groupings.py); this factory only supplies the
    class-balanced elastic-net.
    """
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
    )


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Minimal sPLS-DA: PLSRegression on (0,1) y, threshold at 0.5.

    The sparse truncation of weak loadings is approximated by keeping only the
    top ``n_components`` components.
    """

    _estimator_type = "classifier"

    def __init__(
        self, n_components: int = 10, threshold: float = 0.5, random_state: int = 42
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_comp = max(1, min(self.n_components, min(X.shape) - 1))
        self.inner_ = PLSRegression(n_components=n_comp, scale=True)
        self.inner_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        raw = self.inner_.predict(X).ravel()
        # Clipped, not calibrated: the continuous PLS output is mapped to (0,1).
        p1 = np.clip(raw, 0.0, 1.0)
        return np.column_stack([1 - p1, p1])

    @property
    def x_weights_(self):
        return self.inner_.x_weights_


def _splsda(random_state: int = 42) -> BaseEstimator:
    return PLSDAClassifier(n_components=10, random_state=random_state)


def _lgbm(random_state: int = 42) -> BaseEstimator:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        feature_fraction=0.1,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=10,
        reg_lambda=15.0,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1,
    )


def _lgbm_tuned(
    random_state: int = 42, tuned_params: dict | None = None
) -> BaseEstimator:
    """LightGBM with per-(phenotype, subset) Optuna-tuned hyperparameters.

    The runner injects ``tuned_params``; without cached params this falls back
    to the default literature LGBM.
    """
    if not tuned_params:
        return _lgbm(random_state=random_state)
    from scripts.alternate.ml_comparison.lgbm_tuning import make_tuned_classifier

    return make_tuned_classifier(tuned_params, random_state=random_state)


def _xgb(random_state: int = 42) -> BaseEstimator:
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        colsample_bynode=0.1,
        subsample=0.8,
        reg_lambda=15.0,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )


# Fixed (non-searched) settings for the default-vs-tuned lift study, copied from
# the manuscript defaults in trait_prediction.classifiers.factory so each tuned
# variant differs from the `cb_noeval` / `rf` baselines only in the searched knobs.
_CB_NOEVAL_FIXED = {
    "iterations": 500,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 15,
    "bagging_temperature": 1,
    "rsm": 0.1,
    "bootstrap_type": "Bayesian",
    "task_type": "CPU",
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}
_RF_FIXED = {
    "n_estimators": 1000,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}


def _cb_tuned(
    random_state: int = 42, tuned_params: dict | None = None
) -> BaseEstimator:
    """CatBoost (no early stopping) with Optuna-tuned hyperparameters.

    Falls back to the `cb_noeval` manuscript default when no tuned params are
    supplied. Searched knobs are depth, learning_rate, l2_leaf_reg, rsm and
    iterations.
    """
    from catboost import CatBoostClassifier

    params = {**_CB_NOEVAL_FIXED, "random_state": random_state, **(tuned_params or {})}
    return CatBoostClassifier(**params)


def _cb_libdefault(random_state: int = 42, **_kwargs) -> BaseEstimator:
    """CatBoost with library-default hyperparameters.

    Out-of-the-box baseline for the curated `cb_noeval` configuration. Runs
    single-threaded so the process-parallel sweep does not oversubscribe;
    library defaults use all features per split.
    """
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_state=random_state,
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
        thread_count=1,
    )


def _rf_tuned(
    random_state: int = 42, tuned_params: dict | None = None
) -> BaseEstimator:
    """Random Forest with Optuna-tuned hyperparameters.

    Falls back to the `rf` manuscript default when no tuned params are
    supplied, and keeps the default's class-weight policy (none) so the lift is
    attributable to tuning alone.
    """
    from sklearn.ensemble import RandomForestClassifier

    params = {**_RF_FIXED, "random_state": random_state, **(tuned_params or {})}
    return RandomForestClassifier(**params)


def _enet_tuned(
    random_state: int = 42, tuned_params: dict | None = None
) -> BaseEstimator:
    """Elastic-net LR with Optuna-tuned C and l1_ratio.

    Falls back to the `enet` default (C=1.0, l1_ratio=0.5) when no tuned params
    are supplied, keeping the same saga and balanced-class-weight policy.
    """
    tp = tuned_params or {}
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=tp.get("l1_ratio", 0.5),
                    C=tp.get("C", 1.0),
                    max_iter=5000,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
    )


class GroupLassoLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Group-lasso logistic regression with KEGG-module groupings.

    When the group-lasso backend is unavailable, falls back to plain
    elastic-net LR and the groupings are ignored.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        groups: list[int] | None = None,
        alpha: float = 0.01,
        random_state: int = 42,
    ):
        self.groups = groups
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from group_lasso import LogisticGroupLasso  # type: ignore

            self._backend = "group_lasso"
        except Exception:
            from sklearn.linear_model import LogisticRegression

            self._backend = "fallback_enet"
            self.inner_ = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                tol=1e-3,
                random_state=self.random_state,
            )
            self.inner_.fit(X, y)
            self.classes_ = self.inner_.classes_
            self.coef_ = self.inner_.coef_
            return self
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y).astype(int)
        groups = (
            np.asarray(self.groups, dtype=int)
            if self.groups is not None
            else np.arange(Xa.shape[1])
        )
        self.inner_ = LogisticGroupLasso(
            groups=groups,
            group_reg=self.alpha,
            l1_reg=0.0,
            scale_reg="inverse_group_size",
            supress_warning=True,
            n_iter=200,
            tol=1e-3,
            random_state=self.random_state,
            fit_intercept=True,
        )
        self.inner_.fit(Xa, ya)
        self.classes_ = np.array([0, 1])
        coef = np.asarray(self.inner_.coef_, dtype=float)
        # coef shape is (p, n_classes); take class-1 column for a binary problem
        self.coef_ = (
            coef[:, -1].reshape(1, -1) if coef.ndim == 2 else coef.reshape(1, -1)
        )
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        if self._backend == "fallback_enet":
            return self.inner_.predict_proba(X)
        Xa = np.asarray(X, dtype=float)
        return np.asarray(self.inner_.predict_proba(Xa), dtype=float)


def _glasso_kegg(
    random_state: int = 42, groups: list[int] | None = None
) -> BaseEstimator:
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                GroupLassoLogisticClassifier(
                    groups=groups,
                    alpha=0.01,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _tabpfn(random_state: int = 42) -> BaseEstimator:
    """TabPFN v2 on CPU. The runner slices features down to the stable-cluster
    subset (<=500) before fit."""
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=False,
        random_state=random_state,
    )


_FACTORIES = {
    "lasso": _lasso_lr,
    "enet": _enet_lr,
    "pclr": _pclr,
    "phylo_glmm": _phylo_glmm,
    "splsda": _splsda,
    "lgbm": _lgbm,
    "lgbm_tuned": _lgbm_tuned,
    "xgb": _xgb,
    "tabpfn": _tabpfn,
    "cb_tuned": _cb_tuned,
    "cb_libdefault": _cb_libdefault,
    "rf_tuned": _rf_tuned,
    "enet_tuned": _enet_tuned,
}


def make_alternate(model_type: str, **kwargs) -> BaseEstimator:
    """Return one of the alternate-study classifiers."""
    if model_type == "glasso_kegg":
        groups = kwargs.pop("groups", None)
        random_state = kwargs.pop("random_state", 42)
        return _glasso_kegg(random_state=random_state, groups=groups)
    if model_type not in _FACTORIES:
        raise ValueError(f"unknown alternate model: {model_type!r}")
    return _FACTORIES[model_type](**kwargs)


def patch_make_classifier() -> None:
    """Patch scripts.ml.make_classifier so the alternate aliases dispatch into
    this module before upstream delegation.

    Idempotent; called once by the runner before any fits.
    """
    from scripts import ml as _ml

    if getattr(_ml, "_alternate_patched", False):
        return

    original = _ml.make_classifier

    def patched(model_type: str, **kwargs):
        if model_type in ALTERNATE_MODELS:
            return make_alternate(model_type, **kwargs)
        return original(model_type, **kwargs)

    _ml.make_classifier = patched
    _ml._alternate_patched = True
    # Mirror into ml_splits, which imported make_classifier directly.
    from scripts import ml_splits as _ml_splits

    _ml_splits.make_classifier = patched
