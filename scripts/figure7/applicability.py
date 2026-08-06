"""Label-free reliability helpers: feature-space novelty and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def mean_knn_jaccard_distance(
    test: pd.DataFrame, train: pd.DataFrame, k: int = 5
) -> pd.Series:
    """Mean Jaccard distance from each test row to its ``k`` nearest training rows.

    Operates on binary presence/absence feature matrices aligned on shared columns.

    Parameters
    ----------
    test : pd.DataFrame
        Test genomes x features (0/1), indexed by genome.
    train : pd.DataFrame
        Training genomes x features (0/1).
    k : int
        Number of nearest training neighbours to average over.

    Returns
    -------
    pd.Series
        Mean kNN Jaccard distance per test genome (0 = identical to training).
    """
    cols = train.columns.intersection(test.columns)
    tr = train[cols].to_numpy(dtype=bool)
    te = test[cols].to_numpy(dtype=bool)
    dist = pairwise_distances(te, tr, metric="jaccard")
    kk = min(k, dist.shape[1])
    nearest = np.sort(dist, axis=1)[:, :kk]
    return pd.Series(nearest.mean(axis=1), index=test.index)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error using equal-width bins on P(class == 1).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth 0/1 labels.
    proba : np.ndarray
        Predicted probability of class 1.
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    float
        Weighted mean absolute gap between bin confidence and bin accuracy.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(proba)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (proba > lo) & (proba <= hi) if lo > 0 else (proba >= lo) & (proba <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def calibration_table(per_sample: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Build a reliability-diagram table from a per-sample frame with ``proba``/``y_true``.

    Parameters
    ----------
    per_sample : pd.DataFrame
        Frame with columns ``proba`` (P(class==1)) and ``y_true`` (0/1).
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    pd.DataFrame
        One row per non-empty bin: ``bin_mid``, ``mean_pred``, ``frac_pos``, ``count``.
    """
    proba = per_sample["proba"].to_numpy()
    y = per_sample["y_true"].to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (proba > lo) & (proba <= hi) if lo > 0 else (proba >= lo) & (proba <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_mid": (lo + hi) / 2,
                "mean_pred": float(proba[mask].mean()),
                "frac_pos": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
