"""Phylogeny-aware GLMM via kinship-eigenvector adjustment.

Converts tree distances to a kinship matrix K, eigendecomposes K, and takes the
top-k eigenvectors as additional fixed-effect covariates in a regularized
logistic regression (a low-rank approximation of the LMM
y = X β + Z u + ε with u ~ N(0, σ² K)).

The supplied gtdb-pruned distance matrix covers only ~77% of the 822 genomes in
the KOFAM matrix; samples outside it are dropped at the runner level when the
split is subset for this model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh

DISTANCE_MATRIX = Path("data/processed/phylogeny/distance_matrix.tsv")
N_KINSHIP_PCS = 20


@lru_cache(maxsize=1)
def load_distance_matrix() -> pd.DataFrame:
    if not DISTANCE_MATRIX.exists():
        raise FileNotFoundError(
            f"missing {DISTANCE_MATRIX} — phylo_glmm requires the gtdb-pruned "
            f"tree distance matrix"
        )
    dm = pd.read_csv(DISTANCE_MATRIX, sep="\t", index_col=0)
    # Symmetrize; tree distances should already be symmetric.
    dm = 0.5 * (dm + dm.T)
    return dm


def _distance_to_kinship(D: np.ndarray) -> np.ndarray:
    """Convert a pairwise distance matrix to a Gaussian-kernel kinship matrix.

    K_ij = exp(- D_ij**2 / (2 * sigma**2)),  sigma = median nonzero distance.
    Yields a PSD similarity matrix with K_ii = 1.
    """
    triu = D[np.triu_indices_from(D, k=1)]
    sigma = float(np.median(triu[triu > 0]))
    K = np.exp(-(D**2) / (2.0 * sigma**2))
    return K


def _topk_kinship_pcs(K: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k eigenvectors of K, scaled by sqrt(eigenvalue)."""
    n = K.shape[0]
    # eigh returns ascending eigenvalues; take the top-k from the end.
    eigvals, eigvecs = eigh(K, subset_by_index=(n - k, n - 1))
    # Clip negative eigenvalues from numerical noise.
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs * np.sqrt(eigvals)  # shape (n, k)


def restrict_split_to_covered(split: dict[str, Any]) -> dict[str, Any] | None:
    """Drop samples not in the distance matrix from a split dict."""
    dm = load_distance_matrix()
    covered = set(dm.index)
    out: dict[str, Any] = {}
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        data = split[key]
        keep = [g for g in data.index if g in covered]
        out[key] = data.loc[keep]
    if (
        len(out["X_train"]) < 5
        or len(out["X_test"]) < 10
        or out["y_train"].nunique() < 2
    ):
        return None
    return out


def add_kinship_pcs(split: dict[str, Any], k: int = N_KINSHIP_PCS) -> dict[str, Any]:
    """Append the top-k kinship PCs to each X frame as extra columns.

    The PCs are computed on the union of train+val+test. Restriction to covered
    genomes must have been done already (see :func:`restrict_split_to_covered`).
    """
    dm = load_distance_matrix()
    all_genomes = sorted(
        set(split["X_train"].index)
        | set(split["X_val"].index)
        | set(split["X_test"].index)
    )
    D = dm.loc[all_genomes, all_genomes].values.astype(float)
    K = _distance_to_kinship(D)
    pcs = _topk_kinship_pcs(K, k=min(k, len(all_genomes) - 1))
    pc_cols = [f"treePC{i + 1:02d}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, index=all_genomes, columns=pc_cols)

    out: dict[str, Any] = dict(split)
    for key in ("X_train", "X_val", "X_test"):
        X = split[key]
        out[key] = pd.concat([X, pc_df.loc[X.index]], axis=1)
    return out


def transform_split_for_phylo_glmm(split: dict[str, Any]) -> dict[str, Any] | None:
    """Compose the restrict + augment steps."""
    sub = restrict_split_to_covered(split)
    if sub is None:
        return None
    return add_kinship_pcs(sub, k=N_KINSHIP_PCS)
