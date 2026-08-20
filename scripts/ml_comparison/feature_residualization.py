"""Per-gene feature residualization against dataset + (optionally) phylogeny.

For each KOFAM gene column X_g, a linear regression is fit on the merged
train+val partition:
    X_g  ~  α  +  Σ γ_d · D_d   ( +  Σ β_k · phyloPC_k )
where D_d are dataset-membership dummies and phyloPC_k are the top-k
eigenvectors of the tree-derived kinship matrix. Each value is replaced by its
residual X_g - (α + Σ γ_d · D_d + Σ β_k · phyloPC_k), with the same transform
applied to the test partition, so no test labels enter the residualization.

For cross-dataset evaluation the held-out dataset has no dummy in training, so
its test residual keeps only the intercept and phylogenetic adjustment.

Two variants are supported:
  - "dataset_only"  : covariates = dataset dummies. Works for all genomes.
  - "dataset_phylo" : covariates = dataset dummies + 20 phylo PCs. Requires
                     genomes to be in the GTDB tree (drops ~23% otherwise).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GENOME_DATASET_FILE = Path("data/processed/genome_to_dataset.tsv")
UNKNOWN_DATASET = "unknown"


@lru_cache(maxsize=1)
def _genome_to_dataset() -> dict[str, str]:
    df = pd.read_csv(GENOME_DATASET_FILE, sep="\t", dtype={"genome": str})
    return dict(zip(df["genome"], df["dataset"]))


def _dataset_dummies(genome_ids: list[str]) -> pd.DataFrame:
    """One-hot dataset dummies. Unknown genomes get the UNKNOWN_DATASET column."""
    gd = _genome_to_dataset()
    ds = [gd.get(str(g), UNKNOWN_DATASET) for g in genome_ids]
    return pd.get_dummies(
        pd.Series(ds, index=[str(g) for g in genome_ids], name="dataset"), dtype=float
    )


def _phylo_pcs_for(genome_ids: list[str], n_pcs: int = 20) -> pd.DataFrame | None:
    """Return top-n_pcs kinship-PCs for the given genomes, or None if too many
    are uncovered by the tree."""
    from scripts.ml_comparison.phylo_kinship import (
        _distance_to_kinship,
        _topk_kinship_pcs,
        load_distance_matrix,
    )

    dm = load_distance_matrix()
    covered = set(dm.index)
    genome_ids_str = [str(g) for g in genome_ids]
    missing = [g for g in genome_ids_str if g not in covered]
    if missing:
        return None
    D = dm.loc[genome_ids_str, genome_ids_str].values.astype(float)
    K = _distance_to_kinship(D)
    pcs = _topk_kinship_pcs(K, k=min(n_pcs, len(genome_ids_str) - 1))
    cols = [f"treePC{i + 1:02d}" for i in range(pcs.shape[1])]
    return pd.DataFrame(pcs, index=genome_ids_str, columns=cols)


def _build_covariates(genome_ids: list[str], mode: str) -> pd.DataFrame | None:
    """Build the covariate matrix Z for the residualization regression.

    Returns None if `mode == "dataset_phylo"` and the genome list cannot
    be entirely covered by the tree.
    """
    if mode == "dataset_only":
        Z = _dataset_dummies(genome_ids)
        Z.insert(0, "_intercept", 1.0)
        return Z
    if mode == "dataset_phylo":
        ds = _dataset_dummies(genome_ids)
        phylo = _phylo_pcs_for(genome_ids)
        if phylo is None:
            return None
        Z = pd.concat([ds, phylo.loc[ds.index]], axis=1)
        Z.insert(0, "_intercept", 1.0)
        return Z
    raise ValueError(f"unknown residualization mode: {mode!r}")


def _fit_residualization(X_train: pd.DataFrame, Z_train: pd.DataFrame) -> np.ndarray:
    """Solve Z β = X for each column of X via least squares.

    Returns β as a (n_covariates, n_features) coefficient matrix.
    """
    Z = Z_train.values.astype(float)
    X = X_train.values.astype(float)
    # Ridge-regularized normal equations for numerical stability
    ZtZ = Z.T @ Z + 1e-6 * np.eye(Z.shape[1])
    ZtX = Z.T @ X
    coef = np.linalg.solve(ZtZ, ZtX)  # shape (n_cov, n_feat)
    return coef


def _apply_residualization(
    X: pd.DataFrame,
    Z: pd.DataFrame,
    coef: np.ndarray,
) -> pd.DataFrame:
    """Subtract the fitted Z·coef predictions from X."""
    pred = Z.values.astype(float) @ coef
    return pd.DataFrame(X.values.astype(float) - pred, index=X.index, columns=X.columns)


def _restrict_to_tree(split: dict[str, Any]) -> dict[str, Any] | None:
    """Drop genomes not in the GTDB tree from a split dict (dataset_phylo only)."""
    from scripts.ml_comparison.phylo_kinship import restrict_split_to_covered

    return restrict_split_to_covered(split)


def transform_split_residualize(
    split: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    """Residualize X_train, X_val, X_test against [dataset (+ phylo)]
    covariates fit on the merged train+val partition.

    Returns a new split dict with X_* replaced by residuals (y_* and
    indices unchanged). Returns None if the split is unusable after the
    transform (e.g., dataset_phylo with insufficient tree coverage).
    """
    if mode == "dataset_phylo":
        sub = _restrict_to_tree(split)
        if sub is None:
            return None
        split = sub

    X_train_full = pd.concat([split["X_train"], split["X_val"]], axis=0)
    X_train_full = X_train_full[~X_train_full.index.duplicated(keep="first")]

    Z_train = _build_covariates([str(g) for g in X_train_full.index], mode)
    if Z_train is None:
        return None
    Z_train = Z_train.loc[[str(g) for g in X_train_full.index]]

    coef = _fit_residualization(X_train_full, Z_train)

    out: dict[str, Any] = dict(split)
    for key in ("X_train", "X_val", "X_test"):
        X = split[key]
        Z = _build_covariates([str(g) for g in X.index], mode)
        if Z is None:
            return None
        # Re-align to the training Z columns (missing columns -> 0): under
        # cross-dataset eval the held-out dataset has no dummy in training.
        Z = Z.reindex(columns=Z_train.columns, fill_value=0.0)
        Z = Z.loc[[str(g) for g in X.index]]
        out[key] = _apply_residualization(X, Z, coef)
    return out
