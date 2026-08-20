"""Phylogenetic-PC residualisation for the `pclr` model.

Computes the top-k principal components of the KOFAM presence/absence matrix
and appends them as features alongside the original KOs, so elastic-net LR
absorbs population-structure variance into the PC columns while the KO
coefficients pick up phenotype-specific signal. This is the Patterson-2006
genomic-PC adjustment, used here as a proxy because the repo ships no genome
tree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

N_PCS = 20


def _fit_pcs(X_train: pd.DataFrame) -> tuple[TruncatedSVD, np.ndarray]:
    n_comp = min(N_PCS, min(X_train.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X_train.values.astype(float))
    return svd, svd.transform(X_train.values.astype(float))


def phylo_pc_adjust(split: dict[str, Any]) -> dict[str, Any]:
    """Augment X_train / X_val / X_test with top-N_PCS genomic PCs.

    The PCs are fit on X_train only and projected onto val/test, so there
    is no information leakage from the test set.
    """
    X_train = split["X_train"]
    X_val = split.get("X_val")
    X_test = split["X_test"]

    svd, train_pcs = _fit_pcs(X_train)
    pc_cols = [f"phyloPC{i + 1:02d}" for i in range(train_pcs.shape[1])]

    train_pc_df = pd.DataFrame(train_pcs, index=X_train.index, columns=pc_cols)
    X_train_aug = pd.concat([X_train, train_pc_df], axis=1)

    out: dict[str, Any] = dict(split)
    out["X_train"] = X_train_aug

    if X_val is not None and len(X_val) > 0:
        val_pcs = svd.transform(X_val.values.astype(float))
        val_pc_df = pd.DataFrame(val_pcs, index=X_val.index, columns=pc_cols)
        out["X_val"] = pd.concat([X_val, val_pc_df], axis=1)

    test_pcs = svd.transform(X_test.values.astype(float))
    test_pc_df = pd.DataFrame(test_pcs, index=X_test.index, columns=pc_cols)
    out["X_test"] = pd.concat([X_test, test_pc_df], axis=1)

    return out
