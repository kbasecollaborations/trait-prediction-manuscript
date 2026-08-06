"""Stable-feature selection for TabPFN.

TabPFN v2 caps inputs at ~500 features, so each split is restricted to the
SHAP-stable KO union per phenotype (the same KOs used by the Fig 4C / 5B
comparisons), typically under 100 KOs. A phenotype with no stable KOs falls
back to the top-`max_features` KOs by training-set variance.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

STABLE_KOS_FILE = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")


@lru_cache(maxsize=1)
def _load_stable_ko_index() -> dict[str, list[str]]:
    if not STABLE_KOS_FILE.exists():
        return {}
    with STABLE_KOS_FILE.open() as f:
        clusters = json.load(f)
    return {phen: list(kos.keys()) for phen, kos in clusters.items()}


def _topk_by_variance(X_train: pd.DataFrame, max_features: int) -> list[str]:
    var = X_train.var(axis=0)
    return var.sort_values(ascending=False).head(max_features).index.tolist()


def select_stable_features(
    phenotype: str,
    split: dict[str, Any],
    max_features: int = 500,
) -> dict[str, Any]:
    """Subset the split's X_* frames to (at most) the stable KO union."""
    stable_index = _load_stable_ko_index()
    stable_kos = stable_index.get(phenotype, [])
    X_train = split["X_train"]
    if stable_kos:
        keep = [k for k in stable_kos if k in X_train.columns]
        if len(keep) > max_features:
            # Trim by training variance (deterministic)
            sub = X_train[keep]
            keep = _topk_by_variance(sub, max_features)
    else:
        keep = _topk_by_variance(X_train, max_features)

    out: dict[str, Any] = dict(split)
    for key in ("X_train", "X_val", "X_test"):
        if key in out and isinstance(out[key], pd.DataFrame):
            cols_present = [c for c in keep if c in out[key].columns]
            out[key] = out[key].loc[:, cols_present]
    return out
