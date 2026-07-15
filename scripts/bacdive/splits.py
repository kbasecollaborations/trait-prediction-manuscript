"""Label loading and train/val/test splitting for the BacDive experiments.

All functions return plain genomeID lists (or label Series) so that the parallel
driver can ship lightweight job payloads to workers. No model code here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PHENOTYPE_DIR = Path("data/processed/phenotypes")
GROUP_A_DATASETS = ("atleaf", "lit", "marine", "pmi")


def load_bacdive_labels(phenotype: str) -> pd.Series:
    """Load BacDive 0/1 labels for one phenotype (index = versionless GCF)."""
    path = PHENOTYPE_DIR / "bacdive" / f"{phenotype}.tsv"
    df = pd.read_csv(path, sep="\t", index_col=0, dtype={"genomeID": str})
    s = df.iloc[:, 0].dropna().astype(int)
    s.index = s.index.astype(str)
    return s


def load_group_a_labels(phenotype: str) -> pd.Series:
    """Pooled 0/1 labels for one phenotype across the four Group A datasets."""
    parts = []
    for dataset in GROUP_A_DATASETS:
        path = PHENOTYPE_DIR / dataset / f"{phenotype}.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", index_col=0, dtype={"genomeID": str})
        parts.append(df.iloc[:, 0])
    pooled = pd.concat(parts)
    pooled = pooled[~pooled.index.duplicated(keep="first")].dropna().astype(int)
    pooled.index = pooled.index.astype(str)
    return pooled


def _stratified_holdout(
    y: pd.Series, frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Stratified split of ``y``'s index into (rest, holdout) by ``frac``."""
    rest, holdout = train_test_split(
        y.index.to_numpy(),
        test_size=frac,
        random_state=seed,
        stratify=y.to_numpy(),
    )
    return list(rest), list(holdout)


def random_holdout_split(
    y: pd.Series, seed: int, test_frac: float = 0.15, val_frac: float = 0.15
) -> tuple[list[str], list[str], list[str]]:
    """Label-stratified train/val/test split.

    ``val_frac`` is taken from the post-test remainder so the test fraction is
    exact. Returns ``(train_ids, val_ids, test_ids)``.
    """
    rest_ids, test_ids = _stratified_holdout(y, test_frac, seed)
    rest = y.loc[rest_ids]
    val_rel = val_frac / (1.0 - test_frac)
    train_ids, val_ids = _stratified_holdout(rest, val_rel, seed)
    return train_ids, val_ids, test_ids


def ooc_genus_split(
    y: pd.Series,
    genus_map: dict[str, str],
    seed: int,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]] | None:
    """Leave-whole-genera-out split.

    Whole genera are shuffled (seeded) and accumulated into the test set until
    ~``test_frac`` of samples is reached; the remainder is split into
    train/val (label-stratified). Genomes without a genus are pooled into
    train/val only. Returns ``None`` if the held-out test set lacks both
    classes.
    """
    genera_of = {g: genus_map.get(g) for g in y.index}
    mapped = [g for g in y.index if genera_of[g] is not None]
    unmapped = [g for g in y.index if genera_of[g] is None]

    genera = sorted({genera_of[g] for g in mapped})
    rng = np.random.default_rng(seed)
    rng.shuffle(genera)

    target = int(round(test_frac * len(y)))
    test_ids: list[str] = []
    chosen: set[str] = set()
    by_genus: dict[str, list[str]] = {}
    for g in mapped:
        by_genus.setdefault(genera_of[g], []).append(g)
    for genus in genera:
        if len(test_ids) >= target:
            break
        chosen.add(genus)
        test_ids.extend(by_genus[genus])

    if y.loc[test_ids].nunique() < 2:
        return None

    rest_ids = [g for g in mapped if genera_of[g] not in chosen] + unmapped
    rest = y.loc[rest_ids]
    if rest.nunique() < 2:
        return None
    val_rel = val_frac / (1.0 - len(test_ids) / len(y))
    val_rel = min(max(val_rel, 0.05), 0.4)
    train_ids, val_ids = _stratified_holdout(rest, val_rel, seed)
    return train_ids, val_ids, test_ids


def stratified_subsample(
    ids: list[str], y: pd.Series, size: int, seed: int, min_minority: int = 5
) -> list[str] | None:
    """Label-stratified subsample of ``ids`` to ``size``.

    Returns all ``ids`` if ``size >= len(ids)``. Returns ``None`` if the draw
    would leave fewer than ``min_minority`` minority-class samples or only one
    class.
    """
    if size >= len(ids):
        subset = list(ids)
    else:
        sub_y = y.loc[ids]
        if sub_y.nunique() < 2:
            return None
        if len(ids) - size < 2:
            # holdout too small to stratify; take an unstratified random subset
            rng = np.random.default_rng(seed)
            subset = list(rng.choice(np.asarray(ids), size=size, replace=False))
        else:
            subset, _ = train_test_split(
                np.asarray(ids),
                train_size=size,
                random_state=seed,
                stratify=sub_y.to_numpy(),
            )
            subset = list(subset)
    counts = y.loc[subset].value_counts()
    if len(counts) < 2 or counts.min() < min_minority:
        return None
    return subset


def minority_count(y: pd.Series, ids: list[str]) -> int:
    """Size of the smaller class among ``ids`` (0 if only one class)."""
    counts = y.loc[ids].value_counts()
    return int(counts.min()) if len(counts) == 2 else 0

