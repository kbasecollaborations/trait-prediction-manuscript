#!/usr/bin/env python3
"""Minority-class-in-test exclusion rule: count helpers and row-level filter.

Applied post-hoc at plot/aggregation time, so saved ML result CSVs stay intact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MIN_MINORITY_TEST_SAMPLES: int = 10
"""Minimum minority-class samples required in the held-out test set for a
(phenotype, dataset) cell to be reported. Matches the rule stated in
``sections/methods.tex`` and ``sections/supplementary.tex``."""

_DATASETS_DEFAULT: tuple[str, ...] = ("atleaf", "lit", "marine", "pmi")


def extract_test_dataset(key: str) -> str | None:
    """Return the held-out dataset name from a ``"<phen>_train(...),test(<ds>)"`` key.

    Parameters
    ----------
    key : str
        Key of the form ``"<phenotype>_train(<a>+<b>+<c>),test(<d>)"``.

    Returns
    -------
    str | None
        Held-out dataset name (``"atleaf"``, ``"lit"``, ``"marine"`` or
        ``"pmi"``), or ``None`` when the key does not match this format.
    """
    match = re.search(r"test\(([^)]+)\)", str(key))
    return match.group(1) if match else None


def _load_labels(phenotype_dir: Path, dataset: str, phenotype: str) -> pd.Series | None:
    """Load the experimental label series for one (dataset, phenotype) pair."""
    path = phenotype_dir / dataset / f"{phenotype}.tsv"
    if not path.exists():
        return None
    return (
        pd.read_csv(path, sep="\t", dtype={"genomeID": str})
        .set_index("genomeID")[phenotype]
        .dropna()
    )


def concordant_minority_counts(
    gapmind_file: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv"),
    phenotype_dir: Path = Path("data/processed/phenotypes"),
    datasets: tuple[str, ...] = _DATASETS_DEFAULT,
) -> dict[tuple[str, str], int]:
    """Minority-class concordant sample count per ``(phenotype, dataset)``.

    A sample is concordant when the GapMind prediction equals the experimental
    label. The reported count is ``min(n_pos_concordant, n_neg_concordant)``;
    if only one class is concordant the count is 0.

    Parameters
    ----------
    gapmind_file : Path, optional
        GapMind prediction TSV (default: permissive/loose threshold).
    phenotype_dir : Path, optional
        Directory containing per-dataset phenotype TSVs.
    datasets : tuple[str, ...], optional
        Dataset directory names under ``phenotype_dir``.

    Returns
    -------
    dict[tuple[str, str], int]
        Mapping from ``(phenotype, dataset)`` to minority-class concordant
        sample count.
    """
    gapmind = pd.read_csv(gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str})
    minority: dict[tuple[str, str], int] = {}
    for dataset in datasets:
        dataset_dir = phenotype_dir / dataset
        if not dataset_dir.exists():
            continue
        for phenotype_path in dataset_dir.glob("*.tsv"):
            phenotype = phenotype_path.stem
            if phenotype not in gapmind.columns:
                continue
            labels = _load_labels(phenotype_dir, dataset, phenotype)
            if labels is None:
                continue
            common = labels.index.intersection(gapmind.index)
            labels = labels.loc[common]
            concordant_mask = labels == gapmind.loc[common, phenotype]
            class_counts = labels[concordant_mask].value_counts()
            minority[(phenotype, dataset)] = (
                int(class_counts.min()) if len(class_counts) >= 2 else 0
            )
    return minority


def discordant_minority_counts(
    gapmind_file: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv"),
    phenotype_dir: Path = Path("data/processed/phenotypes"),
    datasets: tuple[str, ...] = _DATASETS_DEFAULT,
) -> dict[tuple[str, str], int]:
    """Minority-class discordant sample count per ``(phenotype, dataset)``.

    Mirrors :func:`concordant_minority_counts` for the discordant subset
    (samples where the GapMind prediction disagrees with the experimental
    label). Used for Figure 5C, whose held-out test set is the discordant
    subset of the held-out dataset.
    """
    gapmind = pd.read_csv(gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str})
    minority: dict[tuple[str, str], int] = {}
    for dataset in datasets:
        dataset_dir = phenotype_dir / dataset
        if not dataset_dir.exists():
            continue
        for phenotype_path in dataset_dir.glob("*.tsv"):
            phenotype = phenotype_path.stem
            if phenotype not in gapmind.columns:
                continue
            labels = _load_labels(phenotype_dir, dataset, phenotype)
            if labels is None:
                continue
            common = labels.index.intersection(gapmind.index)
            labels = labels.loc[common]
            discordant_mask = labels != gapmind.loc[common, phenotype]
            class_counts = labels[discordant_mask].value_counts()
            minority[(phenotype, dataset)] = (
                int(class_counts.min()) if len(class_counts) >= 2 else 0
            )
    return minority


def full_test_minority_counts(
    phenotype_dir: Path = Path("data/processed/phenotypes"),
    datasets: tuple[str, ...] = _DATASETS_DEFAULT,
) -> dict[tuple[str, str], int]:
    """Minority-class count among all labelled samples per ``(phenotype, dataset)``.

    Used for figures whose held-out test set is the full dataset (concordant +
    discordant pooled), such as Figures 3 and 6.
    """
    minority: dict[tuple[str, str], int] = {}
    for dataset in datasets:
        dataset_dir = phenotype_dir / dataset
        if not dataset_dir.exists():
            continue
        for phenotype_path in dataset_dir.glob("*.tsv"):
            phenotype = phenotype_path.stem
            labels = _load_labels(phenotype_dir, dataset, phenotype)
            if labels is None:
                continue
            class_counts = labels.value_counts()
            minority[(phenotype, dataset)] = (
                int(class_counts.min()) if len(class_counts) >= 2 else 0
            )
    return minority


def annotate_minority_test(
    df: pd.DataFrame,
    minority_counts: dict[tuple[str, str], int],
    key_column: str = "key",
    phenotype_column: str = "phenotype",
    test_dataset_column: str | None = None,
    out_column: str = "n_minority_test",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a minority-class-in-test count column.

    The held-out dataset is read from ``test_dataset_column`` when supplied,
    otherwise parsed from ``key_column``. Rows whose held-out dataset cannot be
    determined (e.g., random-split rows) or whose ``(phenotype, dataset)`` pair
    is absent from ``minority_counts`` receive a missing value.

    Parameters
    ----------
    df : pd.DataFrame
        Per-split results table.
    minority_counts : dict[tuple[str, str], int]
        Output of one of the ``*_minority_counts`` helpers, matching the
        figure's test-set definition.
    key_column : str, optional
        Per-split key column, default ``"key"``.
    phenotype_column : str, optional
        Phenotype column, default ``"phenotype"``.
    test_dataset_column : str | None, optional
        If supplied, read the held-out dataset directly from this column.
    out_column : str, optional
        Name of the column to write, default ``"n_minority_test"``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``out_column`` added (overwriting any existing
        column of the same name).
    """

    def lookup(row: pd.Series) -> int | None:
        if test_dataset_column is not None and test_dataset_column in row:
            test_dataset = row[test_dataset_column]
            if isinstance(test_dataset, float) or test_dataset in (None, ""):
                return None
        else:
            test_dataset = extract_test_dataset(row.get(key_column, ""))
            if test_dataset is None:
                return None
        return minority_counts.get((row[phenotype_column], test_dataset))

    out = df.copy()
    if out_column in out.columns:
        out = out.drop(columns=out_column)
    out[out_column] = out.apply(lookup, axis=1)
    return out


def filter_by_minority(
    df: pd.DataFrame,
    minority_counts: dict[tuple[str, str], int],
    min_minority: int = MIN_MINORITY_TEST_SAMPLES,
    key_column: str = "key",
    phenotype_column: str = "phenotype",
    test_dataset_column: str | None = None,
) -> pd.DataFrame:
    """Drop rows whose held-out test set fails the minority-class threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Per-split results table. Must contain ``phenotype_column``. The
        held-out dataset is taken from ``test_dataset_column`` when supplied,
        otherwise extracted from ``key_column`` via :func:`extract_test_dataset`.
    minority_counts : dict[tuple[str, str], int]
        Output of one of the ``*_minority_counts`` helpers above (must match
        the figure's test-set definition).
    min_minority : int, optional
        Minimum required minority count, default ``MIN_MINORITY_TEST_SAMPLES``.
    key_column : str, optional
        Name of the per-split key column. Default ``"key"``.
    phenotype_column : str, optional
        Name of the phenotype column. Default ``"phenotype"``.
    test_dataset_column : str | None, optional
        If supplied, read the held-out dataset directly from this column
        instead of parsing the key string.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``df``. Rows whose key cannot be parsed (e.g.,
        random-split rows) are passed through unchanged.
    """

    def keep(row: pd.Series) -> bool:
        if test_dataset_column is not None:
            test_dataset = row.get(test_dataset_column)
        else:
            test_dataset = extract_test_dataset(row.get(key_column, ""))
        if test_dataset in (None, "", float("nan")):
            return True
        if isinstance(test_dataset, float):
            return True
        return (
            minority_counts.get((row[phenotype_column], test_dataset), 0)
            >= min_minority
        )

    mask = df.apply(keep, axis=1)
    return df.loc[mask].copy()
