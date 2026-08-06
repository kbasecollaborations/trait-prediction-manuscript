"""Rewrite existing train/test split labels from the corrected phenotype files.

Only the label values change; every sample keeps its train/val/test assignment, which
matters because :func:`scripts.create_data_splits.create_phylogeny_splits` calls
``splitter.split()`` without a seed and therefore does not reproduce.

Use this after :mod:`scripts.harmonize_phenotypes` changes any label, then retrain. A
full ``create_data_splits`` regeneration is only needed if the split structure itself
should change (the random split clusters on the label matrix, so a rerun redraws it).

Run with ``uv run python -m scripts.relabel_splits --phenotypes Glucose Galacturonic-Acid``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PHENOTYPE_DIR: Final[Path] = REPO_ROOT / "data/processed/phenotypes"
SPLIT_DIR: Final[Path] = REPO_ROOT / "data/processed/train_test_splits"
DATASETS: Final[tuple[str, ...]] = ("atleaf", "lit", "marine", "pmi")


def load_labels(phenotype: str) -> pd.Series:
    """Load the pooled corrected labels for one phenotype across all datasets.

    Parameters
    ----------
    phenotype
        Published phenotype name.

    Returns
    -------
    pandas.Series
        Labels indexed by ``genomeID``, missing values dropped.
    """
    parts: list[pd.Series] = []
    for dataset in DATASETS:
        path = PHENOTYPE_DIR / dataset / f"{phenotype}.tsv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t", dtype=str).set_index("genomeID")
        parts.append(pd.to_numeric(frame.iloc[:, 0], errors="coerce"))
    pooled = pd.concat(parts)
    return pooled[~pooled.index.duplicated(keep="first")].dropna().astype(int)


def relabel(phenotype: str, dry_run: bool) -> tuple[int, int]:
    """Rewrite every split file for one phenotype with corrected labels.

    Parameters
    ----------
    phenotype
        Published phenotype name.
    dry_run
        When ``True``, count changes without writing.

    Returns
    -------
    tuple[int, int]
        Number of split files rewritten and number of individual labels that flipped.

    Raises
    ------
    KeyError
        If a split references a genome absent from the corrected labels.
    """
    labels = load_labels(phenotype)
    files = sorted(SPLIT_DIR.glob(f"*/{phenotype}/**/y_*.tsv"))
    rewritten = 0
    flipped = 0

    for path in files:
        frame = pd.read_csv(path, sep="\t", dtype=str).set_index("genomeID")
        column = frame.columns[0]
        missing = [g for g in frame.index if g not in labels.index]
        if missing:
            raise KeyError(
                f"{path}: {len(missing)} genomes absent from corrected labels, e.g. {missing[:3]}"
            )
        old = pd.to_numeric(frame[column], errors="coerce")
        new = labels.reindex(frame.index)
        changed = int((old != new).sum())
        if changed:
            flipped += changed
            rewritten += 1
            if not dry_run:
                new.rename(column).to_csv(path, sep="\t", index=True)
    return rewritten, flipped


def main() -> None:
    """Relabel splits for the requested phenotypes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypes", nargs="+", required=True, help="phenotype names to relabel"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for phenotype in args.phenotypes:
        rewritten, flipped = relabel(phenotype, args.dry_run)
        verb = "would rewrite" if args.dry_run else "rewrote"
        print(f"{phenotype}: {verb} {rewritten} split files, {flipped} labels flipped")


if __name__ == "__main__":
    main()
