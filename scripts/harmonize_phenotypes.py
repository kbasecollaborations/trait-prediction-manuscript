"""Regenerate ``data/processed/phenotypes/`` from the raw handoff files.

This replaces the original harmonisation notebook, which stripped stereochemical
descriptors from substrate names with chained ``str.removeprefix`` calls and then let
later source columns silently overwrite earlier ones of the same stripped name. Every
published column is now bound to its source column explicitly, via the table produced
by :mod:`scripts.build_substrate_identity`.

The script is idempotent and verifies its own output: it refuses to write if two
published columns would share a filename, and it reports every file whose contents
differ from what is currently on disk.

Run with ``uv run python -m scripts.harmonize_phenotypes``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Final

import pandas as pd

from scripts.io import index_format_func

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DIR: Final[Path] = REPO_ROOT / "data/raw/phenotypes"
IDENTITY_TABLE: Final[Path] = REPO_ROOT / "data/processed/substrate_identity.csv"
OUTPUT_DIR: Final[Path] = REPO_ROOT / "data/processed/phenotypes"

DATASETS: Final[tuple[str, ...]] = ("lit", "atleaf", "pmi", "marine")


def read_raw(dataset: str) -> pd.DataFrame:
    """Read a raw handoff phenotype matrix.

    Parameters
    ----------
    dataset
        One of ``lit``, ``atleaf``, ``pmi``, ``marine``.

    Returns
    -------
    pandas.DataFrame
        Phenotype matrix indexed by ``genomeID``, normalised with the repository's
        canonical :func:`scripts.io.index_format_func`, with unnamed trailing columns
        dropped.
    """
    frame = pd.read_csv(RAW_DIR / f"{dataset}_phenotypes.tsv", sep="\t", dtype=str)
    frame = frame.rename(columns={frame.columns[0]: "genomeID"})
    frame = frame.loc[:, [c for c in frame.columns if not c.startswith("Unnamed")]]
    frame["genomeID"] = frame["genomeID"].map(index_format_func)
    return frame.set_index("genomeID")


def to_binary(series: pd.Series) -> pd.Series:
    """Coerce a raw phenotype column to nullable integer 0/1.

    Parameters
    ----------
    series
        Raw column, whose values may be ``"1"``, ``"1.0"``, ``"0"``, ``"0.0"`` or missing.

    Returns
    -------
    pandas.Series
        Values as ``Int64``, preserving missing entries.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def write_dataset(dataset: str, identity: pd.DataFrame, output_dir: Path, dry_run: bool) -> dict[str, str]:
    """Write every published phenotype file for one dataset.

    Parameters
    ----------
    dataset
        Dataset key.
    identity
        Substrate identity table filtered to this dataset.
    output_dir
        Destination ``data/processed/phenotypes`` directory.
    dry_run
        When ``True``, compute output but do not write it.

    Returns
    -------
    dict[str, str]
        Mapping of phenotype name to ``"unchanged"``, ``"modified"`` or ``"new"``.

    Raises
    ------
    KeyError
        If the identity table names a source column absent from the raw matrix.
    """
    raw = read_raw(dataset)
    destination = output_dir / dataset
    destination.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    for row in identity.itertuples():
        if row.source_column not in raw.columns:
            raise KeyError(f"{dataset}: source column {row.source_column!r} not in raw matrix")
        series = to_binary(raw[row.source_column]).rename(row.phenotype)
        rendered = series.to_csv(sep="\t").encode()
        target = destination / f"{row.phenotype}.tsv"
        if not target.exists():
            status[row.phenotype] = "new"
        elif hashlib.md5(target.read_bytes()).hexdigest() == hashlib.md5(rendered).hexdigest():
            status[row.phenotype] = "unchanged"
        else:
            status[row.phenotype] = "modified"
        if not dry_run:
            target.write_bytes(rendered)

    if not dry_run:
        expected = {f"{p}.tsv" for p in identity.phenotype}
        for stale in sorted(set(p.name for p in destination.glob("*.tsv")) - expected):
            (destination / stale).unlink()
            status[stale.removesuffix(".tsv")] = "removed"
    return status


def main() -> None:
    """Regenerate all four datasets and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    identity = pd.read_csv(IDENTITY_TABLE)
    for dataset in DATASETS:
        subset = identity[identity.dataset == dataset]
        duplicates = subset.loc[subset.phenotype.duplicated(), "phenotype"].tolist()
        if duplicates:
            raise ValueError(f"{dataset}: duplicate published phenotype names {duplicates}")

        status = write_dataset(dataset, subset, args.output_dir, args.dry_run)
        counts = pd.Series(status).value_counts().to_dict()
        print(f"{dataset}: {len(subset)} phenotypes  {counts}")
        for name, state in sorted(status.items()):
            if state != "unchanged":
                print(f"    {state:9s} {name}")


if __name__ == "__main__":
    main()
