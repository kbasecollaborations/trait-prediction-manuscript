#!/usr/bin/env python3
"""Compute per-phenotype x per-dataset GapMind concordance/discordance counts.

The per-dataset genome universe is the union of ``genomeID`` rows across all
phenotype TSVs in ``data/processed/phenotypes/<dataset>/`` (every genome with at
least one experimental measurement, independent of feature-matrix QC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from scripts.figure5.figure5a_data import load_gapmind_predictions

GAPMIND_FILE: Final[Path] = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR: Final[Path] = Path("data/processed/phenotypes")
OUTPUT_DIR: Final[Path] = Path("data/outputs/concordance_counts")
OUTPUT_FILE: Final[Path] = OUTPUT_DIR / "concordance_counts.tsv"

DATASETS: Final[tuple[str, ...]] = ("atleaf", "lit", "marine", "pmi")


def load_retained_genomes(phenotype_dir: Path, dataset: str) -> set[str]:
    """Load the per-dataset phenotype-file genome universe.

    The universe is the union of all ``genomeID`` values appearing in any
    phenotype TSV under ``phenotype_dir / dataset``, that is, every genome with
    at least one experimental measurement in this dataset.

    Parameters
    ----------
    phenotype_dir : Path
        Root directory containing per-dataset phenotype subdirectories.
    dataset : str
        Name of the dataset subdirectory (e.g. ``"atleaf"``).

    Returns
    -------
    set[str]
        Set of ``genomeID`` strings present in this dataset's phenotype files.

    Raises
    ------
    FileNotFoundError
        If the dataset directory does not exist.
    """
    dataset_dir = phenotype_dir / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    genomes: set[str] = set()
    for path in sorted(dataset_dir.glob("*.tsv")):
        df = pd.read_csv(path, sep="\t", usecols=["genomeID"], dtype={"genomeID": str})
        genomes.update(df["genomeID"].astype(str).unique())
    return genomes


def load_dataset_phenotypes(
    phenotype_dir: Path, dataset: str, retained_genomes: set[str]
) -> dict[str, pd.Series]:
    """Load all phenotype files for a single dataset, reindexed to the dataset universe.

    Each returned Series is reindexed to ``retained_genomes``, so genomes missing
    from a particular phenotype TSV appear as ``NaN`` and are counted as
    ``n_excluded_no_phenotype`` downstream. The row sum across all five count
    columns therefore equals the size of the dataset universe.

    Parameters
    ----------
    phenotype_dir : Path
        Root directory containing per-dataset phenotype subdirectories.
    dataset : str
        Name of the dataset subdirectory (e.g. ``"atleaf"``).
    retained_genomes : set[str]
        Per-dataset phenotype-file genome universe.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from phenotype name to a pandas Series indexed by every
        genome in the dataset universe.

    Raises
    ------
    FileNotFoundError
        If the dataset directory does not exist.
    """
    dataset_dir = phenotype_dir / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    universe_index = pd.Index(sorted(retained_genomes), name="genomeID")
    phenotypes: dict[str, pd.Series] = {}
    for path in sorted(dataset_dir.glob("*.tsv")):
        name = path.stem
        df = pd.read_csv(path, sep="\t", dtype={"genomeID": str})
        df = df.drop_duplicates(subset=["genomeID"], keep="first")
        df = df.set_index("genomeID")
        phenotypes[name] = df[name].reindex(universe_index)
    return phenotypes


def compute_counts_for_pair(
    gapmind: pd.DataFrame,
    phenotype_series: pd.Series,
    phenotype: str,
    dataset: str,
) -> dict[str, int | str]:
    """Compute concordance/discordance counts for a single (phenotype, dataset).

    Parameters
    ----------
    gapmind : pd.DataFrame
        GapMind predictions indexed by ``genomeID``.
    phenotype_series : pd.Series
        Experimental phenotype values for the dataset, indexed by ``genomeID``.
    phenotype : str
        Phenotype name.
    dataset : str
        Dataset name.

    Returns
    -------
    dict[str, int | str]
        Long-form record with counts and identifying labels.
    """
    n_total = len(phenotype_series)

    if phenotype not in gapmind.columns:
        n_no_phenotype = int(phenotype_series.isna().sum())
        return {
            "phenotype": phenotype,
            "dataset": dataset,
            "n_total_genomes": n_total,
            "n_concordant": 0,
            "n_discordant_FP": 0,
            "n_discordant_FN": 0,
            "n_excluded_no_gapmind": n_total - n_no_phenotype,
            "n_excluded_no_phenotype": n_no_phenotype,
            "n_for_training": 0,
        }

    gap = gapmind[phenotype].reindex(phenotype_series.index)
    exp = phenotype_series

    no_phenotype = exp.isna()
    no_gapmind = gap.isna() & ~no_phenotype

    valid = ~no_phenotype & ~no_gapmind
    gap_v = gap[valid].astype(int)
    exp_v = exp[valid].astype(int)

    concordant = gap_v == exp_v
    n_concordant = int(concordant.sum())
    n_fp = int(((gap_v == 1) & (exp_v == 0)).sum())
    n_fn = int(((gap_v == 0) & (exp_v == 1)).sum())

    return {
        "phenotype": phenotype,
        "dataset": dataset,
        "n_total_genomes": n_total,
        "n_concordant": n_concordant,
        "n_discordant_FP": n_fp,
        "n_discordant_FN": n_fn,
        "n_excluded_no_gapmind": int(no_gapmind.sum()),
        "n_excluded_no_phenotype": int(no_phenotype.sum()),
        "n_for_training": n_concordant,
    }


def build_counts_table(
    gapmind: pd.DataFrame,
    phenotype_dir: Path,
    datasets: tuple[str, ...],
    phenotypes: list[str],
) -> pd.DataFrame:
    """Iterate over every (phenotype, dataset) and build the long-form table.

    Parameters
    ----------
    gapmind : pd.DataFrame
        GapMind predictions indexed by ``genomeID``.
    phenotype_dir : Path
        Root directory containing per-dataset phenotype subdirectories. The
        per-dataset phenotype-file genome universe is derived from this
        directory.
    datasets : tuple[str, ...]
        Dataset directory names to iterate over.
    phenotypes : list[str]
        Phenotype names to include (typically the columns of ``gapmind``).

    Returns
    -------
    pd.DataFrame
        Long-form counts table, one row per (phenotype, dataset).
    """
    rows: list[dict[str, int | str]] = []
    for dataset in datasets:
        retained = load_retained_genomes(phenotype_dir, dataset)
        n_universe = len(retained)
        print(f"  {dataset}: {n_universe} genomes in phenotype-file universe")
        dataset_phenotypes = load_dataset_phenotypes(phenotype_dir, dataset, retained)
        for phenotype in phenotypes:
            if phenotype not in dataset_phenotypes:
                # Phenotype not measured in this dataset: the whole universe
                # counts as excluded for a missing phenotype.
                rows.append(
                    {
                        "phenotype": phenotype,
                        "dataset": dataset,
                        "n_total_genomes": n_universe,
                        "n_concordant": 0,
                        "n_discordant_FP": 0,
                        "n_discordant_FN": 0,
                        "n_excluded_no_gapmind": 0,
                        "n_excluded_no_phenotype": n_universe,
                        "n_for_training": 0,
                    }
                )
                continue
            rows.append(
                compute_counts_for_pair(
                    gapmind=gapmind,
                    phenotype_series=dataset_phenotypes[phenotype],
                    phenotype=phenotype,
                    dataset=dataset,
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Build and write the concordance counts table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GapMind predictions from {GAPMIND_FILE} ...")
    gapmind = load_gapmind_predictions(GAPMIND_FILE)
    phenotypes = sorted(gapmind.columns.tolist())
    print(f"  {len(gapmind)} genomes, {len(phenotypes)} phenotypes")

    print(
        f"Loading experimental phenotypes from {PHENOTYPE_DIR}; "
        "the per-dataset universe is the union of genomes in that dataset's "
        "phenotype TSVs (no feature-matrix intersection)."
    )
    counts = build_counts_table(
        gapmind=gapmind,
        phenotype_dir=PHENOTYPE_DIR,
        datasets=DATASETS,
        phenotypes=phenotypes,
    )

    counts = counts.sort_values(["phenotype", "dataset"]).reset_index(drop=True)
    counts.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(counts)} rows to {OUTPUT_FILE}")

    print("\nPer-dataset totals (across all phenotypes):")
    by_dataset = counts.groupby("dataset")[
        [
            "n_concordant",
            "n_discordant_FP",
            "n_discordant_FN",
            "n_excluded_no_gapmind",
            "n_excluded_no_phenotype",
        ]
    ].sum()
    print(by_dataset)

    total_concordant = int(counts["n_concordant"].sum())
    total_fp = int(counts["n_discordant_FP"].sum())
    total_fn = int(counts["n_discordant_FN"].sum())
    total_used = total_concordant + total_fp + total_fn
    pct_concordant = (
        100.0 * total_concordant / total_used if total_used else float("nan")
    )
    print(
        "\nAggregate (sum across all phenotype x dataset cells with both labels):"
        f"\n  concordant = {total_concordant}"
        f"\n  discordant FP = {total_fp}"
        f"\n  discordant FN = {total_fn}"
        f"\n  concordant fraction of labelled = {pct_concordant:.1f}%"
    )


if __name__ == "__main__":
    main()
