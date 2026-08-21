#!/usr/bin/env python3
"""Compute the data for Supplementary Figure S2 (phylogenetic distance splits).

For every phenotype and every split type (random, in-clade, out-of-clade), reads the
persisted train/val/test splits under ``data/processed/train_test_splits`` and
computes, for each held-out test genome, the minimum cophenetic distance to the
reference set (every non-test genome, that is train + val). All five persisted seeds
contribute their test genomes. Writes the tidy long-form table read by
``figureS2_plot.py``.

Run with::

    uv run python -m scripts.figureS2.figureS2_data
"""

from pathlib import Path

import pandas as pd

# The 15 phenotypes shared across all four datasets.
from scripts.create_data_splits import COMMON_PHENOTYPES

SPLITS_DIR = Path("data/processed/train_test_splits")
DISTANCE_FILE = Path("data/processed/phylogeny/distance_matrix.tsv")
OUTPUT_FILE = Path("data/outputs/figureS2/figureS2_data.tsv")

N_SEEDS = 5

# Display split type -> seed-folder template holding y_train/y_val/y_test.tsv.
SPLIT_TYPE_DIRS: dict[str, str] = {
    "random": "random_split/{phenotype}/{seed}",
    "in-clade": "phylogeny_split/{phenotype}/in-clade/{seed}",
    "out-of-clade": "phylogeny_split/{phenotype}/out-of-clade/{seed}",
}


def load_distance_matrix() -> pd.DataFrame:
    """Load the genome-by-genome cophenetic distance matrix.

    Returns
    -------
    pd.DataFrame
        Square distance matrix indexed and columned by genome ID (strings).
    """
    df = pd.read_csv(DISTANCE_FILE, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_genome_ids(split_folder: Path, filename: str) -> list[str]:
    """Read the genome IDs from a split file (its index column).

    Parameters
    ----------
    split_folder : Path
        Folder containing the split TSVs.
    filename : str
        File to read, e.g. ``"y_test.tsv"``.

    Returns
    -------
    list[str]
        Genome IDs, or an empty list if the file does not exist.
    """
    path = split_folder / filename
    if not path.exists():
        return []
    ids = pd.read_csv(path, sep="\t", index_col=0, dtype=str).index
    return ids.astype(str).tolist()


def compute_min_distances(
    distance_df: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
) -> dict[str, float]:
    """Minimum distance from each test genome to the training set.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Cophenetic distance matrix.
    train_ids : list[str]
        Reference (training) genome IDs.
    test_ids : list[str]
        Held-out test genome IDs.

    Returns
    -------
    dict[str, float]
        Mapping from test genome ID to its minimum distance to any training
        genome. Genomes absent from the distance matrix are skipped.
    """
    train = [g for g in train_ids if g in distance_df.index]
    test = [g for g in test_ids if g in distance_df.columns]
    if not train or not test:
        return {}
    sub = distance_df.loc[train, test]
    return sub.min(axis=0).to_dict()


def build_table(distance_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the long-form min-distance table across phenotypes and splits.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Cophenetic distance matrix.

    Returns
    -------
    pd.DataFrame
        Columns: ``phenotype``, ``split_type``, ``genome``, ``min_distance``.
    """
    rows: list[dict[str, object]] = []
    for phenotype in COMMON_PHENOTYPES:
        for split_type, template in SPLIT_TYPE_DIRS.items():
            for seed in range(N_SEEDS):
                folder = SPLITS_DIR / template.format(phenotype=phenotype, seed=seed)
                if not folder.exists():
                    continue
                # Reference set = every non-test genome (train + val).
                train_ids = read_genome_ids(folder, "y_train.tsv") + read_genome_ids(
                    folder, "y_val.tsv"
                )
                test_ids = read_genome_ids(folder, "y_test.tsv")
                for genome, dist in compute_min_distances(
                    distance_df, train_ids, test_ids
                ).items():
                    rows.append(
                        {
                            "phenotype": phenotype,
                            "split_type": split_type,
                            "genome": genome,
                            "min_distance": dist,
                        }
                    )
    return pd.DataFrame(
        rows, columns=["phenotype", "split_type", "genome", "min_distance"]
    )


def main() -> None:
    """Compute the Figure S2 min-distance table and write it to disk."""
    distance_df = load_distance_matrix()
    table = build_table(distance_df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT_FILE}")
    print(table.groupby("split_type")["min_distance"].agg(["count", "median"]).round(3))


if __name__ == "__main__":
    main()
