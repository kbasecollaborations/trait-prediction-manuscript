"""One-time preparation for the BacDive data-volume experiments.

The exact BacDive retrieval date was not retained; the deposited input files
are the fixed snapshot used for the reported analysis.

Produces three cached artefacts (idempotent; skips work that already exists):

1. ``data/processed/bacdive/genome_genus.json`` -- genome -> genus map parsed
   from the ``species`` column of the raw BacDive phenotype table, used for
   out-of-clade (leave-whole-genera-out) splitting.
2. ``data/interim/features/bacdive/kofam_reduced.parquet`` -- BacDive KOFAM
   matrix after variance + correlation filtering (matching ``feature_filtering``).
3. ``data/processed/bacdive/groupA_overlap.txt`` -- version-stripped GCF
   accessions shared between BacDive and Group A (the pooled 4 datasets), used
   to drop leakage genomes from the test side in cross-dataset transfer.

Run: ``uv run python -m scripts.bacdive.prepare_bacdive``
"""

import json
import re
from pathlib import Path

import pandas as pd
from trait_prediction.main import Feature

# Match feature_filtering.py
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
CORRELATION_METHOD = "pearson"  # kofam/rast use pearson; gapmind uses spearman

RAW_PHENOTYPES = Path("data/raw/bacdive_dataset/metabolic_phenotypes.tsv")
BACDIVE_KOFAM = Path("data/interim/features/bacdive/kofam.tsv")
KOFAM_REDUCED = Path("data/interim/features/bacdive/kofam_reduced.parquet")
GROUP_A_KOFAM = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")

OUT_DIR = Path("data/processed/bacdive")
GENUS_MAP = OUT_DIR / "genome_genus.json"
OVERLAP_FILE = OUT_DIR / "groupA_overlap.txt"

_VERSION_RE = re.compile(r"\.\d+$")


def strip_version(accession: str) -> str:
    """Strip a trailing RefSeq version suffix (e.g. ``GCF_x.1`` -> ``GCF_x``)."""
    return _VERSION_RE.sub("", str(accession))


def build_genus_map() -> dict[str, str]:
    """Map each BacDive genome to its genus (first token of the species name)."""
    df = pd.read_csv(RAW_PHENOTYPES, sep="\t", dtype=str)
    genome_col, species_col = df.columns[0], "species"
    genus_map: dict[str, str] = {}
    for genome, species in zip(df[genome_col], df[species_col]):
        if pd.isna(species) or not str(species).strip():
            continue
        genus = str(species).strip().split()[0]
        # Placeholder first tokens: fall back to the next token.
        if genus.lower() in {"uncultured", "candidatus", "unclassified"}:
            tokens = str(species).strip().split()
            genus = tokens[1] if len(tokens) > 1 else genus
        genus_map[str(genome)] = genus
    return genus_map


def reduce_kofam() -> pd.DataFrame:
    """Variance + correlation filter the BacDive KOFAM matrix."""
    print(f"Loading {BACDIVE_KOFAM} ...")
    df = pd.read_csv(BACDIVE_KOFAM, sep="\t", index_col=0, dtype={"genomeID": str})
    df.index = df.index.astype(str)
    print(f"  raw: {df.shape[0]} genomes x {df.shape[1]} KOs")

    filtered, low_var = Feature.remove_features_with_low_variance(
        df, VARIANCE_THRESHOLD
    )
    print(f"  after variance: {filtered.shape[1]} KOs (removed {len(low_var)})")

    filtered, _ = Feature.remove_features_with_high_correlation(
        filtered, CORRELATION_THRESHOLD, parallel=True, method=CORRELATION_METHOD
    )
    print(f"  after correlation: {filtered.shape[1]} KOs")
    return filtered.astype("int8")


def compute_overlap() -> set[str]:
    """Version-stripped GCF accessions present in both BacDive and Group A."""
    bacdive_ids = pd.read_csv(BACDIVE_KOFAM, sep="\t", usecols=[0], dtype=str).iloc[
        :, 0
    ]
    bacdive_stripped = {strip_version(x) for x in bacdive_ids}

    group_a_ids = pd.read_csv(GROUP_A_KOFAM, sep="\t", usecols=[0], dtype=str).iloc[
        :, 0
    ]
    group_a_stripped = {strip_version(x) for x in group_a_ids}

    return bacdive_stripped & group_a_stripped


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if GENUS_MAP.exists():
        print(f"[skip] {GENUS_MAP} exists")
    else:
        genus_map = build_genus_map()
        GENUS_MAP.write_text(json.dumps(genus_map))
        n_genera = len(set(genus_map.values()))
        print(f"Wrote {GENUS_MAP}: {len(genus_map)} genomes, {n_genera} genera")

    if KOFAM_REDUCED.exists():
        print(f"[skip] {KOFAM_REDUCED} exists")
    else:
        reduced = reduce_kofam()
        reduced.to_parquet(KOFAM_REDUCED)
        print(f"Wrote {KOFAM_REDUCED}: {reduced.shape[0]} x {reduced.shape[1]}")

    if OVERLAP_FILE.exists():
        print(f"[skip] {OVERLAP_FILE} exists")
    else:
        overlap = compute_overlap()
        OVERLAP_FILE.write_text("\n".join(sorted(overlap)))
        print(f"Wrote {OVERLAP_FILE}: {len(overlap)} shared (version-stripped) genomes")


if __name__ == "__main__":
    main()
