#!/usr/bin/env python3
"""
Check genome-to-pangenome mapping coverage.

This script analyzes how many genomes in the all_seqs directory have
corresponding species assignments in the pangenome mapping file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Import functions from the main module
from pangenome_completeness import (
    extract_genome_name,
    get_core_genes_path,
)


def main() -> None:
    """Analyze genome mapping coverage."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    all_seqs_dir = project_root / "data" / "raw" / "all_seqs"
    core_genes_dir = project_root / "data" / "processed" / "unique_core_faas"
    mapping_file = project_root / "data" / "processed" / "pangenome" / "assignments.ani.merged_mmseqs90.tsv"

    print("=" * 70)
    print("Genome-to-Pangenome Mapping Analysis")
    print("=" * 70)
    print(f"Genome directory: {all_seqs_dir}")
    print(f"Core genes directory: {core_genes_dir}")
    print(f"Mapping file: {mapping_file}")
    print()

    # Load full mapping file (not just ones with species)
    df = pd.read_csv(mapping_file, sep="\t")
    print(f"Total rows in mapping file: {len(df)}")
    print(f"Rows with species assignment: {df['gtdb_species_clade_id'].notna().sum()}")
    print(f"Rows without species assignment: {df['gtdb_species_clade_id'].isna().sum()}")

    # Create lookup dictionaries
    # All genomes in mapping (genome_name -> species or None)
    all_mapping: dict[str, str | None] = {}
    for _, row in df.iterrows():
        genome_name = row["Genome name"]
        species = row["gtdb_species_clade_id"]
        if pd.notna(species) and species:
            all_mapping[genome_name] = species
        else:
            all_mapping[genome_name] = None

    # Get all genome files
    genome_files = sorted(all_seqs_dir.glob("*.faa"))
    print(f"Genome .faa files in all_seqs: {len(genome_files)}")
    print()

    # Categorize genomes
    in_mapping_with_species: list[str] = []
    in_mapping_no_species: list[str] = []
    not_in_mapping: list[str] = []
    has_core_genes: list[str] = []
    no_core_genes: list[str] = []

    for gf in genome_files:
        name = extract_genome_name(gf)

        if name not in all_mapping:
            not_in_mapping.append(name)
        else:
            species = all_mapping[name]
            if species:
                in_mapping_with_species.append(name)
                # Check for core genes file
                core_path = get_core_genes_path(core_genes_dir, species)
                if core_path:
                    has_core_genes.append(name)
                else:
                    no_core_genes.append(name)
            else:
                in_mapping_no_species.append(name)

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Total genome files:':<45} {len(genome_files):>6}")
    print(f"{'  In mapping with species assignment:':<45} {len(in_mapping_with_species):>6}")
    print(f"    {'With core genes file available:':<43} {len(has_core_genes):>6}")
    print(f"    {'Missing core genes file:':<43} {len(no_core_genes):>6}")
    print(f"{'  In mapping but no species assignment:':<45} {len(in_mapping_no_species):>6}")
    print(f"{'  NOT in mapping file:':<45} {len(not_in_mapping):>6}")
    print()

    # Breakdown by naming convention
    numeric_not_in_mapping = [n for n in not_in_mapping if n[0].isdigit()]
    alpha_not_in_mapping = [n for n in not_in_mapping if not n[0].isdigit()]
    print(f"  Breakdown of NOT in mapping:")
    print(f"    {'Numeric IDs (e.g., 104336.19):':<43} {len(numeric_not_in_mapping):>6}")
    print(f"    {'Alphabetic names:':<43} {len(alpha_not_in_mapping):>6}")
    print()

    # Show sample of missing genomes
    if not_in_mapping:
        print("=" * 70)
        print(f"Sample of genomes NOT in mapping file ({len(not_in_mapping)} total):")
        print("=" * 70)
        for name in sorted(not_in_mapping)[:20]:
            print(f"  {name}")
        if len(not_in_mapping) > 20:
            print(f"  ... and {len(not_in_mapping) - 20} more")
        print()

    # Show genomes in mapping but no species
    if in_mapping_no_species:
        print("=" * 70)
        print(f"Genomes in mapping but NO species assignment ({len(in_mapping_no_species)} total):")
        print("=" * 70)
        for name in sorted(in_mapping_no_species)[:10]:
            print(f"  {name}")
        if len(in_mapping_no_species) > 10:
            print(f"  ... and {len(in_mapping_no_species) - 10} more")
        print()

    # Show genomes missing core gene files
    if no_core_genes:
        print("=" * 70)
        print(f"Genomes with species but missing core gene file ({len(no_core_genes)} total):")
        print("=" * 70)
        for name in sorted(no_core_genes)[:10]:
            species = all_mapping[name]
            print(f"  {name}")
            print(f"    -> species: {species}")
        if len(no_core_genes) > 10:
            print(f"  ... and {len(no_core_genes) - 10} more")
        print()

    # Final stats
    print("=" * 70)
    print("Completeness Calculation Coverage")
    print("=" * 70)
    coverage_pct = 100 * len(has_core_genes) / len(genome_files) if genome_files else 0
    print(f"Genomes that can have completeness calculated: {len(has_core_genes)}/{len(genome_files)} ({coverage_pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
