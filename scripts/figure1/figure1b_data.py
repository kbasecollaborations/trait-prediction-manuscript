#!/usr/bin/env python3
"""Generate the Figure 1B data table (genome counts per phenotype and dataset).

Counts positive and negative genomes for every phenotype shared by the atleaf,
lit, marine and pmi datasets, and writes
``data/outputs/figure1/figure1b_data.csv``.

Run with::

    uv run python -m scripts.figure1.figure1b_data
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.io import read_features, read_phenotypes

feature_files = list(Path("data/interim/features").glob("**/kofam.tsv"))
print(f"Loading {len(feature_files)} kofam feature files...")
feature_set = read_features(feature_files)

phenotype_files_all = list(Path("data/processed/phenotypes").glob("**/*.tsv"))
print(f"Found {len(phenotype_files_all)} phenotype files.")
phenotype_files = []
dataset_phenotype_name_map = defaultdict(set)

for phenotype_file in phenotype_files_all:
    dataset = phenotype_file.parent.stem
    if dataset in ["atleaf", "lit", "marine", "pmi"]:
        phenotype_files.append(phenotype_file)
        dataset_phenotype_name_map[dataset].add(phenotype_file.stem)

COMMON_PHENOTYPES = sorted(set.intersection(*dataset_phenotype_name_map.values()))

phenotype_files = [p for p in phenotype_files if p.stem in COMMON_PHENOTYPES]
phenotype_set = read_phenotypes(phenotype_files)

counts_data = []
for phenotype in phenotype_set.phenotypes:
    phenotype_name = phenotype.pindex.name
    dataset_name = phenotype.pindex.category
    phenotype_data = phenotype.phenotype_data

    value_counts = phenotype_data.value_counts()
    positive_count = value_counts.get(1, 0)
    negative_count = value_counts.get(0, 0)

    counts_data.append(
        {
            "phenotype": phenotype_name,
            "dataset": dataset_name,
            "positive_count": positive_count,
            "negative_count": negative_count,
        }
    )

counts_df = pd.DataFrame(counts_data)

print(f"Number of common phenotypes: {len(COMMON_PHENOTYPES)}")
print(f"Common phenotypes: {COMMON_PHENOTYPES}")
print("\nCounts dataframe:")
print(counts_df)

output_file = Path("data/outputs/figure1/figure1b_data.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
counts_df.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")
