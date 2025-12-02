#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.io import read_features, read_phenotypes

# Load kofam features only
feature_files = list(Path("data/interim/features").glob("**/kofam.tsv"))
print(f"Loading {len(feature_files)} kofam feature files...")
feature_set = read_features(feature_files)

# Find common phenotypes across all 4 datasets
phenotype_files_all = list(Path("data/processed/phenotypes").glob("**/*.tsv"))
print(f"Found {len(phenotype_files_all)} phenotype files.")
phenotype_files = []
dataset_phenotype_name_map = defaultdict(set)

for phenotype_file in phenotype_files_all:
    dataset = phenotype_file.parent.stem
    # Include atleaf, lit, marine, and pmi datasets
    if dataset in ["atleaf", "lit", "marine", "pmi"]:
        phenotype_files.append(phenotype_file)
        dataset_phenotype_name_map[dataset].add(phenotype_file.stem)

# Get phenotypes common to all 4 datasets
COMMON_PHENOTYPES = sorted(set.intersection(*dataset_phenotype_name_map.values()))

# Filter to only include common phenotypes
phenotype_files = [p for p in phenotype_files if p.stem in COMMON_PHENOTYPES]
phenotype_set = read_phenotypes(phenotype_files)

# Create counts dataframe
counts_data = []
for phenotype in phenotype_set.phenotypes:
    phenotype_name = phenotype.pindex.name
    dataset_name = phenotype.pindex.category
    phenotype_data = phenotype.phenotype_data

    # Count positive and negative samples
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

# Create dataframe
counts_df = pd.DataFrame(counts_data)

# Display results
print(f"Number of common phenotypes: {len(COMMON_PHENOTYPES)}")
print(f"Common phenotypes: {COMMON_PHENOTYPES}")
print("\nCounts dataframe:")
print(counts_df)

# Save to CSV
output_file = Path("data/outputs/figure1/figure1c_data.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
counts_df.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")
