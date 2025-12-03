#!/usr/bin/env python3

"""
Generate GapMind phenotype prediction files with strict and loose thresholds.

This script processes raw GapMind phenotype data and creates two binary prediction files:
1. Strict: Only 'complete' status is marked as 1 (present)
2. Loose: Both 'complete' and 'likely_complete' are marked as 1 (present)
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def create_gapmind_predictions(
    input_file: Path,
    output_dir: Path,
    strict_mapping: Dict[str, int],
    loose_mapping: Dict[str, int],
) -> None:
    """
    Create strict and loose GapMind phenotype prediction files.

    Parameters
    ----------
    input_file : Path
        Path to the raw GapMind phenotype data TSV file
    output_dir : Path
        Directory where output files will be saved
    strict_mapping : Dict[str, int]
        Mapping dictionary for strict predictions (only 'complete' = 1)
    loose_mapping : Dict[str, int]
        Mapping dictionary for loose predictions ('complete' and 'likely_complete' = 1)

    Returns
    -------
    None
        Writes two TSV files to output_dir
    """
    # Read the raw GapMind data
    print(f"Reading GapMind data from: {input_file}")
    gapmind_data = pd.read_csv(
        input_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )

    print(f"Loaded data shape: {gapmind_data.shape}")
    print(f"Number of genomes: {len(gapmind_data)}")
    print(f"Number of phenotypes: {len(gapmind_data.columns)}")

    # Create strict predictions (only 'complete' = 1)
    print("\nCreating strict predictions...")
    gapmind_strict = gapmind_data.replace(strict_mapping).astype(np.uint8)
    strict_output = output_dir / "gapmind_phenotypes_strict.tsv"
    gapmind_strict.to_csv(strict_output, sep="\t")
    print(f"Saved strict predictions to: {strict_output}")
    print(f"  Total positive predictions: {gapmind_strict.sum().sum()}")

    # Create loose predictions ('complete' and 'likely_complete' = 1)
    print("\nCreating loose predictions...")
    gapmind_loose = gapmind_data.replace(loose_mapping).astype(np.uint8)
    loose_output = output_dir / "gapmind_phenotypes_loose.tsv"
    gapmind_loose.to_csv(loose_output, sep="\t")
    print(f"Saved loose predictions to: {loose_output}")
    print(f"  Total positive predictions: {gapmind_loose.sum().sum()}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"\nStrict predictions (complete only):")
    print(f"  - Positives per genome (mean): {gapmind_strict.sum(axis=1).mean():.2f}")
    print(
        f"  - Positives per phenotype (mean): {gapmind_strict.sum(axis=0).mean():.2f}"
    )
    print(f"\nLoose predictions (complete + likely_complete):")
    print(f"  - Positives per genome (mean): {gapmind_loose.sum(axis=1).mean():.2f}")
    print(
        f"  - Positives per phenotype (mean): {gapmind_loose.sum(axis=0).mean():.2f}"
    )


def main() -> None:
    """Main execution function."""
    # Define paths
    input_file = Path("data/interim/gapmind/gapmind_phenotype_data_raw.tsv")
    output_dir = Path("data/outputs/figure2")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define mapping dictionaries
    # Strict: Only 'complete' is considered positive
    strict_mapping = {
        "complete": 1,
        "likely_complete": 0,
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }

    # Loose: Both 'complete' and 'likely_complete' are considered positive
    loose_mapping = {
        "complete": 1,
        "likely_complete": 1,
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }

    # Generate the prediction files
    create_gapmind_predictions(input_file, output_dir, strict_mapping, loose_mapping)


if __name__ == "__main__":
    main()
