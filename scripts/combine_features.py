#!/usr/bin/env python3
"""
Combine GapMind, KOFAM, and RAST features with correlation filtering.

This script:
1. Loads GapMind, KOFAM, and RAST features from the reduced features directory
2. Removes RAST features that are correlated with KOFAM features (Pearson correlation)
3. Removes KOFAM features that are correlated with GapMind features (Spearman correlation)
4. Combines the remaining features into a single feature matrix
5. Saves the combined feature matrix and the lists of removed features
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def find_correlated_features(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
) -> dict[str, list[str]]:
    """
    Find features in features_b that are correlated with features in features_a.

    Parameters
    ----------
    features_a : pd.DataFrame
        First feature matrix (reference features to keep).
    features_b : pd.DataFrame
        Second feature matrix (features to potentially remove).
    threshold : float, optional
        Correlation threshold above which features are considered correlated, by default 0.95
    method : str, optional
        Correlation method: 'pearson' or 'spearman', by default "pearson"

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping each feature in features_b to the list of features in
        features_a it is correlated with (if correlation > threshold).
    """
    # Get common samples (genomeIDs) between the two feature sets
    common_samples = features_a.index.intersection(features_b.index)
    features_a_aligned = features_a.loc[common_samples]
    features_b_aligned = features_b.loc[common_samples]

    # Dictionary to store correlated features
    correlated = {}

    # Determine correlation function
    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Check each feature in features_b against all features in features_a
    for col_b in tqdm(
        features_b_aligned.columns, desc=f"Finding {method} correlations"
    ):
        correlated_with = []
        for col_a in features_a_aligned.columns:
            # Calculate correlation
            try:
                corr, _ = corr_func(
                    features_a_aligned[col_a], features_b_aligned[col_b]
                )
                if abs(corr) >= threshold:
                    correlated_with.append(col_a)
            except Exception:
                # Skip if correlation calculation fails (e.g., constant columns)
                pass

        if correlated_with:
            correlated[col_b] = correlated_with

    return correlated


def main() -> None:
    """
    Main function to combine GapMind, KOFAM, and RAST features with correlation filtering.
    """
    # Define paths
    INPUT_DIR = Path("data/processed/features_reduced/combined_datasets")
    OUTPUT_DIR = Path("data/processed/features_reduced/combined_datasets")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = INPUT_DIR / "gapmind.tsv"
    KOFAM_FILE = INPUT_DIR / "kofam.tsv"
    RAST_FILE = INPUT_DIR / "rast.tsv"

    OUTPUT_FILE = OUTPUT_DIR / "gapmind_kofam_rast.tsv"
    CORR_OUTPUT_FILE = OUTPUT_DIR / "correlation_removed_features.json.gz"

    # Correlation thresholds
    CORRELATION_THRESHOLD = 0.95

    print("=" * 80)
    print("Combining GapMind, KOFAM, and RAST features with correlation filtering")
    print("=" * 80)

    # Load feature matrices
    print("\nLoading feature matrices...")
    print(f"  Loading GapMind features from {GAPMIND_FILE}...")
    gapmind_features = pd.read_csv(
        GAPMIND_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"    Shape: {gapmind_features.shape}")

    print(f"  Loading KOFAM features from {KOFAM_FILE}...")
    kofam_features = pd.read_csv(
        KOFAM_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"    Shape: {kofam_features.shape}")

    print(f"  Loading RAST features from {RAST_FILE}...")
    rast_features = pd.read_csv(
        RAST_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"    Shape: {rast_features.shape}")

    # Step 1: Remove RAST features correlated with KOFAM features (Pearson)
    print("\n" + "=" * 80)
    print("Step 1: Remove RAST features correlated with KOFAM features")
    print(f"  Method: Pearson correlation, threshold: {CORRELATION_THRESHOLD}")
    print("=" * 80)

    rast_kofam_correlated = find_correlated_features(
        kofam_features, rast_features, threshold=CORRELATION_THRESHOLD, method="pearson"
    )

    print(f"\nFound {len(rast_kofam_correlated)} RAST features correlated with KOFAM")
    rast_features_filtered = rast_features.drop(
        columns=list(rast_kofam_correlated.keys())
    )
    print(
        f"Remaining RAST features: {rast_features_filtered.shape[1]} / {rast_features.shape[1]}"
    )

    # Step 2: Remove KOFAM features correlated with GapMind features (Spearman)
    print("\n" + "=" * 80)
    print("Step 2: Remove KOFAM features correlated with GapMind features")
    print(f"  Method: Spearman correlation, threshold: {CORRELATION_THRESHOLD}")
    print("=" * 80)

    kofam_gapmind_correlated = find_correlated_features(
        gapmind_features,
        kofam_features,
        threshold=CORRELATION_THRESHOLD,
        method="spearman",
    )

    print(f"\nFound {len(kofam_gapmind_correlated)} KOFAM features correlated with GapMind")
    kofam_features_filtered = kofam_features.drop(
        columns=list(kofam_gapmind_correlated.keys())
    )
    print(
        f"Remaining KOFAM features: {kofam_features_filtered.shape[1]} / {kofam_features.shape[1]}"
    )

    # Step 3: Combine all remaining features
    print("\n" + "=" * 80)
    print("Step 3: Combine all remaining features")
    print("=" * 80)

    # Get all unique genomeIDs across all feature matrices
    all_genomes = (
        gapmind_features.index.union(kofam_features_filtered.index).union(
            rast_features_filtered.index
        )
    )

    print(f"Total unique genomes: {len(all_genomes)}")

    # Reindex all matrices to have the same genomes, filling missing values with 0
    gapmind_features_aligned = gapmind_features.reindex(all_genomes, fill_value=0)
    kofam_features_aligned = kofam_features_filtered.reindex(all_genomes, fill_value=0)
    rast_features_aligned = rast_features_filtered.reindex(all_genomes, fill_value=0)

    # Concatenate along columns
    combined_features = pd.concat(
        [gapmind_features_aligned, kofam_features_aligned, rast_features_aligned],
        axis=1,
    )

    print(f"\nCombined feature matrix shape: {combined_features.shape}")
    print(f"  GapMind features: {gapmind_features_aligned.shape[1]}")
    print(f"  KOFAM features: {kofam_features_aligned.shape[1]}")
    print(f"  RAST features: {rast_features_aligned.shape[1]}")
    print(f"  Total features: {combined_features.shape[1]}")

    # Step 4: Save combined feature matrix
    print("\n" + "=" * 80)
    print("Step 4: Save outputs")
    print("=" * 80)

    print(f"\nSaving combined feature matrix to {OUTPUT_FILE}...")
    combined_features.to_csv(OUTPUT_FILE, sep="\t")
    print("  Done!")

    # Save correlation information
    correlation_info = {
        "rast_kofam_correlated": {
            feature: corr_with for feature, corr_with in rast_kofam_correlated.items()
        },
        "kofam_gapmind_correlated": {
            feature: corr_with for feature, corr_with in kofam_gapmind_correlated.items()
        },
        "summary": {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "rast_features_removed": len(rast_kofam_correlated),
            "kofam_features_removed": len(kofam_gapmind_correlated),
            "original_gapmind_features": gapmind_features.shape[1],
            "original_kofam_features": kofam_features.shape[1],
            "original_rast_features": rast_features.shape[1],
            "final_gapmind_features": gapmind_features_aligned.shape[1],
            "final_kofam_features": kofam_features_aligned.shape[1],
            "final_rast_features": rast_features_aligned.shape[1],
            "total_combined_features": combined_features.shape[1],
        },
    }

    print(f"\nSaving correlation information to {CORR_OUTPUT_FILE}...")
    with gzip.open(CORR_OUTPUT_FILE, "wt", encoding="utf-8") as f:
        json.dump(correlation_info, f, indent=2)
    print("  Done!")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"Removed {len(rast_kofam_correlated)} RAST features (correlated with KOFAM)"
    )
    print(
        f"Removed {len(kofam_gapmind_correlated)} KOFAM features (correlated with GapMind)"
    )
    print(f"Final combined feature matrix: {combined_features.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()
