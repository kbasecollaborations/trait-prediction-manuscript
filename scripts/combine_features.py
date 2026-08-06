#!/usr/bin/env python3
"""Combine GapMind, KOFAM, and RAST features after cross-source correlation filtering."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_correlation_matrix(
    features_a: np.ndarray, features_b: np.ndarray, method: str = "pearson"
) -> np.ndarray:
    """Compute the correlation matrix between two feature matrices.

    Parameters
    ----------
    features_a : np.ndarray
        First feature matrix of shape (n_samples, n_features_a).
    features_b : np.ndarray
        Second feature matrix of shape (n_samples, n_features_b).
    method : str, optional
        Correlation method: 'pearson' or 'spearman', by default "pearson"

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_features_b, n_features_a).
        Element [i, j] is the correlation between features_b[:, i] and features_a[:, j].
    """
    if method == "spearman":
        from scipy.stats import rankdata

        features_a = np.apply_along_axis(rankdata, 0, features_a)
        features_b = np.apply_along_axis(rankdata, 0, features_b)

    a_mean = np.mean(features_a, axis=0, keepdims=True)
    b_mean = np.mean(features_b, axis=0, keepdims=True)

    a_std = np.std(features_a, axis=0, keepdims=True)
    b_std = np.std(features_b, axis=0, keepdims=True)

    # Avoid division by zero: constant columns get std=1 and thus corr=0.
    a_std = np.where(a_std == 0, 1, a_std)
    b_std = np.where(b_std == 0, 1, b_std)

    a_centered = (features_a - a_mean) / a_std
    b_centered = (features_b - b_mean) / b_std

    n_samples = features_a.shape[0]
    corr_matrix = (b_centered.T @ a_centered) / n_samples

    return corr_matrix


def find_correlated_features(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
) -> dict[str, list[str]]:
    """Find features in features_b that are correlated with features in features_a.

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
    common_samples = features_a.index.intersection(features_b.index)
    features_a_aligned = features_a.loc[common_samples]
    features_b_aligned = features_b.loc[common_samples]

    if method not in ["pearson", "spearman"]:
        raise ValueError(f"Unknown correlation method: {method}")

    print(f"  Computing {method} correlation matrix...")
    corr_matrix = compute_correlation_matrix(
        features_a_aligned.values, features_b_aligned.values, method=method
    )

    print(f"  Finding correlations above threshold {threshold}...")
    correlated = {}
    abs_corr_matrix = np.abs(corr_matrix)

    for i, col_b in enumerate(features_b_aligned.columns):
        correlated_indices = np.where(abs_corr_matrix[i, :] >= threshold)[0]

        if len(correlated_indices) > 0:
            correlated_with = [
                features_a_aligned.columns[j] for j in correlated_indices
            ]
            correlated[col_b] = correlated_with

    return correlated


def main() -> None:
    """Combine GapMind, KOFAM, and RAST features with correlation filtering."""
    INPUT_DIR = Path("data/processed/features_reduced/combined_datasets")
    OUTPUT_DIR = Path("data/processed/features_reduced/combined_datasets")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = INPUT_DIR / "gapmind.tsv"
    KOFAM_FILE = INPUT_DIR / "kofam.tsv"
    RAST_FILE = INPUT_DIR / "rast.tsv"

    OUTPUT_FILE = OUTPUT_DIR / "gapmind_kofam_rast.tsv"
    CORR_OUTPUT_FILE = OUTPUT_DIR / "correlation_removed_features.json.gz"

    CORRELATION_THRESHOLD = 0.95

    print("=" * 80)
    print("Combining GapMind, KOFAM, and RAST features with correlation filtering")
    print("=" * 80)

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

    print("\nFinding common genomes across all feature matrices...")
    common_genomes = gapmind_features.index.intersection(
        kofam_features.index
    ).intersection(rast_features.index)
    print(f"  Common genomes: {len(common_genomes)}")
    print(f"    GapMind: {len(gapmind_features)}")
    print(f"    KOFAM: {len(kofam_features)}")
    print(f"    RAST: {len(rast_features)}")

    print("\nFiltering to common genomes...")
    gapmind_features = gapmind_features.loc[common_genomes]
    kofam_features = kofam_features.loc[common_genomes]
    rast_features = rast_features.loc[common_genomes]

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

    print(
        f"\nFound {len(kofam_gapmind_correlated)} KOFAM features correlated with GapMind"
    )
    kofam_features_filtered = kofam_features.drop(
        columns=list(kofam_gapmind_correlated.keys())
    )
    print(
        f"Remaining KOFAM features: {kofam_features_filtered.shape[1]} / {kofam_features.shape[1]}"
    )

    print("\n" + "=" * 80)
    print("Step 3: Combine all remaining features")
    print("=" * 80)

    combined_features = pd.concat(
        [gapmind_features, kofam_features_filtered, rast_features_filtered],
        axis=1,
    )

    print(f"\nCombined feature matrix shape: {combined_features.shape}")
    print(f"  Genomes: {combined_features.shape[0]}")
    print(f"  GapMind features: {gapmind_features.shape[1]}")
    print(f"  KOFAM features: {kofam_features_filtered.shape[1]}")
    print(f"  RAST features: {rast_features_filtered.shape[1]}")
    print(f"  Total features: {combined_features.shape[1]}")

    print("\n" + "=" * 80)
    print("Step 4: Save outputs")
    print("=" * 80)

    print(f"\nSaving combined feature matrix to {OUTPUT_FILE}...")
    combined_features.to_csv(OUTPUT_FILE, sep="\t")
    print("  Done!")

    correlation_info = {
        "rast_kofam_correlated": {
            feature: corr_with for feature, corr_with in rast_kofam_correlated.items()
        },
        "kofam_gapmind_correlated": {
            feature: corr_with
            for feature, corr_with in kofam_gapmind_correlated.items()
        },
        "summary": {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "common_genomes": len(common_genomes),
            "rast_features_removed": len(rast_kofam_correlated),
            "kofam_features_removed": len(kofam_gapmind_correlated),
            "original_gapmind_features": gapmind_features.shape[1],
            "original_kofam_features": kofam_features.shape[1],
            "original_rast_features": rast_features.shape[1],
            "final_gapmind_features": gapmind_features.shape[1],
            "final_kofam_features": kofam_features_filtered.shape[1],
            "final_rast_features": rast_features_filtered.shape[1],
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
    print(f"Removed {len(rast_kofam_correlated)} RAST features (correlated with KOFAM)")
    print(
        f"Removed {len(kofam_gapmind_correlated)} KOFAM features (correlated with GapMind)"
    )
    print(f"Final combined feature matrix: {combined_features.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()
