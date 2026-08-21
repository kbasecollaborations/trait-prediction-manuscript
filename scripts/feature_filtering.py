#!/usr/bin/env python3
"""Apply variance and correlation filtering to combined and per-dataset feature matrices.

Filtering is fitted once on the four datasets pooled, then the retained columns are
written back out per dataset so every matrix carries the same feature set. Reads
``data/interim/features`` and writes ``data/processed/features_reduced``.

Run with::

    uv run python -m scripts.feature_filtering
"""

import gzip
import json
from pathlib import Path

import pandas as pd
from trait_prediction.main import Feature

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
FEATURE_DIR = Path("data/interim/features")
OUTPUT_DIR = Path("data/processed/features_reduced")
DATASET_SUBSET = ["atleaf", "lit", "marine", "pmi"]
FEATURE_TYPES = ["kofam", "rast", "gapmind"]
CORRELATION_METHOD = "spearman"


def load_and_combine_datasets(feature_type: str) -> pd.DataFrame:
    """Load and combine all datasets for a given feature type.

    Parameters
    ----------
    feature_type : str
        Type of feature (kofam, rast, etc.).

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with missing values filled with 0.
    """
    print(f"\nLoading and combining {feature_type} features across datasets...")

    combined_features = []
    for dataset_name in DATASET_SUBSET:
        feature_file = FEATURE_DIR / dataset_name / f"{feature_type}.tsv"
        if feature_file.exists():
            feature_data = pd.read_csv(
                feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
            )
            print(f"  Loaded {dataset_name}: {feature_data.shape}")
            combined_features.append(feature_data)
        else:
            print(f"  Warning: {feature_file} not found, skipping")

    combined_df = pd.concat(combined_features, axis=0, sort=False)
    combined_df = combined_df.fillna(0).astype(int)

    print(f"  Combined shape (before filtering): {combined_df.shape}")

    return combined_df


def filter_combined_features(
    combined_df: pd.DataFrame, feature_type: str
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Apply variance and correlation filtering to combined features.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined feature matrix across all datasets.
    feature_type : str
        Type of feature (kofam, rast, etc.).

    Returns
    -------
    tuple[pd.DataFrame, list[str], dict[str, list[str]]]
        Tuple of (filtered features, low variance features, high correlation features dict).
    """
    print(f"\nFiltering {feature_type} features on combined dataset...")

    filtered_df, low_var_features = Feature.remove_features_with_low_variance(
        combined_df, VARIANCE_THRESHOLD
    )
    print(f"  After variance filtering: {filtered_df.shape[1]} features remaining")

    filtered_df, high_corr_features_dict = (
        Feature.remove_features_with_high_correlation(
            filtered_df, CORRELATION_THRESHOLD, parallel=True, method=CORRELATION_METHOD
        )
    )
    print(f"  After correlation filtering: {filtered_df.shape[1]} features remaining")

    filtered_df = filtered_df.astype(int)

    return filtered_df, low_var_features, high_corr_features_dict


def save_combined_features(
    feature_type: str,
    combined_df: pd.DataFrame,
    low_var_features: list[str],
    high_corr_features_dict: dict[str, list[str]],
) -> None:
    """Save combined filtered features and metadata.

    Parameters
    ----------
    feature_type : str
        Type of feature.
    combined_df : pd.DataFrame
        Combined filtered feature matrix.
    low_var_features : list[str]
        List of removed low variance features.
    high_corr_features_dict : dict[str, list[str]]
        Dictionary of removed high correlation features.
    """
    output_dir = OUTPUT_DIR / "combined_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{feature_type}.tsv"
    combined_df.to_csv(output_file, sep="\t", index=True)
    print(f"  Saved combined filtered features to {output_file}")

    low_var_file = output_dir / f"{feature_type}_low_var_features.txt"
    with open(low_var_file, "w") as f:
        f.write("\n".join(low_var_features))
    print(f"  Saved {len(low_var_features)} low variance features to {low_var_file}")

    corr_file = output_dir / f"{feature_type}_corr_features.json.gz"
    with gzip.open(corr_file, "wt") as f:
        json.dump(high_corr_features_dict, f)
    print(f"  Saved correlation clusters to {corr_file}")


def save_individual_dataset_features(
    feature_type: str, combined_filtered_df: pd.DataFrame
) -> None:
    """Extract and save individual dataset features from combined filtered matrix.

    Parameters
    ----------
    feature_type : str
        Type of feature.
    combined_filtered_df : pd.DataFrame
        Combined filtered feature matrix containing all datasets.
    """
    print(f"\nSaving individual dataset features for {feature_type}...")

    for dataset_name in DATASET_SUBSET:
        feature_file = FEATURE_DIR / dataset_name / f"{feature_type}.tsv"
        if not feature_file.exists():
            print(f"  Warning: {feature_file} not found, skipping {dataset_name}")
            continue

        original_data = pd.read_csv(
            feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
        )

        dataset_samples = original_data.index
        dataset_filtered = combined_filtered_df.loc[dataset_samples]

        output_dir = OUTPUT_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{feature_type}.tsv"
        dataset_filtered.to_csv(output_file, sep="\t", index=True)
        print(f"  Saved {dataset_name}: {dataset_filtered.shape} to {output_file}")


def main() -> None:
    """Filter each feature type and write combined and per-dataset matrices."""
    print("=" * 80)
    print("Feature Filtering Pipeline")
    print("=" * 80)
    print(f"Variance threshold: {VARIANCE_THRESHOLD}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"Datasets: {', '.join(DATASET_SUBSET)}")
    print(f"Feature types: {', '.join(FEATURE_TYPES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    for feature_type in FEATURE_TYPES:
        print("\n" + "=" * 80)
        print(f"Processing {feature_type} features")
        print("=" * 80)

        combined_df = load_and_combine_datasets(feature_type)
        filtered_df, low_var_features, high_corr_features_dict = (
            filter_combined_features(combined_df, feature_type)
        )
        save_combined_features(
            feature_type, filtered_df, low_var_features, high_corr_features_dict
        )
        save_individual_dataset_features(feature_type, filtered_df)

    print("\n" + "=" * 80)
    print("Feature filtering complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
