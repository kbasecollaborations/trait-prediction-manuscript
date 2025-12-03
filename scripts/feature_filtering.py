#!/usr/bin/env python3
"""Script to apply variance and correlation filtering to feature matrices.

This script:
1. Applies variance and correlation filtering to each dataset's features
2. Saves filtered features to data/processed/features_reduced/{dataset}/
3. Creates combined_datasets folder with features combined across all datasets (after filtering)
4. Fills missing columns with 0 and ensures all columns are integers
"""

import gzip
import json
from pathlib import Path

import pandas as pd
from trait_prediction.main import Feature

# Constants
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
FEATURE_DIR = Path("data/interim/features")
OUTPUT_DIR = Path("data/processed/features_reduced")
DATASET_SUBSET = ["atleaf", "lit", "marine", "pmi"]
FEATURE_TYPES = ["kofam", "rast"]


def filter_dataset_features(
    dataset_name: str, feature_type: str
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Apply variance and correlation filtering to a dataset's features.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (atleaf, lit, marine, pmi).
    feature_type : str
        Type of feature (kofam, rast, etc.).

    Returns
    -------
    tuple[pd.DataFrame, list[str], dict[str, list[str]]]
        Tuple of (filtered features, low variance features, high correlation features dict).
    """
    print(f"\nProcessing {dataset_name} - {feature_type}...")

    # Load feature data
    feature_file = FEATURE_DIR / dataset_name / f"{feature_type}.tsv"
    feature_data = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(
        f"  Loaded {feature_data.shape[0]} samples x {feature_data.shape[1]} features"
    )

    # Apply variance filtering using Feature class method
    feature_data, low_var_features = Feature.remove_features_with_low_variance(
        feature_data, VARIANCE_THRESHOLD
    )
    print(f"  After variance filtering: {feature_data.shape[1]} features remaining")

    # Apply correlation filtering using Feature class method
    feature_data, high_corr_features_dict = (
        Feature.remove_features_with_high_correlation(
            feature_data, CORRELATION_THRESHOLD, parallel=True
        )
    )
    print(f"  After correlation filtering: {feature_data.shape[1]} features remaining")

    # Ensure all values are integers
    feature_data = feature_data.astype(int)

    return feature_data, low_var_features, high_corr_features_dict


def save_filtered_features(
    dataset_name: str,
    feature_type: str,
    feature_data: pd.DataFrame,
    low_var_features: list[str],
    high_corr_features_dict: dict[str, list[str]],
) -> None:
    """Save filtered features and metadata.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    feature_type : str
        Type of feature.
    feature_data : pd.DataFrame
        Filtered feature data.
    low_var_features : list[str]
        List of removed low variance features.
    high_corr_features_dict : dict[str, list[str]]
        Dictionary of removed high correlation features.
    """
    output_dir = OUTPUT_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered features
    output_file = output_dir / f"{feature_type}.tsv"
    feature_data.to_csv(output_file, sep="\t", index=True)
    print(f"  Saved filtered features to {output_file}")

    # Save low variance features
    low_var_file = output_dir / f"{feature_type}_low_var_features.txt"
    with open(low_var_file, "w") as f:
        f.write("\n".join(low_var_features))
    print(f"  Saved {len(low_var_features)} low variance features to {low_var_file}")

    # Save high correlation features
    corr_file = output_dir / f"{feature_type}_corr_features.json.gz"
    with gzip.open(corr_file, "wt") as f:
        json.dump(high_corr_features_dict, f)
    print(f"  Saved correlation clusters to {corr_file}")


def combine_datasets_for_feature_type(feature_type: str) -> pd.DataFrame:
    """Combine filtered features across all datasets for a given feature type.

    Parameters
    ----------
    feature_type : str
        Type of feature (kofam, rast, etc.).

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with missing values filled with 0.
    """
    print(f"\nCombining {feature_type} features across datasets...")

    combined_features = []
    for dataset_name in DATASET_SUBSET:
        feature_file = OUTPUT_DIR / dataset_name / f"{feature_type}.tsv"
        if feature_file.exists():
            feature_data = pd.read_csv(
                feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
            )
            print(f"  Loaded {dataset_name}: {feature_data.shape}")
            combined_features.append(feature_data)
        else:
            print(f"  Warning: {feature_file} not found, skipping")

    # Concatenate and fill missing values with 0
    combined_df = pd.concat(combined_features, axis=0, sort=False)
    combined_df = combined_df.fillna(0).astype(int)

    print(f"  Combined shape: {combined_df.shape}")

    return combined_df


def save_combined_features(feature_type: str, combined_df: pd.DataFrame) -> None:
    """Save combined features across datasets.

    Parameters
    ----------
    feature_type : str
        Type of feature.
    combined_df : pd.DataFrame
        Combined feature matrix.
    """
    output_dir = OUTPUT_DIR / "combined_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{feature_type}.tsv"
    combined_df.to_csv(output_file, sep="\t", index=True)
    print(f"  Saved combined features to {output_file}")


def main() -> None:
    """Main function to filter features and create combined datasets."""
    print("=" * 80)
    print("Feature Filtering Pipeline")
    print("=" * 80)
    print(f"Variance threshold: {VARIANCE_THRESHOLD}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"Datasets: {', '.join(DATASET_SUBSET)}")
    print(f"Feature types: {', '.join(FEATURE_TYPES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Step 1: Filter features for each dataset
    print("\n" + "=" * 80)
    print("STEP 1: Filtering features for each dataset")
    print("=" * 80)

    for dataset_name in DATASET_SUBSET:
        for feature_type in FEATURE_TYPES:
            feature_data, low_var_features, high_corr_features_dict = (
                filter_dataset_features(dataset_name, feature_type)
            )
            save_filtered_features(
                dataset_name,
                feature_type,
                feature_data,
                low_var_features,
                high_corr_features_dict,
            )

    # Step 2: Combine features across datasets
    print("\n" + "=" * 80)
    print("STEP 2: Combining features across datasets")
    print("=" * 80)

    for feature_type in FEATURE_TYPES:
        combined_df = combine_datasets_for_feature_type(feature_type)
        save_combined_features(feature_type, combined_df)

    print("\n" + "=" * 80)
    print("Feature filtering complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
