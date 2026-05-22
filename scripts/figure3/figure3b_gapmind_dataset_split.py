#!/usr/bin/env python3
"""
Calculate GapMind metrics for dataset split test sets.

This script calculates GapMind prediction performance metrics specifically
for the test set genomes in each dataset split configuration used in Figure 3B.
This ensures that the GapMind baseline is comparable to the ML model performance,
which is also evaluated only on the test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from scripts.ml_splits import load_split_data


def calculate_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> dict[str, float]:
    """
    Calculate classification metrics for GapMind predictions.

    Parameters
    ----------
    y_true : pd.Series
        True labels (experimental data)
    y_pred : pd.Series
        Predicted labels (GapMind predictions)

    Returns
    -------
    dict[str, float]
        Dictionary of metric names and their values
    """
    # Filter out NaN values (missing experimental data)
    mask = ~y_true.isna()
    y_true_filtered = y_true[mask].astype(int)
    y_pred_filtered = y_pred[mask].astype(int)

    # Skip if no valid data
    if len(y_true_filtered) == 0:
        return {
            "n_samples": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "matthews_corrcoef": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    # Calculate metrics (handle cases where a class might be missing)
    try:
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        balanced_acc = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
        mcc = matthews_corrcoef(y_true_filtered, y_pred_filtered)

        # For precision, recall, F1 - handle cases with no positive predictions
        precision = precision_score(
            y_true_filtered, y_pred_filtered, zero_division=0.0
        )
        recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0.0)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0.0)

        return {
            "n_samples": len(y_true_filtered),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "matthews_corrcoef": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    except Exception as e:
        print(f"    Warning: Could not calculate metrics: {e}")
        return {
            "n_samples": len(y_true_filtered),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "matthews_corrcoef": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }


def extract_phenotype_and_test_dataset(key: str) -> tuple[str, str]:
    """
    Extract phenotype name and test dataset from dataset split key.

    Parameters
    ----------
    key : str
        Key string like "Mannose_train(atleaf+marine+pmi),test(lit)"

    Returns
    -------
    tuple[str, str]
        Tuple of (phenotype_name, test_dataset)
    """
    # Extract phenotype (before first underscore)
    phenotype = key.split("_")[0]

    # Extract test dataset from "test(dataset)" pattern
    test_dataset = key.split("test(")[1].split(")")[0]

    return phenotype, test_dataset


def main() -> None:
    """
    Main function to calculate GapMind metrics for dataset split test sets.
    """
    # Define paths
    SPLITS_DIR = Path("data/processed/train_test_splits")
    GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    OUTPUT_DIR = Path("data/outputs/figure3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load GapMind predictions (loose threshold)
    print(f"Loading GapMind predictions from: {GAPMIND_FILE}")
    gapmind_predictions = pd.read_csv(
        GAPMIND_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"  Loaded {gapmind_predictions.shape[0]} genomes, {gapmind_predictions.shape[1]} phenotypes")

    # Load dataset split data only
    print("\nLoading dataset split configurations...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=["dataset_split"])

    dataset_splits = split_data["dataset_split"]
    print(f"  Loaded {len(dataset_splits)} dataset split configurations")

    # Process each dataset split
    print("\nCalculating GapMind metrics for each dataset split test set...")
    results = []

    for key in tqdm(dataset_splits, desc="Processing dataset splits"):
        split = dataset_splits[key]

        # Get test set data
        y_test = split["y_test"]
        test_genomes = y_test.index.tolist()

        # Extract phenotype and test dataset info
        phenotype, test_dataset = extract_phenotype_and_test_dataset(key)

        # Check if phenotype exists in GapMind predictions
        if phenotype not in gapmind_predictions.columns:
            print(f"\nWarning: Phenotype {phenotype} not found in GapMind predictions, skipping {key}")
            continue

        # Get GapMind predictions for test set genomes
        # Find common genomes between test set and GapMind
        common_genomes = list(set(test_genomes) & set(gapmind_predictions.index))

        if len(common_genomes) == 0:
            print(f"\nWarning: No common genomes found for {key}, skipping")
            continue

        # Get true labels and predictions for common genomes
        y_true = y_test.loc[common_genomes]
        y_pred = gapmind_predictions.loc[common_genomes, phenotype]

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        # Add metadata
        metrics["key"] = key
        metrics["phenotype"] = phenotype
        metrics["test_dataset"] = test_dataset
        metrics["n_test_total"] = len(test_genomes)
        metrics["n_common_genomes"] = len(common_genomes)

        results.append(metrics)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Annotate each row with its full-test minority-class count (Methods).
    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    results_df = annotate_minority_test(results_df, full_test_minority_counts())

    # Save results
    output_file = OUTPUT_DIR / "gapmind_dataset_split_metrics.tsv"
    results_df.to_csv(output_file, sep="\t", index=False)
    print(f"\nSaved results to: {output_file}")

    # Print summary statistics
    print("\nResults summary:")
    print(f"  Total dataset split configurations: {len(results_df)}")
    print(f"\nMean metrics across all dataset splits:")
    for metric in ["accuracy", "balanced_accuracy", "matthews_corrcoef", "precision", "recall", "f1"]:
        mean_val = results_df[metric].mean()
        print(f"    {metric}: {mean_val:.3f}")

    print("\nBy phenotype (mean balanced accuracy):")
    phenotype_summary = (
        results_df.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(phenotype_summary)

    print("\nBy test dataset (mean balanced accuracy):")
    dataset_summary = (
        results_df.groupby("test_dataset")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(dataset_summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
