#!/usr/bin/env python3

"""Generate strict and loose GapMind phenotype prediction files and evaluate them.

Strict marks only 'complete' as present; loose also counts 'likely_complete'.
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


def create_gapmind_predictions(
    input_file: Path,
    output_dir: Path,
    strict_mapping: dict[str, int],
    loose_mapping: dict[str, int],
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
    print(f"Reading GapMind data from: {input_file}")
    gapmind_data = pd.read_csv(
        input_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )

    print(f"Loaded data shape: {gapmind_data.shape}")
    print(f"Number of genomes: {len(gapmind_data)}")
    print(f"Number of phenotypes: {len(gapmind_data.columns)}")

    print("\nCreating strict predictions...")
    gapmind_strict = gapmind_data.replace(strict_mapping).astype(np.uint8)
    strict_output = output_dir / "gapmind_phenotypes_strict.tsv"
    gapmind_strict.to_csv(strict_output, sep="\t")
    print(f"Saved strict predictions to: {strict_output}")
    print(f"  Total positive predictions: {gapmind_strict.sum().sum()}")

    print("\nCreating loose predictions...")
    gapmind_loose = gapmind_data.replace(loose_mapping).astype(np.uint8)
    loose_output = output_dir / "gapmind_phenotypes_loose.tsv"
    gapmind_loose.to_csv(loose_output, sep="\t")
    print(f"Saved loose predictions to: {loose_output}")
    print(f"  Total positive predictions: {gapmind_loose.sum().sum()}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print("\nStrict predictions (complete only):")
    print(f"  - Positives per genome (mean): {gapmind_strict.sum(axis=1).mean():.2f}")
    print(
        f"  - Positives per phenotype (mean): {gapmind_strict.sum(axis=0).mean():.2f}"
    )
    print("\nLoose predictions (complete + likely_complete):")
    print(f"  - Positives per genome (mean): {gapmind_loose.sum(axis=1).mean():.2f}")
    print(f"  - Positives per phenotype (mean): {gapmind_loose.sum(axis=0).mean():.2f}")


def load_experimental_phenotypes(phenotype_dir: Path) -> pd.DataFrame:
    """
    Load and combine experimental phenotype data from all datasets.

    Parameters
    ----------
    phenotype_dir : Path
        Path to the directory containing phenotype subdirectories (atleaf, lit, marine, pmi)

    Returns
    -------
    pd.DataFrame
        Combined phenotype data with genomeID as index and phenotypes as columns
    """
    print(f"\nLoading experimental phenotype data from: {phenotype_dir}")

    phenotype_data = {}

    for dataset_dir in phenotype_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"  Processing dataset: {dataset_name}")

        for phenotype_file in dataset_dir.glob("*.tsv"):
            phenotype_name = phenotype_file.stem

            df = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})

            if phenotype_name not in phenotype_data:
                phenotype_data[phenotype_name] = []

            phenotype_data[phenotype_name].append(df)

    combined_phenotypes = {}
    for phenotype_name, df_list in phenotype_data.items():
        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.drop_duplicates(subset=["genomeID"], keep="first")
        combined = combined.set_index("genomeID")
        combined_phenotypes[phenotype_name] = combined[phenotype_name]

    experimental_data = pd.DataFrame(combined_phenotypes)

    print(f"\nCombined experimental data shape: {experimental_data.shape}")
    print(f"  Genomes: {len(experimental_data)}")
    print(f"  Phenotypes: {len(experimental_data.columns)}")

    return experimental_data


def calculate_metrics(
    y_true: pd.Series, y_pred: pd.Series, phenotype_name: str
) -> dict[str, float]:
    """
    Calculate classification metrics for a single phenotype.

    Parameters
    ----------
    y_true : pd.Series
        True labels (experimental data)
    y_pred : pd.Series
        Predicted labels (GapMind predictions)
    phenotype_name : str
        Name of the phenotype being evaluated

    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and their values
    """
    mask = ~y_true.isna()
    y_true_filtered = y_true[mask].astype(int)
    y_pred_filtered = y_pred[mask].astype(int)

    if len(y_true_filtered) == 0:
        return {
            "phenotype": phenotype_name,
            "n_samples": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "matthews_corrcoef": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    try:
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        balanced_acc = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
        mcc = matthews_corrcoef(y_true_filtered, y_pred_filtered)

        # zero_division=0 guards against splits with no positive predictions
        precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0.0)
        recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0.0)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0.0)

        return {
            "phenotype": phenotype_name,
            "n_samples": len(y_true_filtered),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "matthews_corrcoef": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    except Exception as e:
        print(f"    Warning: Could not calculate metrics for {phenotype_name}: {e}")
        return {
            "phenotype": phenotype_name,
            "n_samples": len(y_true_filtered),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "matthews_corrcoef": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }


def evaluate_predictions(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    prediction_type: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate GapMind predictions against experimental data.

    Parameters
    ----------
    predictions : pd.DataFrame
        GapMind predictions (strict or loose)
    experimental : pd.DataFrame
        Experimental phenotype data
    prediction_type : str
        Type of predictions ('strict' or 'loose')
    output_dir : Path
        Directory to save evaluation results

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each phenotype
    """
    print(f"\nEvaluating {prediction_type} predictions...")

    common_phenotypes = set(predictions.columns) & set(experimental.columns)
    print(f"  Common phenotypes: {len(common_phenotypes)}")

    common_genomes = predictions.index.intersection(experimental.index)
    print(f"  Common genomes: {len(common_genomes)}")

    if len(common_phenotypes) == 0 or len(common_genomes) == 0:
        print(
            f"  Warning: No common phenotypes or genomes found for {prediction_type}!"
        )
        return pd.DataFrame()

    metrics_list = []
    for phenotype in sorted(common_phenotypes):
        y_true = experimental.loc[common_genomes, phenotype]
        y_pred = predictions.loc[common_genomes, phenotype]

        metrics = calculate_metrics(y_true, y_pred, phenotype)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values("phenotype").reset_index(drop=True)

    output_file = output_dir / f"gapmind_{prediction_type}_metrics.tsv"
    metrics_df.to_csv(output_file, sep="\t", index=False)
    print(f"  Saved metrics to: {output_file}")

    print(f"\n  Metrics summary for {prediction_type}:")
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "matthews_corrcoef",
        "precision",
        "recall",
        "f1",
    ]:
        mean_val = metrics_df[metric].mean()
        print(f"    {metric}: {mean_val:.3f}")

    return metrics_df


def main() -> None:
    """Write the strict and loose GapMind predictions and their metrics."""
    input_file = Path("data/interim/gapmind/gapmind_phenotype_data_raw.tsv")
    output_dir = Path("data/outputs/figure2")
    phenotype_dir = Path("data/processed/phenotypes")

    output_dir.mkdir(parents=True, exist_ok=True)

    strict_mapping = {
        "complete": 1,
        "likely_complete": 0,
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }

    loose_mapping = {
        "complete": 1,
        "likely_complete": 1,
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }

    create_gapmind_predictions(input_file, output_dir, strict_mapping, loose_mapping)

    experimental_data = load_experimental_phenotypes(phenotype_dir)

    print("\n" + "=" * 60)
    print("EVALUATING PREDICTIONS AGAINST EXPERIMENTAL DATA")
    print("=" * 60)

    strict_predictions = pd.read_csv(
        output_dir / "gapmind_phenotypes_strict.tsv",
        sep="\t",
        index_col=0,
        dtype={"genomeID": str},
    )
    loose_predictions = pd.read_csv(
        output_dir / "gapmind_phenotypes_loose.tsv",
        sep="\t",
        index_col=0,
        dtype={"genomeID": str},
    )

    strict_metrics = evaluate_predictions(
        strict_predictions, experimental_data, "strict", output_dir
    )

    loose_metrics = evaluate_predictions(
        loose_predictions, experimental_data, "loose", output_dir
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
