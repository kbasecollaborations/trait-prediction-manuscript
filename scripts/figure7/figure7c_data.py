#!/usr/bin/env python3
"""
Generate data for Figure 6C: Impact of filtering problematic samples on P/R/AUPRC.

This script:
1. Identifies problematic samples from Figure 6A analysis
2. Trains ML models on dataset splits
3. Evaluates on full test set and filtered test set (excluding problematic samples)
4. Calculates precision, recall, and AUPRC for both conditions
5. Saves results for comparison plotting
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml


def load_gapmind_predictions(phenotype_dict: dict[str, str]) -> pd.DataFrame:
    """
    Load GapMind predictions.

    Parameters
    ----------
    phenotype_dict : dict[str, str]
        Mapping of phenotype keys to display names.

    Returns
    -------
    pd.DataFrame
        GapMind predictions (0/1 encoded).
    """
    import json

    from scripts.io import index_format_func

    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    if marine_ids_file.exists():
        with open(marine_ids_file, "r") as f:
            marine_ids_map = {v.rsplit("_", 2)[0]: k for k, v in json.load(f).items()}
    else:
        marine_ids_map = {}

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict.keys()]
    datasets = ["s__at-leaf-lit-pmi", "s__marine-seqs"]
    gapmind_data_list = [
        pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{dataset}_categories.csv")
        for dataset in datasets
    ]
    gapmind_data = pd.concat(gapmind_data_list, axis=0)
    gapmind_data["genomeId"] = (
        gapmind_data["genome_id"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    gapmind_data.index = gapmind_data["genomeId"]  # type: ignore
    gapmind_data.index = [marine_ids_map.get(ind, ind) for ind in gapmind_data.index]
    gapmind_data = gapmind_data.loc[:, gapmind_phenotype_subset]
    gapmind_data.columns = gapmind_data.columns.str.replace("Carbon__", "")
    gapmind_data.columns = gapmind_data.columns.map(phenotype_dict)  # type: ignore

    replace_dict = {
        "complete": 1,
        "likely_complete": 1,
        "steps_missing_medium": 0,
        "steps_missing_low": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }
    gapmind_data_replaced = gapmind_data.replace(replace_dict)
    return gapmind_data_replaced.astype(np.uint8)


def load_experimental_phenotypes(phenotype_dict: dict[str, str]) -> pd.DataFrame:
    """
    Load experimental phenotype data.

    Parameters
    ----------
    phenotype_dict : dict[str, str]
        Mapping of phenotype keys to display names.

    Returns
    -------
    pd.DataFrame
        Combined experimental phenotype data.
    """
    from scripts.io import read_phenotypes

    phenotype_files_all = Path("data/processed/phenotypes").glob("**/*.tsv")
    phenotype_files = []
    dataset_phenotype_name_map = defaultdict(set)
    for phenotype_file in phenotype_files_all:
        if phenotype_file.parent.stem in ["bacdive", "combined_phenotypes"]:
            continue
        dataset = phenotype_file.parent.stem
        phenotype_files.append(phenotype_file)
        phenotype_data = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})
        dataset_phenotype_name_map[dataset].add(phenotype_file.stem)

    common_phenotypes = sorted(set.intersection(*dataset_phenotype_name_map.values()))
    phenotype_files = [p for p in phenotype_files if p.stem in common_phenotypes]
    phenotype_set = read_phenotypes(phenotype_files)

    phenotypes_combined_dict = dict()
    for phenotype in phenotype_set:
        phenotype_name = phenotype.pindex.name
        if phenotype_name not in phenotypes_combined_dict:
            phenotypes_combined_dict[phenotype_name] = phenotype.phenotype_data.dropna()
        else:
            phenotypes_combined_dict[phenotype_name] = pd.concat(
                [phenotypes_combined_dict[phenotype_name], phenotype.phenotype_data],
                axis=0,
            )

    for phenotype_name, phenotype_data in phenotypes_combined_dict.items():
        phenotypes_combined_dict[phenotype_name] = phenotype_data[
            ~phenotype_data.index.duplicated(keep="first")
        ]

    return pd.concat(phenotypes_combined_dict.values(), axis=1)


def identify_problematic_samples(
    phenotypes_combined: pd.DataFrame, gapmind_data_pheno: pd.DataFrame
) -> set[str]:
    """
    Identify problematic samples from three GapMind misclassification categories.

    Parameters
    ----------
    phenotypes_combined : pd.DataFrame
        Experimental phenotype data.
    gapmind_data_pheno : pd.DataFrame
        GapMind predictions.

    Returns
    -------
    set[str]
        Set of genome IDs that are problematic (union of all three categories).
    """
    # Category 1: No experimental growth but GapMind predicts growth
    microbes_no_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 0).all(), axis=1)
    ].to_list()

    microbes_gapmind_predicts_growth = []
    for microbe in microbes_no_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if (gapmind_data_pheno.loc[microbe] == 1).any():
                microbes_gapmind_predicts_growth.append(microbe)

    # Category 2: All experimental growth but GapMind incomplete
    microbes_all_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 1).all(), axis=1)
    ].to_list()

    microbes_gapmind_missing_predictions = []
    for microbe in microbes_all_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if not (gapmind_data_pheno.loc[microbe] == 1).all():
                microbes_gapmind_missing_predictions.append(microbe)

    # Category 3: Top 20 most frequently misclassified
    misclassifications = dict()
    phenotype_names = phenotypes_combined.columns

    for phenotype_name in phenotype_names:
        exp_data = phenotypes_combined.loc[:, phenotype_name].dropna().astype(np.uint8)
        gapmind_data_pheno_subset = (
            gapmind_data_pheno.loc[:, phenotype_name].dropna().astype(np.uint8)
        )
        common_inds = exp_data.index.intersection(gapmind_data_pheno_subset.index)
        exp_data = exp_data.loc[common_inds]
        gapmind_data_pheno_subset = gapmind_data_pheno_subset.loc[common_inds]
        misclassified = exp_data[exp_data != gapmind_data_pheno_subset]
        misclassifications[phenotype_name] = misclassified

    missclassified_genomes = []
    for phenotype_name, misclassified in misclassifications.items():
        missclassified_genomes.extend(misclassified.index.unique().tolist())

    missclassified_counts = Counter(missclassified_genomes)
    most_common_misclassified = missclassified_counts.most_common(20)
    top_20_genomes = [genome_id for genome_id, _ in most_common_misclassified]

    # Return union of all three categories
    all_problematic = set(
        microbes_gapmind_predicts_growth
        + microbes_gapmind_missing_predictions
        + top_20_genomes
    )

    print(f"Category 1 (No growth, GM predicts): {len(microbes_gapmind_predicts_growth)}")
    print(f"Category 2 (All growth, GM incomplete): {len(microbes_gapmind_missing_predictions)}")
    print(f"Category 3 (Top 20 misclassified): {len(top_20_genomes)}")
    print(f"Total unique problematic samples: {len(all_problematic)}")

    return all_problematic


def evaluate_with_predictions(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    exclude_samples: set[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate model and calculate precision, recall, AUPRC, and balanced accuracy.

    Parameters
    ----------
    model : Any
        Trained classifier model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test labels.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Validation labels.
    exclude_samples : set[str] | None
        Set of sample IDs to exclude from evaluation. If None, use all samples.

    Returns
    -------
    dict[str, float]
        Dictionary with precision, recall, auprc, balanced_accuracy, and sample counts.
    """
    # Filter samples if needed
    if exclude_samples is not None:
        # Filter test set
        keep_mask_test = ~y_test.index.isin(exclude_samples)
        X_test_filtered = X_test[keep_mask_test]
        y_test_filtered = y_test[keep_mask_test]

        # Filter train set
        keep_mask_train = ~y_train.index.isin(exclude_samples)
        X_train_filtered = X_train[keep_mask_train]
        y_train_filtered = y_train[keep_mask_train]

        # Filter validation set
        keep_mask_val = ~y_val.index.isin(exclude_samples)
        X_val_filtered = X_val[keep_mask_val]
        y_val_filtered = y_val[keep_mask_val]
    else:
        X_test_filtered = X_test
        y_test_filtered = y_test
        X_train_filtered = X_train
        y_train_filtered = y_train
        X_val_filtered = X_val
        y_val_filtered = y_val

    if len(y_test_filtered) == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "auprc": np.nan,
            "balanced_accuracy": np.nan,
            "n_test": len(y_test_filtered),
            "n_train": len(y_train_filtered),
            "n_val": len(y_val_filtered),
        }

    # Check if we have both classes
    if len(np.unique(y_test_filtered)) < 2:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "auprc": np.nan,
            "balanced_accuracy": np.nan,
            "n_test": len(y_test_filtered),
            "n_train": len(y_train_filtered),
            "n_val": len(y_val_filtered),
        }

    # Get predictions
    y_pred = model.predict(X_test_filtered)
    y_pred_proba = model.predict_proba(X_test_filtered)[:, 1]

    # Calculate metrics
    precision = precision_score(y_test_filtered, y_pred, zero_division=0)
    recall = recall_score(y_test_filtered, y_pred, zero_division=0)
    auprc = average_precision_score(y_test_filtered, y_pred_proba)
    balanced_acc = balanced_accuracy_score(y_test_filtered, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
        "balanced_accuracy": balanced_acc,
        "n_test": len(y_test_filtered),
        "n_train": len(y_train_filtered),
        "n_val": len(y_val_filtered),
    }


def run_figure7c_analysis(
    output_dir: Path = Path("data/outputs/figure7"),
    split_type: str = "dataset_split",
) -> None:
    """
    Run Figure 6C analysis: compare metrics before/after filtering problematic samples.

    Parameters
    ----------
    output_dir : Path
        Directory to save output files.
    split_type : str
        Type of split to analyze ("dataset_split" or "random_split").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phenotype mapping
    phenotype_dict = {
        "alanine": "Alanine",
        "arginine": "Arginine",
        "histidine": "Histidine",
        "serine": "Serine",
        "fructose": "Fructose",
        "galactose": "Galactose",
        "glucose": "Glucose",
        "maltose": "Maltose",
        "mannose": "Mannose",
        "sucrose": "Sucrose",
        "myoinositol": "m-Inositol",
        "mannitol": "Mannitol",
        "glycerol": "Glycerol",
        "galacturonate": "Galacturonic-Acid",
        "cellobiose": "Cellobiose",
    }

    print("=" * 80)
    print("Figure 6C: Impact of filtering problematic samples")
    print("=" * 80)

    # Load GapMind and experimental data
    print("\nLoading GapMind predictions...")
    gapmind_data = load_gapmind_predictions(phenotype_dict)

    print("Loading experimental phenotypes...")
    phenotypes_combined = load_experimental_phenotypes(phenotype_dict)

    # Identify problematic samples
    print("\nIdentifying problematic samples...")
    problematic_samples = identify_problematic_samples(phenotypes_combined, gapmind_data)

    # Load splits
    print(f"\nLoading {split_type} data...")
    splits_data = load_split_data(split_types=[split_type])

    if split_type not in splits_data or len(splits_data[split_type]) == 0:
        raise ValueError(f"No data found for split type: {split_type}")

    # Process each split
    results = []
    split_dict = splits_data[split_type]

    for split_key, data in tqdm(split_dict.items(), desc=f"Processing {split_type}"):
        # Extract phenotype name and split number
        parts = split_key.split("_")
        phenotype_name = "_".join(parts[:-1])  # Everything except last part
        split_num = parts[-1]  # Last part is the split number

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Train model
        from scripts.ml import make_classifier

        model = make_classifier("cb", random_state=42)

        # Align validation data
        X_val_aligned = X_val.copy()
        missing_cols_val = X_train.columns.difference(X_val_aligned.columns)
        if len(missing_cols_val) > 0:
            missing_df = pd.DataFrame(0, index=X_val_aligned.index, columns=missing_cols_val)
            X_val_aligned = pd.concat([X_val_aligned, missing_df], axis=1)
        X_val_aligned = X_val_aligned[X_train.columns]

        # Align test data
        X_test_aligned = X_test.copy()
        missing_cols_test = X_train.columns.difference(X_test_aligned.columns)
        if len(missing_cols_test) > 0:
            missing_df = pd.DataFrame(0, index=X_test_aligned.index, columns=missing_cols_test)
            X_test_aligned = pd.concat([X_test_aligned, missing_df], axis=1)
        X_test_aligned = X_test_aligned[X_train.columns]

        # Train
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val_aligned, y_val),
            use_best_model=True,
            verbose=False,
        )

        # Evaluate on full test set
        metrics_full = evaluate_with_predictions(
            model, X_test_aligned, y_test, X_train, y_train, X_val_aligned, y_val, exclude_samples=None
        )

        # Evaluate on filtered test set (excluding problematic samples)
        metrics_filtered = evaluate_with_predictions(
            model, X_test_aligned, y_test, X_train, y_train, X_val_aligned, y_val, exclude_samples=problematic_samples
        )

        # Store results
        for condition, metrics in [("full", metrics_full), ("filtered", metrics_filtered)]:
            results.append(
                {
                    "phenotype": phenotype_name,
                    "split": split_num,
                    "condition": condition,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "auprc": metrics["auprc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "n_test": metrics["n_test"],
                    "n_train": metrics["n_train"],
                    "n_val": metrics["n_val"],
                }
            )

    # Save results
    results_df = pd.DataFrame(results)

    # Annotate each row with its full-test minority-class count (Methods). The
    # held-out dataset is encoded in the train(...),test(...) "split" column.
    if split_type == "dataset_split":
        from scripts.minority_filter import (
            annotate_minority_test,
            full_test_minority_counts,
        )

        results_df = annotate_minority_test(
            results_df, full_test_minority_counts(), key_column="split"
        )

    output_file = output_dir / f"figure7c_{split_type}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    if len(results_df) > 0:
        summary = results_df.groupby(["phenotype", "condition"]).agg(
            {
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "auprc": ["mean", "std"],
                "balanced_accuracy": ["mean", "std"],
                "n_test": "mean",
                "n_train": "mean",
                "n_val": "mean",
            }
        )
        print(summary)
    else:
        print("\nNo results generated. All experiments were skipped.")
        print("Check if test sets are too small or don't have both classes.")


if __name__ == "__main__":
    run_figure7c_analysis()
