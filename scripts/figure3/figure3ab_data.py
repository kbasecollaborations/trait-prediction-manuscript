#!/usr/bin/env python3
"""Evaluate model performance across train-test split types for Figure 3.

Covers random, dataset, and phylogenetic out-of-clade and in-clade splits.

Run with::

    uv run python -m scripts.figure3.figure3ab_data
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml


def run_ml_on_splits(
    split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    model_type: str = "cb",
    random_state: int = 42,
    min_test_samples: int = 10,
) -> pd.DataFrame:
    """Run machine learning on all loaded splits.

    Parameters
    ----------
    split_data : dict
        Nested dictionary from load_split_data() containing all splits
    model_type : str, optional
        Model type to use ('cb', 'rf', 'dt', etc.), by default "cb"
    random_state : int, optional
        Random state for reproducibility, by default 42
    min_test_samples : int, optional
        Minimum number of test samples required to run ML, by default 10

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - All scoring metrics (accuracy, balanced_accuracy, etc.)
        - features: List of top feature names
        - split_type: Type of split (random_split, dataset_split, phylo_ooc, phylo_ic)
        - key: Unique key for the split (e.g., "Alanine_0")
        - phenotype: Phenotype name
        - model_type: Model type used
        - n_train, n_val, n_test: Number of samples in each set
    """
    results = []
    scoring = [
        "accuracy",
        "balanced_accuracy",
        "matthews_corrcoef",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "roc_auc",
    ]

    total_splits = sum(len(splits) for splits in split_data.values())

    with tqdm(total=total_splits, desc="Running ML on splits") as pbar:
        for split_type in split_data:
            for key in split_data[split_type]:
                pbar.set_postfix_str(f"{split_type}/{key}")
                pbar.update(1)

                split = split_data[split_type][key]
                X_train = split["X_train"]
                y_train = split["y_train"]
                X_val = split["X_val"]
                y_val = split["y_val"]
                X_test = split["X_test"]
                y_test = split["y_test"]

                n_test_samples = len(X_test)
                if n_test_samples < min_test_samples:
                    print(
                        f"\nSkipping {split_type}/{key}: test set has only {n_test_samples} samples"
                    )
                    continue

                # CatBoost needs both classes present in train and validation
                if len(y_train.unique()) != 2 or len(y_val.unique()) != 2:
                    print(
                        f"\nSkipping {split_type}/{key}: training or validation set doesn't have 2 classes"
                    )
                    continue

                result = perform_split_ml(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    model_type=model_type,
                    scoring=scoring,
                    random_state=random_state,
                )

                result["split_type"] = split_type
                result["key"] = key
                result["phenotype"] = key.split("_")[0]
                result["model_type"] = model_type
                result["n_train"] = len(X_train)
                result["n_val"] = len(X_val)
                result["n_test"] = len(X_test)

                results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    """Generate Figure 3 data from train-test splits."""
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SPLIT_TYPES = ["random_split", "dataset_split", "phylo_ooc", "phylo_ic"]

    print("Loading train-test splits...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=SPLIT_TYPES)

    print("\nLoaded splits summary:")
    for split_type in split_data:
        print(f"  {split_type}: {len(split_data[split_type])} splits")

    print("\nRunning machine learning on all splits...")
    results = run_ml_on_splits(
        split_data, model_type="cb", random_state=42, min_test_samples=10
    )

    # Annotate each row with its full-test minority-class count (Methods).
    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    results = annotate_minority_test(results, full_test_minority_counts())

    results_file = OUTPUT_DIR / "ml_results.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    print("\nResults summary:")
    print(f"  Total experiments: {len(results)}")
    print("\nBy split type:")
    summary = results.groupby("split_type")["balanced_accuracy"].describe().round(3)
    print(summary)

    print("\nBy phenotype (mean balanced accuracy):")
    phenotype_summary = (
        results.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(phenotype_summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
