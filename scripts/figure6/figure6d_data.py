#!/usr/bin/env python3
"""Generate Figure 6D data: combined vs phenotype-filtered GapMind feature experiments."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.create_data_splits import COMMON_PHENOTYPES
from scripts.ml_splits import load_split_data


def run_ml_on_splits(
    split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    model_type: str = "cb",
    random_state: int = 42,
    min_test_samples: int = 10,
    experiment_name: str = "combined",
) -> pd.DataFrame:
    """
    Run machine learning on all loaded splits.

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
    experiment_name : str, optional
        Name of the experiment (for tracking), by default "combined"

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - All scoring metrics (accuracy, balanced_accuracy, etc.)
        - features: List of top feature names
        - split_type: Type of split (random_split, dataset_split)
        - key: Unique key for the split (e.g., "Alanine_0")
        - phenotype: Phenotype name
        - model_type: Model type used
        - n_train, n_val, n_test: Number of samples in each set
        - n_features: Number of features used
        - experiment: Experiment name (combined or phenotype_filtered)
    """
    from scripts.ml_splits import perform_split_ml

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

    with tqdm(
        total=total_splits, desc=f"Running ML on {experiment_name} features"
    ) as pbar:
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
                result["n_features"] = X_train.shape[1]
                result["experiment"] = experiment_name

                results.append(result)

    return pd.DataFrame(results)


def filter_gapmind_by_phenotype(
    gapmind_features: pd.DataFrame, phenotype_name: str
) -> pd.DataFrame:
    """
    Filter GapMind features by phenotype prefix.

    Parameters
    ----------
    gapmind_features : pd.DataFrame
        Full GapMind feature matrix.
    phenotype_name : str
        Phenotype name to filter by (e.g., "Alanine", "Fructose").

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix containing only features for the specified phenotype.
    """
    phenotype_columns = [
        col for col in gapmind_features.columns if col.startswith(f"{phenotype_name}-")
    ]

    if len(phenotype_columns) == 0:
        raise ValueError(f"No features found for phenotype: {phenotype_name}")

    return gapmind_features[phenotype_columns]


def run_phenotype_filtered_experiment(
    gapmind_file: Path,
    splits_dir: Path,
    split_types: list[str],
    phenotype_list: list[str],
) -> pd.DataFrame:
    """
    Run ML per phenotype using only that phenotype's GapMind feature columns.

    Parameters
    ----------
    gapmind_file : Path
        Path to unreduced GapMind feature matrix.
    splits_dir : Path
        Base directory containing train-test splits.
    split_types : list[str]
        List of split types to use (e.g., ["random_split", "dataset_split"]).
    phenotype_list : list[str]
        List of phenotype names to process.

    Returns
    -------
    pd.DataFrame
        Combined results for all phenotypes.
    """
    print("\nLoading full GapMind feature matrix...")
    gapmind_features = pd.read_csv(
        gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"  Shape: {gapmind_features.shape}")

    all_results = []

    for phenotype in tqdm(phenotype_list, desc="Processing phenotypes"):
        try:
            phenotype_features = filter_gapmind_by_phenotype(
                gapmind_features, phenotype
            )
        except ValueError as e:
            print(f"\n{e}, skipping...")
            continue

        print(
            f"\n  {phenotype}: {phenotype_features.shape[1]} features (out of {gapmind_features.shape[1]})"
        )

        split_data_filtered = {}

        for split_type in split_types:
            split_type_dir = splits_dir / split_type

            if not split_type_dir.exists():
                continue

            split_data_filtered[split_type] = {}

            for phenotype_dir in split_type_dir.iterdir():
                if not phenotype_dir.is_dir():
                    continue

                if phenotype_dir.name != phenotype:
                    continue

                for repeat_dir in phenotype_dir.iterdir():
                    if not repeat_dir.is_dir():
                        continue

                    key = f"{phenotype}_{repeat_dir.name}"

                    from scripts.ml_splits import load_single_split_data

                    data = load_single_split_data(repeat_dir, phenotype_features)
                    split_data_filtered[split_type][key] = data

        if not split_data_filtered or all(
            len(v) == 0 for v in split_data_filtered.values()
        ):
            print(f"  No splits found for {phenotype}, skipping...")
            continue

        results = run_ml_on_splits(
            split_data_filtered,
            model_type="cb",
            random_state=42,
            min_test_samples=10,
            experiment_name="phenotype_filtered",
        )

        all_results.append(results)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def main() -> None:
    """Generate Figure 6D data."""
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure6")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    COMBINED_FEATURES_FILE = Path(
        "data/processed/features_reduced/combined_datasets/gapmind_kofam_rast.tsv"
    )
    GAPMIND_FEATURES_FILE = Path(
        "data/interim/features/combined_datasets/gapmind.tsv"
    )

    SPLIT_TYPES = ["random_split", "dataset_split"]

    # Common phenotypes: taken from the split-generation module rather than a
    # local copy, which had silently dropped Glucose and so excluded it from the
    # phenotype-filtered experiment (the combined experiment reads the splits
    # directory directly and was unaffected).
    PHENOTYPES = list(COMMON_PHENOTYPES)

    print("=" * 80)
    print("Figure 6D: Comparing combined vs phenotype-filtered features")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Experiment 1: Combined features (GapMind + KOFAM + RAST)")
    print("=" * 80)

    if not COMBINED_FEATURES_FILE.exists():
        print(f"\nERROR: Combined features file not found: {COMBINED_FEATURES_FILE}")
        print("Please run scripts/combine_features.py first to generate this file.")
        return

    print("\nLoading train-test splits with combined features...")
    split_data_combined = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=SPLIT_TYPES,
        feature_file=COMBINED_FEATURES_FILE,
    )

    print("\nLoaded splits summary:")
    for split_type in split_data_combined:
        print(f"  {split_type}: {len(split_data_combined[split_type])} splits")

    print("\nRunning machine learning with combined features...")
    results_combined = run_ml_on_splits(
        split_data_combined,
        model_type="cb",
        random_state=42,
        min_test_samples=10,
        experiment_name="combined",
    )

    results_file_combined = OUTPUT_DIR / "figure6d_combined_features_results.csv"
    results_combined.to_csv(results_file_combined, index=False)
    print(f"\nSaved combined features results to: {results_file_combined}")

    print("\n" + "=" * 80)
    print("Experiment 2: Phenotype-filtered features (GapMind only)")
    print("=" * 80)

    if not GAPMIND_FEATURES_FILE.exists():
        print(
            f"\nERROR: GapMind features file not found: {GAPMIND_FEATURES_FILE}"
        )
        return

    results_filtered = run_phenotype_filtered_experiment(
        gapmind_file=GAPMIND_FEATURES_FILE,
        splits_dir=SPLITS_DIR,
        split_types=SPLIT_TYPES,
        phenotype_list=PHENOTYPES,
    )

    if not results_filtered.empty:
        results_file_filtered = OUTPUT_DIR / "figure6d_phenotype_filtered_results.csv"
        results_filtered.to_csv(results_file_filtered, index=False)
        print(f"\nSaved phenotype-filtered results to: {results_file_filtered}")
    else:
        print("\nNo results generated for phenotype-filtered experiment.")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    all_results = pd.concat(
        [results_combined, results_filtered], ignore_index=True
    )

    # Annotate each row with its full-test minority-class count (Methods).
    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    all_results = annotate_minority_test(all_results, full_test_minority_counts())

    results_file_all = OUTPUT_DIR / "figure6d_all_results.csv"
    all_results.to_csv(results_file_all, index=False)
    print(f"\nSaved all results to: {results_file_all}")

    if len(all_results) > 0:
        print("\nBy experiment and split type:")
        summary = (
            all_results.groupby(["experiment", "split_type"])["balanced_accuracy"]
            .describe()
            .round(3)
        )
        print(summary)

        print("\nBy experiment and phenotype (mean balanced accuracy):")
        phenotype_summary = (
            all_results.groupby(["experiment", "phenotype"])["balanced_accuracy"]
            .agg(["mean", "std", "count"])
            .round(3)
            .sort_values(["experiment", "mean"], ascending=[True, False])
        )
        print(phenotype_summary)
    else:
        print("\nNo results generated. All experiments were skipped.")
        print("Check if test sets are too small or don't have both classes.")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
