#!/usr/bin/env python3
"""
Generate ML results on GapMind-concordant samples for Figure 5A.

This script filters train-test splits to only include samples where GapMind
predictions match experimental data (concordant samples), then performs ML
evaluation similar to Figure 3.

Only processes: random_split, dataset_split, and phylo_ooc (excludes phylo_ic).
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml


def load_gapmind_predictions(gapmind_file: Path) -> pd.DataFrame:
    """
    Load GapMind predictions.

    Parameters
    ----------
    gapmind_file : Path
        Path to GapMind predictions TSV file

    Returns
    -------
    pd.DataFrame
        GapMind predictions with genomeID as index and phenotypes as columns
    """
    gapmind_df = pd.read_csv(
        gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    return gapmind_df


def load_experimental_phenotypes(phenotype_dir: Path) -> pd.DataFrame:
    """
    Load and combine experimental phenotype data from all datasets.

    Parameters
    ----------
    phenotype_dir : Path
        Path to the directory containing phenotype subdirectories

    Returns
    -------
    pd.DataFrame
        Combined phenotype data with genomeID as index and phenotypes as columns
    """
    # Dictionary to store data for each phenotype across all datasets
    phenotype_data: dict[str, list[pd.DataFrame]] = {}

    # Iterate through each dataset directory
    for dataset_dir in phenotype_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Load each phenotype file in the dataset
        for phenotype_file in dataset_dir.glob("*.tsv"):
            phenotype_name = phenotype_file.stem

            # Read the phenotype data
            df = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})

            # Skip if phenotype not already in dictionary
            if phenotype_name not in phenotype_data:
                phenotype_data[phenotype_name] = []

            # Add this dataset's data
            phenotype_data[phenotype_name].append(df)

    # Combine all datasets for each phenotype
    combined_phenotypes = {}
    for phenotype_name, df_list in phenotype_data.items():
        # Concatenate all datasets
        combined = pd.concat(df_list, ignore_index=True)

        # Remove duplicates, keeping the first occurrence
        combined = combined.drop_duplicates(subset=["genomeID"], keep="first")

        # Set genomeID as index
        combined = combined.set_index("genomeID")

        combined_phenotypes[phenotype_name] = combined[phenotype_name]

    # Create a single DataFrame with all phenotypes
    experimental_data = pd.DataFrame(combined_phenotypes)

    return experimental_data


def get_concordant_samples(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> set[str]:
    """
    Get set of genome IDs where GapMind predictions match experimental data.

    Parameters
    ----------
    gapmind_predictions : pd.DataFrame
        GapMind predictions
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype data
    phenotype : str
        Phenotype name to check concordance for

    Returns
    -------
    set[str]
        Set of genome IDs with concordant predictions
    """
    # Check if phenotype exists in both datasets
    if phenotype not in gapmind_predictions.columns:
        return set()
    if phenotype not in experimental_phenotypes.columns:
        return set()

    # Get common genomes
    common_genomes = gapmind_predictions.index.intersection(
        experimental_phenotypes.index
    )

    # Filter to genomes with non-NaN experimental data
    exp_data = experimental_phenotypes.loc[common_genomes, phenotype]
    valid_genomes = exp_data.dropna().index

    # Get predictions for valid genomes
    gapmind_vals = gapmind_predictions.loc[valid_genomes, phenotype]
    exp_vals = experimental_phenotypes.loc[valid_genomes, phenotype]

    # Find concordant samples (where predictions match experimental data)
    concordant_mask = gapmind_vals == exp_vals
    concordant_genomes = set(valid_genomes[concordant_mask])

    return concordant_genomes


def filter_split_to_concordant(
    split_data: dict[str, pd.DataFrame | pd.Series],
    concordant_genomes: set[str],
    min_samples: int = 5,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """
    Filter a train/val/test split to only include concordant samples.

    Parameters
    ----------
    split_data : dict
        Dictionary with keys X_train, y_train, X_val, y_val, X_test, y_test
    concordant_genomes : set[str]
        Set of genome IDs that are concordant
    min_samples : int, optional
        Minimum number of samples required in each set, by default 5

    Returns
    -------
    dict | None
        Filtered split data, or None if insufficient samples remain
    """
    filtered_data = {}

    # Filter each dataset to concordant samples
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        data = split_data[key]
        # Find intersection with concordant genomes
        concordant_in_data = set(data.index) & concordant_genomes
        filtered = data.loc[list(concordant_in_data)]
        filtered_data[key] = filtered

    # Check if we have enough samples in each set
    n_train = len(filtered_data["X_train"])
    n_val = len(filtered_data["X_val"])
    n_test = len(filtered_data["X_test"])

    if n_train < min_samples or n_val < min_samples or n_test < min_samples:
        return None

    # Check if we have both classes in train and val
    y_train_unique = filtered_data["y_train"].unique()
    y_val_unique = filtered_data["y_val"].unique()

    if len(y_train_unique) != 2 or len(y_val_unique) != 2:
        return None

    return filtered_data


def run_ml_on_concordant_splits(
    split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    model_type: str = "cb",
    random_state: int = 42,
    min_test_samples: int = 10,
) -> pd.DataFrame:
    """
    Run machine learning on concordant samples from all loaded splits.

    Parameters
    ----------
    split_data : dict
        Nested dictionary from load_split_data() containing all splits
    gapmind_predictions : pd.DataFrame
        GapMind predictions
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype data
    model_type : str, optional
        Model type to use, by default "cb"
    random_state : int, optional
        Random state for reproducibility, by default 42
    min_test_samples : int, optional
        Minimum number of test samples required, by default 10

    Returns
    -------
    pd.DataFrame
        Results dataframe with ML metrics
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

    # Calculate total iterations for progress bar
    total_splits = sum(len(splits) for splits in split_data.values())

    with tqdm(total=total_splits, desc="Running ML on concordant splits") as pbar:
        for split_type in split_data:
            for key in split_data[split_type]:
                pbar.set_postfix_str(f"{split_type}/{key}")
                pbar.update(1)

                split = split_data[split_type][key]
                phenotype = key.split("_")[0]

                # Get concordant samples for this phenotype
                concordant_genomes = get_concordant_samples(
                    gapmind_predictions, experimental_phenotypes, phenotype
                )

                if len(concordant_genomes) == 0:
                    print(
                        f"\nSkipping {split_type}/{key}: no concordant samples found"
                    )
                    continue

                # Filter split to concordant samples
                filtered_split = filter_split_to_concordant(
                    split, concordant_genomes, min_samples=5
                )

                if filtered_split is None:
                    print(
                        f"\nSkipping {split_type}/{key}: insufficient concordant samples"
                    )
                    continue

                X_train = filtered_split["X_train"]
                y_train = filtered_split["y_train"]
                X_val = filtered_split["X_val"]
                y_val = filtered_split["y_val"]
                X_test = filtered_split["X_test"]
                y_test = filtered_split["y_test"]

                # Skip if test set is too small
                n_test_samples = len(X_test)
                if n_test_samples < min_test_samples:
                    print(
                        f"\nSkipping {split_type}/{key}: test set has only {n_test_samples} concordant samples"
                    )
                    continue

                # Run ML
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

                # Add metadata
                result["split_type"] = split_type
                result["key"] = key
                result["phenotype"] = phenotype
                result["model_type"] = model_type
                result["n_train"] = len(X_train)
                result["n_val"] = len(X_val)
                result["n_test"] = len(X_test)
                result["n_concordant_total"] = len(concordant_genomes)

                results.append(result)

    return pd.DataFrame(results)


FEATURE_FILES: dict[str, Path] = {
    "kofam": Path("data/processed/features_reduced/combined_datasets/kofam.tsv"),
    # Filtered (correlation- and variance-reduced) GapMind features. Note that
    # the 0.95 correlation filter consolidates the same transporter / pathway
    # gene across multiple phenotype prefixes, which depletes per-phenotype
    # feature spaces for amino-acid pathways. Kept here for reproducibility.
    "gapmind": Path("data/processed/features_reduced/combined_datasets/gapmind.tsv"),
    # Raw (unfiltered) GapMind features. Preserves every per-phenotype
    # pathway-step column so the ceiling line is a fair pipeline-ceiling per
    # phenotype. Used for the Fig 5A red reference lines.
    "gapmind_raw": Path("data/interim/features/combined_datasets/gapmind.tsv"),
}


def main() -> None:
    """Generate Figure 5A data from concordant samples.

    Accepts a ``--features`` CLI argument selecting which feature matrix to
    use. The default ``kofam`` reproduces the published Fig 5A; ``gapmind``
    produces the pipeline-ceiling reference line that is overlaid on Fig 5A.
    Output filename is suffixed by the feature choice to keep both result
    sets on disk simultaneously.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--features",
        choices=sorted(FEATURE_FILES),
        default="kofam",
        help=(
            "Feature matrix to train on. 'kofam' is the published main analysis;"
            " 'gapmind' produces the GapMind-feature ceiling reference shown on Fig 5A."
        ),
    )
    args = parser.parse_args()

    # Define paths
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    FEATURE_FILE = FEATURE_FILES[args.features]
    PHENOTYPE_DIR = Path("data/processed/phenotypes")

    # Define which split types to process (exclude phylo_ic)
    SPLIT_TYPES = ["random_split", "dataset_split", "phylo_ooc"]

    print(f"Using feature matrix: {args.features} ({FEATURE_FILE})")

    # Load GapMind predictions and experimental phenotypes
    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(f"  Loaded {len(gapmind_predictions)} genomes, {len(gapmind_predictions.columns)} phenotypes")

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"  Loaded {len(experimental_phenotypes)} genomes, {len(experimental_phenotypes.columns)} phenotypes")

    # Load all splits
    print("\nLoading train-test splits...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=SPLIT_TYPES,
        feature_file=FEATURE_FILE,
    )

    # Print summary of loaded data
    print("\nLoaded splits summary:")
    for split_type in split_data:
        print(f"  {split_type}: {len(split_data[split_type])} splits")

    # Run ML on concordant samples only
    print("\nRunning machine learning on concordant samples...")
    results = run_ml_on_concordant_splits(
        split_data,
        gapmind_predictions,
        experimental_phenotypes,
        model_type="cb",
        random_state=42,
        min_test_samples=10,
    )

    # Annotate each row with its concordant minority-class test count so the
    # minority-class filter (Methods) is a column lookup downstream.
    from scripts.minority_filter import (
        annotate_minority_test,
        concordant_minority_counts,
    )

    results = annotate_minority_test(results, concordant_minority_counts())

    # Save results
    suffix = "" if args.features == "kofam" else f"_{args.features}"
    results_file = OUTPUT_DIR / f"figure5a_concordant_ml_results{suffix}.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    # Print summary statistics
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
