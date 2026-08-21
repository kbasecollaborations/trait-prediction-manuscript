#!/usr/bin/env python3
"""Figure 5C/D ML: train on concordant samples, test on discordant and full sets.

Writes ``data/outputs/figure5/figure5c_concordant_train_different_test.csv``.

Run with::

    uv run python -m scripts.figure5.figure5cd_data
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.ml_splits import load_split_data, perform_split_ml


def load_gapmind_predictions(gapmind_file: Path) -> pd.DataFrame:
    """Load GapMind predictions.

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
    """Load and combine experimental phenotype data from all datasets.

    Parameters
    ----------
    phenotype_dir : Path
        Path to the directory containing phenotype subdirectories

    Returns
    -------
    pd.DataFrame
        Combined phenotype data with genomeID as index and phenotypes as columns
    """
    phenotype_data: dict[str, list[pd.DataFrame]] = {}

    for dataset_dir in phenotype_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

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

    return experimental_data


def get_concordant_and_discordant_samples(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> tuple[set[str], set[str]]:
    """Get sets of concordant and discordant genome IDs.

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
    tuple[set[str], set[str]]
        (concordant_genomes, discordant_genomes)
    """
    if phenotype not in gapmind_predictions.columns:
        return set(), set()
    if phenotype not in experimental_phenotypes.columns:
        return set(), set()

    common_genomes = gapmind_predictions.index.intersection(
        experimental_phenotypes.index
    )

    # Restrict to genomes with non-NaN experimental data.
    exp_data = experimental_phenotypes.loc[common_genomes, phenotype]
    valid_genomes = exp_data.dropna().index

    gapmind_vals = gapmind_predictions.loc[valid_genomes, phenotype]
    exp_vals = experimental_phenotypes.loc[valid_genomes, phenotype]

    concordant_mask = gapmind_vals == exp_vals
    concordant_genomes = set(valid_genomes[concordant_mask])
    discordant_mask = ~concordant_mask
    discordant_genomes = set(valid_genomes[discordant_mask])

    return concordant_genomes, discordant_genomes


def filter_split_for_training(
    split_data: dict[str, pd.DataFrame | pd.Series],
    concordant_genomes: set[str],
    min_samples: int = 5,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """Filter train/val sets to concordant samples only; keep the test set as is.

    Parameters
    ----------
    split_data : dict
        Dictionary with keys X_train, y_train, X_val, y_val, X_test, y_test
    concordant_genomes : set[str]
        Set of genome IDs that are concordant
    min_samples : int, optional
        Minimum number of samples required in train/val sets, by default 5

    Returns
    -------
    dict | None
        Filtered split data, or None if insufficient samples
    """
    filtered_data = {}

    for key in ["X_train", "y_train", "X_val", "y_val"]:
        data = split_data[key]
        # Boolean mask, not list(set(...)): set iteration order is salted per
        # process and would reorder rows, making the fitted models irreproducible.
        filtered = data.loc[data.index.isin(concordant_genomes)]
        filtered_data[key] = filtered

    # Test set is left full.
    filtered_data["X_test"] = split_data["X_test"]
    filtered_data["y_test"] = split_data["y_test"]

    n_train = len(filtered_data["X_train"])
    n_val = len(filtered_data["X_val"])

    if n_train < min_samples or n_val < min_samples:
        return None

    y_train_unique = filtered_data["y_train"].unique()
    y_val_unique = filtered_data["y_val"].unique()

    if len(y_train_unique) != 2 or len(y_val_unique) != 2:
        return None

    return filtered_data


def create_discordant_test_split(
    split_data: dict[str, pd.DataFrame | pd.Series],
    discordant_genomes: set[str],
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """Create a split with discordant test samples only.

    Parameters
    ----------
    split_data : dict
        Dictionary with keys X_train, y_train, X_val, y_val, X_test, y_test
    discordant_genomes : set[str]
        Set of genome IDs that are discordant

    Returns
    -------
    dict | None
        Split data with discordant test set, or None if no discordant samples
    """
    filtered_data = {}
    for key in ["X_train", "y_train", "X_val", "y_val"]:
        filtered_data[key] = split_data[key]

    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    discordant_mask = X_test.index.isin(discordant_genomes)
    if not discordant_mask.any():
        return None

    filtered_data["X_test"] = X_test.loc[discordant_mask]
    filtered_data["y_test"] = y_test.loc[discordant_mask]

    return filtered_data


def run_ml_on_concordant_train_with_different_test_sets(
    split_data: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    model_type: str = "cb",
    random_state: int = 42,
    min_test_samples: int = 5,
) -> pd.DataFrame:
    """Train on concordant samples, test on discordant and full samples.

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
        Minimum number of test samples required, by default 5

    Returns
    -------
    pd.DataFrame
        Results dataframe with ML metrics for both test types
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

    with tqdm(
        total=total_splits,
        desc="Running ML: train on concordant, test on discordant/full",
    ) as pbar:
        for split_type in split_data:
            for key in split_data[split_type]:
                pbar.set_postfix_str(f"{split_type}/{key}")
                pbar.update(1)

                split = split_data[split_type][key]
                phenotype = key.split("_")[0]

                concordant_genomes, discordant_genomes = (
                    get_concordant_and_discordant_samples(
                        gapmind_predictions, experimental_phenotypes, phenotype
                    )
                )

                if len(concordant_genomes) == 0:
                    print(
                        f"\nSkipping {split_type}/{key}: no concordant samples for training"
                    )
                    continue

                concordant_split = filter_split_for_training(
                    split, concordant_genomes, min_samples=5
                )

                if concordant_split is None:
                    print(
                        f"\nSkipping {split_type}/{key}: insufficient concordant samples for training"
                    )
                    continue

                # Full test set (concordant + discordant).
                X_train = concordant_split["X_train"]
                y_train = concordant_split["y_train"]
                X_val = concordant_split["X_val"]
                y_val = concordant_split["y_val"]
                X_test_full = concordant_split["X_test"]
                y_test_full = concordant_split["y_test"]

                if (
                    len(X_test_full) >= min_test_samples
                    and len(y_test_full.unique()) == 2
                ):
                    result_full = perform_split_ml(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test_full,
                        y_test_full,
                        model_type=model_type,
                        scoring=scoring,
                        random_state=random_state,
                    )

                    result_full["split_type"] = split_type
                    result_full["key"] = key
                    result_full["phenotype"] = phenotype
                    result_full["model_type"] = model_type
                    result_full["test_type"] = "full"
                    result_full["n_train"] = len(X_train)
                    result_full["n_val"] = len(X_val)
                    result_full["n_test"] = len(X_test_full)
                    result_full["n_concordant_train"] = len(concordant_genomes)
                    result_full["n_discordant_available"] = len(discordant_genomes)

                    results.append(result_full)

                # Discordant test set only.
                if len(discordant_genomes) > 0:
                    discordant_split = create_discordant_test_split(
                        concordant_split, discordant_genomes
                    )

                    if discordant_split is not None:
                        X_test_disc = discordant_split["X_test"]
                        y_test_disc = discordant_split["y_test"]

                        if (
                            len(X_test_disc) >= min_test_samples
                            and len(y_test_disc.unique()) == 2
                        ):
                            result_disc = perform_split_ml(
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_test_disc,
                                y_test_disc,
                                model_type=model_type,
                                scoring=scoring,
                                random_state=random_state,
                            )

                            result_disc["split_type"] = split_type
                            result_disc["key"] = key
                            result_disc["phenotype"] = phenotype
                            result_disc["model_type"] = model_type
                            result_disc["test_type"] = "discordant"
                            result_disc["n_train"] = len(X_train)
                            result_disc["n_val"] = len(X_val)
                            result_disc["n_test"] = len(X_test_disc)
                            result_disc["n_concordant_train"] = len(concordant_genomes)
                            result_disc["n_discordant_available"] = len(
                                discordant_genomes
                            )

                            results.append(result_disc)

    return pd.DataFrame(results)


def main() -> None:
    """Generate Figure 5C data."""
    SPLITS_DIR = Path("data/processed/train_test_splits")
    OUTPUT_DIR = Path("data/outputs/figure5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    KOFAM_FEATURE_FILE = Path(
        "data/processed/features_reduced/combined_datasets/kofam.tsv"
    )
    PHENOTYPE_DIR = Path("data/processed/phenotypes")

    SPLIT_TYPES = ["random_split", "dataset_split"]

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(
        f"  Loaded {len(gapmind_predictions)} genomes, {len(gapmind_predictions.columns)} phenotypes"
    )

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(
        f"  Loaded {len(experimental_phenotypes)} genomes, {len(experimental_phenotypes.columns)} phenotypes"
    )

    print("\nLoading train-test splits...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=SPLIT_TYPES,
        feature_file=KOFAM_FEATURE_FILE,
    )

    print("\nLoaded splits summary:")
    for split_type in split_data:
        print(f"  {split_type}: {len(split_data[split_type])} splits")

    print("\nRunning ML: train on concordant, test on discordant/full...")
    results = run_ml_on_concordant_train_with_different_test_sets(
        split_data,
        gapmind_predictions,
        experimental_phenotypes,
        model_type="cb",
        random_state=42,
        min_test_samples=5,
    )

    # Figure 5C tests on the discordant subset, so the minority-class count is
    # taken over discordant samples (Methods).
    from scripts.minority_filter import (
        annotate_minority_test,
        discordant_minority_counts,
    )

    results = annotate_minority_test(results, discordant_minority_counts())

    results_file = OUTPUT_DIR / "figure5c_concordant_train_different_test.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    print("\nResults summary:")
    print(f"  Total experiments: {len(results)}")
    print("\nBy split type and test type:")
    summary = (
        results.groupby(["split_type", "test_type"])["balanced_accuracy"]
        .describe()
        .round(3)
    )
    print(summary)

    print("\nBy test type (overall):")
    test_type_summary = (
        results.groupby("test_type")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(test_type_summary)

    print("\nBy phenotype and test type (mean balanced accuracy):")
    phenotype_summary = (
        results.groupby(["phenotype", "test_type"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(phenotype_summary.head(20))

    print("\nDone!")


if __name__ == "__main__":
    main()
