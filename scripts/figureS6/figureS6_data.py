#!/usr/bin/env python3
"""Generate Figure S6 data: model performance versus training sample size.

Sweeps training size, training type (full vs concordant), split type, and test
subset for Histidine and Galactose, with 3 repeats per configuration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.ml import _get_scores, make_classifier

PHENOTYPES_TO_ANALYZE = ["Histidine", "Galactose"]
SAMPLE_SIZES = [50, 100, 200, 500, "full"]
N_REPEATS = 3
SPLIT_TYPES = ["random_split", "dataset_split", "phylo_ooc"]
RANDOM_STATE = 42

# One of "gapmind", "kofam" or "rast".
FEATURE_TYPE = "kofam"

SCORING = [
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


def load_gapmind_predictions() -> pd.DataFrame:
    """Load and process GapMind predictions.

    Returns
    -------
    pd.DataFrame
        GapMind predictions as binary (0/1) for each phenotype.
    """
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

    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    with open(marine_ids_file, "r") as f:
        marine_ids_map = {v.rsplit("_", 2)[0]: k for k, v in json.load(f).items()}

    from scripts.io import index_format_func

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict]
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
        "steps_missing_low": 0,
        "steps_missing_medium": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }
    gapmind_data_binary = gapmind_data.replace(replace_dict).astype(np.uint8)

    return gapmind_data_binary


def load_split_files(base_dir: Path) -> dict[str, dict[str, dict[str, pd.Series]]]:
    """Load train/test/val split files (labels only, not features).

    Parameters
    ----------
    base_dir : Path
        Base directory containing split folders.

    Returns
    -------
    dict[str, dict[str, dict[str, pd.Series]]]
        Nested dictionary: {split_type: {key: {y_train, y_val, y_test}}}
    """
    result: dict[str, dict[str, dict[str, pd.Series]]] = {}

    if "random_split" in SPLIT_TYPES:
        random_split_dir = base_dir / "random_split"
        if random_split_dir.exists():
            random_split_data = {}
            for phenotype_dir in random_split_dir.iterdir():
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                if phenotype_name not in PHENOTYPES_TO_ANALYZE:
                    continue
                for repeat_dir in phenotype_dir.iterdir():
                    if not repeat_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_{repeat_dir.name}"
                    y_train = pd.read_csv(
                        repeat_dir / "y_train.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_val = pd.read_csv(
                        repeat_dir / "y_val.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_test = pd.read_csv(
                        repeat_dir / "y_test.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    random_split_data[key] = {
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test,
                    }
            result["random_split"] = random_split_data

    if "dataset_split" in SPLIT_TYPES:
        dataset_split_dir = base_dir / "dataset_split"
        if dataset_split_dir.exists():
            dataset_split_data = {}
            for phenotype_dir in dataset_split_dir.iterdir():
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                if phenotype_name not in PHENOTYPES_TO_ANALYZE:
                    continue
                for split_dir in phenotype_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_{split_dir.name}"
                    y_train = pd.read_csv(
                        split_dir / "y_train.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_val = pd.read_csv(
                        split_dir / "y_val.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_test = pd.read_csv(
                        split_dir / "y_test.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    dataset_split_data[key] = {
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test,
                    }
            result["dataset_split"] = dataset_split_data

    if "phylo_ooc" in SPLIT_TYPES:
        phylo_split_dir = base_dir / "phylogeny_split"
        if phylo_split_dir.exists():
            phylo_ooc_data = {}
            for phenotype_dir in phylo_split_dir.iterdir():
                if not phenotype_dir.is_dir():
                    continue
                phenotype_name = phenotype_dir.name
                if phenotype_name not in PHENOTYPES_TO_ANALYZE:
                    continue
                ooc_dir = phenotype_dir / "out-of-clade"
                if not ooc_dir.exists():
                    continue
                for split_dir in ooc_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_name}_ooc_{split_dir.name}"
                    y_train = pd.read_csv(
                        split_dir / "y_train.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_val = pd.read_csv(
                        split_dir / "y_val.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    y_test = pd.read_csv(
                        split_dir / "y_test.tsv",
                        sep="\t",
                        index_col=0,
                        dtype={"genomeID": str},
                    ).iloc[:, 0]
                    phylo_ooc_data[key] = {
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test,
                    }
            result["phylo_ooc"] = phylo_ooc_data

    return result


def subsample_indices(
    indices: pd.Index, y: pd.Series, n_samples: int | str, random_state: int
) -> pd.Index:
    """Subsample indices using stratified sampling.

    Parameters
    ----------
    indices : pd.Index
        Indices to subsample from.
    y : pd.Series
        Labels for stratification.
    n_samples : int | str
        Number of samples to select, or "full" for all samples.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    pd.Index
        Subsampled indices.
    """
    if n_samples == "full":
        return indices

    y_subset = y.loc[indices]

    if len(y_subset) <= n_samples:
        return indices

    if len(y_subset.unique()) == 1:
        sampled_indices = y_subset.sample(
            n=n_samples, replace=False, random_state=random_state
        ).index
    else:
        # sklearn stratification needs at least one sample per class in the
        # holdout portion.
        n_classes = len(y_subset.unique())
        test_size = len(y_subset) - n_samples

        if test_size >= n_classes:
            sampled_indices, _ = train_test_split(
                y_subset.index,
                train_size=n_samples,
                stratify=y_subset,
                random_state=random_state,
            )
        else:
            sampled_indices = y_subset.sample(
                n=n_samples, replace=False, random_state=random_state
            ).index

        sampled_indices = pd.Index(sampled_indices)

    return sampled_indices


def run_data_requirements_analysis(
    split_data: dict[str, dict[str, dict[str, pd.Series]]],
    feature_data: pd.DataFrame,
    gapmind_data: pd.DataFrame,
    checkpoint_file: Path | None = None,
) -> pd.DataFrame:
    """Run data requirements analysis across all splits and configurations.

    Parameters
    ----------
    split_data : dict
        Nested dictionary of splits from load_split_files().
    feature_data : pd.DataFrame
        Feature matrix with samples as rows and features as columns.
    gapmind_data : pd.DataFrame
        GapMind predictions for identifying concordant samples.
    checkpoint_file : Path, optional
        Path to incremental checkpoint CSV. If it exists, already-completed
        (split_type, key) combinations are skipped and the file is appended to.

    Returns
    -------
    pd.DataFrame
        Results dataframe with performance metrics.
    """
    results = []

    done_keys: set[tuple[str, str]] = set()
    if checkpoint_file is not None and checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)
        done_keys = set(
            (row["split_type"], row["key"])
            for _, row in existing[["split_type", "key"]].drop_duplicates().iterrows()
        )
        results.extend(existing.to_dict("records"))
        print(
            f"  Resuming: {len(done_keys)} (split_type, key) combinations already done."
        )

    total_iterations = 0
    for split_type in split_data:
        for key in split_data[split_type]:
            for training_type in ["full", "concordant"]:
                for sample_size in SAMPLE_SIZES:
                    total_iterations += N_REPEATS

    with tqdm(total=total_iterations, desc="Running analysis") as pbar:
        for split_type in split_data:
            for key in split_data[split_type]:
                phenotype_name = key.split("_")[0]
                pbar.set_postfix_str(f"{split_type}/{key}")

                if (split_type, key) in done_keys:
                    pbar.update(N_REPEATS * len(SAMPLE_SIZES) * 2)
                    continue

                y_train = split_data[split_type][key]["y_train"]
                y_val = split_data[split_type][key]["y_val"]
                y_test = split_data[split_type][key]["y_test"]

                train_indices = y_train.index.intersection(feature_data.index)
                val_indices = y_val.index.intersection(feature_data.index)
                test_indices = y_test.index.intersection(feature_data.index)

                X_train_full = feature_data.loc[train_indices]
                y_train_full = y_train.loc[train_indices]
                X_val_full = feature_data.loc[val_indices]
                y_val_full = y_val.loc[val_indices]
                X_test = feature_data.loc[test_indices]
                y_test_subset = y_test.loc[test_indices]

                if len(X_test) < 10:
                    pbar.update(N_REPEATS * len(SAMPLE_SIZES) * 2)
                    continue

                # Concordant = experimental label equals the GapMind call.
                gapmind_train = gapmind_data.loc[
                    train_indices.intersection(gapmind_data.index), phenotype_name
                ]
                concordant_train_mask = (
                    y_train_full.loc[gapmind_train.index] == gapmind_train
                )
                concordant_train_indices = gapmind_train[concordant_train_mask].index

                gapmind_val = gapmind_data.loc[
                    val_indices.intersection(gapmind_data.index), phenotype_name
                ]
                concordant_val_mask = y_val_full.loc[gapmind_val.index] == gapmind_val
                concordant_val_indices = gapmind_val[concordant_val_mask].index

                gapmind_test = gapmind_data.loc[
                    test_indices.intersection(gapmind_data.index), phenotype_name
                ]
                concordant_test_mask = (
                    y_test_subset.loc[gapmind_test.index] == gapmind_test
                )
                concordant_test_indices = gapmind_test[concordant_test_mask].index
                discordant_test_indices = gapmind_test[~concordant_test_mask].index

                for training_type in ["full", "concordant"]:
                    if training_type == "full":
                        base_train_indices = train_indices
                        base_val_indices = val_indices
                    else:
                        base_train_indices = concordant_train_indices
                        base_val_indices = concordant_val_indices

                    for sample_size in SAMPLE_SIZES:
                        for repeat_idx in range(N_REPEATS):
                            pbar.update(1)

                            repeat_random_state = RANDOM_STATE + repeat_idx
                            sampled_train_indices = subsample_indices(
                                base_train_indices,
                                y_train_full,
                                sample_size,
                                repeat_random_state,
                            )

                            if len(sampled_train_indices) < 5:
                                continue

                            X_train = X_train_full.loc[sampled_train_indices]
                            y_train = y_train_full.loc[sampled_train_indices]

                            if len(y_train.unique()) != 2:
                                continue

                            # Validation set is not subsampled.
                            X_val = X_val_full.loc[base_val_indices]
                            y_val = y_val_full.loc[base_val_indices]

                            if len(y_val.unique()) != 2:
                                continue

                            model = make_classifier("cb", random_state=RANDOM_STATE)

                            # Align validation features to the training columns.
                            X_val_aligned = X_val.copy()
                            missing_cols = X_train.columns.difference(
                                X_val_aligned.columns
                            )
                            if len(missing_cols) > 0:
                                missing_df = pd.DataFrame(
                                    0, index=X_val_aligned.index, columns=missing_cols
                                )
                                X_val_aligned = pd.concat(
                                    [X_val_aligned, missing_df], axis=1
                                )
                            X_val_aligned = X_val_aligned[X_train.columns]

                            model.fit(
                                X_train,
                                y_train,
                                eval_set=(X_val_aligned, y_val),
                                use_best_model=True,
                                verbose=False,
                            )

                            # Align test features to the training columns.
                            X_test_aligned = X_test.copy()
                            missing_cols = X_train.columns.difference(
                                X_test_aligned.columns
                            )
                            if len(missing_cols) > 0:
                                missing_df = pd.DataFrame(
                                    0, index=X_test_aligned.index, columns=missing_cols
                                )
                                X_test_aligned = pd.concat(
                                    [X_test_aligned, missing_df], axis=1
                                )
                            X_test_aligned = X_test_aligned[X_train.columns]

                            result_full = _get_scores(
                                model, X_test_aligned, y_test_subset, SCORING
                            )
                            result_full["test_subset"] = "full"
                            result_full["n_test_samples"] = len(y_test_subset)
                            result_full["n_train_samples"] = len(y_train)
                            result_full["sample_size"] = sample_size
                            result_full["split_type"] = split_type
                            result_full["key"] = key
                            result_full["phenotype"] = phenotype_name
                            result_full["training_type"] = training_type
                            result_full["repeat"] = repeat_idx
                            results.append(result_full)

                            if len(concordant_test_indices) >= 5:
                                X_test_concordant = X_test_aligned.loc[
                                    concordant_test_indices
                                ]
                                y_test_concordant = y_test_subset.loc[
                                    concordant_test_indices
                                ]
                                result_concordant = _get_scores(
                                    model, X_test_concordant, y_test_concordant, SCORING
                                )
                                result_concordant["test_subset"] = "concordant"
                                result_concordant["n_test_samples"] = len(
                                    y_test_concordant
                                )
                                result_concordant["n_train_samples"] = len(y_train)
                                result_concordant["sample_size"] = sample_size
                                result_concordant["split_type"] = split_type
                                result_concordant["key"] = key
                                result_concordant["phenotype"] = phenotype_name
                                result_concordant["training_type"] = training_type
                                result_concordant["repeat"] = repeat_idx
                                results.append(result_concordant)

                            if len(discordant_test_indices) >= 5:
                                X_test_discordant = X_test_aligned.loc[
                                    discordant_test_indices
                                ]
                                y_test_discordant = y_test_subset.loc[
                                    discordant_test_indices
                                ]
                                result_discordant = _get_scores(
                                    model, X_test_discordant, y_test_discordant, SCORING
                                )
                                result_discordant["test_subset"] = "discordant"
                                result_discordant["n_test_samples"] = len(
                                    y_test_discordant
                                )
                                result_discordant["n_train_samples"] = len(y_train)
                                result_discordant["sample_size"] = sample_size
                                result_discordant["split_type"] = split_type
                                result_discordant["key"] = key
                                result_discordant["phenotype"] = phenotype_name
                                result_discordant["training_type"] = training_type
                                result_discordant["repeat"] = repeat_idx
                                results.append(result_discordant)

                if checkpoint_file is not None and len(results) > 0:
                    pd.DataFrame(results).to_csv(checkpoint_file, index=False)

    return pd.DataFrame(results)


def main() -> None:
    """Generate Figure S6 data."""
    SPLITS_DIR = Path("data/processed/train_test_splits")
    FEATURE_FILE = Path(
        f"data/processed/features_reduced/combined_datasets/{FEATURE_TYPE}.tsv"
    )
    OUTPUT_DIR = Path("data/outputs/figureS6")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {FEATURE_TYPE.upper()} feature data...")
    feature_data = pd.read_csv(
        FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"  Feature data shape: {feature_data.shape}")

    print("\nLoading GapMind predictions...")
    gapmind_data = load_gapmind_predictions()
    print(f"  GapMind predictions shape: {gapmind_data.shape}")

    print("\nLoading train-test splits...")
    split_data = load_split_files(SPLITS_DIR)
    print("\nLoaded splits summary:")
    for split_type in split_data:
        print(f"  {split_type}: {len(split_data[split_type])} splits")

    print("\nRunning data requirements analysis...")
    print(f"  Feature type: {FEATURE_TYPE.upper()}")
    print(f"  Phenotypes: {PHENOTYPES_TO_ANALYZE}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  Repeats per configuration: {N_REPEATS}")
    print("  Training types: full, concordant")
    print("  Test subsets: full, concordant, discordant")

    checkpoint_file = (
        OUTPUT_DIR / f"figure_s6_data_requirements_{FEATURE_TYPE}_checkpoint.csv"
    )
    results = run_data_requirements_analysis(
        split_data, feature_data, gapmind_data, checkpoint_file=checkpoint_file
    )

    results["feature_type"] = FEATURE_TYPE

    results_file = OUTPUT_DIR / f"figure_s6_data_requirements_{FEATURE_TYPE}.csv"
    results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    print("\nResults summary:")
    print(f"  Total experiments: {len(results)}")

    if len(results) > 0:
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
    else:
        print("\nNo results generated. All experiments were skipped.")
        print("Check if test sets are too small or don't have both classes.")

    print("\nDone!")


if __name__ == "__main__":
    main()
