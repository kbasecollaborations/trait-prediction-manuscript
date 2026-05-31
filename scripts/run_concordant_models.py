#!/usr/bin/env python3
"""Train CatBoost models on random_split data filtered to GapMind-concordant samples."""

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.figure5.figure5a_data import (
    filter_split_to_concordant,
    get_concordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml_splits import (
    get_feature_importances,
    load_split_data,
    perform_split_ml_with_model,
)

# Paths are relative to the repo root.
SPLITS_DIR = Path("data/processed/train_test_splits")
GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR = Path("data/processed/phenotypes")
OUTPUT_BASE = Path("data/outputs/concordant_models")
SPLIT_TYPES = ["random_split"]
MIN_SAMPLES = 5
MIN_TEST_SAMPLES = 10
RANDOM_STATE = 42
MODEL_TYPE = "cb"

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


def _sanitize_suffix(s: str) -> str:
    """Replace non-alphanumeric characters with underscore for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def _model_suffix_for_key(key: str) -> str:
    """Build filesystem-safe suffix for a random_split key (e.g. Alanine_0 -> random_split_0)."""
    parts = key.split("_", 1)
    if len(parts) == 1:
        id_part = "0"
    else:
        id_part = _sanitize_suffix(parts[1])
    return f"random_split_{id_part}"


def _save_model_artifacts(
    out_dir: Path,
    phenotype: str,
    suffix: str,
    model: object,
    X_train: pd.DataFrame,
    result: dict,
    split_type: str,
    key: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_concordant_total: int,
) -> None:
    """Save .cbm, _metadata.json, _feature_importances.csv, _selected_features.txt."""
    base_name = f"{phenotype}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cbm_path = out_dir / f"{base_name}.cbm"
    if hasattr(model, "save_model"):
        model.save_model(str(cbm_path), format="cbm")

    # Request all features; trait_prediction otherwise defaults to the top 10.
    n_feats = X_train.shape[1]
    importances = get_feature_importances(model, X_train, n_features=n_feats)
    fi_path = out_dir / f"{base_name}_feature_importances.csv"
    importances_df = importances.reset_index()
    importances_df.columns = ["feature", "importance"]
    importances_df.to_csv(fi_path, index=False)

    features = result.get("features", importances.index.tolist())
    sf_path = out_dir / f"{base_name}_selected_features.txt"
    with open(sf_path, "w") as f:
        for feat in features:
            f.write(f"{feat}\n")

    metadata = {
        "model_name": base_name,
        "phenotype_name": phenotype,
        "split_type": split_type,
        "key": key,
        "n_samples_train": n_train,
        "n_samples_val": n_val,
        "n_samples_test": n_test,
        "n_concordant_total": n_concordant_total,
        "n_features": int(X_train.shape[1]),
        "random_state": RANDOM_STATE,
        "selected_features": features,
    }
    for metric in SCORING:
        if metric in result:
            metadata[f"test_{metric}"] = float(result[metric])
    meta_path = out_dir / f"{base_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    """Run concordant analysis on random_split only and save models."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(f"  Loaded {len(gapmind_predictions)} genomes, {len(gapmind_predictions.columns)} phenotypes")

    print("\nLoading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"  Loaded {len(experimental_phenotypes)} genomes, {len(experimental_phenotypes.columns)} phenotypes")

    print("\nLoading random_split train-test splits...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=SPLIT_TYPES)
    if "random_split" not in split_data or not split_data["random_split"]:
        print("  No random_split data found. Exiting.")
        return
    n_splits = len(split_data["random_split"])
    print(f"  Loaded {n_splits} random splits")

    results_list = []
    with tqdm(total=n_splits, desc="Concordant ML + save models") as pbar:
        for key, split in split_data["random_split"].items():
            pbar.set_postfix_str(key)
            pbar.update(1)

            phenotype = key.split("_", 1)[0]
            concordant_genomes = get_concordant_samples(
                gapmind_predictions, experimental_phenotypes, phenotype
            )
            if len(concordant_genomes) == 0:
                print(f"\nSkipping {key}: no concordant samples found")
                continue

            filtered_split = filter_split_to_concordant(
                split, concordant_genomes, min_samples=MIN_SAMPLES
            )
            if filtered_split is None:
                print(f"\nSkipping {key}: insufficient concordant samples")
                continue

            X_train = filtered_split["X_train"]
            y_train = filtered_split["y_train"]
            X_val = filtered_split["X_val"]
            y_val = filtered_split["y_val"]
            X_test = filtered_split["X_test"]
            y_test = filtered_split["y_test"]

            if len(X_test) < MIN_TEST_SAMPLES:
                print(f"\nSkipping {key}: test set has only {len(X_test)} concordant samples")
                continue

            result, model = perform_split_ml_with_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                model_type=MODEL_TYPE,
                scoring=SCORING,
                random_state=RANDOM_STATE,
            )

            suffix = _model_suffix_for_key(key)
            out_dir = OUTPUT_BASE / phenotype
            _save_model_artifacts(
                out_dir=out_dir,
                phenotype=phenotype,
                suffix=suffix,
                model=model,
                X_train=X_train,
                result=result,
                split_type="random_split",
                key=key,
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
                n_concordant_total=len(concordant_genomes),
            )

            row = {**result, "split_type": "random_split", "key": key, "phenotype": phenotype}
            results_list.append(row)

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_path = OUTPUT_BASE / "concordant_ml_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved metrics to {results_path}")

    print(f"\nModels and artifacts saved under {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
