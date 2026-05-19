#!/usr/bin/env python3
"""
Generate per-sample SHAP value arrays for the Figure S11 beeswarm plot.

For each target phenotype the script:
1. Loads the canonical random-split fold (fold 0) for the phenotype.
2. Restricts both training and held-out test samples to GapMind-concordant
   genomes (as in Figure 5B).
3. Uses KOFAM annotations as the feature space; GapMind is used only to define
   concordance.
4. Trains a single CatBoost classifier with ``make_classifier("cb_noeval")``.
5. Computes SHAP values on the concordant held-out test set with
   ``shap.TreeExplainer``.
6. Persists ``shap_values``, ``feature_values``, ``feature_names``,
   ``predictions``, and ``y_true`` to a compressed ``.npz`` file.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

from scripts.figure5.figure5b_data import (
    get_concordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data

warnings.filterwarnings("ignore")

DEFAULT_PHENOTYPES: tuple[str, ...] = ("Histidine", "Galactose")
SPLIT_FOLD: str = "0"
RANDOM_STATE: int = 42

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SPLITS_DIR: Path = REPO_ROOT / "data/processed/train_test_splits/random_split"
FEATURE_FILE: Path = (
    REPO_ROOT / "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
GAPMIND_FILE: Path = REPO_ROOT / "data/outputs/figure2/gapmind_phenotypes_loose.tsv"
PHENOTYPE_DIR: Path = REPO_ROOT / "data/processed/phenotypes"
OUTPUT_DIR: Path = REPO_ROOT / "data/outputs/figureS11"


def load_concordant_train_test(
    phenotype: str,
    split_dir: Path,
    feature_data: pd.DataFrame,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load random-split data and filter train+val and test sets to concordant samples.

    Parameters
    ----------
    phenotype : str
        Phenotype name (e.g. ``"Histidine"``).
    split_dir : Path
        Path to a single random-split fold directory containing
        ``y_train.tsv``, ``y_val.tsv``, ``y_test.tsv``.
    feature_data : pd.DataFrame
        KOFAM feature matrix indexed by genomeID.
    gapmind_predictions : pd.DataFrame
        GapMind prediction table with genomes as index, phenotypes as columns.
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype table with genomes as index, phenotypes as columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        ``(X_train, y_train, X_test, y_test)`` restricted to concordant genomes.
        ``X_train``/``y_train`` combine the original train and validation folds.

    Raises
    ------
    ValueError
        If filtering leaves fewer than two classes in train or test, or fewer
        than 10 samples in either class of the training set.
    """
    split_data = load_single_split_data(split_dir, feature_data)

    X_trainval = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
    y_trainval = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    concordant_genomes = get_concordant_samples(
        gapmind_predictions, experimental_phenotypes, phenotype
    )
    if len(concordant_genomes) == 0:
        raise ValueError(f"No concordant genomes found for phenotype {phenotype}")

    train_keep = sorted(set(X_trainval.index) & concordant_genomes)
    test_keep = sorted(set(X_test.index) & concordant_genomes)

    X_train = X_trainval.loc[train_keep]
    y_train = y_trainval.loc[train_keep]
    X_test = X_test.loc[test_keep]
    y_test = y_test.loc[test_keep]

    if y_train.nunique() != 2:
        raise ValueError(
            f"Concordant training set for {phenotype} has only "
            f"{y_train.nunique()} class(es)."
        )
    if y_test.nunique() < 1:
        raise ValueError(f"Concordant test set for {phenotype} is empty.")
    class_counts = y_train.value_counts()
    if class_counts.min() < 10:
        raise ValueError(
            f"Concordant training set for {phenotype} has minority class size "
            f"{class_counts.min()} (< 10)."
        )

    return X_train, y_train, X_test, y_test


def compute_shap_values(
    model: CatBoostClassifier, X: pd.DataFrame
) -> np.ndarray:
    """
    Compute per-sample SHAP values for a fitted CatBoost classifier.

    Parameters
    ----------
    model : CatBoostClassifier
        A fitted CatBoost binary classifier.
    X : pd.DataFrame
        Feature matrix on which to compute SHAP values.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_features)`` with SHAP contributions for
        the positive class.
    """
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)
    values = shap_explanation.values
    if values.ndim == 3:
        values = values[..., 1]
    return np.asarray(values)


def generate_phenotype_data(
    phenotype: str,
    feature_data: pd.DataFrame,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Train a CatBoost model and persist SHAP arrays for one phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype name to process.
    feature_data : pd.DataFrame
        KOFAM feature matrix (combined datasets).
    gapmind_predictions : pd.DataFrame
        GapMind predictions used for concordance filtering.
    experimental_phenotypes : pd.DataFrame
        Experimental phenotype table used for concordance filtering.
    output_dir : Path
        Directory in which to write ``<phenotype>_shap_values.npz``.

    Returns
    -------
    Path
        Path to the written ``.npz`` file.

    Raises
    ------
    FileNotFoundError
        If the random-split fold directory for the phenotype does not exist.
    """
    split_dir = SPLITS_DIR / phenotype / SPLIT_FOLD
    if not split_dir.exists():
        raise FileNotFoundError(f"Random-split fold not found: {split_dir}")

    print(f"\n[{phenotype}] Loading concordant train/test splits...")
    X_train, y_train, X_test, y_test = load_concordant_train_test(
        phenotype,
        split_dir,
        feature_data,
        gapmind_predictions,
        experimental_phenotypes,
    )
    print(
        f"[{phenotype}] Train: n={len(X_train)} (pos={int((y_train == 1).sum())}, "
        f"neg={int((y_train == 0).sum())}); "
        f"Test: n={len(X_test)} (pos={int((y_test == 1).sum())}, "
        f"neg={int((y_test == 0).sum())}); "
        f"features={X_train.shape[1]}"
    )

    print(f"[{phenotype}] Training CatBoost (cb_noeval, seed={RANDOM_STATE})...")
    model = make_classifier("cb_noeval", random_state=RANDOM_STATE)
    model.fit(X_train, y_train, verbose=False)

    print(f"[{phenotype}] Predicting on held-out concordant test set...")
    predictions = model.predict(X_test).astype(int).ravel()

    print(f"[{phenotype}] Computing SHAP values via TreeExplainer...")
    shap_values = compute_shap_values(model, X_test)

    out_path = output_dir / f"{phenotype}_shap_values.npz"
    np.savez_compressed(
        out_path,
        shap_values=shap_values.astype(np.float32),
        feature_values=X_test.to_numpy().astype(np.int8),
        feature_names=np.asarray(X_test.columns.tolist(), dtype=object),
        predictions=predictions.astype(np.int8),
        y_true=y_test.to_numpy().astype(np.int8),
    )
    print(f"[{phenotype}] Wrote {out_path}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    print(f"[{phenotype}] Top-5 features by mean |SHAP|:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank}. {X_test.columns[idx]}  (mean |SHAP|={mean_abs[idx]:.4f})")

    return out_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attribute ``phenotype`` (str | None).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a CatBoost model on GapMind-concordant samples and persist "
            "per-sample SHAP arrays for the Figure S11 beeswarm plot."
        )
    )
    parser.add_argument(
        "--phenotype",
        type=str,
        default=None,
        choices=list(DEFAULT_PHENOTYPES),
        help=(
            "Restrict generation to a single phenotype (default: process all "
            f"of {list(DEFAULT_PHENOTYPES)})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the Figure S11 SHAP data generation script."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading KOFAM feature matrix (combined datasets)...")
    feature_data = pd.read_csv(
        FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"  Loaded {feature_data.shape[0]} genomes x {feature_data.shape[1]} features")

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(f"  Loaded {len(gapmind_predictions)} genomes")

    print("Loading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"  Loaded {len(experimental_phenotypes)} genomes")

    phenotypes: tuple[str, ...] = (
        (args.phenotype,) if args.phenotype else DEFAULT_PHENOTYPES
    )

    for phenotype in phenotypes:
        generate_phenotype_data(
            phenotype,
            feature_data,
            gapmind_predictions,
            experimental_phenotypes,
            OUTPUT_DIR,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
