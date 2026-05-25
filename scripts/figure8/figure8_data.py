#!/usr/bin/env python3
"""
Generate Figure 8 data: selective prediction / applicability
domain for concordant-trained models on the full held-out cross-dataset test set.

The manuscript shows that concordant-trained models retain only intermediate
balanced accuracy on the full, natural-composition held-out test set
(Figure 5D). This script asks a complementary, deployment-oriented question:
*for a new genome whose concordance is unknown, can the model itself flag which
predictions to trust?*

For every ``dataset_split`` (train on three datasets, test on the held-out
fourth) the script re-fits the concordant-trained CatBoost model exactly as in
``figure5d_full_test_data.py`` but records per-sample predicted probabilities.
It then builds two label-free abstention signals --- both computable without
knowing the experimental outcome of the test genome:

    1. Model confidence: ``max(p, 1 - p)`` from ``predict_proba``.
    2. GapMind-ML agreement: whether the GapMind call and the ML prediction
       coincide.

From these it produces risk-coverage curves (balanced accuracy on the retained
subset versus the fraction of genomes the model commits to).

Outputs (all under ``data/outputs/figure8/``):
    - ``figure8_per_sample.tsv``                  one row per held-out test genome
    - ``figure8_risk_coverage.tsv``               pooled balanced accuracy vs. coverage
    - ``figure8_risk_coverage_by_phenotype.tsv``  per-phenotype risk-coverage curves
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    get_concordant_and_discordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.figure8.applicability import (
    calibration_table,
    expected_calibration_error,
)
from scripts.ml import make_classifier
from scripts.ml_splits import load_split_data
from trait_prediction.pipeline import align_columns

SPLITS_DIR: Path = Path("data/processed/train_test_splits")
OUTPUT_DIR: Path = Path("data/outputs/figure8")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
# KOFAM feature matrix, matching the concordant-trained models of Figure 5.
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")

RANDOM_STATE: int = 42
MIN_TRAIN_SAMPLES: int = 5

HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")

# Coverage grid for the risk-coverage curve (fraction of genomes retained).
COVERAGE_GRID: np.ndarray = np.round(np.arange(0.10, 1.0001, 0.05), 2)


def parse_held_out_dataset(key: str) -> str | None:
    """
    Extract the held-out dataset name from a ``dataset_split`` key.

    Parameters
    ----------
    key : str
        Split key of the form ``"<phenotype>_train(...),test(<dataset>)"``.

    Returns
    -------
    str | None
        Held-out dataset name, or ``None`` if the key does not match.
    """
    match = HELD_OUT_RE.search(key)
    return match.group(1) if match else None


def fit_concordant_model_and_predict_proba(
    split: Mapping[str, pd.DataFrame | pd.Series],
    concordant_genomes: set[str],
) -> pd.DataFrame | None:
    """
    Fit a concordant-trained CatBoost model and predict the full held-out test.

    Mirrors ``figure5d_full_test_data.fit_concordant_model_and_predict`` (sorted
    index ordering and ``random_state=42`` for determinism) but returns the
    predicted probability of growth in addition to the hard label.

    Parameters
    ----------
    split : Mapping[str, pd.DataFrame | pd.Series]
        Output of ``load_single_split_data`` for one split.
    concordant_genomes : set[str]
        Genome IDs that are GapMind-concordant for the phenotype.

    Returns
    -------
    pd.DataFrame | None
        Frame indexed by held-out genome ID with columns ``proba`` (P(growth))
        and ``y_pred`` (0/1), or ``None`` when the concordant training subset is
        too small or single-class.
    """
    train_idx = sorted(set(split["X_train"].index) & concordant_genomes)
    val_idx = sorted(set(split["X_val"].index) & concordant_genomes)

    if len(train_idx) < MIN_TRAIN_SAMPLES or len(val_idx) < MIN_TRAIN_SAMPLES:
        return None

    X_train = split["X_train"].loc[train_idx]
    y_train = split["y_train"].loc[train_idx]
    X_val = split["X_val"].loc[val_idx]
    y_val = split["y_val"].loc[val_idx]

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        return None

    X_test = split["X_test"]
    if len(X_test) == 0:
        return None

    model = make_classifier("cb", random_state=RANDOM_STATE)
    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val_aligned, y_val),
        use_best_model=True,
        verbose=False,
    )

    # CatBoost orders predict_proba columns by sorted class label, so column 1
    # is P(class == 1) == P(growth).
    proba = np.asarray(model.predict_proba(X_test_aligned))[:, 1]
    return pd.DataFrame(
        {"proba": proba, "y_pred": (proba >= 0.5).astype(int)},
        index=X_test.index,
    )


def fit_full_data_model_and_predict_proba(
    split: Mapping[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame | None:
    """
    Fit a full-data CatBoost model (all training samples) and predict the held-out test.

    Mirrors :func:`fit_concordant_model_and_predict_proba` but trains on every training
    genome with a valid label rather than the GapMind-concordant subset.

    Parameters
    ----------
    split : Mapping[str, pd.DataFrame | pd.Series]
        Output of ``load_split_data`` for one ``dataset_split`` key.

    Returns
    -------
    pd.DataFrame | None
        Frame indexed by held-out genome ID with columns ``proba`` and ``y_pred``,
        or ``None`` when the training data is too small or single-class.
    """
    y_train = split["y_train"].dropna()
    y_val = split["y_val"].dropna()
    if len(y_train) < MIN_TRAIN_SAMPLES or len(y_val) < MIN_TRAIN_SAMPLES:
        return None
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        return None
    X_test = split["X_test"]
    if len(X_test) == 0:
        return None

    X_train = split["X_train"].loc[y_train.index]
    X_val = split["X_val"].loc[y_val.index]
    model = make_classifier("cb", random_state=RANDOM_STATE)
    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)
    model.fit(
        X_train, y_train, eval_set=(X_val_aligned, y_val),
        use_best_model=True, verbose=False,
    )
    proba = np.asarray(model.predict_proba(X_test_aligned))[:, 1]
    return pd.DataFrame(
        {"proba": proba, "y_pred": (proba >= 0.5).astype(int)}, index=X_test.index
    )


def collect_per_sample_predictions(
    split_data: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the pooled per-sample prediction table across all dataset splits.

    Parameters
    ----------
    split_data : Mapping
        Nested split dictionary from ``load_split_data`` (``dataset_split`` only).
    gapmind_predictions : pd.DataFrame
        Loose-mode GapMind predictions (genomes x phenotypes, 0/1).
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype table (genomes x phenotypes, 0/1).

    Returns
    -------
    pd.DataFrame
        One row per held-out test genome with columns ``phenotype``,
        ``held_out_dataset``, ``genome``, ``y_true``, ``y_pred``, ``proba``,
        ``confidence``, ``gapmind_pred``.
    """
    records: list[dict[str, object]] = []
    splits = split_data["dataset_split"]

    for key, split in tqdm(splits.items(), total=len(splits), desc="S15 splits"):
        held_out = parse_held_out_dataset(key)
        if held_out is None:
            continue
        phenotype = key.split("_", 1)[0]

        concordant_genomes, _ = get_concordant_and_discordant_samples(
            gapmind_predictions, experimental_phenotypes, phenotype
        )
        if len(concordant_genomes) == 0:
            continue

        predictions = fit_concordant_model_and_predict_proba(
            split, concordant_genomes
        )
        if predictions is None:
            continue

        y_test = split["y_test"]
        gm_col = (
            gapmind_predictions[phenotype]
            if phenotype in gapmind_predictions.columns
            else None
        )

        for genome in predictions.index:
            y_true = y_test.loc[genome]
            if pd.isna(y_true):
                continue
            proba = float(predictions.loc[genome, "proba"])
            y_pred = int(predictions.loc[genome, "y_pred"])

            gm_pred: float | int = np.nan
            if gm_col is not None and genome in gm_col.index:
                gm_val = gm_col.loc[genome]
                if not pd.isna(gm_val):
                    gm_pred = int(gm_val)

            records.append(
                {
                    "phenotype": phenotype,
                    "held_out_dataset": held_out,
                    "genome": genome,
                    "y_true": int(y_true),
                    "y_pred": y_pred,
                    "proba": proba,
                    "confidence": max(proba, 1.0 - proba),
                    "gapmind_pred": gm_pred,
                }
            )

    return pd.DataFrame.from_records(records)


def collect_full_data_per_sample(
    split_data: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame | pd.Series]]],
    gapmind_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the pooled full-data-model per-sample table across all dataset splits."""
    records: list[dict[str, object]] = []
    splits = split_data["dataset_split"]
    for key, split in tqdm(splits.items(), total=len(splits), desc="full-data splits"):
        held_out = parse_held_out_dataset(key)
        if held_out is None:
            continue
        phenotype = key.split("_", 1)[0]
        predictions = fit_full_data_model_and_predict_proba(split)
        if predictions is None:
            continue
        y_test = split["y_test"]
        gm_col = (
            gapmind_predictions[phenotype]
            if phenotype in gapmind_predictions.columns else None
        )
        for genome in predictions.index:
            y_true = y_test.loc[genome]
            if pd.isna(y_true):
                continue
            proba = float(predictions.loc[genome, "proba"])
            gm_pred: float | int = np.nan
            if gm_col is not None and genome in gm_col.index and not pd.isna(gm_col.loc[genome]):
                gm_pred = int(gm_col.loc[genome])
            records.append({
                "phenotype": phenotype, "held_out_dataset": held_out, "genome": genome,
                "y_true": int(y_true), "y_pred": int(predictions.loc[genome, "y_pred"]),
                "proba": proba, "confidence": max(proba, 1.0 - proba), "gapmind_pred": gm_pred,
            })
    return pd.DataFrame.from_records(records)


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy that falls back to plain accuracy on a single-class set.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Balanced accuracy when both classes are present, plain accuracy when a
        single class is present, ``np.nan`` when the input is empty.
    """
    if len(y_true) == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float((y_true == y_pred).mean())
    return float(balanced_accuracy_score(y_true, y_pred))


def build_risk_coverage(per_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pooled risk-coverage curve using model confidence.

    Test genomes are ranked by ``confidence`` (descending); at each coverage
    level the most-confident fraction is retained and balanced accuracy is
    recomputed on that subset.

    Parameters
    ----------
    per_sample : pd.DataFrame
        Output of :func:`collect_per_sample_predictions`.

    Returns
    -------
    pd.DataFrame
        One row per coverage level with retained-subset balanced accuracy,
        accuracy, sample count, and confidence threshold.
    """
    ordered = per_sample.sort_values(
        "confidence", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    y_true = ordered["y_true"].to_numpy()
    y_pred = ordered["y_pred"].to_numpy()
    n_total = len(ordered)

    rows: list[dict[str, object]] = []
    for coverage in COVERAGE_GRID:
        k = max(1, int(round(coverage * n_total)))
        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": k,
                "confidence_threshold": float(ordered["confidence"].iloc[k - 1]),
                "balanced_accuracy": safe_balanced_accuracy(
                    y_true[:k], y_pred[:k]
                ),
                "accuracy": float((y_true[:k] == y_pred[:k]).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_risk_coverage_by_phenotype(per_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-phenotype risk-coverage curves using class-stratified confidence.

    Single-class abstention by raw ``max(p, 1 - p)`` is degenerate on
    per-phenotype subsets when the model is class-skewed: the top-confidence
    subset can be entirely one predicted class, collapsing balanced accuracy
    to 0.5 regardless of how accurate the model actually is. To avoid that,
    at each coverage level we retain the top fraction of predictions
    *within each predicted class*: the most-confident ``y_pred = 1`` calls
    (highest ``proba``) and the most-confident ``y_pred = 0`` calls (lowest
    ``proba``) are kept in parallel. This guarantees both predicted classes
    are represented at every coverage level, so balanced accuracy is
    well-defined throughout.

    Parameters
    ----------
    per_sample : pd.DataFrame
        Output of :func:`collect_per_sample_predictions`.

    Returns
    -------
    pd.DataFrame
        One row per (phenotype, coverage) with retained-subset balanced
        accuracy, accuracy, sample count, and per-class retained counts.
    """
    rows: list[dict[str, object]] = []
    for phenotype, group in per_sample.groupby("phenotype", sort=True):
        pos = (
            group[group["y_pred"] == 1]
            .sort_values("proba", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        neg = (
            group[group["y_pred"] == 0]
            .sort_values("proba", ascending=True, kind="mergesort")
            .reset_index(drop=True)
        )
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 and n_neg == 0:
            continue
        for coverage in COVERAGE_GRID:
            k_pos = max(1, int(round(coverage * n_pos))) if n_pos else 0
            k_neg = max(1, int(round(coverage * n_neg))) if n_neg else 0
            retained = pd.concat(
                [pos.iloc[:k_pos], neg.iloc[:k_neg]], ignore_index=True
            )
            y_true = retained["y_true"].to_numpy()
            y_pred = retained["y_pred"].to_numpy()
            rows.append(
                {
                    "phenotype": phenotype,
                    "coverage": float(coverage),
                    "n_retained": len(retained),
                    "n_retained_pos": k_pos,
                    "n_retained_neg": k_neg,
                    "balanced_accuracy": safe_balanced_accuracy(y_true, y_pred),
                    "accuracy": float((y_true == y_pred).mean())
                    if len(retained)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Build and persist the Supplementary Figure S15 data tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GapMind predictions (loose)...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    print(
        f"  {len(gapmind_predictions)} genomes, "
        f"{len(gapmind_predictions.columns)} phenotypes"
    )

    print("Loading experimental phenotypes...")
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(
        f"  {len(experimental_phenotypes)} genomes, "
        f"{len(experimental_phenotypes.columns)} phenotypes"
    )

    print("Loading dataset_split train-test splits...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=["dataset_split"],
        feature_file=KOFAM_FEATURE_FILE,
    )

    per_sample = collect_per_sample_predictions(
        split_data, gapmind_predictions, experimental_phenotypes
    )
    if per_sample.empty:
        raise RuntimeError("No per-sample predictions were produced.")

    per_sample = per_sample.sort_values(
        ["phenotype", "held_out_dataset", "genome"]
    ).reset_index(drop=True)

    risk_coverage = build_risk_coverage(per_sample)

    per_sample.to_csv(OUTPUT_DIR / "figure8_per_sample.tsv", sep="\t", index=False)
    risk_coverage.to_csv(
        OUTPUT_DIR / "figure8_risk_coverage.tsv", sep="\t", index=False
    )

    full_per_sample = collect_full_data_per_sample(split_data, gapmind_predictions)
    full_per_sample = full_per_sample.sort_values(
        ["phenotype", "held_out_dataset", "genome"]
    ).reset_index(drop=True)
    full_per_sample.to_csv(
        OUTPUT_DIR / "figure8_per_sample_fulldata.tsv", sep="\t", index=False
    )
    print(f"  full-data per-sample genomes: {len(full_per_sample)}")

    calib_rows = []
    for model_name, frame in [("concordant", per_sample), ("full_data", full_per_sample)]:
        ct = calibration_table(frame, n_bins=10)
        ct.insert(0, "model", model_name)
        ct["ece"] = expected_calibration_error(
            frame["y_true"].to_numpy(), frame["proba"].to_numpy(), n_bins=10
        )
        calib_rows.append(ct)
    pd.concat(calib_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "figure8_calibration.tsv", sep="\t", index=False
    )

    rc_rows = []
    for model_name, frame in [("concordant", per_sample), ("full_data", full_per_sample)]:
        rc = build_risk_coverage_by_phenotype(frame)
        rc.insert(0, "model", model_name)
        rc_rows.append(rc)
    pd.concat(rc_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "figure8_risk_coverage_by_phenotype.tsv", sep="\t", index=False
    )

    print(f"\nSaved Supplementary Figure S15 data to {OUTPUT_DIR}")
    print(f"  pooled held-out test genomes: {len(per_sample)}")
    full = risk_coverage[risk_coverage["coverage"] == 1.0].iloc[0]
    half = risk_coverage.iloc[(risk_coverage["coverage"] - 0.5).abs().argmin()]
    print(
        f"  balanced accuracy @ coverage 1.00: {full['balanced_accuracy']:.3f}"
    )
    print(
        f"  balanced accuracy @ coverage {half['coverage']:.2f}: "
        f"{half['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
