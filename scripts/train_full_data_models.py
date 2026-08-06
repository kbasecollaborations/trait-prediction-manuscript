#!/usr/bin/env python3
"""Train and save full-data CatBoost deployment models (one per phenotype).

Trains one CatBoost classifier per phenotype on all available labelled genomes
using the reduced KOFAM feature matrix. The saved checkpoints predict a
phenotype for a new genome from its KOFAM presence/absence vector without
retraining.

Outputs (under ``data/outputs/full_data_models/<Phenotype>/``):
``<Phenotype>.cbm``, ``<Phenotype>_metadata.json``,
``<Phenotype>_feature_importances.csv``, ``<Phenotype>_selected_features.txt``,
plus a top-level ``full_data_models_summary.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.figure5.figure5a_data import load_experimental_phenotypes
from scripts.ml import make_classifier
from scripts.ml_splits import get_feature_importances

FEATURE_FILE = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
PHENOTYPE_DIR = Path("data/processed/phenotypes")
OUTPUT_BASE = Path("data/outputs/full_data_models")
RANDOM_STATE = 42
MODEL_TYPE = "cb_noeval"  # no early stopping: full data has no held-out eval set
MIN_MINORITY = 5

# The 15 carbon-source phenotypes analysed in the manuscript.
PHENOTYPES: list[str] = [
    "Alanine",
    "Arginine",
    "Cellobiose",
    "Fructose",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Glycerol",
    "Histidine",
    "m-Inositol",
    "Maltose",
    "Mannitol",
    "Mannose",
    "Serine",
    "Sucrose",
]


def _save_model_artifacts(
    out_dir: Path,
    phenotype: str,
    model: object,
    X: pd.DataFrame,
    n_pos: int,
    n_neg: int,
) -> dict[str, int | str]:
    """Save the ``.cbm`` checkpoint and its companion metadata files.

    Parameters
    ----------
    out_dir : Path
        Per-phenotype output directory (created if missing).
    phenotype : str
        Phenotype name, used as the artifact base name.
    model : object
        Fitted CatBoost classifier exposing ``save_model``.
    X : pd.DataFrame
        Training feature matrix (genomes x KOFAM features).
    n_pos, n_neg : int
        Positive/negative label counts in the training set.

    Returns
    -------
    dict[str, int | str]
        Summary row for the aggregate summary table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cbm_path = out_dir / f"{phenotype}.cbm"
    if hasattr(model, "save_model"):
        model.save_model(str(cbm_path), format="cbm")

    n_feats = X.shape[1]
    importances = get_feature_importances(model, X, n_features=n_feats)
    importances_df = importances.reset_index()
    importances_df.columns = ["feature", "importance"]
    importances_df.to_csv(out_dir / f"{phenotype}_feature_importances.csv", index=False)

    features = importances.index.tolist()
    with open(out_dir / f"{phenotype}_selected_features.txt", "w") as fh:
        fh.writelines(f"{feat}\n" for feat in features)

    metadata: dict[str, int | str] = {
        "model_name": phenotype,
        "phenotype_name": phenotype,
        "training": "full_data (all labelled genomes)",
        "feature_set": "kofam_reduced",
        "feature_file": str(FEATURE_FILE),
        "model_type": MODEL_TYPE,
        "n_samples": int(n_pos + n_neg),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "n_features": int(n_feats),
        "random_state": RANDOM_STATE,
    }
    with open(out_dir / f"{phenotype}_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def main() -> None:
    """Train and persist one full-data CatBoost model per phenotype."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(FEATURE_FILE, sep="\t", index_col=0)
    phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    print(f"Features: {X.shape[0]} genomes x {X.shape[1]} KOFAM features")

    summary: list[dict[str, int | str]] = []
    for phenotype in tqdm(PHENOTYPES, desc="Training full-data models"):
        if phenotype not in phenotypes.columns:
            print(f"  Skipping {phenotype}: not found in phenotype data")
            continue

        y = phenotypes[phenotype].dropna().astype(int)
        common = X.index.intersection(y.index)
        y = y.loc[common]
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        if min(n_pos, n_neg) < MIN_MINORITY:
            print(
                f"  Skipping {phenotype}: minority class < {MIN_MINORITY} (pos={n_pos}, neg={n_neg})"
            )
            continue

        X_pheno = X.loc[common]
        model = make_classifier(MODEL_TYPE, random_state=RANDOM_STATE)
        model.fit(X_pheno, y)
        meta = _save_model_artifacts(
            OUTPUT_BASE / phenotype, phenotype, model, X_pheno, n_pos, n_neg
        )
        summary.append(meta)

    if summary:
        pd.DataFrame(summary).to_csv(
            OUTPUT_BASE / "full_data_models_summary.csv", index=False
        )
    print(f"\nSaved {len(summary)} full-data models under {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
