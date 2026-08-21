#!/usr/bin/env python3
"""Generate data for Supplementary Figure S5: KOFAM features on concordant samples.

GapMind concordance selects training samples only; KOFAM annotations provide the
feature space. Concordant training and full training are compared across
random_split, dataset_split and phylo_ooc, on the full, concordant and
discordant test subsets.

Writes data/outputs/figureS5/figureS5_kofam_concordant_results.csv.

Run with::

    uv run python -m scripts.figureS5.figureS5_data
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.ml import _get_scores, make_classifier

COMMON_PHENOTYPES = [
    "Alanine",
    "Arginine",
    "Cellobiose",
    "Fructose",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Glycerol",
    "Histidine",
    "Maltose",
    "Mannitol",
    "Mannose",
    "Serine",
    "Sucrose",
    "m-Inositol",
]
SPLIT_TYPES = ["random_split", "dataset_split", "phylo_ooc"]
RANDOM_STATE = 42

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
    """Load GapMind predictions and convert to binary concordance labels.

    Returns
    -------
    pd.DataFrame
        Binary GapMind predictions (1 = pathway complete, 0 = incomplete)
        indexed by genomeID with phenotype columns.
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
        pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{ds}_categories.csv")
        for ds in datasets
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
    return gapmind_data.replace(replace_dict).astype(np.uint8)


def load_split_files(
    base_dir: Path,
) -> dict[str, dict[str, dict[str, pd.Series]]]:
    """Load train/val/test split label files for all 15 shared phenotypes.

    Parameters
    ----------
    base_dir : Path
        Root of the train_test_splits directory.

    Returns
    -------
    dict[str, dict[str, dict[str, pd.Series]]]
        ``{split_type: {key: {"y_train": ..., "y_val": ..., "y_test": ...}}}``
    """
    result: dict[str, dict[str, dict[str, pd.Series]]] = {}

    def _read_y(path: Path) -> pd.Series:
        return pd.read_csv(path, sep="\t", index_col=0, dtype={"genomeID": str}).iloc[
            :, 0
        ]

    if "random_split" in SPLIT_TYPES:
        rdir = base_dir / "random_split"
        data: dict[str, dict[str, pd.Series]] = {}
        if rdir.exists():
            for pdir in rdir.iterdir():
                if not pdir.is_dir() or pdir.name not in COMMON_PHENOTYPES:
                    continue
                for rpt in pdir.iterdir():
                    if not rpt.is_dir():
                        continue
                    key = f"{pdir.name}_{rpt.name}"
                    data[key] = {
                        "y_train": _read_y(rpt / "y_train.tsv"),
                        "y_val": _read_y(rpt / "y_val.tsv"),
                        "y_test": _read_y(rpt / "y_test.tsv"),
                    }
        result["random_split"] = data

    if "dataset_split" in SPLIT_TYPES:
        ddir = base_dir / "dataset_split"
        data = {}
        if ddir.exists():
            for pdir in ddir.iterdir():
                if not pdir.is_dir() or pdir.name not in COMMON_PHENOTYPES:
                    continue
                for sdir in pdir.iterdir():
                    if not sdir.is_dir():
                        continue
                    key = f"{pdir.name}_{sdir.name}"
                    data[key] = {
                        "y_train": _read_y(sdir / "y_train.tsv"),
                        "y_val": _read_y(sdir / "y_val.tsv"),
                        "y_test": _read_y(sdir / "y_test.tsv"),
                    }
        result["dataset_split"] = data

    if "phylo_ooc" in SPLIT_TYPES:
        phylo_dir = base_dir / "phylogeny_split"
        data = {}
        if phylo_dir.exists():
            for pdir in phylo_dir.iterdir():
                if not pdir.is_dir() or pdir.name not in COMMON_PHENOTYPES:
                    continue
                ooc = pdir / "out-of-clade"
                if not ooc.exists():
                    continue
                for sdir in ooc.iterdir():
                    if not sdir.is_dir():
                        continue
                    key = f"{pdir.name}_ooc_{sdir.name}"
                    data[key] = {
                        "y_train": _read_y(sdir / "y_train.tsv"),
                        "y_val": _read_y(sdir / "y_val.tsv"),
                        "y_test": _read_y(sdir / "y_test.tsv"),
                    }
        result["phylo_ooc"] = data

    return result


def run_kofam_concordant_analysis(
    split_data: dict[str, dict[str, dict[str, pd.Series]]],
    feature_data: pd.DataFrame,
    gapmind_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train models on full vs concordant samples using KOFAM features.

    For each split, trains two models (full training set, concordant-only
    training set), then evaluates on the full, concordant and discordant test
    subsets.

    Parameters
    ----------
    split_data : dict
        Nested split dictionary from :func:`load_split_files`.
    feature_data : pd.DataFrame
        KOFAM feature matrix (genomes x features).
    gapmind_data : pd.DataFrame
        Binary GapMind predictions for concordance identification.

    Returns
    -------
    pd.DataFrame
        Results with columns for metrics, split metadata, training type,
        and test subset type.
    """
    results: list[dict[str, Any]] = []

    total = sum(len(v) for v in split_data.values())
    with tqdm(total=total * 2, desc="KOFAM concordant analysis") as pbar:
        for split_type, splits in split_data.items():
            for key, split in splits.items():
                phenotype = key.split("_")[0]
                pbar.set_postfix_str(f"{split_type}/{key}")

                y_train = split["y_train"]
                y_val = split["y_val"]
                y_test = split["y_test"]

                train_idx = y_train.index.intersection(feature_data.index)
                val_idx = y_val.index.intersection(feature_data.index)
                test_idx = y_test.index.intersection(feature_data.index)

                X_train_full = feature_data.loc[train_idx]
                y_train_full = y_train.loc[train_idx]
                X_val_full = feature_data.loc[val_idx]
                y_val_full = y_val.loc[val_idx]
                X_test = feature_data.loc[test_idx]
                y_test_sub = y_test.loc[test_idx]

                if len(X_test) < 10:
                    pbar.update(2)
                    continue

                # Concordant = experimental label equals the GapMind call.
                gm_train = gapmind_data.loc[
                    train_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_train = gm_train[
                    y_train_full.loc[gm_train.index] == gm_train
                ].index

                gm_val = gapmind_data.loc[
                    val_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_val = gm_val[y_val_full.loc[gm_val.index] == gm_val].index

                gm_test = gapmind_data.loc[
                    test_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_test_mask = y_test_sub.loc[gm_test.index] == gm_test
                conc_test = gm_test[conc_test_mask].index
                disc_test = gm_test[~conc_test_mask].index

                for training_type in ["full", "concordant"]:
                    pbar.update(1)

                    if training_type == "full":
                        ti, vi = train_idx, val_idx
                    else:
                        ti, vi = conc_train, conc_val

                    X_tr = feature_data.loc[ti]
                    y_tr = y_train_full.loc[ti]
                    X_vl = feature_data.loc[vi]
                    y_vl = y_val_full.loc[vi]

                    if (
                        len(X_tr) < 10
                        or len(X_vl) < 5
                        or y_tr.nunique() != 2
                        or y_vl.nunique() != 2
                    ):
                        continue

                    model = make_classifier("cb", random_state=RANDOM_STATE)

                    # Align to the training columns.
                    X_vl_aligned = X_vl.reindex(columns=X_tr.columns, fill_value=0)
                    X_test_aligned = X_test.reindex(columns=X_tr.columns, fill_value=0)

                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=(X_vl_aligned, y_vl),
                        use_best_model=True,
                        verbose=False,
                    )

                    test_subsets = [
                        ("full", test_idx, y_test_sub),
                        (
                            "concordant",
                            conc_test,
                            y_test_sub.loc[conc_test] if len(conc_test) >= 5 else None,
                        ),
                        (
                            "discordant",
                            disc_test,
                            y_test_sub.loc[disc_test] if len(disc_test) >= 5 else None,
                        ),
                    ]
                    for subset_name, subset_idx, y_sub in test_subsets:
                        if y_sub is None or len(y_sub) < 5:
                            continue
                        X_sub = X_test_aligned.loc[subset_idx]
                        scores = _get_scores(model, X_sub, y_sub, SCORING)
                        scores.update(
                            {
                                "split_type": split_type,
                                "key": key,
                                "phenotype": phenotype,
                                "training_type": training_type,
                                "test_subset": subset_name,
                                "n_train": len(y_tr),
                                "n_val": len(y_vl),
                                "n_test": len(y_sub),
                            }
                        )
                        results.append(scores)

    return pd.DataFrame(results)


def main() -> None:
    """Run the KOFAM-on-concordant analysis and save results."""
    splits_dir = Path("data/processed/train_test_splits")
    feature_file = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
    output_dir = Path("data/outputs/figureS5")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading KOFAM features...")
    features = pd.read_csv(feature_file, sep="\t", index_col=0, dtype={"genomeID": str})
    print(f"  Shape: {features.shape}")

    print("Loading GapMind predictions...")
    gapmind = load_gapmind_predictions()
    print(f"  Shape: {gapmind.shape}")

    print("Loading splits...")
    splits = load_split_files(splits_dir)
    for st, sv in splits.items():
        print(f"  {st}: {len(sv)} splits")

    print("\nRunning analysis...")
    results = run_kofam_concordant_analysis(splits, features, gapmind)
    results["feature_type"] = "kofam"

    out_file = output_dir / "figureS5_kofam_concordant_results.csv"
    results.to_csv(out_file, index=False)
    print(f"\nSaved {len(results)} rows to {out_file}")

    if len(results) > 0:
        print("\nMean balanced accuracy by training type and split:")
        summary = (
            results[results["test_subset"] == "full"]
            .groupby(["split_type", "training_type"])["balanced_accuracy"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        print(summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
