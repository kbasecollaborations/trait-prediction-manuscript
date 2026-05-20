#!/usr/bin/env python3
"""
Generate data for Supplementary Figure S6: Learning curves for all 15 phenotypes.

Extends the Figure 7 analysis (Histidine and Galactose only) to all 15 shared
phenotypes, providing the evidence needed to characterise the distribution of
saturation points rather than relying on a single best-case example.

Design mirrors figure7_data.py exactly, differing only in:
  - PHENOTYPES_TO_ANALYZE: all 15 instead of 2
  - Output directory: data/outputs/figureS6/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.ml import _get_scores, make_classifier


# ── Analysis parameters ──────────────────────────────────────────────────────
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
SAMPLE_SIZES: list[int | str] = [50, 100, 200, 500, "full"]
N_REPEATS = 3
SPLIT_TYPES = ["random_split", "dataset_split", "phylo_ooc"]
RANDOM_STATE = 42
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
    """
    Load GapMind predictions and convert to binary labels.

    Returns
    -------
    pd.DataFrame
        Binary predictions indexed by genomeID.
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
    gapmind_data = pd.concat(
        [
            pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{ds}_categories.csv")
            for ds in datasets
        ],
        axis=0,
    )
    gapmind_data["genomeId"] = (
        gapmind_data["genome_id"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    gapmind_data.index = gapmind_data["genomeId"]  # type: ignore
    gapmind_data.index = [marine_ids_map.get(i, i) for i in gapmind_data.index]
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
    """
    Load split label files for all 15 shared phenotypes.

    Parameters
    ----------
    base_dir : Path
        Root of train_test_splits directory.

    Returns
    -------
    dict[str, dict[str, dict[str, pd.Series]]]
        ``{split_type: {key: {"y_train", "y_val", "y_test"}}}``
    """
    result: dict[str, dict[str, dict[str, pd.Series]]] = {}

    def _read_y(p: Path) -> pd.Series:
        return pd.read_csv(p, sep="\t", index_col=0, dtype={"genomeID": str}).iloc[
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


def subsample_indices(
    indices: pd.Index,
    y: pd.Series,
    n_samples: int | str,
    random_state: int,
) -> pd.Index:
    """
    Stratified subsampling of indices.

    Parameters
    ----------
    indices : pd.Index
        Candidate indices.
    y : pd.Series
        Labels used for stratification.
    n_samples : int | str
        Target count, or ``"full"`` for no subsampling.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.Index
        Selected indices.
    """
    if n_samples == "full":
        return indices

    y_sub = y.loc[indices]
    if len(y_sub) <= n_samples:
        return indices

    if y_sub.nunique() == 1:
        return pd.Index(y_sub.sample(n=n_samples, replace=False, random_state=random_state).index)

    test_size = len(y_sub) - n_samples
    if test_size >= y_sub.nunique():
        sampled, _ = train_test_split(
            y_sub.index,
            train_size=n_samples,
            stratify=y_sub,
            random_state=random_state,
        )
        return pd.Index(sampled)

    return pd.Index(
        y_sub.sample(n=n_samples, replace=False, random_state=random_state).index
    )


def run_learning_curve_analysis(
    split_data: dict[str, dict[str, dict[str, pd.Series]]],
    feature_data: pd.DataFrame,
    gapmind_data: pd.DataFrame,
    chunk_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run learning-curve analysis for all 15 phenotypes.

    For each split × phenotype × training type × sample size × repeat,
    trains a CatBoost model and evaluates on full / concordant / discordant
    test subsets. If ``chunk_dir`` is provided, results for each
    (split_type, key, training_type) combination are written to a per-chunk
    CSV inside that directory, and chunks already present on disk are
    skipped on a subsequent run (resume support).

    Parameters
    ----------
    split_data : dict
        Nested split dictionary from :func:`load_split_files`.
    feature_data : pd.DataFrame
        Feature matrix (genomes × features).
    gapmind_data : pd.DataFrame
        Binary GapMind predictions.
    chunk_dir : Path, optional
        Directory to write/read per-chunk checkpoint CSVs. When ``None``,
        all results are accumulated in memory and no checkpoints are
        written.

    Returns
    -------
    pd.DataFrame
        One row per evaluation, with metric columns and metadata.
    """
    results: list[dict] = []

    total = sum(
        len(keys) * 2 * len(SAMPLE_SIZES) * N_REPEATS
        for keys in split_data.values()
    )

    with tqdm(total=total, desc="Learning curves (all phenotypes)") as pbar:
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
                    pbar.update(2 * len(SAMPLE_SIZES) * N_REPEATS)
                    continue

                # Concordant masks
                gm_tr = gapmind_data.loc[
                    train_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_tr = gm_tr[y_train_full.loc[gm_tr.index] == gm_tr].index

                gm_vl = gapmind_data.loc[
                    val_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_vl = gm_vl[y_val_full.loc[gm_vl.index] == gm_vl].index

                gm_te = gapmind_data.loc[
                    test_idx.intersection(gapmind_data.index), phenotype
                ]
                conc_te_mask = y_test_sub.loc[gm_te.index] == gm_te
                conc_te = gm_te[conc_te_mask].index
                disc_te = gm_te[~conc_te_mask].index

                for training_type in ["full", "concordant"]:
                    base_tr = train_idx if training_type == "full" else conc_tr
                    base_vl = val_idx if training_type == "full" else conc_vl

                    chunk_file: Path | None = None
                    if chunk_dir is not None:
                        chunk_file = chunk_dir / f"{split_type}__{key}__{training_type}.csv"
                        if chunk_file.exists():
                            # Resume: load cached chunk and skip recomputation.
                            try:
                                cached = pd.read_csv(chunk_file).to_dict("records")
                                results.extend(cached)
                                pbar.update(len(SAMPLE_SIZES) * N_REPEATS)
                                continue
                            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                                # Corrupt/empty checkpoint; recompute.
                                chunk_file.unlink(missing_ok=True)

                    chunk_results: list[dict] = []
                    for sample_size in SAMPLE_SIZES:
                        for rep in range(N_REPEATS):
                            pbar.update(1)
                            seed = RANDOM_STATE + rep

                            sampled_tr = subsample_indices(
                                base_tr, y_train_full, sample_size, seed
                            )
                            if len(sampled_tr) < 5:
                                continue

                            X_tr = X_train_full.loc[sampled_tr]
                            y_tr = y_train_full.loc[sampled_tr]

                            if y_tr.nunique() != 2:
                                continue

                            X_vl = X_val_full.loc[base_vl]
                            y_vl = y_val_full.loc[base_vl]
                            if y_vl.nunique() != 2:
                                continue

                            model = make_classifier("cb", random_state=RANDOM_STATE)
                            X_vl_a = X_vl.reindex(columns=X_tr.columns, fill_value=0)
                            model.fit(
                                X_tr,
                                y_tr,
                                eval_set=(X_vl_a, y_vl),
                                use_best_model=True,
                                verbose=False,
                            )

                            X_te_a = X_test.reindex(columns=X_tr.columns, fill_value=0)

                            test_subsets = [
                                ("full", test_idx, y_test_sub),
                                (
                                    "concordant",
                                    conc_te,
                                    y_test_sub.loc[conc_te]
                                    if len(conc_te) >= 5
                                    else None,
                                ),
                                (
                                    "discordant",
                                    disc_te,
                                    y_test_sub.loc[disc_te]
                                    if len(disc_te) >= 5
                                    else None,
                                ),
                            ]
                            for sub_name, sub_idx, y_sub in test_subsets:
                                if y_sub is None or len(y_sub) < 5:
                                    continue
                                scores = _get_scores(
                                    model, X_te_a.loc[sub_idx], y_sub, SCORING
                                )
                                scores.update(
                                    {
                                        "split_type": split_type,
                                        "key": key,
                                        "phenotype": phenotype,
                                        "training_type": training_type,
                                        "test_subset": sub_name,
                                        "sample_size": sample_size,
                                        "n_train_samples": len(y_tr),
                                        "n_test_samples": len(y_sub),
                                        "repeat": rep,
                                    }
                                )
                                chunk_results.append(scores)
                                results.append(scores)

                    if chunk_file is not None:
                        # Always write the chunk file (even if empty) so a
                        # resumed run does not redo this combination.
                        pd.DataFrame(chunk_results).to_csv(chunk_file, index=False)

    return pd.DataFrame(results)


def main() -> None:
    """Run learning-curve analysis for all 15 phenotypes and save results."""
    splits_dir = Path("data/processed/train_test_splits")
    feature_file = Path(
        f"data/processed/features_reduced/combined_datasets/{FEATURE_TYPE}.tsv"
    )
    output_dir = Path("data/outputs/figureS6")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / f"_chunks_{FEATURE_TYPE}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {FEATURE_TYPE.upper()} features...")
    features = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    print(f"  Shape: {features.shape}")

    print("Loading GapMind predictions...")
    gapmind = load_gapmind_predictions()
    print(f"  Shape: {gapmind.shape}")

    print("Loading splits...")
    splits = load_split_files(splits_dir)
    for st, sv in splits.items():
        print(f"  {st}: {len(sv)} splits")

    print(f"\nRunning learning curves for {len(COMMON_PHENOTYPES)} phenotypes...")
    print(f"Checkpoint chunks: {chunk_dir}")
    results = run_learning_curve_analysis(
        splits, features, gapmind, chunk_dir=chunk_dir
    )
    results["feature_type"] = FEATURE_TYPE

    out_file = output_dir / f"figureS6_learning_curves_{FEATURE_TYPE}.csv"
    results.to_csv(out_file, index=False)
    print(f"\nSaved {len(results)} rows to {out_file}")

    if len(results) > 0:
        print("\nMean balanced accuracy by phenotype (concordant, dataset_split, full test):")
        sub = results[
            (results["training_type"] == "concordant")
            & (results["split_type"] == "dataset_split")
            & (results["test_subset"] == "full")
        ]
        if len(sub) > 0:
            summary = (
                sub.groupby(["phenotype", "sample_size"])["balanced_accuracy"]
                .mean()
                .unstack("sample_size")
                .round(3)
            )
            print(summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
