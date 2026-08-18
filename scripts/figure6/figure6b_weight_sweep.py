#!/usr/bin/env python3
"""Phase 2 weight sweep: train CatBoost on y_soft-filtered splits for four weight configs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scripts.figure6.figure6b_data import (
    K_NEIGHBORS,
    load_gapmind_confidence,
    load_phylogenetic_data,
)
from scripts.figure6.figure6b_parameter_exploration import (
    WeightConfig,
    phylo_knn_confidence,
)
from scripts.ml_splits import load_split_data, perform_split_ml

WeightingMode = Literal["label_confidence", "boundary_certainty"]
DEFAULT_WEIGHTING_MODE: WeightingMode = "label_confidence"

OUTPUT_DIR = Path("data/outputs/figure6")
SPLITS_DIR = Path("data/processed/train_test_splits")
SPLIT_TYPES = ["random_split", "dataset_split"]
MIN_TEST_SAMPLES = 10

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

PHASE2_CONFIGS: list[WeightConfig] = [
    WeightConfig("free_balanced", w_phylo=0.4, w_gapmind=0.0, w_exp=0.6),
    WeightConfig("current", w_phylo=0.2, w_gapmind=0.3, w_exp=0.5),
    WeightConfig("high_mech", w_phylo=0.15, w_gapmind=0.4, w_exp=0.45),
    WeightConfig("very_high_mech", w_phylo=0.1, w_gapmind=0.5, w_exp=0.4),
]


@dataclass(frozen=True)
class FilteredSplit:
    """One confidence-weighted split ready for ML.

    Attributes
    ----------
    split_type : str
        ``random_split`` or ``dataset_split``.
    key : str
        Original split key, e.g. ``Histidine_0`` or ``Histidine_train(...)``.
    phenotype : str
        Phenotype name.
    X_train, y_train, X_val, y_val, X_test, y_test : pd.DataFrame | pd.Series
        Phylogenetically scored train/val and original test.
    train_weights, val_weights : pd.Series
        Per-row confidence weights for CatBoost fitting and early stopping.
    """

    split_type: str
    key: str
    phenotype: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_weights: pd.Series
    val_weights: pd.Series


def compute_split_y_soft(
    config: WeightConfig,
    split: dict[str, Any],
    phenotype_name: str,
    conf_mech: dict[str, pd.Series] | None,
    distance_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Compute train and validation soft labels without held-out test labels.

    Parameters
    ----------
    config : WeightConfig
        Weights to apply.
    split : dict[str, Any]
        One train/validation/test split.
    phenotype_name : str
        Phenotype represented by the split.
    conf_mech : dict[str, pd.Series] | None
        GapMind confidence by phenotype and genome. May be omitted when
        ``w_gapmind`` is zero.
    distance_df : pd.DataFrame
        Precomputed phylogenetic distance matrix.

    Returns
    -------
    dict[str, pd.Series]
        Soft-label series for ``train`` and ``val``, clipped to ``(0.01, 0.99)``.
    """
    y_train = cast(pd.Series, split["y_train"]).dropna()
    scores: dict[str, pd.Series] = {}
    for set_name in ("train", "val"):
        y_exp = cast(pd.Series, split[f"y_{set_name}"]).dropna()
        conf_phylo = phylo_knn_confidence(
            y_train,
            distance_df,
            k=K_NEIGHBORS,
            query_index=y_exp.index,
        )
        common = conf_phylo.index.intersection(y_exp.index)
        mech_conf: pd.Series | None = None
        if config.w_gapmind != 0.0:
            if conf_mech is None or phenotype_name not in conf_mech:
                raise ValueError(
                    f"GapMind confidence is required for {config.name!r}"
                )
            mech_conf = conf_mech[phenotype_name]
            common = common.intersection(mech_conf.index)
        soft = (
            conf_phylo.loc[common] * config.w_phylo
            + y_exp.loc[common] * config.w_exp
        )
        if mech_conf is not None:
            soft += mech_conf.loc[common] * config.w_gapmind
        scores[set_name] = np.clip(soft, 0.01, 1 - 0.01)
    return scores


def sample_weights_from_soft_labels(
    y_exp: pd.Series,
    y_soft: pd.Series,
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> pd.Series:
    """Convert composite growth probabilities to per-label confidence weights.

    Parameters
    ----------
    y_exp : pd.Series
        Observed binary labels.
    y_soft : pd.Series
        Composite probabilities of growth, aligned by genome ID.
    weighting_mode : {"label_confidence", "boundary_certainty"}, optional
        ``label_confidence`` assigns the composite support for the observed
        label. ``boundary_certainty`` assigns twice the distance from 0.5 with
        a nonzero floor.

    Returns
    -------
    pd.Series
        Sample weights for genomes present in both inputs.

    Raises
    ------
    ValueError
        If ``weighting_mode`` is unsupported.
    """
    common = y_exp.index.intersection(y_soft.index)
    observed = y_exp.loc[common].astype(float)
    soft = y_soft.loc[common].astype(float)
    if weighting_mode == "label_confidence":
        weights = 1.0 - (observed - soft).abs()
    elif weighting_mode == "boundary_certainty":
        weights = np.clip(2.0 * (soft - 0.5).abs(), 0.01, 1.0)
    else:
        raise ValueError(f"unsupported weighting mode: {weighting_mode!r}")
    return pd.Series(weights, index=common, dtype=float)


def compute_split_sample_weights(
    config: WeightConfig,
    split: dict[str, Any],
    phenotype_name: str,
    conf_mech: dict[str, pd.Series] | None,
    distance_df: pd.DataFrame,
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> dict[str, pd.Series]:
    """Compute aligned train and validation weights without test labels.

    Parameters
    ----------
    config : WeightConfig
        Composite-score weights.
    split : dict[str, Any]
        One train/validation/test split.
    phenotype_name : str
        Phenotype represented by the split.
    conf_mech : dict[str, pd.Series] | None
        GapMind confidence, omitted for mechanism-free weighting.
    distance_df : pd.DataFrame
        Precomputed phylogenetic distance matrix.
    weighting_mode : {"label_confidence", "boundary_certainty"}, optional
        Mapping from composite probability to sample weight.

    Returns
    -------
    dict[str, pd.Series]
        Per-row weights for the tree-scored train and validation genomes.
    """
    soft_by_set = compute_split_y_soft(
        config, split, phenotype_name, conf_mech, distance_df
    )
    return {
        set_name: sample_weights_from_soft_labels(
            cast(pd.Series, split[f"y_{set_name}"]).dropna(),
            soft,
            weighting_mode,
        )
        for set_name, soft in soft_by_set.items()
    }


def filter_splits(
    split_data: dict[str, dict[str, dict[str, Any]]],
    config: WeightConfig,
    conf_mech: dict[str, pd.Series] | None,
    distance_df: pd.DataFrame,
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> list[FilteredSplit]:
    """Prepare confidence-weighted train and validation data for every split.

    Parameters
    ----------
    split_data : dict
        From ``load_split_data``.
    config : WeightConfig
        Weights to apply.
    conf_mech : dict[str, pd.Series] | None
        GapMind confidence by phenotype and genome. May be omitted for the
        mechanism-free configuration.
    distance_df : pd.DataFrame
        Precomputed phylogenetic distance matrix.
    weighting_mode : {"label_confidence", "boundary_certainty"}, optional
        Mapping from composite probability to sample weight.

    Returns
    -------
    list[FilteredSplit]
        Weighted splits, skipping any that lack a class in train or val.
    """
    filtered: list[FilteredSplit] = []
    for split_type, splits in split_data.items():
        for key, split in splits.items():
            phenotype_name = key.split("_")[0]
            if config.w_gapmind != 0.0 and (
                conf_mech is None or phenotype_name not in conf_mech
            ):
                continue

            weights_by_set = compute_split_sample_weights(
                config,
                split,
                phenotype_name,
                conf_mech,
                distance_df,
                weighting_mode,
            )
            kept_idx: dict[str, pd.Index] = {}
            for set_name in ("train", "val"):
                y = split[f"y_{set_name}"]
                kept_idx[set_name] = y.index.intersection(
                    weights_by_set[set_name].index
                )

            X_train = split["X_train"].loc[kept_idx["train"]]
            y_train = split["y_train"].loc[kept_idx["train"]]
            X_val = split["X_val"].loc[kept_idx["val"]]
            y_val = split["y_val"].loc[kept_idx["val"]]
            X_test = split["X_test"]
            y_test = split["y_test"]

            if len(X_test) < MIN_TEST_SAMPLES:
                continue
            if len(y_train.unique()) != 2 or len(y_val.unique()) != 2:
                continue

            filtered.append(
                FilteredSplit(
                    split_type=split_type,
                    key=key,
                    phenotype=phenotype_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    train_weights=weights_by_set["train"].loc[y_train.index],
                    val_weights=weights_by_set["val"].loc[y_val.index],
                )
            )
    return filtered


def fit_one(
    fs: FilteredSplit,
    config_name: str,
    thread_count: int,
) -> dict[str, Any]:
    """Train and score CatBoost on a single filtered split.

    Parameters
    ----------
    fs : FilteredSplit
        One filtered train/val/test bundle.
    config_name : str
        Weight-config tag stored in the result row.
    thread_count : int
        CatBoost ``thread_count`` parameter, used to cap per-worker threads
        when this function is called inside ``joblib.Parallel``.

    Returns
    -------
    dict[str, Any]
        Result row including all metrics in ``SCORING`` and metadata.
    """
    result = perform_split_ml(
        fs.X_train,
        fs.y_train,
        fs.X_val,
        fs.y_val,
        fs.X_test,
        fs.y_test,
        model_type="cb",
        scoring=SCORING,
        train_sample_weight=fs.train_weights,
        val_sample_weight=fs.val_weights,
        random_state=42,
        thread_count=thread_count,
    )
    result.update(
        {
            "config": config_name,
            "split_type": fs.split_type,
            "key": fs.key,
            "phenotype": fs.phenotype,
            "model_type": "cb",
            "n_train": len(fs.X_train),
            "n_val": len(fs.X_val),
            "n_test": len(fs.X_test),
            "train_weight_sum": float(fs.train_weights.sum()),
            "val_weight_sum": float(fs.val_weights.sum()),
        }
    )
    return result


def run_config(
    config: WeightConfig,
    split_data: dict[str, dict[str, dict[str, Any]]],
    conf_mech: dict[str, pd.Series],
    distance_df: pd.DataFrame,
    n_jobs: int,
    thread_count: int,
    weighting_mode: WeightingMode,
    output_suffix: str = "",
) -> pd.DataFrame:
    """Run all ML fits for one weight config.

    Parameters
    ----------
    config : WeightConfig
        Weights to apply.
    split_data : dict
        Output of ``load_split_data`` (shared across configs).
    conf_mech : dict[str, pd.Series]
        GapMind confidence by phenotype and genome.
    distance_df : pd.DataFrame
        Precomputed phylogenetic distance matrix.
    n_jobs : int
        Parallel workers. ``1`` runs sequentially.
    thread_count : int
        CatBoost threads per worker.
    weighting_mode : {"label_confidence", "boundary_certainty"}
        Mapping from composite probability to sample weight.
    output_suffix : str, optional
        Suffix used to keep sensitivity-analysis outputs separate.

    Returns
    -------
    pd.DataFrame
        One row per filtered split with metrics and metadata.
    """
    print(
        f"\n=== Config: {config.name} "
        f"(w_phylo={config.w_phylo}, w_gapmind={config.w_gapmind}, "
        f"w_exp={config.w_exp}) ==="
    )
    filtered = filter_splits(
        split_data, config, conf_mech, distance_df, weighting_mode
    )
    print(
        f"  {len(filtered)} filtered splits (n_jobs={n_jobs}, thread_count={thread_count})"
    )

    t0 = time.time()
    if n_jobs == 1:
        rows = [
            fit_one(fs, config.name, thread_count)
            for fs in tqdm(filtered, desc=config.name)
        ]
    else:
        rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(fit_one)(fs, config.name, thread_count) for fs in filtered
        )
        rows = cast(list[dict[str, Any]], rows)
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df["weighting_mode"] = weighting_mode
    out_file = OUTPUT_DIR / (
        f"figure6b_weight_sweep_{config.name}{output_suffix}.csv"
    )
    df.to_csv(out_file, index=False)
    print(f"  Saved {out_file}")
    return df


def main(
    n_jobs: int = 4,
    thread_count: int = 3,
    weighting_mode: WeightingMode = DEFAULT_WEIGHTING_MODE,
) -> None:
    """Run all Phase 2 configs and write a combined CSV.

    Parameters
    ----------
    n_jobs : int
        Parallel workers across ML fits. Use ``1`` to disable parallelism.
    thread_count : int
        CatBoost ``thread_count`` per worker.
    weighting_mode : {"label_confidence", "boundary_certainty"}, optional
        Mapping from composite probability to sample weight. Non-default
        sensitivity outputs receive a filename suffix.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading splits from {SPLITS_DIR} ...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=SPLIT_TYPES)

    print("Loading phylogenetic distances and GapMind confidence ...")
    _tree, distance_df = load_phylogenetic_data()
    conf_mech = load_gapmind_confidence()
    output_suffix = (
        "" if weighting_mode == DEFAULT_WEIGHTING_MODE else f"_{weighting_mode}"
    )

    all_results: list[pd.DataFrame] = []
    for config in PHASE2_CONFIGS:
        df = run_config(
            config,
            split_data,
            conf_mech,
            distance_df,
            n_jobs,
            thread_count,
            weighting_mode,
            output_suffix,
        )
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    combined = annotate_minority_test(combined, full_test_minority_counts())
    combined_file = OUTPUT_DIR / (
        f"figure6b_weight_sweep_combined{output_suffix}.csv"
    )
    combined.to_csv(combined_file, index=False)
    print(f"\nSaved combined results: {combined_file}")

    print("\n=== Summary: mean balanced_accuracy by config and split_type ===")
    print(
        combined.groupby(["config", "split_type"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )


if __name__ == "__main__":
    n_jobs_env = int(os.environ.get("PHASE2_N_JOBS", "4"))
    thread_count_env = int(os.environ.get("PHASE2_THREAD_COUNT", "3"))
    weighting_mode_env = cast(
        WeightingMode,
        os.environ.get("FIGURE6_WEIGHTING_MODE", DEFAULT_WEIGHTING_MODE),
    )
    main(
        n_jobs=n_jobs_env,
        thread_count=thread_count_env,
        weighting_mode=weighting_mode_env,
    )
