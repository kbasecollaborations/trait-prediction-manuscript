#!/usr/bin/env python3
"""Phase 2 weight sweep: train CatBoost on y_soft-filtered splits for four weight configs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scripts.figure6.figure6b_parameter_exploration import (
    WeightConfig,
    build_inputs,
)
from scripts.ml_splits import load_split_data, perform_split_ml

CONFIDENCE_THRESHOLD_LOW = 0.4
CONFIDENCE_THRESHOLD_HIGH = 0.6

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
    """One filtered split ready for ML.

    Attributes
    ----------
    split_type : str
        ``random_split`` or ``dataset_split``.
    key : str
        Original split key, e.g. ``Histidine_0`` or ``Histidine_train(...)``.
    phenotype : str
        Phenotype name.
    X_train, y_train, X_val, y_val, X_test, y_test : pd.DataFrame | pd.Series
        Filtered train/val and original test.
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


def compute_y_soft(
    config: WeightConfig,
    inputs: dict[str, dict[str, pd.Series]],
) -> dict[str, pd.Series]:
    """Compute ``y_soft`` per phenotype for a given weight config.

    Parameters
    ----------
    config : WeightConfig
        Weights to apply.
    inputs : dict[str, dict[str, pd.Series]]
        Phase 1 cached per-phenotype series (``conf_phylo``, ``conf_mech``,
        ``y_exp``).

    Returns
    -------
    dict[str, pd.Series]
        Mapping phenotype name -> ``y_soft`` series clipped to ``(0.01, 0.99)``.
    """
    y_soft: dict[str, pd.Series] = {}
    for phenotype_name, parts in inputs.items():
        conf_phylo = parts["conf_phylo"]
        conf_mech = parts["conf_mech"]
        y_exp = parts["y_exp"]
        common = conf_phylo.index.intersection(conf_mech.index).intersection(
            y_exp.index
        )
        if len(common) == 0:
            continue
        soft = (
            conf_phylo.loc[common] * config.w_phylo
            + conf_mech.loc[common] * config.w_gapmind
            + y_exp.loc[common] * config.w_exp
        )
        y_soft[phenotype_name] = np.clip(soft, 0.01, 1 - 0.01)
    return y_soft


def filter_splits(
    split_data: dict[str, dict[str, dict[str, Any]]],
    y_soft: dict[str, pd.Series],
) -> list[FilteredSplit]:
    """Apply ``y_soft`` filter to train+val of every split.

    Parameters
    ----------
    split_data : dict
        From ``load_split_data``.
    y_soft : dict[str, pd.Series]
        From ``compute_y_soft``.

    Returns
    -------
    list[FilteredSplit]
        Filtered splits, skipping any that lose a class in train or val.
    """
    filtered: list[FilteredSplit] = []
    for split_type, splits in split_data.items():
        for key, split in splits.items():
            phenotype_name = key.split("_")[0]
            if phenotype_name not in y_soft:
                continue

            soft = y_soft[phenotype_name]
            kept_idx: dict[str, pd.Index] = {}
            for set_name in ("train", "val"):
                y = split[f"y_{set_name}"]
                common = y.index.intersection(soft.index)
                soft_set = soft.loc[common]
                confident = (soft_set < CONFIDENCE_THRESHOLD_LOW) | (
                    soft_set > CONFIDENCE_THRESHOLD_HIGH
                )
                kept_idx[set_name] = soft_set[confident].index

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
        }
    )
    return result


def run_config(
    config: WeightConfig,
    split_data: dict[str, dict[str, dict[str, Any]]],
    inputs: dict[str, dict[str, pd.Series]],
    n_jobs: int,
    thread_count: int,
) -> pd.DataFrame:
    """Run all ML fits for one weight config.

    Parameters
    ----------
    config : WeightConfig
        Weights to apply.
    split_data : dict
        Output of ``load_split_data`` (shared across configs).
    inputs : dict[str, dict[str, pd.Series]]
        Phase 1 cached per-phenotype series.
    n_jobs : int
        Parallel workers. ``1`` runs sequentially.
    thread_count : int
        CatBoost threads per worker.

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
    y_soft = compute_y_soft(config, inputs)
    filtered = filter_splits(split_data, y_soft)
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
    out_file = OUTPUT_DIR / f"figure6b_weight_sweep_{config.name}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved {out_file}")
    return df


def main(n_jobs: int = 4, thread_count: int = 3) -> None:
    """Run all Phase 2 configs and write a combined CSV.

    Parameters
    ----------
    n_jobs : int
        Parallel workers across ML fits. Use ``1`` to disable parallelism.
    thread_count : int
        CatBoost ``thread_count`` per worker.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading splits from {SPLITS_DIR} ...")
    split_data = load_split_data(base_dir=SPLITS_DIR, split_types=SPLIT_TYPES)

    # Routed through build_inputs() rather than unpickling the cache directly, so
    # the freshness check against the phenotype labels applies here too.
    inputs = build_inputs()

    all_results: list[pd.DataFrame] = []
    for config in PHASE2_CONFIGS:
        df = run_config(config, split_data, inputs, n_jobs, thread_count)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    from scripts.minority_filter import (
        annotate_minority_test,
        full_test_minority_counts,
    )

    combined = annotate_minority_test(combined, full_test_minority_counts())
    combined_file = OUTPUT_DIR / "figure6b_weight_sweep_combined.csv"
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
    main(n_jobs=n_jobs_env, thread_count=thread_count_env)
