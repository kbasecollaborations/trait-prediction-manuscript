#!/usr/bin/env python3
"""Phase 1 soft-label weight sweep: report y_soft retention and concordant overlap, no ML."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.figure6.figure6b_data import (
    K_NEIGHBORS,
    load_gapmind_confidence,
    load_gapmind_data,
    load_phenotype_data,
    load_phylogenetic_data,
)
from scripts.io import cache_is_fresh


def phylo_knn_confidence(
    y_exp: pd.Series, distance_df: pd.DataFrame, k: int
) -> pd.Series:
    """Mean experimental label of each genome's k nearest tree neighbours.

    Parameters
    ----------
    y_exp : pd.Series
        Binary experimental labels indexed by genome ID.
    distance_df : pd.DataFrame
        Square pairwise tree-distance matrix, diagonal set to ``inf``.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    pd.Series
        Mean label of the k nearest neighbours, indexed like ``y_exp``.

    Notes
    -----
    Bypasses ``NearestNeighborClassifier.fit`` because the installed
    ``trait_prediction`` version recomputes distances from a newick file path
    rather than reusing the already-loaded ``distance_df``.
    """
    common = y_exp.index.intersection(distance_df.index)
    y_train = y_exp.loc[common].astype(float)
    distances_train = distance_df.loc[common, common]

    conf = pd.Series(index=common, dtype=float)
    for genome_id in common:
        node_distances = distances_train.loc[genome_id]
        nearest = node_distances.nsmallest(k).index
        conf.loc[genome_id] = float(y_train.loc[nearest].mean())
    return conf

CONFIDENCE_THRESHOLD_LOW = 0.4
CONFIDENCE_THRESHOLD_HIGH = 0.6

OUTPUT_DIR = Path("data/outputs/figure6")
CACHE_FILE = OUTPUT_DIR / "phase1_inputs_cache.pkl"

PHENOTYPE_DIR = Path("data/processed/phenotypes")
"""Experimental labels the cached ``y_exp`` and phylogenetic k-NN confidence
derive from. Tracked for cache freshness: a label correction must invalidate the
cache, otherwise the sweep silently trains on superseded soft labels."""


@dataclass(frozen=True)
class WeightConfig:
    """One soft-label weight configuration to test.

    Parameters
    ----------
    name : str
        Short label used in the output table.
    w_phylo : float
        Weight on the phylogenetic k-NN confidence.
    w_gapmind : float
        Weight on the GapMind mechanistic confidence. ``0`` means no
        mechanistic predictor is used.
    w_exp : float
        Weight on the experimental label.
    """

    name: str
    w_phylo: float
    w_gapmind: float
    w_exp: float

    def __post_init__(self) -> None:
        total = self.w_phylo + self.w_gapmind + self.w_exp
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0; got {total:.4f} for config {self.name!r}"
            )


CONFIGS: list[WeightConfig] = [
    WeightConfig("free_phylo_heavy", w_phylo=0.5, w_gapmind=0.0, w_exp=0.5),
    WeightConfig("free_balanced", w_phylo=0.4, w_gapmind=0.0, w_exp=0.6),
    WeightConfig("free_exp_leaning", w_phylo=0.3, w_gapmind=0.0, w_exp=0.7),
    WeightConfig("low_mech", w_phylo=0.3, w_gapmind=0.15, w_exp=0.55),
    WeightConfig("mid_mech", w_phylo=0.25, w_gapmind=0.2, w_exp=0.55),
    WeightConfig("current", w_phylo=0.2, w_gapmind=0.3, w_exp=0.5),
    WeightConfig("high_mech", w_phylo=0.15, w_gapmind=0.4, w_exp=0.45),
    WeightConfig("very_high_mech", w_phylo=0.1, w_gapmind=0.5, w_exp=0.4),
]


def build_inputs(fresh: bool = False) -> dict[str, dict[str, pd.Series]]:
    """Compute ``conf_phylo``, ``conf_mech``, and ``y_exp`` once per phenotype.

    The result is cached in ``CACHE_FILE``. Because the cache is derived from the
    experimental labels, it is reused only when it post-dates
    ``PHENOTYPE_DIR``; a stale cache is rebuilt rather than silently returned.

    Parameters
    ----------
    fresh : bool, optional
        Ignore any existing cache and recompute, default ``False``.

    Returns
    -------
    dict[str, dict[str, pd.Series]]
        Outer dict keyed by phenotype name; inner dict has keys ``conf_phylo``,
        ``conf_mech``, ``y_exp`` and ``gapmind_pred`` (binary GapMind call used
        to define the concordant set).
    """
    if not fresh and cache_is_fresh(CACHE_FILE, PHENOTYPE_DIR):
        print(f"Loading cached Phase 1 inputs from {CACHE_FILE} ...")
        with open(CACHE_FILE, "rb") as f:
            return cast(dict[str, dict[str, pd.Series]], pickle.load(f))
    if CACHE_FILE.exists():
        print(
            f"Cache {CACHE_FILE} predates {PHENOTYPE_DIR}; rebuilding from "
            "current labels."
        )

    print("Building Phase 1 inputs (slow: phylogenetic k-NN per phenotype) ...")
    _tree, distance_df = load_phylogenetic_data()
    conf_mech = load_gapmind_confidence()
    gapmind_binary = load_gapmind_data()
    phenotype_data = load_phenotype_data()

    inputs: dict[str, dict[str, pd.Series]] = {}
    for phenotype_name, y_exp in tqdm(
        phenotype_data.items(), desc="Computing phylo confidence"
    ):
        if phenotype_name not in conf_mech:
            print(f"Warning: {phenotype_name} not in GapMind confidence, skipping")
            continue
        if phenotype_name not in gapmind_binary.columns:
            print(f"Warning: {phenotype_name} not in GapMind binary, skipping")
            continue

        conf_phylo = phylo_knn_confidence(
            y_exp, distance_df, k=K_NEIGHBORS
        )
        inputs[phenotype_name] = {
            "conf_phylo": conf_phylo,
            "conf_mech": conf_mech[phenotype_name],
            "y_exp": y_exp,
            "gapmind_pred": gapmind_binary[phenotype_name],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(inputs, f)
    print(f"Cached Phase 1 inputs to {CACHE_FILE}")
    return inputs


def evaluate_config(
    config: WeightConfig,
    inputs: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """Apply the ``y_soft`` filter for one config and report retention.

    Parameters
    ----------
    config : WeightConfig
        Weights to use.
    inputs : dict[str, dict[str, pd.Series]]
        Per-phenotype phylo / mech / exp / gapmind_pred series.

    Returns
    -------
    pd.DataFrame
        One row per phenotype with columns ``n_total``, ``n_kept``,
        ``frac_kept``, ``n_concordant``, ``frac_kept_in_concordant``,
        ``frac_concordant_in_kept``, ``n_filtered_only_in_concordant``,
        ``n_kept_only_in_discordant``.
    """
    rows: list[dict[str, float | int | str]] = []
    for phenotype_name, parts in inputs.items():
        conf_phylo = parts["conf_phylo"]
        conf_mech = parts["conf_mech"]
        y_exp = parts["y_exp"]
        gapmind_pred = parts["gapmind_pred"]

        common = (
            conf_phylo.index
            .intersection(conf_mech.index)
            .intersection(y_exp.index)
            .intersection(gapmind_pred.index)
        )
        if len(common) == 0:
            continue

        conf_phylo_s = conf_phylo.loc[common]
        conf_mech_s = conf_mech.loc[common]
        y_exp_s = y_exp.loc[common]
        gapmind_pred_s = gapmind_pred.loc[common]

        y_soft = (
            conf_phylo_s * config.w_phylo
            + conf_mech_s * config.w_gapmind
            + y_exp_s * config.w_exp
        )
        y_soft = np.clip(y_soft, 0.01, 1 - 0.01)

        kept_mask = (y_soft < CONFIDENCE_THRESHOLD_LOW) | (
            y_soft > CONFIDENCE_THRESHOLD_HIGH
        )
        kept = set(common[kept_mask])
        concordant = set(common[y_exp_s.values == gapmind_pred_s.values])

        n_total = len(common)
        n_kept = len(kept)
        n_concordant = len(concordant)
        kept_and_concordant = kept & concordant
        kept_only = kept - concordant
        concordant_only = concordant - kept

        rows.append(
            {
                "config": config.name,
                "phenotype": phenotype_name,
                "n_total": n_total,
                "n_kept": n_kept,
                "frac_kept": n_kept / n_total if n_total > 0 else float("nan"),
                "n_concordant": n_concordant,
                "frac_concordant": (
                    n_concordant / n_total if n_total > 0 else float("nan")
                ),
                "n_kept_and_concordant": len(kept_and_concordant),
                "frac_kept_in_concordant": (
                    len(kept_and_concordant) / n_concordant
                    if n_concordant > 0
                    else float("nan")
                ),
                "frac_concordant_in_kept": (
                    len(kept_and_concordant) / n_kept
                    if n_kept > 0
                    else float("nan")
                ),
                "n_kept_only": len(kept_only),
                "n_concordant_only": len(concordant_only),
            }
        )

    return pd.DataFrame(rows)


def pairwise_jaccard_filtered_out(
    per_config_filtered_out: dict[str, set[str]],
) -> pd.DataFrame:
    """Jaccard overlap between the *removed* sets across configs.

    Parameters
    ----------
    per_config_filtered_out : dict[str, set[str]]
        Mapping config name -> set of genome IDs removed by that config
        (pooled across phenotypes).

    Returns
    -------
    pd.DataFrame
        Square symmetric Jaccard matrix.
    """
    names = list(per_config_filtered_out)
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sa, sb = per_config_filtered_out[a], per_config_filtered_out[b]
            union = sa | sb
            mat[i, j] = len(sa & sb) / len(union) if union else float("nan")
    return pd.DataFrame(mat, index=names, columns=names)


def main(fresh: bool = False) -> None:
    """Run Phase 1 sweep and write diagnostic CSVs.

    Parameters
    ----------
    fresh : bool, optional
        Recompute the Phase 1 input cache even if it is up to date.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(fresh=fresh)

    all_rows: list[pd.DataFrame] = []
    filtered_out_sets: dict[str, set[str]] = {}

    for config in CONFIGS:
        print(f"\n=== Config: {config.name} (w_phylo={config.w_phylo}, "
              f"w_gapmind={config.w_gapmind}, w_exp={config.w_exp}) ===")
        df = evaluate_config(config, inputs)
        all_rows.append(df)

        per_phenotype = df.set_index("phenotype")
        print(per_phenotype[["n_total", "n_kept", "frac_kept",
                              "frac_kept_in_concordant",
                              "frac_concordant_in_kept"]].round(3))

        removed_pool: set[str] = set()
        for phenotype_name, parts in inputs.items():
            conf_phylo = parts["conf_phylo"]
            conf_mech = parts["conf_mech"]
            y_exp = parts["y_exp"]
            common = (
                conf_phylo.index
                .intersection(conf_mech.index)
                .intersection(y_exp.index)
            )
            conf_phylo_s = conf_phylo.loc[common]
            conf_mech_s = conf_mech.loc[common]
            y_exp_s = y_exp.loc[common]
            y_soft = (
                conf_phylo_s * config.w_phylo
                + conf_mech_s * config.w_gapmind
                + y_exp_s * config.w_exp
            )
            y_soft = np.clip(y_soft, 0.01, 1 - 0.01)
            removed_mask = (y_soft >= CONFIDENCE_THRESHOLD_LOW) & (
                y_soft <= CONFIDENCE_THRESHOLD_HIGH
            )
            removed_pool.update(f"{phenotype_name}::{gid}"
                                 for gid in common[removed_mask])
        filtered_out_sets[config.name] = removed_pool

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(OUTPUT_DIR / "phase1_retention_per_phenotype.csv", index=False)

    pooled = (
        long_df.groupby("config")
        .agg(
            n_total=("n_total", "sum"),
            n_kept=("n_kept", "sum"),
            n_concordant=("n_concordant", "sum"),
            n_kept_and_concordant=("n_kept_and_concordant", "sum"),
            mean_frac_kept=("frac_kept", "mean"),
            mean_frac_kept_in_concordant=("frac_kept_in_concordant", "mean"),
            mean_frac_concordant_in_kept=("frac_concordant_in_kept", "mean"),
        )
        .reset_index()
    )
    pooled["frac_kept_pool"] = pooled["n_kept"] / pooled["n_total"]
    pooled["frac_concordant_pool"] = pooled["n_concordant"] / pooled["n_total"]
    pooled["frac_kept_in_concordant_pool"] = (
        pooled["n_kept_and_concordant"] / pooled["n_concordant"]
    )
    pooled["frac_concordant_in_kept_pool"] = (
        pooled["n_kept_and_concordant"] / pooled["n_kept"]
    )
    pooled.to_csv(OUTPUT_DIR / "phase1_retention_pooled.csv", index=False)

    print("\n=== Pooled retention across all phenotypes ===")
    print(
        pooled[
            [
                "config",
                "frac_kept_pool",
                "frac_concordant_pool",
                "frac_kept_in_concordant_pool",
                "frac_concordant_in_kept_pool",
            ]
        ].round(3)
    )

    jaccard = pairwise_jaccard_filtered_out(filtered_out_sets)
    jaccard.to_csv(OUTPUT_DIR / "phase1_jaccard_filtered_out.csv")
    print("\n=== Pairwise Jaccard overlap of *removed* sets across configs ===")
    print(jaccard.round(3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recompute the Phase 1 input cache even if it is up to date.",
    )
    main(fresh=parser.parse_args().fresh)
