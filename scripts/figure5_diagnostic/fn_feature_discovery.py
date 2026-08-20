#!/usr/bin/env python3
"""Stable features introduced by GapMind false negatives, per phenotype.

Three train-set regimes run through the multi-seed SHAP stable-feature pipeline
of Figure 5B / Tables S3-S4, differing only in which samples are selected:

* ``full``           : all training samples (Figure 4 / Table S3 regime).
* ``concordant``     : GapMind-concordant samples only (Figure 5B / Table S4).
* ``concordant_fn``  : concordant samples plus GapMind false negatives
  (GapMind = 0, experiment = 1).

For every phenotype and leave-one-dataset-out split the pipeline re-splits the
selected samples 80/20, screens to the top 300 KOFAM features, then over 20
seeds fits ``cb_noeval`` and keeps the top-10 mean-|SHAP| features; features
appearing in >= 70% of seeds are stable.

Features stable in ``concordant_fn`` but not in ``concordant`` are written to
``fn_introduced_features.csv`` with whether ``full`` also surfaces them, a
canonical/non-canonical flag (GapMind step KO set union KEGG reference map) and
an uncharacterised flag (hypothetical / putative / unknown-function KOs).

Run with::

    uv run python -m scripts.figure5_diagnostic.fn_feature_discovery [options]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import scripts.figure5.figure5b_data as f5b
from scripts.figure5_diagnostic.fn_mechanism_shap import (
    build_symbol_to_ko,
    canonical_ko_set,
    load_ko_descriptions,
)
from scripts.figure5.figure5b_data import (
    get_consistent_features,
    get_screened_split_data,
    train_and_get_top_features_split,
)
from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml_splits import load_single_split_data

THREADS: int = 4
SPLITS_DIR: Path = Path("data/processed/train_test_splits")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
GAPMIND_STEP_FILE: Path = Path("data/interim/features/combined_datasets/gapmind.tsv")
KO_DICT_FILE: Path = Path("data/external/mapping/KO_dictionary.json")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
DEFAULT_OUT_DIR: Path = Path("data/outputs/figure5_fn_discovery")

N_SEEDS: int = 20
THRESHOLD: float = 0.7
N_FEATURES: int = 10
N_CANDIDATE: int = 300
MIN_SAMPLES: int = 20
MIN_MINORITY: int = 10

Regime = str
REGIMES: tuple[Regime, ...] = ("full", "concordant", "concordant_fn")

UNCHAR_MARKERS: tuple[str, ...] = (
    "uncharacterized",
    "uncharacterised",
    "putative",
    "hypothetical",
    "unknown function",
    "DUF",
    "UPF",
    "domain-containing protein",
    "membrane protein",
)


def labeled_sets(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    phenotype: str,
) -> tuple[set[str], set[str]]:
    """Return the concordant and false-negative genome sets for a phenotype.

    Parameters
    ----------
    gapmind_predictions : pd.DataFrame
        GapMind loose predictions (genomes x phenotypes).
    experimental_phenotypes : pd.DataFrame
        Experimental phenotypes (genomes x phenotypes).
    phenotype : str
        Phenotype name.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(concordant, false_negative)`` genome-id sets. Both empty if the
        phenotype is missing from either table.
    """
    if phenotype not in gapmind_predictions.columns:
        return set(), set()
    if phenotype not in experimental_phenotypes.columns:
        return set(), set()
    common = gapmind_predictions.index.intersection(experimental_phenotypes.index)
    exp = experimental_phenotypes.loc[common, phenotype].dropna()
    gm = gapmind_predictions.loc[exp.index, phenotype].dropna()
    exp = exp.loc[gm.index]
    concordant = set(gm.index[gm == exp])
    false_negative = set(gm.index[(gm == 0) & (exp == 1)])
    return concordant, false_negative


def select_samples(
    index: Sequence[str],
    regime: Regime,
    concordant: set[str],
    false_negative: set[str],
) -> list[str]:
    """Select the training genome ids for a regime.

    Parameters
    ----------
    index : Sequence[str]
        Combined train+val genome ids available in the split.
    regime : Regime
        One of ``"full"``, ``"concordant"``, ``"concordant_fn"``.
    concordant : set[str]
        Concordant genome set.
    false_negative : set[str]
        False-negative genome set.

    Returns
    -------
    list[str]
        Selected genome ids.
    """
    idx = set(index)
    if regime == "full":
        return sorted(idx)
    if regime == "concordant":
        return sorted(idx & concordant)
    if regime == "concordant_fn":
        return sorted(idx & (concordant | false_negative))
    raise ValueError(f"Unknown regime: {regime}")


def stable_features_for_selection(
    x_combined: pd.DataFrame,
    y_combined: pd.Series,
    sample_ids: Sequence[str],
) -> list[str] | None:
    """Compute stable top SHAP features for a selected sample set.

    Mirrors ``figure5b_data.analyze_combined_splits``: size guards, 80/20
    re-split, screen to ``N_CANDIDATE`` features, then a ``N_SEEDS`` SHAP
    top-``N_FEATURES`` loop kept at ``THRESHOLD`` consistency.

    Parameters
    ----------
    x_combined : pd.DataFrame
        Combined train+val feature matrix.
    y_combined : pd.Series
        Combined train+val labels.
    sample_ids : Sequence[str]
        Selected genome ids.

    Returns
    -------
    list[str] | None
        Stable feature names, or ``None`` if the selection was too small or
        single-class.
    """
    if len(sample_ids) < MIN_SAMPLES:
        return None
    x_sel = x_combined.loc[list(sample_ids)]
    y_sel = y_combined.loc[list(sample_ids)]
    if y_sel.nunique() != 2 or y_sel.value_counts().min() < MIN_MINORITY:
        return None

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_sel, y_sel, train_size=0.8, stratify=y_sel, random_state=42, shuffle=True
    )
    split_data = {"X_train": x_tr, "y_train": y_tr, "X_val": x_val, "y_val": y_val}
    split_data = get_screened_split_data(split_data, n_candidate_features=N_CANDIDATE)

    feature_lists: list[list[str]] = []
    for seed in range(N_SEEDS):
        try:
            feature_lists.append(
                train_and_get_top_features_split(
                    split_data, random_state=seed, n_features=N_FEATURES
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep sweeping other seeds
            print(f"    seed {seed} failed: {exc}", flush=True)
    if not feature_lists:
        return None
    return get_consistent_features(feature_lists, threshold=THRESHOLD)


def run_sweep(
    phenotypes: Sequence[str] | None,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> dict[Regime, dict[str, list[str]]]:
    """Compute stable features per regime for every phenotype x split.

    Parameters
    ----------
    phenotypes : Sequence[str] | None
        Restrict to these phenotypes; ``None`` runs all found on disk.
    gapmind_predictions : pd.DataFrame
        GapMind loose predictions.
    experimental_phenotypes : pd.DataFrame
        Experimental phenotypes.
    feature_data : pd.DataFrame
        Reduced KOFAM feature matrix.

    Returns
    -------
    dict[Regime, dict[str, list[str]]]
        ``results[regime][key]`` -> stable feature list.
    """
    dataset_split_dir = SPLITS_DIR / "dataset_split"
    found = [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]
    pheno_list = [p for p in found if phenotypes is None or p in set(phenotypes)]

    results: dict[Regime, dict[str, list[str]]] = {r: {} for r in REGIMES}
    for phenotype in tqdm(sorted(pheno_list), desc="phenotypes"):
        concordant, false_negative = labeled_sets(
            gapmind_predictions, experimental_phenotypes, phenotype
        )
        phenotype_dir = dataset_split_dir / phenotype
        for split_type in [d.name for d in phenotype_dir.iterdir() if d.is_dir()]:
            key = f"{phenotype}_{split_type}"
            try:
                split_data = load_single_split_data(
                    phenotype_dir / split_type, feature_data
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  load {key} failed: {exc}", flush=True)
                continue
            x_combined = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
            y_combined = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)
            for regime in REGIMES:
                ids = select_samples(
                    x_combined.index, regime, concordant, false_negative
                )
                stable = stable_features_for_selection(x_combined, y_combined, ids)
                if stable is not None:
                    results[regime][key] = stable
    return results


def is_uncharacterized(description: str) -> bool:
    """Return whether a KO description denotes an uncharacterised gene.

    Parameters
    ----------
    description : str
        KO ``symbol; name [EC:...]`` description string.

    Returns
    -------
    bool
        ``True`` if the KO has no gene symbol or matches an uncharacterised marker.
    """
    if not description:
        return True
    head = description.split(";", 1)[0].strip()
    if head.startswith("K") and head[1:].isdigit():
        return True  # no gene symbol, only the KO id repeated
    low = description.lower()
    return any(m.lower() in low for m in UNCHAR_MARKERS)


def phenotype_of(key: str) -> str:
    """Return the phenotype prefix of a split key."""
    return key.split("_", 1)[0]


def build_comparison(
    results: Mapping[Regime, Mapping[str, list[str]]],
    canonical_by_phenotype: Mapping[str, set[str]],
    gapmind_by_phenotype: Mapping[str, set[str]],
    ko_descriptions: Mapping[str, str],
) -> pd.DataFrame:
    """Isolate FN-introduced features and annotate them.

    Parameters
    ----------
    results : Mapping
        ``results[regime][key]`` stable feature lists.
    canonical_by_phenotype : Mapping[str, set[str]]
        Canonical KO set per phenotype.
    gapmind_by_phenotype : Mapping[str, set[str]]
        GapMind-step KO subset per phenotype.
    ko_descriptions : Mapping[str, str]
        KO -> description map.

    Returns
    -------
    pd.DataFrame
        One row per (phenotype, KO) that is FN-introduced in >= 1 fold.
    """
    keys_by_pheno: dict[str, list[str]] = {}
    for key in results["concordant_fn"]:
        keys_by_pheno.setdefault(phenotype_of(key), []).append(key)

    rows: list[dict[str, object]] = []
    for phenotype, keys in keys_by_pheno.items():
        canonical = canonical_by_phenotype.get(phenotype, set())
        gapmind_kos = gapmind_by_phenotype.get(phenotype, set())
        cfn_counter: Counter[str] = Counter()
        conc_all: set[str] = set()
        full_all: set[str] = set()
        for key in keys:
            cfn_counter.update(results["concordant_fn"].get(key, []))
            conc_all.update(results["concordant"].get(key, []))
            full_all.update(results["full"].get(key, []))
        for ko, n_folds in cfn_counter.items():
            if ko in conc_all:
                continue  # not FN-introduced: concordant already had it
            desc = ko_descriptions.get(ko, "")
            rows.append(
                {
                    "phenotype": phenotype,
                    "ko": ko,
                    "description": desc,
                    "n_folds_fn": n_folds,
                    "in_full": ko in full_all,
                    "is_canonical": ko in canonical,
                    "in_gapmind_steps": ko in gapmind_kos,
                    "is_uncharacterized": is_uncharacterized(desc),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Candidate priority: uncharacterised and FN-specific first, then recurrence.
    frame["fn_specific"] = ~frame["in_full"]
    frame = frame.sort_values(
        ["is_uncharacterized", "fn_specific", "n_folds_fn", "phenotype"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--phenotypes", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the three-regime stable-feature sweep and the FN-introduced comparison."""
    args = parse_args()
    f5b._THREAD_COUNT = args.threads  # cap CatBoost threads inside the shared helpers
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phenotypes = (
        [p.strip() for p in args.phenotypes.split(",") if p.strip()]
        if args.phenotypes
        else None
    )

    print(f"Threads: {args.threads} | phenotypes: {phenotypes or 'all'}")
    ko_descriptions = load_ko_descriptions(KO_DICT_FILE)
    symbol_to_ko = build_symbol_to_ko(ko_descriptions)
    step_columns = pd.read_csv(GAPMIND_STEP_FILE, sep="\t", nrows=0).columns.tolist()

    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    feature_data = pd.read_csv(
        KOFAM_FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str}
    )

    print("Running three-regime stable-feature sweep...")
    results = run_sweep(
        phenotypes, gapmind_predictions, experimental_phenotypes, feature_data
    )
    for regime in REGIMES:
        out = args.out_dir / f"{regime}_stable_features.json"
        out.write_text(json.dumps(results[regime], indent=2))
        print(f"  {regime}: {len(results[regime])} splits -> {out}")

    canonical_by_phenotype: dict[str, set[str]] = {}
    gapmind_by_phenotype: dict[str, set[str]] = {}
    all_phenos = {phenotype_of(k) for k in results["concordant_fn"]}
    for phenotype in all_phenos:
        canonical, gapmind_kos, _ = canonical_ko_set(
            phenotype, step_columns, symbol_to_ko
        )
        canonical_by_phenotype[phenotype] = canonical
        gapmind_by_phenotype[phenotype] = gapmind_kos

    comparison = build_comparison(
        results, canonical_by_phenotype, gapmind_by_phenotype, ko_descriptions
    )
    comp_path = args.out_dir / "fn_introduced_features.csv"
    comparison.to_csv(comp_path, index=False)
    print(
        f"\nFN-introduced features (in concordant_fn, not in concordant): {len(comparison)}"
    )
    print(f"Wrote {comp_path}")

    if not comparison.empty:
        unchar = comparison[comparison["is_uncharacterized"]]
        print(
            f"\nUncharacterised FN-introduced candidates: {len(unchar)} "
            f"(of which FN-specific / not in full: {(~unchar['in_full']).sum()})"
        )
        cols = [
            "phenotype",
            "ko",
            "n_folds_fn",
            "in_full",
            "is_canonical",
            "is_uncharacterized",
            "description",
        ]
        print("\nTop 30 candidates (uncharacterised & FN-specific first):")
        print(comparison.head(30)[cols].to_string(max_colwidth=64))


if __name__ == "__main__":
    main()
