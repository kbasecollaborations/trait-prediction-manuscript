#!/usr/bin/env python3
"""Size- and class-matched random-subset control for the concordance feature-recovery claim.

This is a robustness control for Figure 5B / Table 1. The main analysis reports
that cross-dataset SHAP feature stability roughly doubles (mean shared cluster
count 0.5 -> 1.3; phenotypes sharing >=1 cluster 6 -> 12) when training is
restricted to GapMind-concordant samples. A skeptical reading is that *any*
equal-size subset of the training pool, not concordance specifically, would
concentrate stable features. This script answers that by re-running the exact
Figure 5B stability pipeline on **random** subsets that are matched, per
phenotype and per data partition, to the concordant subset's size and class
balance, then comparing the resulting cluster-stability statistics against the
concordant values.

The pipeline (screen to a broad CatBoost candidate set, 20-seed SHAP top-10,
>=70% recurrence -> "stable", SHAP-supervised redundancy clustering, then
combined-of-three vs held-out-alone cluster intersection) is reused verbatim
from ``scripts.figure5.figure5b_data``; only the sample-selection rule changes
from "concordant" to "random, size- and class-matched".

Important: matching is on the number of TRAINING SAMPLES (rows), not on the
number of features (columns). The KOFAM feature space is held fixed and is
identical to the concordant and full-data analyses; concordance filtering does
not change the feature space, it removes about 30% of the training samples, so
the relevant confound to control for is sample count and composition, not
feature-space size. This control therefore varies only the sample-selection rule
(concordant vs random) at matched sample count and class balance.

Interpretation
--------------
If concordant training yields higher shared-cluster counts and/or higher
pathway concentration than the size- and class-matched random control, the
feature-stability doubling is specific to concordance rather than a generic
consequence of subsetting. If the random control matches the concordant value,
the doubling is a subsetting artefact and the recovery claim must be softened.

Run with::

    uv run python -m scripts.figure5.figure5b_random_control --n-control-seeds 10

This is compute-heavy (each control seed re-runs the full 20-seed stability
sweep over all phenotypes x splits). It does not retrain any published model and
writes only to ``data/outputs/figure5/figure5b_random_control/``.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scripts.figure5 import figure5b_data as _f5b
from scripts.figure5.figure5b_data import (
    _kos_to_clusters,
    compare_features,
    get_concordant_samples,
    get_consistent_features,
    get_screened_feature_names,
    get_screened_split_data,
    load_all_datasets_combined,
    load_experimental_phenotypes,
    load_gapmind_predictions,
    load_individual_dataset,
    train_and_get_top_features_individual,
    train_and_get_top_features_split,
)
from scripts.ml_splits import load_single_split_data


def _cell_rng(control_seed: int, key: str) -> np.random.Generator:
    """Deterministic, thread-safe per-cell RNG.

    Each (control_seed, cell) pair gets its own generator so cells can run in
    parallel without sharing a single mutable RNG. Equivalent in distribution to
    drawing one matched random subset per cell.
    """
    cell_hash = zlib.crc32(key.encode()) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([control_seed, cell_hash]))

# Match the concordant pipeline's eligibility gates so the control runs on the
# same (phenotype, split) cells as Figure 5B.
MIN_SUBSET = 20
MIN_PER_CLASS = 10
DATASETS = ("atleaf", "lit", "marine")
HELD_OUT_DATASETS = ("atleaf", "lit", "marine")


def class_matched_random_indices(
    y: pd.Series, n_pos: int, n_neg: int, rng: np.random.Generator
) -> list[str]:
    """Draw a random, class-balance-matched set of genome indices from ``y``.

    Parameters
    ----------
    y : pd.Series
        Binary labels for the full eligible pool, indexed by genome ID.
    n_pos : int
        Number of positive (growth) samples to draw.
    n_neg : int
        Number of negative (no-growth) samples to draw.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    list[str]
        Sampled genome indices (``min(n_pos, available)`` positives and
        ``min(n_neg, available)`` negatives, without replacement).
    """
    pos_idx = y.index[y == 1].to_numpy()
    neg_idx = y.index[y == 0].to_numpy()
    take_pos = min(n_pos, len(pos_idx))
    take_neg = min(n_neg, len(neg_idx))
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=take_pos, replace=False),
            rng.choice(neg_idx, size=take_neg, replace=False),
        ]
    )
    return chosen.tolist()


def _matched_counts(y: pd.Series, concordant_genomes: set[str]) -> tuple[int, int] | None:
    """Return ``(n_pos, n_neg)`` of the concordant subset of ``y``, or None if ineligible.

    Applies the same gates as the concordant Figure 5B pipeline (at least
    ``MIN_SUBSET`` concordant samples, both classes present, and at least
    ``MIN_PER_CLASS`` in the minority class).

    Parameters
    ----------
    y : pd.Series
        Labels for the data partition, indexed by genome ID.
    concordant_genomes : set[str]
        Concordant genome IDs for this phenotype.

    Returns
    -------
    tuple[int, int] | None
        Concordant per-class counts, or None if the cell would be skipped.
    """
    concordant_mask = y.index.isin(concordant_genomes)
    if concordant_mask.sum() < MIN_SUBSET:
        return None
    y_conc = y.loc[concordant_mask]
    if len(y_conc.unique()) != 2:
        return None
    counts = y_conc.value_counts()
    if counts.min() < MIN_PER_CLASS:
        return None
    return int((y_conc == 1).sum()), int((y_conc == 0).sum())


def _stability_on_subset(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    *,
    is_split: bool,
    seed: int,
    n_seeds: int,
    threshold: float,
    n_features: int,
    n_candidate_features: int,
) -> list[str] | None:
    """Run the Figure 5B seeded SHAP stability sweep on a single subset.

    Parameters
    ----------
    X_sub, y_sub : pd.DataFrame, pd.Series
        Matched random subset.
    is_split : bool
        If True, use the train/val re-split path (``train_and_get_top_features_split``);
        if False, use the single-matrix path (``train_and_get_top_features_individual``).
    seed : int
        Control-replicate seed, used to vary the 80/20 re-split.
    n_seeds : int
        Number of SHAP stability seeds.
    threshold : float
        Recurrence threshold for "stable" features.
    n_features : int
        Top features per run.
    n_candidate_features : int
        CatBoost candidate-screen size.

    Returns
    -------
    list[str] | None
        Stable feature list, or None if the subset is degenerate.
    """
    if len(y_sub.unique()) != 2 or y_sub.value_counts().min() < 2:
        return None

    if is_split:
        # Mirror the concordant combined path: 80/20 train/val re-split, screen,
        # then seeded stability.
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub,
            y_sub,
            train_size=0.8,
            stratify=y_sub,
            random_state=seed,
            shuffle=True,
        )
        screened = get_screened_split_data(
            {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val},
            n_candidate_features=n_candidate_features,
        )
        feature_lists = [
            train_and_get_top_features_split(screened, random_state=s, n_features=n_features)
            for s in range(n_seeds)
        ]
    else:
        candidate_features = get_screened_feature_names(
            X_sub, y_sub, n_candidate_features=n_candidate_features
        )
        X_screened = X_sub.loc[:, candidate_features]
        feature_lists = [
            train_and_get_top_features_individual(
                X_screened, y_sub, random_state=s, n_features=n_features
            )
            for s in range(n_seeds)
        ]

    if not feature_lists:
        return None
    return get_consistent_features(feature_lists, threshold=threshold)


def analyze_combined_splits_random(
    splits_dir: Path,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    control_seed: int,
    *,
    n_seeds: int,
    threshold: float,
    n_features: int,
    n_candidate_features: int,
    n_jobs: int = 1,
) -> dict[str, list[str]]:
    """Combined-of-three stable features on size/class-matched random subsets.

    Mirrors ``figure5b_data.analyze_combined_splits`` but, for each dataset_split
    cell, replaces the concordant subset with a random subset matched to the
    concordant subset's size and class balance. Cells are independent and run in
    parallel (threading backend) when ``n_jobs > 1``.
    """
    dataset_split_dir = splits_dir / "dataset_split"
    phenotypes = [d.name for d in dataset_split_dir.iterdir() if d.is_dir()]

    feature_file = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
    feature_data = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )

    cells = [
        (phenotype, split_type)
        for phenotype in phenotypes
        for split_type in [
            d.name for d in (dataset_split_dir / phenotype).iterdir() if d.is_dir()
        ]
    ]

    def _process(phenotype: str, split_type: str) -> tuple[str, list[str] | None]:
        key = f"{phenotype}_{split_type}"
        try:
            split_data = load_single_split_data(
                dataset_split_dir / phenotype / split_type, feature_data
            )
        except Exception as exc:  # noqa: BLE001 - mirror figure5b tolerance
            print(f"Error loading {key}: {exc}")
            return key, None

        X_pool = pd.concat([split_data["X_train"], split_data["X_val"]], axis=0)
        y_pool = pd.concat([split_data["y_train"], split_data["y_val"]], axis=0)

        concordant = get_concordant_samples(
            gapmind_predictions, experimental_phenotypes, phenotype
        )
        matched = _matched_counts(y_pool, concordant)
        if matched is None:
            return key, None
        n_pos, n_neg = matched

        sel = class_matched_random_indices(
            y_pool, n_pos, n_neg, _cell_rng(control_seed, key)
        )
        stable = _stability_on_subset(
            X_pool.loc[sel],
            y_pool.loc[sel],
            is_split=True,
            seed=control_seed,
            n_seeds=n_seeds,
            threshold=threshold,
            n_features=n_features,
            n_candidate_features=n_candidate_features,
        )
        return key, stable

    pairs = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process)(ph, st)
        for ph, st in tqdm(
            cells, desc=f"[seed {control_seed}] combined (random)", leave=False
        )
    )
    return {key: stable for key, stable in pairs if stable is not None}


def analyze_individual_datasets_random(
    phenotypes: list[str],
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    control_seed: int,
    *,
    n_seeds: int,
    threshold: float,
    n_features: int,
    n_candidate_features: int,
    n_jobs: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """Held-out-alone stable features on size/class-matched random subsets.

    Mirrors ``figure5b_data.analyze_individual_datasets`` with random matched
    subsets in place of concordant ones; (dataset, phenotype) cells run in
    parallel (threading backend) when ``n_jobs > 1``.
    """
    cells = [(dataset, phenotype) for dataset in DATASETS for phenotype in phenotypes]

    def _process(dataset: str, phenotype: str) -> tuple[str, str, list[str] | None]:
        key = f"{dataset}_{phenotype}"
        try:
            X, y = load_individual_dataset(dataset, phenotype)
        except Exception as exc:  # noqa: BLE001 - mirror figure5b tolerance
            print(f"Error loading {dataset}/{phenotype}: {exc}")
            return dataset, phenotype, None

        concordant = get_concordant_samples(
            gapmind_predictions, experimental_phenotypes, phenotype
        )
        matched = _matched_counts(y, concordant)
        if matched is None:
            return dataset, phenotype, None
        n_pos, n_neg = matched

        sel = class_matched_random_indices(
            y, n_pos, n_neg, _cell_rng(control_seed, key)
        )
        stable = _stability_on_subset(
            X.loc[sel],
            y.loc[sel],
            is_split=False,
            seed=control_seed,
            n_seeds=n_seeds,
            threshold=threshold,
            n_features=n_features,
            n_candidate_features=n_candidate_features,
        )
        return dataset, phenotype, stable

    triples = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process)(ds, ph)
        for ds, ph in tqdm(
            cells, desc=f"[seed {control_seed}] individual (random)", leave=False
        )
    )
    results: dict[str, dict[str, list[str]]] = {ds: {} for ds in DATASETS}
    for dataset, phenotype, stable in triples:
        if stable is not None:
            results[dataset][phenotype] = stable
    return results


def summarise_control(
    summary_df: pd.DataFrame,
) -> dict[str, float]:
    """Reduce one control replicate's comparison table to headline statistics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compare_features`` for one control replicate (cluster-level
        columns required).

    Returns
    -------
    dict[str, float]
        ``mean_shared_clusters`` (per-comparison mean, comparable to the
        manuscript's 0.5/1.3) and ``n_phenotypes_sharing`` (phenotypes with at
        least one shared cluster, summed over held-out comparisons, comparable
        to the manuscript's 6/12).
    """
    if summary_df.empty or "n_intersection_clusters" not in summary_df.columns:
        return {"mean_shared_clusters": float("nan"), "n_phenotypes_sharing": 0.0}
    per_phenotype = summary_df.groupby("phenotype")["n_intersection_clusters"].sum()
    return {
        "mean_shared_clusters": float(summary_df["n_intersection_clusters"].mean()),
        "n_phenotypes_sharing": float((per_phenotype > 0).sum()),
    }


def reaggregate_clusters(
    output_dir: Path,
    ko_clusters_by_phenotype: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Recompute the cluster-level columns from the retained KO lists, no refitting.

    The per-seed comparison CSVs store the KO membership of every comparison as
    ``intersection``, ``unique_to_individual`` and ``unique_to_combined``, so the
    combined and individual KO sets are recoverable exactly
    (``combined = intersection | unique_to_combined``, and likewise for
    individual). Only the cluster columns depend on
    ``ko_clusters_shap_hclust.json``, so a change to the cluster map can be
    propagated by re-deriving those five columns instead of repeating the
    multi-hour SHAP refit.

    Parameters
    ----------
    output_dir : Path
        Directory holding ``random_control_comparison_seed*.csv``.
    ko_clusters_by_phenotype : dict[str, dict[str, int]]
        Current per-phenotype KO-to-cluster mapping.

    Returns
    -------
    pd.DataFrame
        Rebuilt per-seed summary (the contents of ``random_control_summary.csv``).

    Raises
    ------
    FileNotFoundError
        If no per-seed comparison CSVs are present.
    ValueError
        If a row's KO lists do not reproduce its stored KO-level counts, which
        would mean the lists are not a faithful record of the fitted sets.
    """
    paths = sorted(
        output_dir.glob("random_control_comparison_seed*.csv"),
        key=lambda p: int(p.stem.rsplit("seed", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError(
            f"no per-seed comparison CSVs in {output_dir}; run the full control first"
        )

    def kos(cell: object) -> list[str]:
        if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "":
            return []
        return str(cell).split(";")

    per_seed: list[dict[str, float]] = []
    for path in paths:
        seed = int(path.stem.rsplit("seed", 1)[1])
        df = pd.read_csv(path)
        for position, row in df.iterrows():
            intersection = kos(row["intersection"])
            combined = intersection + kos(row["unique_to_combined"])
            individual = intersection + kos(row["unique_to_individual"])
            if (
                len(combined) != int(row["n_combined_features"])
                or len(individual) != int(row["n_individual_features"])
            ):
                raise ValueError(
                    f"{path.name}: KO lists for {row['comparison']} do not "
                    "reproduce the stored feature counts; reaggregation would "
                    "not be faithful"
                )
            ko_to_cluster = ko_clusters_by_phenotype.get(row["phenotype"])
            combined_clusters = _kos_to_clusters(combined, ko_to_cluster)
            individual_clusters = _kos_to_clusters(individual, ko_to_cluster)
            df.loc[position, "n_intersection_clusters"] = len(
                combined_clusters & individual_clusters
            )
            df.loc[position, "n_unique_to_individual_clusters"] = len(
                individual_clusters - combined_clusters
            )
            df.loc[position, "n_unique_to_combined_clusters"] = len(
                combined_clusters - individual_clusters
            )
            df.loc[position, "n_combined_clusters"] = len(combined_clusters)
            df.loc[position, "n_individual_clusters"] = len(individual_clusters)
        df.to_csv(path, index=False)
        stats = summarise_control(df)
        stats["control_seed"] = seed
        per_seed.append(stats)
        print(
            f"  seed {seed}: mean shared clusters = "
            f"{stats['mean_shared_clusters']:.3f}; phenotypes sharing >=1 = "
            f"{stats['n_phenotypes_sharing']:.0f}"
        )

    summary = pd.DataFrame(per_seed)
    summary.to_csv(output_dir / "random_control_summary.csv", index=False)
    return summary


def main() -> None:
    """Run the size/class-matched random-subset control and write a summary."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--reaggregate-only",
        action="store_true",
        help="Skip all model fitting: recompute only the cluster-level columns "
        "of the existing per-seed CSVs against the current cluster map, then "
        "rebuild the summary. Use after regenerating "
        "ko_clusters_shap_hclust.json when the KO-level results are current.",
    )
    parser.add_argument(
        "--n-control-seeds",
        type=int,
        default=10,
        help="Number of independent random-subset replicates (default: 10).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="SHAP stability seeds per subset, matching Figure 5B (default: 20).",
    )
    parser.add_argument(
        "--n-candidate-features",
        type=int,
        default=300,
        help="CatBoost candidate-screen size, matching Figure 5B (default: 300).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Cells to run in parallel (threading backend). 1 = sequential "
        "(default). On small subsets, parallel cells with a low --thread-count "
        "is far faster than one all-core fit at a time.",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=-1,
        help="CatBoost/SHAP threads per fit (default -1 = all cores). Set low "
        "(1-2) when --n-jobs > 1 to avoid oversubscribing the machine.",
    )
    args = parser.parse_args()

    # Cap per-fit threads so many small-subset cells can run concurrently without
    # oversubscription (default -1 leaves Figure 5B's own behaviour unchanged).
    _f5b._THREAD_COUNT = args.thread_count

    splits_dir = Path("data/processed/train_test_splits")
    output_dir = Path("data/outputs/figure5/figure5b_random_control")
    output_dir.mkdir(parents=True, exist_ok=True)

    gapmind_file = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
    phenotype_dir = Path("data/processed/phenotypes")
    cluster_file = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")

    threshold = 0.7
    n_features = 10

    print("Loading GapMind predictions (loose) and experimental phenotypes...")
    gapmind_predictions = load_gapmind_predictions(gapmind_file)
    experimental_phenotypes = load_experimental_phenotypes(phenotype_dir)

    common_phenotypes = sorted(
        d.name for d in (splits_dir / "dataset_split").iterdir() if d.is_dir()
    )

    ko_clusters_by_phenotype: dict[str, dict[str, int]] | None = None
    if cluster_file.exists():
        with open(cluster_file) as handle:
            ko_clusters_by_phenotype = json.load(handle)
    else:
        print(f"WARNING: cluster mapping missing at {cluster_file}; cluster counts skipped.")

    if args.reaggregate_only:
        if ko_clusters_by_phenotype is None:
            raise FileNotFoundError(
                f"--reaggregate-only needs the cluster mapping at {cluster_file}"
            )
        print("Reaggregating cluster columns from retained KO lists (no fitting)...")
        summary = reaggregate_clusters(output_dir, ko_clusters_by_phenotype)
        _report_control(summary, output_dir)
        return

    per_seed_summaries: list[dict[str, float]] = []
    for control_seed in range(args.n_control_seeds):
        print(f"\n=== Control replicate {control_seed + 1}/{args.n_control_seeds} ===")

        combined = analyze_combined_splits_random(
            splits_dir,
            gapmind_predictions,
            experimental_phenotypes,
            control_seed,
            n_seeds=args.n_seeds,
            threshold=threshold,
            n_features=n_features,
            n_candidate_features=args.n_candidate_features,
            n_jobs=args.n_jobs,
        )
        individual = analyze_individual_datasets_random(
            common_phenotypes,
            gapmind_predictions,
            experimental_phenotypes,
            control_seed,
            n_seeds=args.n_seeds,
            threshold=threshold,
            n_features=n_features,
            n_candidate_features=args.n_candidate_features,
            n_jobs=args.n_jobs,
        )

        summary_df = compare_features(
            combined, individual, ko_clusters_by_phenotype=ko_clusters_by_phenotype
        )
        summary_df.to_csv(
            output_dir / f"random_control_comparison_seed{control_seed}.csv",
            index=False,
        )
        stats = summarise_control(summary_df)
        stats["control_seed"] = control_seed
        per_seed_summaries.append(stats)
        print(
            f"  mean shared clusters = {stats['mean_shared_clusters']:.3f}; "
            f"phenotypes sharing >=1 cluster = {stats['n_phenotypes_sharing']:.0f}"
        )

    summary = pd.DataFrame(per_seed_summaries)
    summary.to_csv(output_dir / "random_control_summary.csv", index=False)
    _report_control(summary, output_dir)


def _report_control(summary: pd.DataFrame, output_dir: Path) -> None:
    """Print the control-vs-baseline comparison for a per-seed summary table."""
    print("\n" + "=" * 70)
    print("Random-subset control vs concordant baseline (Figure 5B / Table 1)")
    print("=" * 70)
    print(
        "Random control (mean +/- sd over replicates):\n"
        f"  mean shared clusters   = {summary['mean_shared_clusters'].mean():.3f}"
        f" +/- {summary['mean_shared_clusters'].std():.3f}"
        "   (concordant baseline: 1.3; full-data: 0.5)\n"
        f"  phenotypes sharing >=1 = {summary['n_phenotypes_sharing'].mean():.1f}"
        f" +/- {summary['n_phenotypes_sharing'].std():.1f}"
        "   (concordant baseline: 12; full-data: 6)"
    )
    print(
        "\nInterpretation: if the random-control numbers sit near the full-data "
        "baseline (0.5 / 6) and below the concordant value (1.3 / 12), the "
        "feature-stability gain is specific to concordance, not to subsetting."
    )
    print(f"\nWrote summary to {output_dir / 'random_control_summary.csv'}")


if __name__ == "__main__":
    main()
