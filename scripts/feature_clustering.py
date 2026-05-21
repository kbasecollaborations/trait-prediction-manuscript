#!/usr/bin/env python3
"""Cluster KOFAM features into target-aware redundancy groups for SHAP comparison.

Pre-computes a per-phenotype mapping ``KO -> cluster_id`` used downstream by
``scripts/figure4/figure4c_data.py`` and ``scripts/figure5/figure5b_data.py``.
The cluster space is fixed per phenotype so that Figure 4 (full data) and
Figure 5 (concordant data) compare on identical cluster identities.

Main method (Lundberg-style supervised redundancy):
    Pairwise distance is ``1 - R^2`` from univariate XGBoost models predicting
    the phenotype, as implemented in ``shap.utils.hclust``. Two KOs cluster
    if they share more than ``1 - cutoff`` of their univariate explanatory
    power with respect to the phenotype.

Supplementary methods:
    Jaccard distance with average linkage (UPGMA), the pan-genome convention
    for binary presence/absence data (Snipen & Liland 2010).

Notes
-----
- KOFAM features are 0/1 (presence/absence).
- The pooled concordant feature matrix is used for clustering; this is the
  matrix the manuscript is reasoning about under the applicability-domain
  framing.
- Clustering scope per phenotype is the union of every KO that appears in any
  stable feature list across Fig 4 and Fig 5 outputs, plus the union of those
  with non-zero variance in the pooled matrix.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import shap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")


PHENOTYPES_DEFAULT: tuple[str, ...] = (
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
)

DATASETS_DEFAULT: tuple[str, ...] = ("atleaf", "lit", "marine", "pmi")


def load_gapmind_predictions(gapmind_file: Path) -> pd.DataFrame:
    """Load GapMind phenotype predictions.

    Parameters
    ----------
    gapmind_file : Path
        Path to the GapMind predictions TSV file with ``genomeID`` index.

    Returns
    -------
    pd.DataFrame
        GapMind predictions, indexed by genome ID.
    """
    return pd.read_csv(
        gapmind_file, sep="\t", index_col=0, dtype={"genomeID": str}
    )


def load_pooled_matrix(
    phenotype: str,
    datasets: Sequence[str],
    gapmind_predictions: pd.DataFrame,
    concordant_only: bool,
    features_dir: Path = Path("data/processed/features_reduced"),
    phenotypes_dir: Path = Path("data/processed/phenotypes"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Pool KOFAM features and phenotype labels across datasets.

    Parameters
    ----------
    phenotype : str
        Phenotype name (e.g., ``"Histidine"``).
    datasets : Sequence[str]
        Dataset directory names to pool over.
    gapmind_predictions : pd.DataFrame
        Loaded GapMind predictions used to filter to concordant samples.
    concordant_only : bool
        If True, restrict to samples where the GapMind prediction equals the
        experimental phenotype value.
    features_dir : Path, optional
        Directory containing per-dataset ``kofam.tsv`` files.
    phenotypes_dir : Path, optional
        Directory containing per-dataset ``{phenotype}.tsv`` files.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix (genomes by KOFAM presence/absence) and aligned label
        series for the phenotype.
    """
    feature_frames: list[pd.DataFrame] = []
    label_series: list[pd.Series] = []

    for dataset in datasets:
        feature_path = features_dir / dataset / "kofam.tsv"
        phenotype_path = phenotypes_dir / dataset / f"{phenotype}.tsv"
        if not (feature_path.exists() and phenotype_path.exists()):
            continue
        features = pd.read_csv(
            feature_path, sep="\t", index_col=0, dtype={"genomeID": str}
        )
        labels = (
            pd.read_csv(phenotype_path, sep="\t", dtype={"genomeID": str})
            .set_index("genomeID")[phenotype]
            .dropna()
        )
        common = (
            features.index.intersection(labels.index).intersection(
                gapmind_predictions.index
            )
        )
        labels = labels.loc[common]
        features = features.loc[common]
        if concordant_only:
            keep = gapmind_predictions.loc[common, phenotype] == labels
            keep_idx = keep[keep].index
            features = features.loc[keep_idx]
            labels = labels.loc[keep_idx]
        feature_frames.append(features)
        label_series.append(labels)

    X = pd.concat(feature_frames, axis=0)
    X = X[~X.index.duplicated(keep="first")]
    y = pd.concat(label_series, axis=0)
    y = y[~y.index.duplicated(keep="first")]
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


def collect_stable_ko_union(phenotype: str) -> set[str]:
    """Collect the union of stable KOs ever picked for a phenotype.

    Pools stable feature lists across both Figure 4 (full data) and Figure 5
    (concordant data), and across both combined-split and individual-dataset
    analyses, so the cluster space is fixed for downstream comparisons.

    Parameters
    ----------
    phenotype : str
        Phenotype name.

    Returns
    -------
    set[str]
        Union of KO identifiers appearing in any stable feature list.
    """
    union: set[str] = set()
    sources = [
        Path("data/outputs/figure4/combined_splits_shap_features.json"),
        Path("data/outputs/figure5/figure5b_combined_splits_shap_features.json"),
    ]
    for src in sources:
        if not src.exists():
            continue
        with src.open() as handle:
            combined = json.load(handle)
        for key, features in combined.items():
            if key.startswith(f"{phenotype}_"):
                union.update(features)

    ind_sources = [
        Path("data/outputs/figure4/individual_datasets_shap_features.json"),
        Path("data/outputs/figure5/figure5b_individual_datasets_shap_features.json"),
    ]
    for src in ind_sources:
        if not src.exists():
            continue
        with src.open() as handle:
            individual = json.load(handle)
        for dataset_features in individual.values():
            union.update(dataset_features.get(phenotype, []))
    return union


def cluster_shap_hclust(
    X: pd.DataFrame,
    y: pd.Series,
    ko_union: set[str],
    cutoff: float = 0.5,
) -> dict[str, int]:
    """Cluster features using SHAP's supervised redundancy linkage.

    Pairwise distance is ``1 - R^2`` from univariate XGBoost models
    (``shap.utils.hclust``), so features with similar predictive power
    w.r.t. ``y`` are placed close together.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples by features).
    y : pd.Series
        Binary phenotype labels aligned to ``X.index``.
    ko_union : set[str]
        Candidate KO identifiers to cluster over.
    cutoff : float, optional
        Distance threshold for ``fcluster`` (default 0.5).

    Returns
    -------
    dict[str, int]
        Mapping from KO identifier to integer cluster ID.
    """
    kos = sorted(set(ko_union) & set(X.columns))
    X_sub = X[kos].astype(float)
    non_constant = list(X_sub.columns[X_sub.var() > 0])
    X_sub = X_sub[non_constant]
    if len(non_constant) < 2:
        return {ko: idx for idx, ko in enumerate(non_constant)}
    Z = shap.utils.hclust(X_sub.values, y.values.astype(int))
    clusters = fcluster(Z, t=cutoff, criterion="distance")
    return dict(zip(non_constant, clusters))


def cluster_jaccard(
    X: pd.DataFrame,
    ko_union: set[str],
    cutoff: float = 0.3,
) -> dict[str, int]:
    """Cluster binary features using Jaccard distance and average linkage.

    Used as a domain-standard robustness check (pan-genome convention).

    Parameters
    ----------
    X : pd.DataFrame
        Binary feature matrix.
    ko_union : set[str]
        Candidate KO identifiers.
    cutoff : float, optional
        Distance threshold for ``fcluster`` (default 0.3, i.e., Jaccard >= 0.7).

    Returns
    -------
    dict[str, int]
        Mapping from KO identifier to integer cluster ID.
    """
    kos = sorted(set(ko_union) & set(X.columns))
    X_sub = X[kos].astype(int)
    non_constant = list(X_sub.columns[X_sub.var() > 0])
    X_sub = X_sub[non_constant]
    if len(non_constant) < 2:
        return {ko: idx for idx, ko in enumerate(non_constant)}
    distances = pdist(X_sub.T.values, metric="jaccard")
    Z = linkage(distances, method="average")
    clusters = fcluster(Z, t=cutoff, criterion="distance")
    return dict(zip(non_constant, clusters))


Method = Literal["shap_hclust", "jaccard"]


def build_phenotype_clusters(
    method: Method,
    phenotypes: Sequence[str],
    datasets: Sequence[str],
    gapmind_file: Path,
    cutoff: float,
    concordant_only: bool,
) -> dict[str, dict[str, int]]:
    """Build per-phenotype KO cluster mappings.

    Parameters
    ----------
    method : Literal["shap_hclust", "jaccard"]
        Clustering method.
    phenotypes : Sequence[str]
        Phenotypes to process.
    datasets : Sequence[str]
        Datasets to pool features over.
    gapmind_file : Path
        GapMind predictions TSV.
    cutoff : float
        Distance threshold passed to ``fcluster``.
    concordant_only : bool
        Whether to restrict the pooled matrix to concordant samples.

    Returns
    -------
    dict[str, dict[str, int]]
        ``{phenotype: {ko: cluster_id}}``.
    """
    gapmind = load_gapmind_predictions(gapmind_file)
    results: dict[str, dict[str, int]] = {}
    for phenotype in phenotypes:
        ko_union = collect_stable_ko_union(phenotype)
        if not ko_union:
            print(f"  [skip] {phenotype}: no stable KOs found")
            continue
        X, y = load_pooled_matrix(
            phenotype=phenotype,
            datasets=datasets,
            gapmind_predictions=gapmind,
            concordant_only=concordant_only,
        )
        if method == "shap_hclust":
            mapping = cluster_shap_hclust(X, y, ko_union, cutoff=cutoff)
        elif method == "jaccard":
            mapping = cluster_jaccard(X, ko_union, cutoff=cutoff)
        else:
            raise ValueError(f"Unknown method: {method}")
        mapping_str = {ko: int(cid) for ko, cid in mapping.items()}
        n_clusters = len(set(mapping_str.values()))
        n_kos = len(mapping_str)
        print(
            f"  {phenotype}: {n_kos} KOs -> {n_clusters} clusters "
            f"(reduction {n_kos - n_clusters})"
        )
        results[phenotype] = mapping_str
    return results


def main() -> None:
    """Run clustering across phenotypes and write JSON outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/clustering"),
        help="Directory to write cluster JSONs.",
    )
    parser.add_argument(
        "--gapmind",
        type=Path,
        default=Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv"),
    )
    parser.add_argument(
        "--cutoff-shap",
        type=float,
        default=0.5,
        help="Distance cutoff for shap_hclust (default 0.5).",
    )
    parser.add_argument(
        "--cutoff-jaccard",
        type=float,
        default=0.3,
        help="Distance cutoff for Jaccard linkage (default 0.3 -> J>=0.7).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["shap_hclust", "jaccard"],
        default=["shap_hclust", "jaccard"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, float]] = []
    for method in args.methods:
        cutoff = args.cutoff_shap if method == "shap_hclust" else args.cutoff_jaccard
        configs.append((method, cutoff))

    for method, cutoff in configs:
        print(f"\n=== Clustering with {method} (cutoff={cutoff}) ===")
        results = build_phenotype_clusters(
            method=method,  # type: ignore[arg-type]
            phenotypes=PHENOTYPES_DEFAULT,
            datasets=DATASETS_DEFAULT,
            gapmind_file=args.gapmind,
            cutoff=cutoff,
            concordant_only=True,
        )
        out_path = args.output_dir / f"ko_clusters_{method}.json"
        with out_path.open("w") as handle:
            json.dump(results, handle, indent=2)
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
