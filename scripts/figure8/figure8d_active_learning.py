#!/usr/bin/env python3
"""
Pilot retrospective active-learning analysis for experimental target selection.

This script simulates selecting a small batch of held-out genome--phenotype
experiments, adding those labels to the training set, and measuring whether the
selected batch improves cross-dataset performance more than random selection.
It is intentionally small and exploratory so it can be run quickly before a full
analysis is designed.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    get_concordant_and_discordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data
from trait_prediction.pipeline import align_columns


FEATURE_FILE: Path = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
SPLITS_DIR: Path = Path("data/processed/train_test_splits/dataset_split")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
OUTPUT_DIR: Path = Path("data/outputs/active_learning_pilot")
HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the short active-learning pilot."""

    phenotypes: tuple[str, ...]
    budget: int
    seeds: tuple[int, ...]
    max_splits_per_phenotype: int
    candidate_fraction: float
    iterations: int
    min_eval_samples: int
    max_diversity_pool_multiplier: int


def parse_args() -> PilotConfig:
    """Parse command-line arguments.

    Returns
    -------
    PilotConfig
        Parsed analysis configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=["Glucose", "Histidine"],
        help="Phenotypes to include in the pilot.",
    )
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-splits-per-phenotype", type=int, default=2)
    parser.add_argument("--candidate-fraction", type=float, default=0.50)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--min-eval-samples", type=int, default=20)
    parser.add_argument("--max-diversity-pool-multiplier", type=int, default=4)
    args = parser.parse_args()
    return PilotConfig(
        phenotypes=tuple(args.phenotypes),
        budget=args.budget,
        seeds=tuple(args.seeds),
        max_splits_per_phenotype=args.max_splits_per_phenotype,
        candidate_fraction=args.candidate_fraction,
        iterations=args.iterations,
        min_eval_samples=args.min_eval_samples,
        max_diversity_pool_multiplier=args.max_diversity_pool_multiplier,
    )


def safe_balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy, returning NaN if both classes are not present.

    Parameters
    ----------
    y_true
        True binary labels.
    y_pred
        Predicted binary labels.

    Returns
    -------
    float
        Balanced accuracy or NaN when the evaluation set is single-class.
    """
    if y_true.nunique() < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def held_out_dataset(split_dir: Path) -> str:
    """Extract the held-out dataset name from a split directory.

    Parameters
    ----------
    split_dir
        Dataset-split directory.

    Returns
    -------
    str
        Held-out dataset name.

    Raises
    ------
    ValueError
        If the split directory name does not contain a held-out dataset.
    """
    match = HELD_OUT_RE.search(split_dir.name)
    if match is None:
        raise ValueError(f"Could not parse held-out dataset from {split_dir}")
    return match.group(1)


def split_candidate_and_eval(
    y_test: pd.Series,
    candidate_fraction: float,
    seed: int,
    min_eval_samples: int,
    budget: int,
) -> tuple[pd.Index, pd.Index] | None:
    """Split held-out test genomes into candidate and evaluation pools.

    Parameters
    ----------
    y_test
        Held-out test labels.
    candidate_fraction
        Fraction assigned to the candidate pool.
    seed
        Random seed.
    min_eval_samples
        Minimum required evaluation-pool size.
    budget
        Number of candidates to select for simulated phenotyping.

    Returns
    -------
    tuple[pd.Index, pd.Index] | None
        Candidate and evaluation indices, or None if the split is too small.
    """
    if len(y_test) < budget + min_eval_samples:
        return None
    stratify = y_test if y_test.nunique() == 2 and y_test.value_counts().min() >= 2 else None
    candidate_idx, eval_idx = train_test_split(
        y_test.index,
        train_size=candidate_fraction,
        random_state=seed,
        stratify=stratify,
    )
    candidate_idx = pd.Index(candidate_idx)
    eval_idx = pd.Index(eval_idx)
    if len(candidate_idx) < budget or len(eval_idx) < min_eval_samples:
        return None
    if y_test.loc[eval_idx].nunique() < 2:
        return None
    return candidate_idx, eval_idx


def fit_predict_proba(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    iterations: int,
) -> np.ndarray | None:
    """Fit a short CatBoost model and return P(growth) for test rows.

    Parameters
    ----------
    X_train
        Training feature matrix.
    y_train
        Training labels.
    X_test
        Test feature matrix.
    seed
        Random seed.
    iterations
        CatBoost iteration cap for the pilot.

    Returns
    -------
    np.ndarray | None
        Predicted probability for class 1, or None if training is single-class.
    """
    if y_train.nunique() < 2:
        return None
    model = make_classifier(
        "cb_noeval",
        random_state=seed,
        iterations=iterations,
        depth=4,
        learning_rate=0.05,
    )
    model.fit(X_train, y_train, verbose=False)
    X_test_aligned = align_columns(X_train, X_test)
    return np.asarray(model.predict_proba(X_test_aligned))[:, 1]


def rank_by_strategy(
    strategy: str,
    candidate_scores: pd.DataFrame,
    X_candidate: pd.DataFrame,
    budget: int,
    seed: int,
    diversity_pool_multiplier: int,
) -> pd.Index:
    """Select candidate genomes according to an acquisition strategy.

    Parameters
    ----------
    strategy
        Acquisition strategy name.
    candidate_scores
        Candidate table containing confidence, uncertainty, and disagreement.
    X_candidate
        Candidate feature matrix.
    budget
        Number of genomes to select.
    seed
        Random seed for random acquisition.
    diversity_pool_multiplier
        Size multiplier for the diversity pre-pool.

    Returns
    -------
    pd.Index
        Selected candidate genome IDs.
    """
    if strategy == "random":
        return pd.Index(
            candidate_scores.sample(n=budget, random_state=seed).index
        )
    if strategy == "uncertainty":
        return pd.Index(
            candidate_scores.sort_values("uncertainty", ascending=False)
            .head(budget)
            .index
        )
    if strategy == "disagreement":
        return pd.Index(
            candidate_scores.sort_values(
                ["disagreement", "uncertainty"], ascending=[False, False]
            )
            .head(budget)
            .index
        )
    if strategy != "combined_diverse":
        raise ValueError(f"Unknown strategy: {strategy}")

    score = candidate_scores["uncertainty"] + 0.5 * candidate_scores["disagreement"]
    prepool_size = min(len(candidate_scores), budget * diversity_pool_multiplier)
    prepool = score.sort_values(ascending=False).head(prepool_size).index
    return greedy_diverse_selection(X_candidate.loc[prepool], score.loc[prepool], budget)


def greedy_diverse_selection(
    X_pool: pd.DataFrame,
    score: pd.Series,
    budget: int,
) -> pd.Index:
    """Greedily select high-scoring but feature-diverse candidates.

    Parameters
    ----------
    X_pool
        Candidate feature matrix.
    score
        Acquisition score indexed like ``X_pool``.
    budget
        Number of candidates to select.

    Returns
    -------
    pd.Index
        Selected genome IDs.
    """
    selected: list[str] = [str(score.sort_values(ascending=False).index[0])]
    remaining: list[str] = [str(idx) for idx in X_pool.index if idx not in selected]
    while remaining and len(selected) < budget:
        distances = pairwise_distances(
            X_pool.loc[remaining].to_numpy(),
            X_pool.loc[selected].to_numpy(),
            metric="hamming",
        )
        min_distance = pd.Series(distances.min(axis=1), index=remaining)
        normalized_score = score.loc[remaining]
        if normalized_score.max() > normalized_score.min():
            normalized_score = (normalized_score - normalized_score.min()) / (
                normalized_score.max() - normalized_score.min()
            )
        combined = normalized_score + min_distance
        next_id = str(combined.sort_values(ascending=False).index[0])
        selected.append(next_id)
        remaining.remove(next_id)
    return pd.Index(selected)


def iter_split_dirs(phenotype: str, max_splits: int) -> Iterable[Path]:
    """Yield split directories for one phenotype.

    Parameters
    ----------
    phenotype
        Phenotype name.
    max_splits
        Maximum number of split directories to yield.

    Yields
    ------
    Path
        Dataset split directory.
    """
    phenotype_dir = SPLITS_DIR / phenotype
    split_dirs = sorted(path for path in phenotype_dir.iterdir() if path.is_dir())
    yield from split_dirs[:max_splits]


def run_pilot(config: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the retrospective active-learning pilot.

    Parameters
    ----------
    config
        Pilot configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Detailed run table, selected-candidate table, and strategy summary table.
    """
    features = pd.read_csv(FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str})
    gapmind = load_gapmind_predictions(GAPMIND_FILE)
    phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    strategies = ("random", "uncertainty", "disagreement", "combined_diverse")
    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for phenotype in tqdm(config.phenotypes, desc="Phenotypes"):
        concordant, _ = get_concordant_and_discordant_samples(
            gapmind, phenotypes, phenotype
        )
        if not concordant:
            continue
        for split_dir in iter_split_dirs(phenotype, config.max_splits_per_phenotype):
            split = load_single_split_data(split_dir, features)
            held_out = held_out_dataset(split_dir)
            X_base = pd.concat([split["X_train"], split["X_val"]], axis=0)
            y_base = pd.concat([split["y_train"], split["y_val"]], axis=0)
            base_idx = sorted(set(X_base.index) & concordant)
            if len(base_idx) < 20:
                continue
            X_base = X_base.loc[base_idx]
            y_base = y_base.loc[base_idx]
            if y_base.nunique() < 2:
                continue

            for seed in config.seeds:
                split_indices = split_candidate_and_eval(
                    split["y_test"],
                    config.candidate_fraction,
                    seed,
                    config.min_eval_samples,
                    config.budget,
                )
                if split_indices is None:
                    continue
                candidate_idx, eval_idx = split_indices
                X_candidate = split["X_test"].loc[candidate_idx]
                y_candidate = split["y_test"].loc[candidate_idx]
                X_eval = split["X_test"].loc[eval_idx]
                y_eval = split["y_test"].loc[eval_idx]

                candidate_proba = fit_predict_proba(
                    X_base, y_base, X_candidate, seed, config.iterations
                )
                eval_proba = fit_predict_proba(
                    X_base, y_base, X_eval, seed, config.iterations
                )
                if candidate_proba is None or eval_proba is None:
                    continue
                initial_eval_pred = (eval_proba >= 0.5).astype(int)
                initial_ba = safe_balanced_accuracy(y_eval, initial_eval_pred)
                if np.isnan(initial_ba):
                    continue

                candidate_scores = pd.DataFrame(
                    {
                        "proba": candidate_proba,
                        "confidence": np.maximum(candidate_proba, 1 - candidate_proba),
                    },
                    index=candidate_idx,
                )
                candidate_scores["uncertainty"] = 1.0 - candidate_scores["confidence"]
                candidate_scores["y_pred"] = (candidate_scores["proba"] >= 0.5).astype(
                    int
                )
                gm_values = gapmind[phenotype].reindex(candidate_scores.index)
                candidate_scores["gapmind_pred"] = gm_values
                candidate_scores["disagreement"] = (
                    gm_values.notna()
                    & (gm_values.astype("float") != candidate_scores["y_pred"])
                ).astype(int)

                for strategy in strategies:
                    selected = rank_by_strategy(
                        strategy,
                        candidate_scores,
                        X_candidate,
                        config.budget,
                        seed,
                        config.max_diversity_pool_multiplier,
                    )
                    X_aug = pd.concat([X_base, X_candidate.loc[selected]], axis=0)
                    y_aug = pd.concat([y_base, y_candidate.loc[selected]], axis=0)
                    for rank, genome in enumerate(selected, start=1):
                        scores = candidate_scores.loc[genome]
                        selected_rows.append(
                            {
                                "phenotype": phenotype,
                                "held_out_dataset": held_out,
                                "seed": seed,
                                "strategy": strategy,
                                "rank": rank,
                                "genome": genome,
                                "true_label": int(y_candidate.loc[genome]),
                                "initial_ml_prediction": int(scores["y_pred"]),
                                "initial_proba": float(scores["proba"]),
                                "initial_confidence": float(scores["confidence"]),
                                "initial_uncertainty": float(scores["uncertainty"]),
                                "gapmind_prediction": scores["gapmind_pred"],
                                "gapmind_ml_disagree": int(scores["disagreement"]),
                            }
                        )
                    updated_proba = fit_predict_proba(
                        X_aug, y_aug, X_eval, seed, config.iterations
                    )
                    if updated_proba is None:
                        continue
                    updated_pred = (updated_proba >= 0.5).astype(int)
                    updated_ba = safe_balanced_accuracy(y_eval, updated_pred)
                    rows.append(
                        {
                            "phenotype": phenotype,
                            "held_out_dataset": held_out,
                            "seed": seed,
                            "strategy": strategy,
                            "budget": config.budget,
                            "n_base_train": len(X_base),
                            "n_candidate_pool": len(X_candidate),
                            "n_eval": len(X_eval),
                            "selected_disagreement_fraction": float(
                                candidate_scores.loc[selected, "disagreement"].mean()
                            ),
                            "selected_mean_uncertainty": float(
                                candidate_scores.loc[selected, "uncertainty"].mean()
                            ),
                            "initial_balanced_accuracy": initial_ba,
                            "updated_balanced_accuracy": updated_ba,
                            "delta_balanced_accuracy": updated_ba - initial_ba,
                        }
                    )

    detailed = pd.DataFrame(rows)
    selected_detail = pd.DataFrame(selected_rows)
    if detailed.empty:
        return detailed, selected_detail, pd.DataFrame()
    summary = (
        detailed.groupby("strategy", as_index=False)
        .agg(
            n_runs=("delta_balanced_accuracy", "size"),
            mean_initial_ba=("initial_balanced_accuracy", "mean"),
            mean_updated_ba=("updated_balanced_accuracy", "mean"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            median_delta_ba=("delta_balanced_accuracy", "median"),
            mean_selected_disagreement_fraction=(
                "selected_disagreement_fraction",
                "mean",
            ),
            mean_selected_uncertainty=("selected_mean_uncertainty", "mean"),
        )
        .sort_values("mean_delta_ba", ascending=False)
    )
    return detailed, selected_detail, summary


def main() -> None:
    """Run the pilot and save detailed and summary outputs."""
    config = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed, selected_detail, summary = run_pilot(config)
    detailed_path = OUTPUT_DIR / "active_learning_pilot_detailed.tsv"
    selected_path = OUTPUT_DIR / "active_learning_pilot_selected_candidates.tsv"
    summary_path = OUTPUT_DIR / "active_learning_pilot_summary.tsv"
    detailed.to_csv(detailed_path, sep="\t", index=False)
    selected_detail.to_csv(selected_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Saved detailed results to {detailed_path}")
    print(f"Saved selected candidates to {selected_path}")
    print(f"Saved summary results to {summary_path}")
    if summary.empty:
        print("No valid pilot runs were produced.")
    else:
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
