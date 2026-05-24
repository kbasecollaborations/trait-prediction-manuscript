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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    get_concordant_and_discordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.figure6.applicability import mean_knn_jaccard_distance
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data
from trait_prediction.pipeline import align_columns


FEATURE_FILE: Path = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
SPLITS_DIR: Path = Path("data/processed/train_test_splits/dataset_split")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
OUTPUT_DIR: Path = Path("data/outputs/figure6")
HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")

# Label-free candidate-selection strategies compared in Figure 8 Panel C.
STRATEGIES: tuple[str, ...] = ("low_confidence", "high_ood", "diversity", "random")


def select_candidates(
    candidates: pd.DataFrame, k: int, strategy: str, rng: np.random.Generator
) -> list[str]:
    """
    Pick ``k`` candidate genome IDs to label under a label-free strategy.

    Parameters
    ----------
    candidates : pd.DataFrame
        Indexed by genome; must contain ``confidence`` and ``ood`` columns.
    k : int
        Number to select.
    strategy : str
        One of :data:`STRATEGIES`.
    rng : np.random.Generator
        Source of randomness (``random`` selection and ``diversity`` spread).

    Returns
    -------
    list[str]
        Selected genome IDs.

    Raises
    ------
    ValueError
        If ``strategy`` is not recognised.
    """
    k = min(k, len(candidates))
    if strategy == "low_confidence":
        return candidates.nsmallest(k, "confidence").index.tolist()
    if strategy == "high_ood":
        return candidates.nlargest(k, "ood").index.tolist()
    if strategy == "random":
        return rng.choice(candidates.index.to_numpy(), size=k, replace=False).tolist()
    if strategy == "diversity":
        order = candidates["ood"].sort_values().index.to_numpy()
        idx = np.linspace(0, len(order) - 1, k).round().astype(int)
        return order[idx].tolist()
    raise ValueError(f"unknown strategy: {strategy}")


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
        default=[
            "m-Inositol",
            "Histidine",
            "Glucose",
            "Cellobiose",
            "Mannose",
            "Maltose",
        ],
        help="Phenotypes to include in the pilot.",
    )
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-splits-per-phenotype", type=int, default=4)
    parser.add_argument("--candidate-fraction", type=float, default=0.50)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--min-eval-samples", type=int, default=20)
    args = parser.parse_args()
    return PilotConfig(
        phenotypes=tuple(args.phenotypes),
        budget=args.budget,
        seeds=tuple(args.seeds),
        max_splits_per_phenotype=args.max_splits_per_phenotype,
        candidate_fraction=args.candidate_fraction,
        iterations=args.iterations,
        min_eval_samples=args.min_eval_samples,
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

                # Label-free candidate scores: model confidence and feature-space
                # novelty (mean kNN Jaccard distance to the training set). Both are
                # computable without knowing the candidate's experimental outcome.
                confidence = np.maximum(candidate_proba, 1 - candidate_proba)
                ood = mean_knn_jaccard_distance(X_candidate, X_base, k=5)
                candidates = pd.DataFrame(
                    {
                        "confidence": pd.Series(confidence, index=candidate_idx),
                        "ood": ood.reindex(candidate_idx),
                    },
                    index=candidate_idx,
                )
                candidates["proba"] = pd.Series(candidate_proba, index=candidate_idx)
                candidates["y_pred"] = (candidates["proba"] >= 0.5).astype(int)

                for strategy in STRATEGIES:
                    rng = np.random.default_rng(seed)
                    selected = pd.Index(
                        select_candidates(candidates, config.budget, strategy, rng)
                    )
                    X_aug = pd.concat([X_base, X_candidate.loc[selected]], axis=0)
                    y_aug = pd.concat([y_base, y_candidate.loc[selected]], axis=0)
                    for rank, genome in enumerate(selected, start=1):
                        scores = candidates.loc[genome]
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
                                "ood": float(scores["ood"]),
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
                            "strategy": strategy,
                            "phenotype": phenotype,
                            "held_out_dataset": held_out,
                            "seed": seed,
                            "n_added": len(selected),
                            "delta_balanced_accuracy": updated_ba - initial_ba,
                            "n_base_train": len(X_base),
                            "n_candidate_pool": len(X_candidate),
                            "n_eval": len(X_eval),
                            "selected_mean_confidence": float(
                                candidates.loc[selected, "confidence"].mean()
                            ),
                            "selected_mean_ood": float(
                                candidates.loc[selected, "ood"].mean()
                            ),
                            "initial_balanced_accuracy": initial_ba,
                            "updated_balanced_accuracy": updated_ba,
                        }
                    )

    detailed = pd.DataFrame(rows)
    selected_detail = pd.DataFrame(selected_rows)
    if detailed.empty:
        return detailed, selected_detail, pd.DataFrame()
    final = detailed[detailed["n_added"] == detailed["n_added"].max()]
    by_phenotype = (
        final.groupby(["phenotype", "strategy"], as_index=False)
        .agg(
            n_added=("n_added", "max"),
            n_runs=("delta_balanced_accuracy", "size"),
            mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
        )
        .sort_values(["phenotype", "mean_delta_balanced_accuracy"], ascending=[True, False])
    )
    return detailed, selected_detail, by_phenotype


def main() -> None:
    """Run the prioritization simulation and save Figure 8 Panel C tables."""
    config = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed, _selected_detail, by_phenotype = run_pilot(config)
    detailed_path = OUTPUT_DIR / "figure6_prioritization.tsv"
    by_phenotype_path = OUTPUT_DIR / "figure6_prioritization_by_phenotype.tsv"
    detailed.to_csv(detailed_path, sep="\t", index=False)
    by_phenotype.to_csv(by_phenotype_path, sep="\t", index=False)
    print(f"Saved prioritization results to {detailed_path}")
    print(f"Saved per-phenotype summary to {by_phenotype_path}")
    if detailed.empty:
        print("No valid prioritization runs were produced.")
    else:
        final = detailed[detailed["n_added"] == detailed["n_added"].max()]
        per_strategy = (
            final.groupby("strategy")["delta_balanced_accuracy"].mean().sort_values(ascending=False)
        )
        print("Mean delta balanced accuracy by strategy (final n_added):")
        print(per_strategy.round(4).to_string())


if __name__ == "__main__":
    main()
