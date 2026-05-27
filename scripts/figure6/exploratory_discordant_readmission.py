#!/usr/bin/env python3
"""
Exploratory pilot: does selectively re-admitting a small number of *discordant*
training samples back into the concordant-only training set improve cross-dataset
balanced accuracy on the **full** held-out test set, without paying the cost of
collecting new labels?

Setup
-----
For each (phenotype, dataset split, seed) we:

1. Train a baseline concordant-only model on the train+val pool restricted to
   concordant samples and evaluate on the full ``y_test``.
2. Score every discordant sample in the train+val pool with several label-free
   selection signals (low model confidence, high feature-space novelty, evenly
   spaced diversity along the novelty axis, random) and a single label-aware
   reference signal (``hard_for_ml`` -- the concordant model is wrong about the
   experimental label).
3. For budget ``K in {25, 50}`` augment the concordant training set with the
   top-K discordants under each strategy, retrain, and re-evaluate on the same
   full ``y_test``. The recorded delta is ``augmented_ba - baseline_ba``.
4. As an upper bound, the same configuration is run with every available
   discordant re-admitted (= full-data training).

The four label-free strategies mirror Figure 7C exactly; the ``hard_for_ml``
strategy is included as a label-aware reference to bound how much information
the label-free signals are leaving on the table.

Speed-tuned to match ``scripts/figure7/figure7d_active_learning.py``
(``cb_noeval`` with ``iterations=120``, ``depth=4``, ``learning_rate=0.05``).
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
from tqdm import tqdm

from scripts.figure5.figure5cd_data import (
    get_concordant_and_discordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.figure7.applicability import mean_knn_jaccard_distance
from scripts.ml import make_classifier
from scripts.ml_splits import load_single_split_data
from trait_prediction.pipeline import align_columns

FEATURE_FILE: Path = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
SPLITS_DIR: Path = Path("data/processed/train_test_splits/dataset_split")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
OUTPUT_DIR: Path = Path("data/outputs/figure6/readmission_exploratory")
HELD_OUT_RE: re.Pattern[str] = re.compile(r"test\(([^)]+)\)")

# Label-free strategies mirror Figure 7C; ``hard_for_ml`` is a label-aware
# reference (uses the experimental label at selection time).
STRATEGIES: tuple[str, ...] = (
    "random",
    "low_confidence",
    "high_ood",
    "diversity",
    "hard_for_ml",
)


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the discordant-readmission pilot."""

    phenotypes: tuple[str, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    max_splits_per_phenotype: int
    iterations: int
    min_discordant_pool: int


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
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[25, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-splits-per-phenotype", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--min-discordant-pool", type=int, default=10)
    args = parser.parse_args()
    return PilotConfig(
        phenotypes=tuple(args.phenotypes),
        budgets=tuple(sorted(args.budgets)),
        seeds=tuple(args.seeds),
        max_splits_per_phenotype=args.max_splits_per_phenotype,
        iterations=args.iterations,
        min_discordant_pool=args.min_discordant_pool,
    )


def held_out_dataset(split_dir: Path) -> str:
    """Extract the held-out dataset name from a split directory.

    Parameters
    ----------
    split_dir : Path
        Dataset-split directory whose name contains ``test(<dataset>)``.

    Returns
    -------
    str
        Held-out dataset name.

    Raises
    ------
    ValueError
        If the directory name does not contain a ``test(<dataset>)`` token.
    """
    match = HELD_OUT_RE.search(split_dir.name)
    if match is None:
        raise ValueError(f"Could not parse held-out dataset from {split_dir}")
    return match.group(1)


def iter_split_dirs(phenotype: str, max_splits: int) -> Iterable[Path]:
    """Yield up to ``max_splits`` dataset-split directories for ``phenotype``.

    Parameters
    ----------
    phenotype : str
        Phenotype name.
    max_splits : int
        Maximum number of split directories to yield.

    Yields
    ------
    Path
        Dataset-split directory.
    """
    phenotype_dir = SPLITS_DIR / phenotype
    if not phenotype_dir.exists():
        return
    split_dirs = sorted(p for p in phenotype_dir.iterdir() if p.is_dir())
    yield from split_dirs[:max_splits]


def safe_balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy, returning NaN if both classes are not present.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    float
        Balanced accuracy or NaN when the evaluation set is single-class.
    """
    if y_true.nunique() < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def fit_predict_proba(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    iterations: int,
) -> np.ndarray | None:
    """Fit a short CatBoost model and return ``P(class == 1)`` on ``X_test``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels (0/1).
    X_test : pd.DataFrame
        Test feature matrix.
    seed : int
        Random seed.
    iterations : int
        CatBoost iteration cap.

    Returns
    -------
    np.ndarray | None
        Predicted probability of class 1, or ``None`` if training is single-class.
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


def select_discordants(
    discordants: pd.DataFrame, k: int, strategy: str, rng: np.random.Generator
) -> list[str]:
    """Pick ``k`` discordant genome IDs under a selection strategy.

    Parameters
    ----------
    discordants : pd.DataFrame
        Indexed by genome. Required columns: ``confidence``, ``ood``,
        ``ml_wrong`` (used only by the label-aware ``hard_for_ml`` strategy).
    k : int
        Number to select.
    strategy : str
        One of :data:`STRATEGIES`.
    rng : np.random.Generator
        Source of randomness for ``random``.

    Returns
    -------
    list[str]
        Selected genome IDs.

    Raises
    ------
    ValueError
        If ``strategy`` is not recognised.
    """
    k = min(k, len(discordants))
    if strategy == "random":
        return rng.choice(discordants.index.to_numpy(), size=k, replace=False).tolist()
    if strategy == "low_confidence":
        return discordants.nsmallest(k, "confidence").index.tolist()
    if strategy == "high_ood":
        return discordants.nlargest(k, "ood").index.tolist()
    if strategy == "diversity":
        order = discordants["ood"].sort_values().index.to_numpy()
        idx = np.linspace(0, len(order) - 1, k).round().astype(int)
        return order[idx].tolist()
    if strategy == "hard_for_ml":
        # Label-aware reference: pick discordants where the concordant model's
        # prediction disagrees with the experimental label, ranked by how
        # wrong (lowest predicted probability of the true class).
        wrong = discordants[discordants["ml_wrong"]]
        if len(wrong) == 0:
            return []
        return wrong.nsmallest(k, "prob_of_true").index.tolist()
    raise ValueError(f"unknown strategy: {strategy}")


def run_pilot(config: PilotConfig) -> pd.DataFrame:
    """Run the discordant-readmission pilot.

    Parameters
    ----------
    config : PilotConfig
        Pilot configuration.

    Returns
    -------
    pd.DataFrame
        One row per (phenotype, held-out dataset, seed, strategy, budget).
        Includes a special strategy ``full_readmit`` with all discordants added
        as an upper-bound reference.
    """
    features = pd.read_csv(FEATURE_FILE, sep="\t", index_col=0, dtype={"genomeID": str})
    gapmind = load_gapmind_predictions(GAPMIND_FILE)
    phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    rows: list[dict[str, object]] = []

    for phenotype in tqdm(config.phenotypes, desc="Phenotypes"):
        concordant_set, discordant_set = get_concordant_and_discordant_samples(
            gapmind, phenotypes, phenotype
        )
        if not concordant_set or not discordant_set:
            continue
        for split_dir in iter_split_dirs(phenotype, config.max_splits_per_phenotype):
            split = load_single_split_data(split_dir, features)
            held_out = held_out_dataset(split_dir)
            X_pool = pd.concat([split["X_train"], split["X_val"]], axis=0)
            y_pool = pd.concat([split["y_train"], split["y_val"]], axis=0)
            X_test, y_test = split["X_test"], split["y_test"]

            conc_idx = sorted(set(X_pool.index) & concordant_set)
            disc_idx = sorted(set(X_pool.index) & discordant_set)
            if len(conc_idx) < 20 or len(disc_idx) < config.min_discordant_pool:
                continue

            X_conc, y_conc = X_pool.loc[conc_idx], y_pool.loc[conc_idx]
            X_disc, y_disc = X_pool.loc[disc_idx], y_pool.loc[disc_idx]
            if y_conc.nunique() < 2 or y_test.nunique() < 2:
                continue

            for seed in config.seeds:
                # ----- baseline concordant-only model -----
                base_proba_test = fit_predict_proba(
                    X_conc, y_conc, X_test, seed, config.iterations
                )
                if base_proba_test is None:
                    continue
                baseline_ba = safe_balanced_accuracy(
                    y_test, (base_proba_test >= 0.5).astype(int)
                )
                if np.isnan(baseline_ba):
                    continue

                # ----- score discordant pool with the same baseline model -----
                disc_proba = fit_predict_proba(
                    X_conc, y_conc, X_disc, seed, config.iterations
                )
                if disc_proba is None:
                    continue
                confidence = np.maximum(disc_proba, 1 - disc_proba)
                ood = mean_knn_jaccard_distance(X_disc, X_conc, k=5).reindex(disc_idx)
                disc_pred = (disc_proba >= 0.5).astype(int)
                ml_wrong = disc_pred != y_disc.to_numpy()
                prob_of_true = np.where(y_disc.to_numpy() == 1, disc_proba, 1 - disc_proba)
                discordants = pd.DataFrame(
                    {
                        "confidence": confidence,
                        "ood": ood.to_numpy(),
                        "ml_wrong": ml_wrong,
                        "prob_of_true": prob_of_true,
                    },
                    index=disc_idx,
                )

                # ----- upper bound: re-admit all discordants -----
                full_proba = fit_predict_proba(
                    pd.concat([X_conc, X_disc], axis=0),
                    pd.concat([y_conc, y_disc], axis=0),
                    X_test,
                    seed,
                    config.iterations,
                )
                if full_proba is not None:
                    full_ba = safe_balanced_accuracy(
                        y_test, (full_proba >= 0.5).astype(int)
                    )
                    rows.append({
                        "phenotype": phenotype,
                        "held_out_dataset": held_out,
                        "seed": seed,
                        "strategy": "full_readmit",
                        "budget": len(disc_idx),
                        "n_concordant": len(conc_idx),
                        "n_discordant_available": len(disc_idx),
                        "baseline_ba": baseline_ba,
                        "augmented_ba": full_ba,
                        "delta_ba": full_ba - baseline_ba,
                    })

                # ----- selective readmission across strategies and budgets -----
                for strategy in STRATEGIES:
                    rng = np.random.default_rng(seed)
                    for budget in config.budgets:
                        selected = select_discordants(discordants, budget, strategy, rng)
                        if len(selected) == 0:
                            continue
                        X_aug = pd.concat([X_conc, X_disc.loc[selected]], axis=0)
                        y_aug = pd.concat([y_conc, y_disc.loc[selected]], axis=0)
                        aug_proba = fit_predict_proba(
                            X_aug, y_aug, X_test, seed, config.iterations
                        )
                        if aug_proba is None:
                            continue
                        aug_ba = safe_balanced_accuracy(
                            y_test, (aug_proba >= 0.5).astype(int)
                        )
                        rows.append({
                            "phenotype": phenotype,
                            "held_out_dataset": held_out,
                            "seed": seed,
                            "strategy": strategy,
                            "budget": len(selected),
                            "n_concordant": len(conc_idx),
                            "n_discordant_available": len(disc_idx),
                            "baseline_ba": baseline_ba,
                            "augmented_ba": aug_ba,
                            "delta_ba": aug_ba - baseline_ba,
                        })

    return pd.DataFrame(rows)


def summarise(detailed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-run results into per-strategy and per-phenotype tables.

    Parameters
    ----------
    detailed : pd.DataFrame
        Output of :func:`run_pilot`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``by_strategy`` -- mean and SEM of ``delta_ba`` per
        (strategy, budget) across all runs; ``by_phenotype_strategy`` --
        mean ``delta_ba`` per (phenotype, strategy, budget).
    """
    if detailed.empty:
        empty = pd.DataFrame()
        return empty, empty

    def _agg(group: pd.DataFrame) -> pd.Series:
        vals = group["delta_ba"].to_numpy()
        n = len(vals)
        sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return pd.Series({
            "n_runs": n,
            "mean_delta_ba": float(np.mean(vals)),
            "sem_delta_ba": sem,
            "median_delta_ba": float(np.median(vals)),
            "frac_runs_positive": float(np.mean(vals > 0)),
        })

    by_strategy = (
        detailed.groupby(["strategy", "budget"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
        .sort_values(["budget", "mean_delta_ba"], ascending=[True, False])
    )
    by_phenotype_strategy = (
        detailed.groupby(["phenotype", "strategy", "budget"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
        .sort_values(["phenotype", "budget", "mean_delta_ba"], ascending=[True, True, False])
    )
    return by_strategy, by_phenotype_strategy


def main() -> None:
    """Run the discordant-readmission pilot and persist results."""
    config = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed = run_pilot(config)

    detailed_path = OUTPUT_DIR / "readmission_detailed.tsv"
    detailed.to_csv(detailed_path, sep="\t", index=False)
    print(f"\nSaved detailed results to {detailed_path} ({len(detailed)} rows)")

    if detailed.empty:
        print("No valid runs produced.")
        return

    by_strategy, by_phenotype_strategy = summarise(detailed)
    strategy_path = OUTPUT_DIR / "readmission_by_strategy.tsv"
    phen_path = OUTPUT_DIR / "readmission_by_phenotype_strategy.tsv"
    by_strategy.to_csv(strategy_path, sep="\t", index=False)
    by_phenotype_strategy.to_csv(phen_path, sep="\t", index=False)
    print(f"Saved per-strategy summary to {strategy_path}")
    print(f"Saved per-phenotype summary to {phen_path}")

    print("\nMean delta balanced accuracy by strategy and budget:")
    print(by_strategy.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
