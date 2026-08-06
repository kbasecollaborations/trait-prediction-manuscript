"""Decompose the Figure 5A cross-dataset performance shortfall into
majority-class collapse, class-balance shift, random-vs-dataset gap, and
GapMind-feature rescue. All numbers respect the minority-class filter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from scripts.minority_filter import (
    MIN_MINORITY_TEST_SAMPLES,
    concordant_minority_counts,
    extract_test_dataset,
    filter_by_minority,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS: tuple[str, ...] = ("atleaf", "lit", "marine", "pmi")
KOFAM_CSV: Path = REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results.csv"
GAPMIND_RAW_CSV: Path = (
    REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results_gapmind_raw.csv"
)
GAPMIND_CSV: Path = (
    REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results_gapmind.csv"
)
GAPMIND_PRED_TSV: Path = REPO_ROOT / "data/outputs/figure2/gapmind_phenotypes_loose.tsv"
PHENOTYPE_DIR: Path = REPO_ROOT / "data/processed/phenotypes"

MAJORITY_COLLAPSE_SENS_THRESHOLD: float = 0.95
MAJORITY_COLLAPSE_SPEC_THRESHOLD: float = 0.10


def load_concordant_labels(
    phenotype: str, datasets: Iterable[str] = DATASETS
) -> dict[str, pd.Series]:
    """Return ``{dataset: concordant-label-series}`` for a phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype name (column of the GapMind TSV and stem of the per-dataset
        phenotype TSVs).
    datasets : Iterable[str], optional
        Datasets to load.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of dataset to series of experimental labels restricted to
        concordant samples (i.e., experimental label equals GapMind loose
        prediction).
    """
    gapmind = pd.read_csv(
        GAPMIND_PRED_TSV, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    out: dict[str, pd.Series] = {}
    if phenotype not in gapmind.columns:
        return out
    for dataset in datasets:
        path = PHENOTYPE_DIR / dataset / f"{phenotype}.tsv"
        if not path.exists():
            continue
        labels = (
            pd.read_csv(path, sep="\t", dtype={"genomeID": str})
            .set_index("genomeID")[phenotype]
            .dropna()
            .astype(int)
        )
        common = labels.index.intersection(gapmind.index)
        labels = labels.loc[common]
        concordant = labels[labels == gapmind.loc[common, phenotype]]
        out[dataset] = concordant
    return out


def parse_train_datasets(key: str) -> list[str]:
    """Parse the train datasets from a ``"<phen>_train(a+b+c),test(d)"`` key."""
    match = re.search(r"train\(([^)]+)\)", str(key))
    return match.group(1).split("+") if match else []


def per_fold_class_balance(phenotype: str) -> pd.DataFrame:
    """Return per-(phenotype, held-out-dataset) train/test positive fractions.

    Parameters
    ----------
    phenotype : str
        Phenotype name.

    Returns
    -------
    pd.DataFrame
        Columns: ``phenotype``, ``test_dataset``, ``train_pos_frac``,
        ``test_pos_frac``, ``train_n``, ``test_n``, ``train_majority``,
        ``test_majority``.
    """
    concordant = load_concordant_labels(phenotype)
    rows: list[dict[str, object]] = []
    for held_out in DATASETS:
        if held_out not in concordant:
            continue
        train_series = pd.concat(
            [concordant[d] for d in DATASETS if d != held_out and d in concordant]
        )
        test_series = concordant[held_out]
        if train_series.empty or test_series.empty:
            continue
        train_pos = float((train_series == 1).mean())
        test_pos = float((test_series == 1).mean())
        train_majority = int(train_series.value_counts().idxmax())
        test_class_counts = test_series.value_counts()
        test_majority = (
            int(test_class_counts.idxmax()) if len(test_class_counts) else None
        )
        rows.append(
            {
                "phenotype": phenotype,
                "test_dataset": held_out,
                "train_pos_frac": train_pos,
                "test_pos_frac": test_pos,
                "train_n": int(train_series.size),
                "test_n": int(test_series.size),
                "train_majority": train_majority,
                "test_majority": test_majority,
                "train_test_majority_match": train_majority == test_majority,
                "abs_balance_shift": abs(train_pos - test_pos),
            }
        )
    return pd.DataFrame(rows)


def majority_collapse(row: pd.Series) -> bool:
    """Return True if a fold's confusion matrix is degenerate."""
    sens = float(row["sensitivity"])
    spec = float(row["specificity"])
    high_sens_low_spec = (
        sens > MAJORITY_COLLAPSE_SENS_THRESHOLD
        and spec < MAJORITY_COLLAPSE_SPEC_THRESHOLD
    )
    high_spec_low_sens = (
        spec > MAJORITY_COLLAPSE_SENS_THRESHOLD
        and sens < MAJORITY_COLLAPSE_SPEC_THRESHOLD
    )
    return bool(high_sens_low_spec or high_spec_low_sens)


def load_and_filter(
    csv_path: Path, minority: dict[tuple[str, str], int]
) -> pd.DataFrame:
    """Load a Figure 5A result CSV and apply the minority-test filter.

    Parameters
    ----------
    csv_path : Path
        CSV file path.
    minority : dict[tuple[str, str], int]
        Concordant minority counts per ``(phenotype, dataset)``.

    Returns
    -------
    pd.DataFrame
        Filtered results.
    """
    df = pd.read_csv(csv_path)
    return filter_by_minority(df, minority)


def summarize_phenotype(
    kofam: pd.DataFrame,
    gapmind_features: pd.DataFrame,
    balance: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-phenotype summary table for Figure 5A diagnostics."""
    rows: list[dict[str, object]] = []
    for phenotype, sub in kofam.groupby("phenotype"):
        random = sub[sub["split_type"] == "random_split"]
        cross = sub[sub["split_type"] == "dataset_split"]
        phylo = sub[sub["split_type"] == "phylo_ooc"]

        # Pooled concordant size and positive fraction from labels, not the CSV.
        concordant_series = pd.concat(
            list(load_concordant_labels(str(phenotype)).values())
        )
        n_concordant = int(concordant_series.size)
        pooled_pos = (
            float((concordant_series == 1).mean()) if n_concordant else float("nan")
        )

        collapsed = (
            cross.apply(majority_collapse, axis=1)
            if len(cross)
            else pd.Series(dtype=bool)
        )
        collapse_frac = float(collapsed.mean()) if len(collapsed) else float("nan")

        gapmind_cross = gapmind_features[
            (gapmind_features["phenotype"] == phenotype)
            & (gapmind_features["split_type"] == "dataset_split")
        ]

        rows.append(
            {
                "phenotype": phenotype,
                "n_concordant": n_concordant,
                "pooled_pos_frac": pooled_pos,
                "mean_BA_random_KOFAM": (
                    float(random["balanced_accuracy"].mean())
                    if len(random)
                    else float("nan")
                ),
                "mean_BA_dataset_KOFAM": (
                    float(cross["balanced_accuracy"].mean())
                    if len(cross)
                    else float("nan")
                ),
                "mean_BA_phylo_KOFAM": (
                    float(phylo["balanced_accuracy"].mean())
                    if len(phylo)
                    else float("nan")
                ),
                "mean_BA_dataset_GapMind": (
                    float(gapmind_cross["balanced_accuracy"].mean())
                    if len(gapmind_cross)
                    else float("nan")
                ),
                "gap_random_minus_dataset": (
                    float(random["balanced_accuracy"].mean())
                    - float(cross["balanced_accuracy"].mean())
                    if len(random) and len(cross)
                    else float("nan")
                ),
                "gap_GapMind_minus_KOFAM_cross": (
                    float(gapmind_cross["balanced_accuracy"].mean())
                    - float(cross["balanced_accuracy"].mean())
                    if len(gapmind_cross) and len(cross)
                    else float("nan")
                ),
                "n_dataset_folds": len(cross),
                "n_collapsed_folds": int(collapsed.sum()) if len(collapsed) else 0,
                "majority_collapse_frac": collapse_frac,
                "mean_abs_balance_shift": (
                    float(
                        balance.loc[
                            balance["phenotype"] == phenotype, "abs_balance_shift"
                        ].mean()
                    )
                    if not balance.empty
                    else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("mean_BA_dataset_KOFAM").reset_index(drop=True)
    )


def per_fold_detail(
    kofam: pd.DataFrame,
    balance: pd.DataFrame,
) -> pd.DataFrame:
    """Return per (phenotype, held-out-dataset) fold-level detail."""
    cross = kofam[kofam["split_type"] == "dataset_split"].copy()
    cross["test_dataset"] = cross["key"].apply(extract_test_dataset)
    cross["majority_collapse"] = cross.apply(majority_collapse, axis=1)
    merged = cross.merge(balance, on=["phenotype", "test_dataset"], how="left")
    return (
        merged[
            [
                "phenotype",
                "test_dataset",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "majority_collapse",
                "train_pos_frac",
                "test_pos_frac",
                "abs_balance_shift",
                "train_majority",
                "test_majority",
                "train_test_majority_match",
                "n_train",
                "n_test",
                "n_minority_test",
            ]
        ]
        .sort_values(["phenotype", "test_dataset"])
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the decomposition and write summary CSVs."""
    minority = concordant_minority_counts(
        gapmind_file=GAPMIND_PRED_TSV,
        phenotype_dir=PHENOTYPE_DIR,
        datasets=DATASETS,
    )
    kofam = load_and_filter(KOFAM_CSV, minority)
    gapmind_features = load_and_filter(GAPMIND_RAW_CSV, minority)

    phenotypes = sorted(kofam["phenotype"].unique())
    balance = pd.concat(
        [per_fold_class_balance(p) for p in phenotypes], ignore_index=True
    )

    out_dir = REPO_ROOT / "data/outputs/figure5_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_phenotype(kofam, gapmind_features, balance)
    folds = per_fold_detail(kofam, balance)

    summary.to_csv(out_dir / "phenotype_summary.csv", index=False)
    folds.to_csv(out_dir / "fold_detail.csv", index=False)
    balance.to_csv(out_dir / "class_balance_per_fold.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(
        f"\nMinority-test threshold applied: MIN_MINORITY_TEST_SAMPLES = {MIN_MINORITY_TEST_SAMPLES}"
    )
    print("Per-phenotype summary (sorted by dataset-split KOFAM BA):\n")
    print(summary.to_string(index=False))

    print("\n\nPer-fold detail (dataset_split rows passing minority filter):\n")
    print(folds.to_string(index=False))

    worst = ["Glucose", "Serine", "Galacturonic-Acid"]
    best = ["Histidine", "m-Inositol", "Mannitol"]
    print("\n\nWorst performers per-fold (Glucose/Serine/Galacturonic-Acid):")
    print(folds[folds["phenotype"].isin(worst)].to_string(index=False))
    print("\n\nBest performers per-fold (Histidine/m-Inositol/Mannitol):")
    print(folds[folds["phenotype"].isin(best)].to_string(index=False))


if __name__ == "__main__":
    main()
