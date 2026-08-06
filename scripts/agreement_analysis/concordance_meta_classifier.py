#!/usr/bin/env python3
"""Per-genome concordance meta-classifier, evaluated leave-one-dataset-out.

For each phenotype, predicts concordance (whether the GapMind call agrees with
the experimental phenotype) from KOFAM genome features alone, scored both pooled
across phenotypes and per-phenotype against the per-row ``confidence`` column as
a baseline AUC.

Three cuts:

1. ``binary``: concordant (1) vs. all discordant (0).
2. ``fn``: concordant vs. false-negative discordant only (GapMind says no growth,
   experiment says growth). FP-discordant rows are dropped.
3. ``fp``: concordant vs. false-positive discordant only (GapMind says growth,
   experiment says no growth). FN-discordant rows are dropped.

Run with::

    EXPERIMENT_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \\
        uv run python -m scripts.agreement_analysis.concordance_meta_classifier
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from trait_prediction.pipeline import align_columns

from scripts.ml import make_classifier

PER_SAMPLE_FILE: Path = Path("data/outputs/figure7/figure7_per_sample.tsv")
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
DEFAULT_OUTPUT_DIR: Path = Path("data/outputs/concordance_meta")

RANDOM_STATE: int = 42
# CatBoost early stopping needs a validation set carved from the training data.
VAL_FRACTION: float = 0.2
# Minimum genomes (and per-class minimum) required to fit / score a split.
MIN_TRAIN_SAMPLES: int = 10
MIN_TEST_SAMPLES: int = 5

THREADS: int = int(os.environ.get("EXPERIMENT_THREADS", "4"))

Cut = Literal["binary", "fn", "fp"]
ALL_CUTS: tuple[Cut, ...] = ("binary", "fn", "fp")


def load_per_sample(per_sample_file: Path) -> pd.DataFrame:
    """Load the Figure 7 per-sample table and derive concordance labels.

    Parameters
    ----------
    per_sample_file : Path
        Path to ``figure7_per_sample.tsv`` with columns ``phenotype``,
        ``held_out_dataset``, ``genome``, ``y_true``, ``y_pred``, ``proba``,
        ``confidence``, ``gapmind_pred``.

    Returns
    -------
    pd.DataFrame
        Rows with a non-NaN GapMind call, augmented with three integer columns:
        ``concordant`` (gapmind_pred == y_true), ``is_fp`` (gapmind_pred == 1 and
        y_true == 0), and ``is_fn`` (gapmind_pred == 0 and y_true == 1).

    Raises
    ------
    FileNotFoundError
        If ``per_sample_file`` does not exist.
    """
    if not per_sample_file.exists():
        raise FileNotFoundError(f"Per-sample table not found: {per_sample_file}")

    df = pd.read_csv(per_sample_file, sep="\t", dtype={"genome": str})

    # gapmind_pred is a float column because missing calls are NaN; drop those
    # rows before casting.
    df = df[df["gapmind_pred"].notna()].copy()
    df["gapmind_pred"] = df["gapmind_pred"].astype(int)
    df["y_true"] = df["y_true"].astype(int)

    df["concordant"] = (df["gapmind_pred"] == df["y_true"]).astype(int)
    df["is_fp"] = ((df["gapmind_pred"] == 1) & (df["y_true"] == 0)).astype(int)
    df["is_fn"] = ((df["gapmind_pred"] == 0) & (df["y_true"] == 1)).astype(int)

    return df.reset_index(drop=True)


def load_kofam_features(kofam_file: Path) -> pd.DataFrame:
    """Load the reduced KOFAM feature matrix.

    The matrix is used as-is; it is already correlation- and variance-filtered.

    Parameters
    ----------
    kofam_file : Path
        Path to the reduced combined-dataset KOFAM TSV (index ``genomeID``).

    Returns
    -------
    pd.DataFrame
        Genome-by-KO 0/1 feature matrix indexed by genome id (string).

    Raises
    ------
    FileNotFoundError
        If ``kofam_file`` does not exist.
    """
    if not kofam_file.exists():
        raise FileNotFoundError(f"KOFAM feature file not found: {kofam_file}")

    kofam = pd.read_csv(kofam_file, sep="\t", index_col=0, dtype={"genomeID": str})
    kofam.index = kofam.index.astype(str)
    return kofam


def select_cut_rows(rows: pd.DataFrame, cut: Cut) -> pd.DataFrame:
    """Restrict per-sample rows to those used by a given cut.

    Parameters
    ----------
    rows : pd.DataFrame
        Per-sample rows for a single phenotype (output of :func:`load_per_sample`).
    cut : {"binary", "fn", "fp"}
        ``binary`` keeps all rows (concordant vs. all discordant). ``fn`` keeps
        concordant and FN-discordant rows (drops FP). ``fp`` keeps concordant and
        FP-discordant rows (drops FN).

    Returns
    -------
    pd.DataFrame
        The subset of ``rows`` belonging to the cut.
    """
    if cut == "binary":
        return rows
    if cut == "fn":
        return rows[(rows["concordant"] == 1) | (rows["is_fn"] == 1)]
    if cut == "fp":
        return rows[(rows["concordant"] == 1) | (rows["is_fp"] == 1)]
    raise ValueError(f"Unknown cut: {cut!r}")


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC that returns ``np.nan`` on a single-class target."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (AUPRC) that returns ``np.nan`` on a single-class target."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def fit_predict_meta(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threads: int,
) -> np.ndarray | None:
    """Fit a concordance meta-classifier and predict concordance probability.

    A validation set is carved from the training data for CatBoost early stopping,
    and ``X_test`` columns are aligned to the training columns before prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        KOFAM features for the non-held-out genomes.
    y_train : pd.Series
        Concordance labels for the non-held-out genomes.
    X_test : pd.DataFrame
        KOFAM features for the held-out genomes.
    threads : int
        ``thread_count`` passed to the CatBoost classifier.

    Returns
    -------
    np.ndarray | None
        P(concordant) for each held-out genome, or ``None`` when the training data
        is too small, single-class, or cannot be split into two-class train/val
        folds.
    """
    if len(X_train) < MIN_TRAIN_SAMPLES or y_train.nunique() != 2:
        return None

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=VAL_FRACTION,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    except ValueError:
        return None

    if y_tr.nunique() != 2 or y_val.nunique() != 2:
        return None

    model = make_classifier("cb", random_state=RANDOM_STATE, thread_count=threads)
    X_val_aligned = align_columns(X_tr, X_val)
    X_test_aligned = align_columns(X_tr, X_test)
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val_aligned, y_val),
        use_best_model=True,
        verbose=False,
    )

    # CatBoost orders predict_proba columns by sorted class label, so column 1 is
    # P(class == 1) == P(concordant).
    return np.asarray(model.predict_proba(X_test_aligned))[:, 1]


def evaluate_phenotype_cut(
    phenotype: str,
    rows: pd.DataFrame,
    kofam: pd.DataFrame,
    cut: Cut,
    threads: int,
) -> list[dict[str, object]]:
    """Run leave-one-dataset-out evaluation for one (phenotype, cut).

    Parameters
    ----------
    phenotype : str
        Phenotype name.
    rows : pd.DataFrame
        All per-sample rows for this phenotype (before the cut restriction).
    kofam : pd.DataFrame
        KOFAM feature matrix indexed by genome id.
    cut : {"binary", "fn", "fp"}
        Concordance cut to evaluate.
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    list[dict[str, object]]
        One record per held-out dataset with meta and baseline AUC/AUPRC and
        sample counts. Held-out datasets that cannot be scored are skipped.
    """
    cut_rows = select_cut_rows(rows, cut)
    cut_rows = cut_rows[cut_rows["genome"].isin(kofam.index)]
    records: list[dict[str, object]] = []

    for held_out in sorted(cut_rows["held_out_dataset"].unique()):
        train_rows = cut_rows[cut_rows["held_out_dataset"] != held_out]
        test_rows = cut_rows[cut_rows["held_out_dataset"] == held_out]

        if len(train_rows) < MIN_TRAIN_SAMPLES or len(test_rows) < MIN_TEST_SAMPLES:
            continue
        if test_rows["concordant"].nunique() < 2:
            continue

        X_train = kofam.loc[train_rows["genome"]]
        y_train = pd.Series(
            train_rows["concordant"].to_numpy(), index=train_rows["genome"]
        )
        X_test = kofam.loc[test_rows["genome"]]
        y_test = test_rows["concordant"].to_numpy()

        proba = fit_predict_meta(X_train, y_train, X_test, threads)
        if proba is None:
            continue

        records.append(
            {
                "phenotype": phenotype,
                "held_out_dataset": held_out,
                "cut": cut,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_test_concordant": int(test_rows["concordant"].sum()),
                "n_test_discordant": int((1 - test_rows["concordant"]).sum()),
                "meta_auc": _safe_auc(y_test, proba),
                "meta_auprc": _safe_auprc(y_test, proba),
                "confidence_auc": _safe_auc(y_test, test_rows["confidence"].to_numpy()),
                "confidence_auprc": _safe_auprc(
                    y_test, test_rows["confidence"].to_numpy()
                ),
            }
        )

    return records


def pooled_summary(
    per_sample: pd.DataFrame,
    kofam: pd.DataFrame,
    cuts: tuple[Cut, ...],
    threads: int,
) -> pd.DataFrame:
    """Compute pooled (cross-phenotype) meta vs. confidence AUC/AUPRC per cut.

    Predictions are gathered per (phenotype, held-out dataset), so each meta model
    sees only its own training datasets, then concatenated to score one pooled
    curve per cut.

    Parameters
    ----------
    per_sample : pd.DataFrame
        Full per-sample table with concordance labels.
    kofam : pd.DataFrame
        KOFAM feature matrix.
    cuts : tuple of {"binary", "fn", "fp"}
        Cuts to evaluate.
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    pd.DataFrame
        One row per cut with pooled meta and confidence AUC/AUPRC and the pooled
        sample count.
    """
    summary_rows: list[dict[str, object]] = []

    for cut in cuts:
        pooled_y: list[int] = []
        pooled_meta: list[float] = []
        pooled_conf: list[float] = []

        for phenotype, rows in per_sample.groupby("phenotype", sort=True):
            cut_rows = select_cut_rows(rows, cut)
            cut_rows = cut_rows[cut_rows["genome"].isin(kofam.index)]

            for held_out in sorted(cut_rows["held_out_dataset"].unique()):
                train_rows = cut_rows[cut_rows["held_out_dataset"] != held_out]
                test_rows = cut_rows[cut_rows["held_out_dataset"] == held_out]
                if (
                    len(train_rows) < MIN_TRAIN_SAMPLES
                    or len(test_rows) < MIN_TEST_SAMPLES
                ):
                    continue
                if test_rows["concordant"].nunique() < 2:
                    continue

                X_train = kofam.loc[train_rows["genome"]]
                y_train = pd.Series(
                    train_rows["concordant"].to_numpy(), index=train_rows["genome"]
                )
                X_test = kofam.loc[test_rows["genome"]]

                proba = fit_predict_meta(X_train, y_train, X_test, threads)
                if proba is None:
                    continue

                pooled_y.extend(test_rows["concordant"].tolist())
                pooled_meta.extend(proba.tolist())
                pooled_conf.extend(test_rows["confidence"].tolist())

        y_arr = np.asarray(pooled_y)
        summary_rows.append(
            {
                "cut": cut,
                "n_pooled": len(y_arr),
                "n_concordant": int(y_arr.sum()) if len(y_arr) else 0,
                "n_discordant": int((1 - y_arr).sum()) if len(y_arr) else 0,
                "meta_auc": _safe_auc(y_arr, np.asarray(pooled_meta)),
                "meta_auprc": _safe_auprc(y_arr, np.asarray(pooled_meta)),
                "confidence_auc": _safe_auc(y_arr, np.asarray(pooled_conf)),
                "confidence_auprc": _safe_auprc(y_arr, np.asarray(pooled_conf)),
            }
        )

    return pd.DataFrame(summary_rows)


def print_pooled_table(pooled: pd.DataFrame) -> None:
    """Print the pooled meta-classifier vs. confidence-baseline AUC table.

    Parameters
    ----------
    pooled : pd.DataFrame
        Output of :func:`pooled_summary`.
    """
    print("\nPooled (cross-phenotype) AUC: meta-classifier vs. confidence baseline")
    print(
        f"{'cut':<8} {'n':>6} {'meta_auc':>9} {'conf_auc':>9} "
        f"{'meta_auprc':>11} {'conf_auprc':>11}"
    )
    for _, r in pooled.iterrows():
        print(
            f"{r['cut']:<8} {int(r['n_pooled']):>6} "
            f"{r['meta_auc']:>9.3f} {r['confidence_auc']:>9.3f} "
            f"{r['meta_auprc']:>11.3f} {r['confidence_auprc']:>11.3f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``threads``, ``phenotypes``, ``cuts``, ``out``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help="CatBoost thread_count (default: EXPERIMENT_THREADS env or 4).",
    )
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=None,
        help="Optional subset of phenotypes (smoke test).",
    )
    parser.add_argument(
        "--cuts",
        nargs="+",
        choices=list(ALL_CUTS),
        default=list(ALL_CUTS),
        help="Cuts to evaluate (default: all three).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--held-out",
        nargs="+",
        default=None,
        help="Optional subset of held-out datasets (smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the concordance meta-classifier experiment and persist results."""
    args = parse_args()
    cuts: tuple[Cut, ...] = tuple(args.cuts)
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    per_sample = load_per_sample(PER_SAMPLE_FILE)
    kofam = load_kofam_features(KOFAM_FEATURE_FILE)

    if args.phenotypes is not None:
        per_sample = per_sample[per_sample["phenotype"].isin(args.phenotypes)]
    if args.held_out is not None:
        per_sample = per_sample[per_sample["held_out_dataset"].isin(args.held_out)]

    print(
        f"Loaded {len(per_sample)} per-sample rows "
        f"({per_sample['phenotype'].nunique()} phenotypes, "
        f"{per_sample['held_out_dataset'].nunique()} held-out datasets) "
        f"after dropping NaN GapMind calls."
    )
    print(f"KOFAM matrix: {kofam.shape[0]} genomes x {kofam.shape[1]} features.")
    print(f"Using thread_count={args.threads} for CatBoost.")

    detail_records: list[dict[str, object]] = []
    for phenotype, rows in per_sample.groupby("phenotype", sort=True):
        for cut in cuts:
            detail_records.extend(
                evaluate_phenotype_cut(phenotype, rows, kofam, cut, args.threads)
            )
    detail = pd.DataFrame(detail_records)

    detail_file = args.out / "concordance_meta_per_split.csv"
    detail.to_csv(detail_file, index=False)
    print(f"\nSaved per-split results to {detail_file} ({len(detail)} rows).")

    pooled = pooled_summary(per_sample, kofam, cuts, args.threads)
    pooled_file = args.out / "concordance_meta_pooled.csv"
    pooled.to_csv(pooled_file, index=False)
    print(f"Saved pooled summary to {pooled_file}.")

    print_pooled_table(pooled)

    if not detail.empty:
        print("\nPer-phenotype mean meta_auc vs. confidence_auc (by cut):")
        per_pheno = (
            detail.groupby(["cut", "phenotype"])[["meta_auc", "confidence_auc"]]
            .mean()
            .round(3)
        )
        print(per_pheno)

    elapsed = time.perf_counter() - start
    print(f"\nElapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
