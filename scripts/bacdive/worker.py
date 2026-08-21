"""Worker-side code for the parallel BacDive sweep.

Kept out of the ``-m`` ``__main__`` driver so that ``joblib``/``loky`` pickles
the worker functions by reference rather than by value and workers resolve the
cached loaders correctly.
"""

from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.bacdive.splits import load_bacdive_labels, load_group_a_labels
from scripts.ml_splits import perform_split_ml

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


@cache
def load_matrix(path: str) -> pd.DataFrame:
    """Load a feature matrix once per process (Parquet or TSV by extension)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, sep="\t", index_col=0, dtype={"genomeID": str})
    df.index = df.index.astype(str)
    return df


@cache
def load_labels(group: str, phenotype: str) -> pd.Series:
    """Load 0/1 labels for one phenotype from ``bacdive`` or Group A, cached per process."""
    return (
        load_bacdive_labels(phenotype)
        if group == "bacdive"
        else load_group_a_labels(phenotype)
    )


def run_job(job: dict[str, Any], thread_count: int) -> dict[str, Any]:
    """Train and score one CatBoost fit; never raises (errors return a row)."""
    meta = {k: job[k] for k in job if k not in {"train_ids", "val_ids", "test_ids"}}
    try:
        train_mat = load_matrix(job["train_matrix"])
        test_mat = load_matrix(job["test_matrix"])
        y_tr = load_labels(job["train_group"], job["phenotype"])
        y_te = load_labels(job["test_group"], job["phenotype"])

        result = perform_split_ml(
            train_mat.loc[job["train_ids"]],
            y_tr.loc[job["train_ids"]],
            train_mat.loc[job["val_ids"]],
            y_tr.loc[job["val_ids"]],
            test_mat.loc[job["test_ids"]],
            y_te.loc[job["test_ids"]],
            model_type="cb",
            scoring=SCORING,
            random_state=job["seed"],
            thread_count=thread_count,
            # BacDive carbon sources are strongly positive-skewed; the
            # unweighted manuscript default collapses to majority-class
            # prediction (balanced accuracy ~0.5) here.
            auto_class_weights="Balanced",
        )
        result.pop("features", None)
        return {**meta, **result, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
        return {**meta, "status": f"error: {type(exc).__name__}: {exc}"}


def _gather(parts: list, phenotype: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix + label vector from multiple (path, group, ids) parts.

    Parts are aligned to the union of their feature columns (missing filled 0),
    so manuscript (KOFAM) and BacDive (reduced KOFAM) rows can be stacked.
    """
    Xs, ys = [], []
    for matrix_path, group, ids in parts:
        if not ids:
            continue
        mat = load_matrix(matrix_path)
        lab = load_labels(group, phenotype)
        Xs.append(mat.loc[ids])
        ys.append(lab.loc[ids])
    cols = sorted(set().union(*[set(X.columns) for X in Xs]))
    Xs = [X.reindex(columns=cols, fill_value=0).astype("int16") for X in Xs]
    return pd.concat(Xs), pd.concat(ys)


def run_mixed_job(job: dict[str, Any], thread_count: int) -> dict[str, Any]:
    """Train/score one fit whose training pool may mix data sources.

    ``job`` carries ``train_parts``/``val_parts`` (lists of [path, group, ids])
    and ``test_part`` ([path, group, ids]); columns are auto-aligned by
    ``perform_split_ml``.
    """
    meta = {
        k: job[k] for k in job if k not in {"train_parts", "val_parts", "test_part"}
    }
    try:
        ph = job["phenotype"]
        X_tr, y_tr = _gather(job["train_parts"], ph)
        X_va, y_va = _gather(job["val_parts"], ph)
        X_te, y_te = _gather([job["test_part"]], ph)
        result = perform_split_ml(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            y_te,
            model_type="cb",
            scoring=SCORING,
            random_state=job["seed"],
            thread_count=thread_count,
            auto_class_weights="Balanced",
        )
        result.pop("features", None)
        return {**meta, **result, "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {**meta, "status": f"error: {type(exc).__name__}: {exc}"}
