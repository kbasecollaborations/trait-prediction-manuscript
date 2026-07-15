#!/usr/bin/env python3
"""Cross-dataset accuracy AND feature stability for the C / F / FB strategies.

The head-to-head sweep recorded only metrics. To compare *feature stability*
across the three deployment strategies that matter for the manuscript question
("does adding BacDive volume help over full or concordant training in the
cross-dataset setting?") we re-run those fits and keep the top-K feature
importances per fit so stability can be measured downstream.

Strategies (tested on each held-out manuscript dataset's full labels):
    C   concordant samples of the other 3 manuscript datasets
    F   full other 3 manuscript datasets
    FB  full other 3 + all BacDive (volume added)

Output: ``data/outputs/bacdive/strategy_stability.csv`` with one row per
(phenotype, held_out, strategy, seed): the scoring metrics plus a ``top_features``
column (``;``-joined KOFAM KO ids, ranked by CatBoost importance).

Run (from the repository root):
    uv run python -m scripts.bacdive.strategy_stability
"""

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from scripts.bacdive.head_to_head import build_jobs
from scripts.bacdive.worker import _gather
from scripts.ml_splits import perform_split_ml

SCORING = [
    "accuracy", "balanced_accuracy", "matthews_corrcoef",
    "precision", "recall", "f1", "sensitivity", "specificity", "roc_auc",
]
STRATEGIES = {"C", "F", "FB"}
TOP_K = 25
OUTPUT = Path("data/outputs/bacdive/strategy_stability.csv")
N_JOBS = int(os.environ.get("BACDIVE_N_JOBS", os.cpu_count() or 4))


def run_feature_job(job: dict[str, Any], thread_count: int = 1) -> dict[str, Any]:
    """Train one CatBoost fit and keep its top-K feature importances.

    Parameters
    ----------
    job : dict
        A strategy job from :func:`build_jobs` (carries train/val/test parts).
    thread_count : int, optional
        CatBoost thread count, by default 1.

    Returns
    -------
    dict
        Metadata, scoring metrics, and a ``top_features`` string; ``status`` is
        ``"ok"`` or an error description (the sweep never crashes on one fit).
    """
    meta = {k: job[k] for k in job
            if k not in {"train_parts", "val_parts", "test_part"}}
    try:
        ph = job["phenotype"]
        X_tr, y_tr = _gather(job["train_parts"], ph)
        X_va, y_va = _gather(job["val_parts"], ph)
        X_te, y_te = _gather([job["test_part"]], ph)
        res = perform_split_ml(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            model_type="cb", scoring=SCORING, random_state=job["seed"],
            thread_count=thread_count, auto_class_weights="Balanced",
        )
        feats = list(res.pop("features", []))[:TOP_K]
        return {**meta, **res, "top_features": ";".join(map(str, feats)), "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - record, do not crash the sweep
        return {**meta, "status": f"error: {type(exc).__name__}: {exc}"}


def main() -> None:
    """Run the C/F/FB fits with feature capture and save the result table."""
    jobs, _ = build_jobs()
    jobs = [j for j in jobs
            if j["analysis"] == "strategy" and j["strategy"] in STRATEGIES]
    print(f"Running {len(jobs)} fits (C/F/FB x 13 phenotypes x 4 held-out x 5 seeds), "
          f"n_jobs={N_JOBS}")
    t0 = time.time()
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_feature_job)(job, 1) for job in jobs
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    ok = (df["status"] == "ok").sum()
    print(f"saved {OUTPUT} ({ok} ok, {len(df) - ok} error) in {(time.time()-t0)/60:.1f} min")
    if ok:
        m = df[df.status == "ok"].groupby("strategy")["balanced_accuracy"].agg(
            ["mean", "count"]).round(3)
        print(m.to_string())


if __name__ == "__main__":
    main()
