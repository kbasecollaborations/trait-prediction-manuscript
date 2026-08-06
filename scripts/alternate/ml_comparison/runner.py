#!/usr/bin/env python3
"""
Unified runner for the alternative-ML model comparison.

For each (model, phenotype, subset, split_type, repeat) combination, fit on
train+val, evaluate on test, and write the per-fit JSON. A fit is skipped when
its output JSON already exists and records status "ok".

Outputs
-------
data/outputs/ml_comparison/
    per_fit/{model}/{subset}/{split_type}/{Phenotype}_{repeat}.json
    manifest.jsonl           # append-only audit log of completed/failed fits
    results.csv              # aggregated long-form table (regenerated periodically)
    aggregation.json         # bookkeeping for what's in results.csv

CLI
---
    uv run python -m scripts.alternate.ml_comparison.runner \\
        --models lasso enet lgbm \\
        --subsets full concordant \\
        --split-types random_split dataset_split \\
        --phenotypes Histidine Glucose Cellobiose \\
        --resume

Default: all models, both subsets, both split_types, all 15 phenotypes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trait_prediction.pipeline import align_columns, get_scores

from scripts import ml as _ml
from scripts.alternate.ml_comparison.models import (
    importances_from_model,
    patch_make_classifier,
)
from scripts.figure5.figure5a_data import (
    filter_split_to_concordant,
    get_concordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.ml_splits import load_split_data

PHENOTYPES = (
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
DEFAULT_MODELS = (
    "cb_noeval",
    "rf",  # baselines under matched (train+val merged) protocol
    "lasso",
    "enet",
    "splsda",
    "lgbm",
    "xgb",
    "pclr",
    "glasso_kegg",
)
DEFAULT_SUBSETS = ("full", "concordant")
DEFAULT_SPLIT_TYPES = ("random_split", "dataset_split")
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
OUTPUT_BASE = Path("data/outputs/ml_comparison")
SPLITS_DIR = Path("data/processed/train_test_splits")
FEATURE_FILE = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
GAPMIND_FILE = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR = Path("data/processed/phenotypes")
MIN_SAMPLES = 5
MIN_TEST_SAMPLES = 10
RANDOM_STATE = 42
AGGREGATE_EVERY = 10  # rewrite results.csv every N completed fits
TOP_FEATURES_SAVED = 50


def per_fit_path(
    model: str, subset: str, split_type: str, phenotype: str, repeat: str
) -> Path:
    return (
        OUTPUT_BASE
        / "per_fit"
        / model
        / subset
        / split_type
        / f"{phenotype}_{repeat}.json"
    )


def manifest_path() -> Path:
    return OUTPUT_BASE / "manifest.jsonl"


def results_path() -> Path:
    return OUTPUT_BASE / "results.csv"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def append_manifest(entry: dict[str, Any]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload.get("status") == "ok" and "balanced_accuracy" in payload
    except Exception:
        return False


def split_repeats_for(split_type: str, phenotype: str) -> list[str]:
    """Return the repeat-folder names available for (split_type, phenotype)."""
    base = SPLITS_DIR / split_type / phenotype
    if not base.exists():
        return []
    repeats = sorted([p.name for p in base.iterdir() if p.is_dir()])
    return repeats


def _fit_one(
    *,
    model_type: str,
    phenotype: str,
    subset: str,
    split_type: str,
    repeat: str,
    split: dict[str, Any],
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and score a single fit, returning the payload dict to be written."""
    extra_kwargs = extra_kwargs or {}
    started = time.time()

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    # Merge train + val for non-early-stopping models.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    X_train_full = X_train_full[~X_train_full.index.duplicated(keep="first")]
    y_train_full = y_train_full[~y_train_full.index.duplicated(keep="first")]

    X_test_aligned = align_columns(X_train_full, X_test)

    model = _ml.make_classifier(
        _resolve_model_alias(model_type),
        random_state=RANDOM_STATE,
        **extra_kwargs,
    )
    model.fit(X_train_full, y_train_full)
    scores = get_scores(model, X_test_aligned, y_test, SCORING)

    feat_series = importances_from_model(model, list(X_train_full.columns))
    top_features = feat_series.head(TOP_FEATURES_SAVED)

    payload: dict[str, Any] = {
        "status": "ok",
        "model": model_type,
        "phenotype": phenotype,
        "subset": subset,
        "split_type": split_type,
        "repeat": repeat,
        "n_train": len(X_train_full),
        "n_test": len(X_test),
        "n_features": int(X_train_full.shape[1]),
        "fit_seconds": round(time.time() - started, 3),
        "top_features": top_features.index.tolist(),
        "top_importances": top_features.values.tolist(),
    }
    for metric in SCORING:
        payload[metric] = scores.get(metric)
    return payload


def _load_concordant_resources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GAPMIND_FILE.exists():
        raise FileNotFoundError(
            f"missing {GAPMIND_FILE} — need to run figure2 gapmind script first"
        )
    if not PHENOTYPE_DIR.exists():
        raise FileNotFoundError(f"missing {PHENOTYPE_DIR}")
    return (
        load_gapmind_predictions(GAPMIND_FILE),
        load_experimental_phenotypes(PHENOTYPE_DIR),
    )


def _materialize_subset_split(
    *,
    raw_split: dict[str, Any],
    subset: str,
    phenotype: str,
    concordant_resources: tuple[pd.DataFrame, pd.DataFrame] | None,
) -> dict[str, Any] | None:
    """Return the train/val/test dict for the requested subset, or None when the
    subset leaves too few samples to fit or test."""
    if subset == "full":
        n_test = len(raw_split.get("X_test", []))
        if n_test < MIN_TEST_SAMPLES:
            return None
        return raw_split
    if subset == "concordant":
        gapmind, exp = concordant_resources  # type: ignore[misc]
        concordant = get_concordant_samples(gapmind, exp, phenotype)
        if not concordant:
            return None
        filtered = filter_split_to_concordant(
            raw_split, concordant, min_samples=MIN_SAMPLES
        )
        if filtered is None:
            return None
        if len(filtered["X_test"]) < MIN_TEST_SAMPLES:
            return None
        return filtered
    raise ValueError(f"unknown subset: {subset}")


def aggregate_results() -> None:
    """Rebuild results.csv from all per_fit JSON files."""
    rows: list[dict[str, Any]] = []
    root = OUTPUT_BASE / "per_fit"
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        try:
            with path.open() as f:
                payload = json.load(f)
        except Exception:
            continue
        if payload.get("status") != "ok":
            continue
        row = {
            k: payload[k]
            for k in (
                "model",
                "phenotype",
                "subset",
                "split_type",
                "repeat",
                "n_train",
                "n_test",
                "n_features",
                "fit_seconds",
                *SCORING,
            )
            if k in payload
        }
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(
        ["model", "subset", "split_type", "phenotype", "repeat"]
    )
    out = results_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    bookkeeping = {
        "n_rows": len(df),
        "n_models": df["model"].nunique(),
        "n_phenotypes": df["phenotype"].nunique(),
        "regenerated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(OUTPUT_BASE / "aggregation.json", bookkeeping)


def _build_work_list(args) -> list[tuple[str, str, str, str, str]]:
    work: list[tuple[str, str, str, str, str]] = []
    for split_type in args.split_types:
        for phenotype in args.phenotypes:
            repeats = split_repeats_for(split_type, phenotype)
            for repeat in repeats:
                for subset in args.subsets:
                    for model in args.models:
                        work.append((model, subset, split_type, phenotype, repeat))
    return work


def _load_kofam_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"missing {FEATURE_FILE} — link data/processed or run feature_filtering.py"
        )
    df = pd.read_csv(FEATURE_FILE, sep="\t", index_col=0)
    return df


def _model_extra_kwargs(
    model: str, phenotype: str, subset: str = "full"
) -> dict[str, Any]:
    """Per-model extra kwargs, such as groupings or cached tuned parameters."""
    if model == "glasso_kegg":
        from scripts.alternate.ml_comparison.groupings import (
            kegg_module_groupings_for_phenotype,
        )

        return {"groups": kegg_module_groupings_for_phenotype(phenotype)}
    if model == "lgbm_tuned":
        import json

        from scripts.alternate.ml_comparison.lgbm_tuning import cache_path_for

        path = cache_path_for(phenotype, subset)
        if path.exists():
            with path.open() as f:
                return {"tuned_params": json.load(f)}
        return {}
    if model in ("cb_tuned", "rf_tuned", "enet_tuned"):
        import json

        from scripts.alternate.ml_comparison.hpo_tuning import cache_path_for

        path = cache_path_for(model, phenotype, subset)
        if path.exists():
            with path.open() as f:
                return {"tuned_params": json.load(f)}
        return {}
    return {}


# Model name -> classifier alias, for models that reuse another model's fit but
# apply a different split-level transform.
PHYLO_RESTRICTED_BASELINES = {
    "cb_phylo": "cb_noeval",
    "enet_phylo": "enet",
    "lgbm_phylo": "lgbm",
}
# Residualised-feature variants: dataset-only residualization.
RESIDUALIZED_DATASET_ONLY = {
    "cb_resid_d": "cb_noeval",
    "lgbm_resid_d": "lgbm",
    "enet_resid_d": "enet",
}
# Residualised-feature variants: dataset + phylo PC residualization, restricted
# to GTDB-tree-covered genomes.
RESIDUALIZED_DATASET_PHYLO = {
    "cb_resid_dp": "cb_noeval",
    "lgbm_resid_dp": "lgbm",
    "enet_resid_dp": "enet",
}


def _maybe_transform_split(
    model: str, phenotype: str, split: dict[str, Any]
) -> dict[str, Any] | None:
    """Apply the per-model preprocessing transform.

    Returns None when the transform leaves too few samples to use (for example,
    phylo_glmm restricted to tree-covered genomes).
    """
    if model == "pclr":
        from scripts.alternate.ml_comparison.phylo_pcs import phylo_pc_adjust

        return phylo_pc_adjust(split)
    if model == "phylo_glmm":
        from scripts.alternate.ml_comparison.phylo_kinship import (
            transform_split_for_phylo_glmm,
        )

        return transform_split_for_phylo_glmm(split)
    if model in PHYLO_RESTRICTED_BASELINES:
        from scripts.alternate.ml_comparison.phylo_kinship import (
            restrict_split_to_covered,
        )

        return restrict_split_to_covered(split)
    if model in RESIDUALIZED_DATASET_ONLY:
        from scripts.alternate.ml_comparison.feature_residualization import (
            transform_split_residualize,
        )

        return transform_split_residualize(split, mode="dataset_only")
    if model in RESIDUALIZED_DATASET_PHYLO:
        from scripts.alternate.ml_comparison.feature_residualization import (
            transform_split_residualize,
        )

        return transform_split_residualize(split, mode="dataset_phylo")
    if model == "tabpfn":
        from scripts.alternate.ml_comparison.tabpfn_features import (
            select_stable_features,
        )

        return select_stable_features(phenotype, split, max_features=500)
    return split


def _resolve_model_alias(model: str) -> str:
    """Resolve a transform-only alias (e.g. cb_phylo -> cb_noeval) to the alias
    ``make_classifier`` accepts."""
    if model in PHYLO_RESTRICTED_BASELINES:
        return PHYLO_RESTRICTED_BASELINES[model]
    if model in RESIDUALIZED_DATASET_ONLY:
        return RESIDUALIZED_DATASET_ONLY[model]
    if model in RESIDUALIZED_DATASET_PHYLO:
        return RESIDUALIZED_DATASET_PHYLO[model]
    return model


def run(args) -> None:
    patch_make_classifier()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"[runner] loading KOFAM feature matrix from {FEATURE_FILE}...")
    feature_data = _load_kofam_features()
    print(
        f"[runner]   loaded {feature_data.shape[0]} samples x {feature_data.shape[1]} features"
    )

    print("[runner] loading splits...")
    split_data = load_split_data(
        base_dir=SPLITS_DIR,
        feature_file=FEATURE_FILE,
        split_types=list(set(args.split_types)),
    )

    concordant_resources = None
    if "concordant" in args.subsets:
        print("[runner] loading concordant-data resources...")
        concordant_resources = _load_concordant_resources()

    work = _build_work_list(args)
    print(f"[runner] total fits planned: {len(work)}")

    completed_since_aggregate = 0
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, (model, subset, split_type, phenotype, repeat) in enumerate(work):
        out = per_fit_path(model, subset, split_type, phenotype, repeat)
        if args.resume and is_done(out):
            n_skipped += 1
            continue

        key = f"{phenotype}_{repeat}" if split_type == "random_split" else None
        raw_split = None
        if split_type in split_data:
            if split_type == "random_split":
                raw_split = split_data[split_type].get(key)
            else:
                # The loader keys dataset/phylo splits by
                # f"{phenotype}_{folder_name}", with folder names taken verbatim.
                for cand_key, cand in split_data[split_type].items():
                    if cand_key.startswith(phenotype + "_") and cand_key.endswith(
                        repeat
                    ):
                        raw_split = cand
                        break

        if raw_split is None:
            n_failed += 1
            append_manifest(
                {
                    "status": "missing_split",
                    "model": model,
                    "phenotype": phenotype,
                    "subset": subset,
                    "split_type": split_type,
                    "repeat": repeat,
                }
            )
            continue

        try:
            materialized = _materialize_subset_split(
                raw_split=raw_split,
                subset=subset,
                phenotype=phenotype,
                concordant_resources=concordant_resources,
            )
            if materialized is None:
                payload = {
                    "status": "skipped_insufficient_samples",
                    "model": model,
                    "phenotype": phenotype,
                    "subset": subset,
                    "split_type": split_type,
                    "repeat": repeat,
                }
                atomic_write_json(out, payload)
                append_manifest(payload)
                n_skipped += 1
                continue

            materialized = _maybe_transform_split(model, phenotype, materialized)
            if materialized is None:
                payload = {
                    "status": "skipped_transform_dropped",
                    "model": model,
                    "phenotype": phenotype,
                    "subset": subset,
                    "split_type": split_type,
                    "repeat": repeat,
                }
                atomic_write_json(out, payload)
                append_manifest(payload)
                n_skipped += 1
                continue

            payload = _fit_one(
                model_type=model,
                phenotype=phenotype,
                subset=subset,
                split_type=split_type,
                repeat=repeat,
                split=materialized,
                extra_kwargs=_model_extra_kwargs(model, phenotype, subset),
            )
            atomic_write_json(out, payload)
            append_manifest(
                {
                    "status": "ok",
                    "model": model,
                    "phenotype": phenotype,
                    "subset": subset,
                    "split_type": split_type,
                    "repeat": repeat,
                    "balanced_accuracy": payload.get("balanced_accuracy"),
                    "fit_seconds": payload.get("fit_seconds"),
                }
            )
            n_done += 1
            completed_since_aggregate += 1
        except Exception as e:  # noqa: BLE001
            err_payload = {
                "status": "error",
                "model": model,
                "phenotype": phenotype,
                "subset": subset,
                "split_type": split_type,
                "repeat": repeat,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            with suppress(Exception):
                atomic_write_json(out.with_suffix(".error.json"), err_payload)
            append_manifest(
                {
                    k: err_payload[k]
                    for k in (
                        "status",
                        "model",
                        "phenotype",
                        "subset",
                        "split_type",
                        "repeat",
                        "error_type",
                        "error",
                    )
                }
            )
            n_failed += 1

        if completed_since_aggregate >= AGGREGATE_EVERY:
            aggregate_results()
            completed_since_aggregate = 0

        if (i + 1) % 5 == 0 or i + 1 == len(work):
            print(
                f"[runner] progress {i + 1}/{len(work)} done={n_done} "
                f"skipped={n_skipped} failed={n_failed}"
            )

    aggregate_results()
    print(f"[runner] final: done={n_done} skipped={n_skipped} failed={n_failed}")
    print(f"[runner] results -> {results_path()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        choices=["full", "concordant"],
    )
    p.add_argument(
        "--split-types",
        nargs="+",
        default=list(DEFAULT_SPLIT_TYPES),
        choices=["random_split", "dataset_split"],
    )
    p.add_argument("--phenotypes", nargs="+", default=list(PHENOTYPES))
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip fits whose per_fit json is already 'ok'",
    )
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
