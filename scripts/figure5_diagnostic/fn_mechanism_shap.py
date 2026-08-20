#!/usr/bin/env python3
"""Rank KOFAM genes by SHAP on the GapMind false negatives an fp_only model recovers.

For each phenotype and each leave-one-dataset-out fold: fit an ``fp_only`` and a
``concordant`` CatBoost model on the same split (only the train/val filter
differs), take the held-out false negatives (GapMind = 0, experiment = 1) that
the fp_only model predicts as growers, compute signed per-genome SHAP for both
models on those same genomes, and pool across folds. Each ranked KO is flagged
canonical or non-canonical against the GapMind-step and KEGG reference-map KO
sets.

Writes ``data/outputs/figure5_fn_mechanism/<phenotype>_fn_shap_ranking.csv``.

Run with::

    uv run python -m scripts.figure5_diagnostic.fn_mechanism_shap [options]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.figure5.fp_only_filter import build_filter_masks, filter_train_val
from scripts.ml_splits import load_split_data, perform_split_ml_with_model

try:  # KEGG reference-map KO sets.
    from scripts.tables.kegg_module_coverage import pathway_kos_for_phenotype
except Exception:  # pragma: no cover - fallback if the import moves.

    def pathway_kos_for_phenotype(phenotype: str) -> set[str]:  # type: ignore[misc]
        return set()


THREADS: int = int(os.environ.get("EXPERIMENT_THREADS", "4"))

SPLITS_DIR: Path = Path("data/processed/train_test_splits")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
KOFAM_FEATURE_FILE: Path = Path(
    "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
GAPMIND_STEP_FILE: Path = Path("data/interim/features/combined_datasets/gapmind.tsv")
KO_DICT_FILE: Path = Path("data/external/mapping/KO_dictionary.json")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
DEFAULT_OUT_DIR: Path = Path("data/outputs/figure5_fn_mechanism")

RANDOM_STATE: int = 42
DEFAULT_PHENOTYPES: tuple[str, ...] = ("Fructose", "Mannose")
TOP_K: int = 25

# GapMind step pseudo-names the gene-symbol crosswalk cannot resolve on its own.
STEP_ALIAS_TO_KO: dict[str, set[str]] = {
    "man-isomerase": {"K29027"},  # D-mannose isomerase (yihS)
    "mannokinase": {"K00847", "K00845"},  # ROK/hexose kinases to M6P
    "1pfk": {"K00882"},  # 1-phosphofructokinase (fruK synonym)
}

KO_RE: re.Pattern[str] = re.compile(r"K\d{5}")


def load_ko_descriptions(path: Path) -> dict[str, str]:
    """Load the KO -> human-readable name/symbol/EC map.

    Parameters
    ----------
    path : Path
        Path to ``KO_dictionary.json`` (``term_hash`` keyed by ``KO:K#####``).

    Returns
    -------
    dict[str, str]
        Mapping from bare KO id (``K#####``) to its ``symbol; description [EC:...]``
        string.
    """
    raw = json.loads(path.read_text())
    term_hash: Mapping[str, Mapping[str, str]] = raw["term_hash"]
    return {v["id"]: v["name"] for v in term_hash.values()}


def build_symbol_to_ko(ko_descriptions: Mapping[str, str]) -> dict[str, set[str]]:
    """Index gene symbols to the KO(s) that carry them.

    Each KO name is of the form ``"sym1, sym2, name; longer description [EC:...]"``.
    The gene symbols are the comma-separated tokens before the first semicolon.

    Parameters
    ----------
    ko_descriptions : Mapping[str, str]
        Output of :func:`load_ko_descriptions`.

    Returns
    -------
    dict[str, set[str]]
        Lower-cased gene symbol -> set of KO ids.
    """
    symbol_to_ko: dict[str, set[str]] = {}
    for ko, name in ko_descriptions.items():
        head = name.split(";", 1)[0]
        for token in head.split(","):
            symbol = token.strip().lower()
            if symbol:
                symbol_to_ko.setdefault(symbol, set()).add(ko)
    return symbol_to_ko


def gapmind_step_symbols(step_columns: Sequence[str], phenotype: str) -> list[str]:
    """Extract the bare GapMind step names for a phenotype.

    Parameters
    ----------
    step_columns : Sequence[str]
        Column names of the GapMind step matrix (``<Phenotype>-<step>``).
    phenotype : str
        Phenotype (substrate) name.

    Returns
    -------
    list[str]
        Step names with the ``<Phenotype>-`` prefix stripped.
    """
    prefix = f"{phenotype}-"
    return [c[len(prefix) :] for c in step_columns if c.startswith(prefix)]


def canonical_ko_set(
    phenotype: str,
    step_columns: Sequence[str],
    symbol_to_ko: Mapping[str, set[str]],
) -> tuple[set[str], set[str], list[str]]:
    """Build the canonical KO set for a substrate and report crosswalk coverage.

    Canonical = KOs backing any GapMind step for the phenotype (gene-symbol
    crosswalk plus the alias supplement) union the KEGG reference-map KOs.

    Parameters
    ----------
    phenotype : str
        Phenotype (substrate) name.
    step_columns : Sequence[str]
        Column names of the GapMind step matrix.
    symbol_to_ko : Mapping[str, set[str]]
        Output of :func:`build_symbol_to_ko`.

    Returns
    -------
    tuple[set[str], set[str], list[str]]
        ``(canonical, gapmind_kos, unmapped_steps)`` where ``canonical`` is the
        full canonical KO set, ``gapmind_kos`` is the GapMind-step subset only,
        and ``unmapped_steps`` lists GapMind steps with no KO match.
    """
    steps = gapmind_step_symbols(step_columns, phenotype)
    gapmind_kos: set[str] = set()
    unmapped: list[str] = []
    for step in steps:
        key = step.lower()
        matched = set(symbol_to_ko.get(key, set())) | set(
            STEP_ALIAS_TO_KO.get(key, set())
        )
        if matched:
            gapmind_kos |= matched
        else:
            unmapped.append(step)
    kegg_kos = pathway_kos_for_phenotype(phenotype)
    canonical = gapmind_kos | kegg_kos
    return canonical, gapmind_kos, unmapped


def per_genome_shap(
    model: CatBoostClassifier,
    x_subset: pd.DataFrame,
    feature_order: Sequence[str],
    threads: int,
) -> pd.DataFrame:
    """Return the signed per-genome SHAP matrix (toward the positive class).

    Parameters
    ----------
    model : CatBoostClassifier
        Fitted CatBoost model.
    x_subset : pd.DataFrame
        Genomes (rows) to explain; columns are aligned to ``feature_order``.
    feature_order : Sequence[str]
        Training feature column order the model expects.
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    pd.DataFrame
        ``(n_genomes, n_features)`` signed SHAP values, indexed by genome id.
    """
    aligned = x_subset.reindex(columns=list(feature_order), fill_value=0)
    pool = Pool(data=aligned)
    shap = model.get_feature_importance(
        data=pool, type="ShapValues", thread_count=threads
    )
    shap = shap[:, :-1]  # drop the base-value column
    return pd.DataFrame(shap, index=aligned.index, columns=list(feature_order))


def parse_held_out(key: str) -> str | None:
    """Extract the held-out dataset from a ``dataset_split`` key.

    Parameters
    ----------
    key : str
        Split key of the form ``"<phenotype>_train(...),test(<dataset>)"``.

    Returns
    -------
    str | None
        Held-out dataset name, or ``None`` if the key does not match.
    """
    match = re.search(r"test\(([^)]+)\)", key)
    return match.group(1) if match else None


def analyse_phenotype(
    phenotype: str,
    dataset_splits: Mapping[str, Mapping[str, pd.DataFrame | pd.Series]],
    masks: Mapping[str, set[str]],
    kofam_features: pd.DataFrame,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    """Run the fp_only-versus-concordant FN-partition SHAP analysis for one phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype (substrate) name.
    dataset_splits : Mapping
        The ``dataset_split`` dictionary from :func:`load_split_data`.
    masks : Mapping[str, set[str]]
        Output of :func:`build_filter_masks` for the phenotype.
    kofam_features : pd.DataFrame
        Reduced KOFAM feature matrix (genomes x KOs), for prevalence stats.
    threads : int
        CatBoost ``thread_count``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]] | None
        ``(per_ko_ranking, diagnostics)`` or ``None`` if no recovered false
        negatives were found across folds.
    """
    fn_set = masks["false_negative"]
    fp_fn_shap: list[pd.DataFrame] = []  # fp_only model SHAP on recovered FN genomes
    cc_fn_shap: list[pd.DataFrame] = []  # concordant model SHAP on the same genomes
    recovered_fn: list[str] = []
    fold_report: list[dict[str, object]] = []

    keys = sorted(k for k in dataset_splits if k.split("_", 1)[0] == phenotype)
    for key in keys:
        held_out = parse_held_out(key)
        split = dataset_splits[key]
        x_test: pd.DataFrame = split["X_test"]  # type: ignore[assignment]

        test_fn = sorted(set(x_test.index) & fn_set)
        if not test_fn:
            continue

        fitted: dict[str, CatBoostClassifier] = {}
        for mode in ("fp_only", "concordant"):
            filtered = filter_train_val(split, mode, masks)  # type: ignore[arg-type]
            if filtered is None:
                break
            x_tr, y_tr, x_val, y_val = filtered
            _, model = perform_split_ml_with_model(
                x_tr,
                y_tr,
                x_val,
                y_val,
                split["X_test"],  # type: ignore[arg-type]
                split["y_test"],  # type: ignore[arg-type]
                model_type="cb",
                scoring=["balanced_accuracy"],
                random_state=RANDOM_STATE,
                thread_count=threads,
            )
            fitted[mode] = model  # type: ignore[assignment]
        if len(fitted) != 2:
            continue

        fp_model = fitted["fp_only"]
        features = list(fp_model.feature_names_)
        x_fn = x_test.loc[test_fn]
        preds = fp_model.predict(x_fn.reindex(columns=features, fill_value=0))
        recovered = [
            g for g, p in zip(test_fn, np.asarray(preds).ravel()) if int(p) == 1
        ]
        if not recovered:
            fold_report.append(
                {"held_out": held_out, "n_test_fn": len(test_fn), "n_recovered": 0}
            )
            continue

        x_rec = x_test.loc[recovered]
        fp_fn_shap.append(per_genome_shap(fp_model, x_rec, features, threads))
        cc_fn_shap.append(
            per_genome_shap(fitted["concordant"], x_rec, features, threads)
        )
        recovered_fn.extend(recovered)
        fold_report.append(
            {
                "held_out": held_out,
                "n_test_fn": len(test_fn),
                "n_recovered": len(recovered),
            }
        )

    if not fp_fn_shap:
        return None

    fp_shap = pd.concat(fp_fn_shap).groupby(level=0).mean()
    cc_shap = (
        pd.concat(cc_fn_shap).reindex(columns=fp_shap.columns).groupby(level=0).mean()
    )

    ranking = pd.DataFrame(
        {
            "fp_mean_signed_shap": fp_shap.mean(axis=0),
            "fp_mean_abs_shap": fp_shap.abs().mean(axis=0),
            "cc_mean_signed_shap": cc_shap.mean(axis=0),
            "cc_mean_abs_shap": cc_shap.abs().mean(axis=0),
        }
    )
    ranking["fp_minus_cc_abs_shap"] = (
        ranking["fp_mean_abs_shap"] - ranking["cc_mean_abs_shap"]
    )

    diagnostics: dict[str, object] = {
        "n_recovered_fn": len(set(recovered_fn)),
        "recovered_genomes": sorted(set(recovered_fn)),
        "folds": fold_report,
    }
    return ranking, diagnostics


def add_prevalence(
    ranking: pd.DataFrame,
    recovered_genomes: Sequence[str],
    phenotype: str,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    kofam_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach KO prevalence in recovered-FN versus concordant-negative genomes.

    Parameters
    ----------
    ranking : pd.DataFrame
        Per-KO SHAP ranking (index = KO id).
    recovered_genomes : Sequence[str]
        Recovered false-negative genome ids.
    phenotype : str
        Phenotype name.
    gapmind_predictions : pd.DataFrame
        GapMind loose predictions (genomes x phenotypes).
    experimental_phenotypes : pd.DataFrame
        Experimental phenotypes (genomes x phenotypes).
    kofam_features : pd.DataFrame
        Reduced KOFAM feature matrix.

    Returns
    -------
    pd.DataFrame
        ``ranking`` with ``prev_fn``, ``prev_conc_neg`` and ``prev_enrichment``
        columns added.
    """
    common = gapmind_predictions.index.intersection(experimental_phenotypes.index)
    gm = gapmind_predictions.loc[common, phenotype]
    exp = experimental_phenotypes.loc[common, phenotype]
    conc_neg = sorted(set(gm[(gm == 0) & (exp == 0)].index) & set(kofam_features.index))
    rec = sorted(set(recovered_genomes) & set(kofam_features.index))

    kos = [k for k in ranking.index if k in kofam_features.columns]
    prev_fn = (
        (kofam_features.loc[rec, kos] > 0).mean(axis=0)
        if rec
        else pd.Series(dtype=float)
    )
    prev_neg = (
        (kofam_features.loc[conc_neg, kos] > 0).mean(axis=0)
        if conc_neg
        else pd.Series(dtype=float)
    )
    ranking = ranking.copy()
    ranking["prev_fn"] = prev_fn.reindex(ranking.index)
    ranking["prev_conc_neg"] = prev_neg.reindex(ranking.index)
    ranking["prev_enrichment"] = ranking["prev_fn"] - ranking["prev_conc_neg"]
    return ranking


def annotate(
    ranking: pd.DataFrame,
    canonical: set[str],
    gapmind_kos: set[str],
    ko_descriptions: Mapping[str, str],
) -> pd.DataFrame:
    """Add canonical/non-canonical flags and KO descriptions.

    Parameters
    ----------
    ranking : pd.DataFrame
        Per-KO ranking (index = KO id).
    canonical : set[str]
        Canonical KO set (GapMind crosswalk union KEGG map).
    gapmind_kos : set[str]
        GapMind-step KO subset only.
    ko_descriptions : Mapping[str, str]
        KO -> description map.

    Returns
    -------
    pd.DataFrame
        Annotated ranking with ``is_canonical``, ``in_gapmind_steps`` and
        ``description`` columns.
    """
    ranking = ranking.copy()
    ranking["in_gapmind_steps"] = ranking.index.isin(gapmind_kos)
    ranking["is_canonical"] = ranking.index.isin(canonical)
    ranking["description"] = [ko_descriptions.get(k, "") for k in ranking.index]
    return ranking


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--phenotypes",
        type=str,
        default=",".join(DEFAULT_PHENOTYPES),
        help="Comma-separated phenotypes (default: Fructose,Mannose).",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the FN-partition SHAP analysis for the requested phenotypes."""
    args = parse_args()
    phenotypes = [p.strip() for p in args.phenotypes.split(",") if p.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Threads: {args.threads} | phenotypes: {phenotypes}")

    print("Loading KO dictionary and GapMind step columns...")
    ko_descriptions = load_ko_descriptions(KO_DICT_FILE)
    symbol_to_ko = build_symbol_to_ko(ko_descriptions)
    step_columns = pd.read_csv(GAPMIND_STEP_FILE, sep="\t", nrows=0).columns.tolist()

    print("Loading GapMind predictions, experimental phenotypes, KOFAM features...")
    gapmind_predictions = load_gapmind_predictions(GAPMIND_FILE)
    experimental_phenotypes = load_experimental_phenotypes(PHENOTYPE_DIR)
    kofam_features = pd.read_csv(KOFAM_FEATURE_FILE, sep="\t", index_col=0)
    kofam_features.index = kofam_features.index.astype(str)

    print("Loading dataset splits (KOFAM features)...")
    dataset_splits = load_split_data(
        base_dir=SPLITS_DIR,
        split_types=["dataset_split"],
        feature_file=KOFAM_FEATURE_FILE,
    )["dataset_split"]

    for phenotype in phenotypes:
        print(f"\n{'=' * 70}\n{phenotype}\n{'=' * 70}")
        canonical, gapmind_kos, unmapped = canonical_ko_set(
            phenotype, step_columns, symbol_to_ko
        )
        print(
            f"  GapMind steps mapped to {len(gapmind_kos)} KOs; "
            f"{len(unmapped)} steps unmapped: {unmapped}"
        )
        print(f"  Canonical KO set (GapMind union KEGG map): {len(canonical)} KOs")

        masks = build_filter_masks(
            gapmind_predictions, experimental_phenotypes, phenotype
        )
        result = analyse_phenotype(
            phenotype, dataset_splits, masks, kofam_features, args.threads
        )
        if result is None:
            print("  No recovered false negatives; skipping.")
            continue
        ranking, diagnostics = result
        ranking = add_prevalence(
            ranking,
            diagnostics["recovered_genomes"],  # type: ignore[arg-type]
            phenotype,
            gapmind_predictions,
            experimental_phenotypes,
            kofam_features,
        )
        ranking = annotate(ranking, canonical, gapmind_kos, ko_descriptions)
        ranking = ranking.sort_values("fp_mean_signed_shap", ascending=False)

        out_path = args.out_dir / f"{phenotype}_fn_shap_ranking.csv"
        ranking.to_csv(out_path)
        print(
            f"  Recovered FN genomes pooled: {diagnostics['n_recovered_fn']} "
            f"(folds: {diagnostics['folds']})"
        )
        print(f"  Wrote {out_path}")

        top = ranking.head(args.top_k)
        noncanon = top[~top["is_canonical"]]
        print(
            f"\n  Top {args.top_k} KOs by SHAP-toward-growth on recovered FN genomes:"
        )
        print(
            top[
                [
                    "fp_mean_signed_shap",
                    "cc_mean_signed_shap",
                    "prev_fn",
                    "prev_conc_neg",
                    "is_canonical",
                    "in_gapmind_steps",
                    "description",
                ]
            ].to_string(max_colwidth=60)
        )
        print(
            f"\n  --> {len(noncanon)} of top {args.top_k} are NON-canonical candidates."
        )


if __name__ == "__main__":
    main()
