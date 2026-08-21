"""Diagnose why some KOFAM phenotypes underperform on Figure 5A.

Reports per-phenotype cross-dataset balanced accuracy (KOFAM versus GapMind),
the single-feature AUROC ceiling on the pooled concordant subset, and the top
stable SHAP features per phenotype.

Writes ``ba_table.csv``, ``kofam_signal.csv``, ``gapmind_signal.csv`` and
``shap_top.csv`` under ``scripts/figure5_diagnostic/``.

Run with::

    uv run python -m scripts.figure5_diagnostic.feature_signal_analysis
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.minority_filter import (
    concordant_minority_counts,
    filter_by_minority,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS: tuple[str, ...] = ("atleaf", "lit", "marine", "pmi")

KOFAM_CSV: Path = REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results.csv"
GAPMIND_RAW_CSV: Path = (
    REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results_gapmind_raw.csv"
)
KOFAM_FEATURES: Path = (
    REPO_ROOT / "data/processed/features_reduced/combined_datasets/kofam.tsv"
)
GAPMIND_RAW_FEATURES: Path = (
    REPO_ROOT / "data/interim/features/combined_datasets/gapmind.tsv"
)
GAPMIND_PRED_TSV: Path = REPO_ROOT / "data/outputs/figure2/gapmind_phenotypes_loose.tsv"
PHENOTYPE_DIR: Path = REPO_ROOT / "data/processed/phenotypes"
SHAP_JSON: Path = (
    REPO_ROOT / "data/outputs/figure5/figure5b_combined_splits_shap_features.json"
)

POOR_PHENOTYPES: tuple[str, ...] = ("Glucose", "Serine", "Galacturonic-Acid")
GOOD_PHENOTYPES: tuple[str, ...] = ("Histidine", "m-Inositol", "Mannitol")
ALL_PHENOTYPES: tuple[str, ...] = POOR_PHENOTYPES + GOOD_PHENOTYPES


def _aggregate_ba(csv: Path, split_type: str) -> pd.DataFrame:
    """Return per-phenotype mean/std BA after minority filter."""
    df = pd.read_csv(csv)
    if split_type == "dataset_split":
        df = df[df["split_type"] == "dataset_split"]
        df = filter_by_minority(df, concordant_minority_counts())
    else:
        df = df[df["split_type"] == split_type]
    agg = (
        df.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": f"ba_mean_{split_type}",
                "std": f"ba_std_{split_type}",
                "count": f"n_{split_type}",
            }
        )
    )
    return agg


def build_ba_table() -> pd.DataFrame:
    kofam_ds = _aggregate_ba(KOFAM_CSV, "dataset_split")
    kofam_rs = _aggregate_ba(KOFAM_CSV, "random_split")
    gm_ds = _aggregate_ba(GAPMIND_RAW_CSV, "dataset_split")
    table = kofam_ds.join(gm_ds, lsuffix="_kofam", rsuffix="_gapmind").join(
        kofam_rs[["ba_mean_random_split"]]
    )
    table["delta_gapmind_minus_kofam"] = (
        table["ba_mean_dataset_split_gapmind"] - table["ba_mean_dataset_split_kofam"]
    )
    return table


def load_concordant_labels(phenotype: str) -> pd.Series:
    """Return a Series of concordant binary labels keyed by genomeID,
    pooled across all four datasets."""
    gapmind = pd.read_csv(
        GAPMIND_PRED_TSV, sep="\t", index_col=0, dtype={"genomeID": str}
    )
    if phenotype not in gapmind.columns:
        raise KeyError(phenotype)
    labels = []
    for ds in DATASETS:
        p = PHENOTYPE_DIR / ds / f"{phenotype}.tsv"
        if not p.exists():
            continue
        s = (
            pd.read_csv(p, sep="\t", dtype={"genomeID": str})
            .set_index("genomeID")[phenotype]
            .dropna()
        )
        labels.append(s)
    if not labels:
        return pd.Series(dtype=int)
    pooled = pd.concat(labels)
    pooled = pooled[~pooled.index.duplicated(keep="first")]
    common = pooled.index.intersection(gapmind.index)
    pooled = pooled.loc[common]
    concordant_mask = pooled == gapmind.loc[common, phenotype]
    return pooled[concordant_mask].astype(int)


def variance_filter(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop features whose variance is below ``threshold``."""
    var = X.var()
    return X.loc[:, var >= threshold]


def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop features that are >=threshold correlated with another kept feature."""
    if X.shape[1] <= 1:
        return X
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if (upper[col] >= threshold).any()]
    return X.drop(columns=to_drop)


def restrict_to_phenotype_columns(
    features: pd.DataFrame, phenotype: str
) -> pd.DataFrame:
    """For GapMind-feature matrix, keep columns whose prefix matches the phenotype.

    GapMind column names look like ``"Glucose-ptsG"``; for KOFAM there is no
    phenotype prefix so this is a no-op.
    """
    cols = [c for c in features.columns if c.startswith(f"{phenotype}-")]
    if not cols:
        return features
    return features[cols]


def single_feature_auroc(
    X: pd.DataFrame, y: pd.Series, top_n: int = 5
) -> tuple[float, list[tuple[str, float]]]:
    """Return (max AUROC, top-N (feature, AUROC) list).

    Skips features with zero variance after restriction to ``y.index``.
    """
    common = X.index.intersection(y.index)
    X_ = X.loc[common]
    y_ = y.loc[common]
    if y_.nunique() < 2 or len(y_) < 10:
        return float("nan"), []
    scores: list[tuple[str, float]] = []
    for col in X_.columns:
        x = X_[col].values
        if np.unique(x).size < 2:
            continue
        try:
            auc = roc_auc_score(y_.values, x)
        except Exception:
            continue
        # One-sided AUROC: either direction is informative for a single feature.
        scores.append((col, max(auc, 1.0 - auc)))
    if not scores:
        return float("nan"), []
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[0][1], scores[:top_n]


def feature_signal_table(
    feature_file: Path,
    phenotypes: list[str],
    *,
    restrict_prefix: bool = False,
    apply_corr_var_filter: bool = True,
) -> pd.DataFrame:
    """Compute per-phenotype single-feature AUROC ceiling and feature counts."""
    print(f"Loading features from {feature_file}...")
    X = pd.read_csv(feature_file, sep="\t", index_col=0, dtype={"genomeID": str})
    print(f"  raw shape: {X.shape}")

    rows = []
    for phen in phenotypes:
        try:
            y = load_concordant_labels(phen)
        except KeyError:
            print(f"  {phen}: not in GapMind preds, skip")
            continue
        if len(y) < 20 or y.nunique() < 2:
            print(f"  {phen}: too few concordant labels ({len(y)})")
            continue
        common = X.index.intersection(y.index)
        X_p = X.loc[common]
        y_p = y.loc[common]

        if restrict_prefix:
            X_p = restrict_to_phenotype_columns(X_p, phen)
        n_features_pre = X_p.shape[1]

        if apply_corr_var_filter:
            X_p = variance_filter(X_p, 0.01)
            n_after_var = X_p.shape[1]
            X_p = correlation_filter(X_p, 0.95)
            n_after_corr = X_p.shape[1]
        else:
            n_after_var = n_features_pre
            n_after_corr = X_p.shape[1]

        max_auc, top = single_feature_auroc(X_p, y_p, top_n=5)
        rows.append(
            {
                "phenotype": phen,
                "n_concordant": len(y_p),
                "positive_fraction": float(y_p.mean()),
                "n_features_pre": n_features_pre,
                "n_after_var": n_after_var,
                "n_after_corr": n_after_corr,
                "max_single_feature_auroc": max_auc,
                "top5_features": "; ".join(f"{n}({v:.3f})" for n, v in top),
            }
        )
    return pd.DataFrame(rows).set_index("phenotype")


def top_shap_features(phenotypes: list[str]) -> pd.DataFrame:
    with open(SHAP_JSON) as f:
        j = json.load(f)
    rows = []
    for phen in phenotypes:
        feats: list[str] = []
        for k, v in j.items():
            if k.split("_")[0] == phen:
                feats.extend(v)
        c = Counter(feats)
        rows.append(
            {
                "phenotype": phen,
                "n_unique_stable_features": len(set(feats)),
                "top5_stable": "; ".join(f"{f}(x{n})" for f, n in c.most_common(5)),
            }
        )
    return pd.DataFrame(rows).set_index("phenotype")


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("=" * 70)
    print("PART 1: Per-phenotype BA — KOFAM vs GapMind (raw), concordant")
    print("=" * 70)
    ba_table = build_ba_table()
    cols = [
        "ba_mean_dataset_split_kofam",
        "ba_std_dataset_split_kofam",
        "ba_mean_dataset_split_gapmind",
        "ba_std_dataset_split_gapmind",
        "delta_gapmind_minus_kofam",
        "ba_mean_random_split",
    ]
    out = ba_table[cols].round(3).sort_values("ba_mean_dataset_split_kofam")
    print(out)
    out.to_csv(REPO_ROOT / "scripts/figure5_diagnostic/ba_table.csv")

    print()
    print("=" * 70)
    print("PART 2: Single-feature AUROC ceiling on pooled concordant subset")
    print("=" * 70)
    print("\n[KOFAM features — global matrix, var@0.01 + corr@0.95]")
    kofam_sig = feature_signal_table(
        KOFAM_FEATURES,
        list(ALL_PHENOTYPES),
        restrict_prefix=False,
        apply_corr_var_filter=True,
    )
    print(kofam_sig)
    kofam_sig.to_csv(REPO_ROOT / "scripts/figure5_diagnostic/kofam_signal.csv")

    print(
        "\n[GapMind raw step features — restricted to '<phenotype>-*' cols, NO corr/var filter]"
    )
    gm_sig = feature_signal_table(
        GAPMIND_RAW_FEATURES,
        list(POOR_PHENOTYPES) + list(GOOD_PHENOTYPES),
        restrict_prefix=True,
        apply_corr_var_filter=False,
    )
    print(gm_sig)
    gm_sig.to_csv(REPO_ROOT / "scripts/figure5_diagnostic/gapmind_signal.csv")

    print()
    print("=" * 70)
    print("PART 3: Top stable SHAP KOFAM features per phenotype")
    print("=" * 70)
    shap_tbl = top_shap_features(list(ALL_PHENOTYPES))
    print(shap_tbl)
    shap_tbl.to_csv(REPO_ROOT / "scripts/figure5_diagnostic/shap_top.csv")


if __name__ == "__main__":
    main()
