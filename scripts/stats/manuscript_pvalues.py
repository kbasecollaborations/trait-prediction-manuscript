#!/usr/bin/env python3
"""Recompute every p-value cited in the manuscript and apply BH correction.

Writes a TSV of each test's raw p-value, paired sample size, and BH q-value.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from scripts.minority_filter import (
    concordant_minority_counts,
    discordant_minority_counts,
    filter_by_minority,
    full_test_minority_counts,
)

OUTPUT_DIR = Path("data/outputs/stats")
OUTPUT_FILE = OUTPUT_DIR / "manuscript_pvalues.tsv"

COMMON_PHENOTYPES: tuple[str, ...] = (
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


def _paired_wilcoxon(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    """Two-sided paired Wilcoxon signed-rank on aligned phenotype means."""
    common = sorted(set(left.index) & set(right.index) & set(COMMON_PHENOTYPES))
    if len(common) < 2:
        return float("nan"), len(common)
    res = wilcoxon(
        left.loc[common].to_numpy(),
        right.loc[common].to_numpy(),
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(res.pvalue), len(common)


def _per_phenotype_mean(df: pd.DataFrame) -> pd.Series:
    return df.groupby("phenotype")["balanced_accuracy"].mean()


def _load_gapmind_random() -> pd.Series:
    df = pd.read_csv(
        "data/outputs/figure3/gapmind_random_split_metrics.tsv", sep="\t"
    )
    return _per_phenotype_mean(df)


def _load_gapmind_dataset() -> pd.Series:
    df = pd.read_csv(
        "data/outputs/figure3/gapmind_dataset_split_metrics.tsv", sep="\t"
    )
    return _per_phenotype_mean(df)


def test_fig3a_ml_vs_gapmind_random() -> tuple[float, int]:
    """Figure 3A: full-data ML (KOFAM) vs GapMind, random holdout."""
    ml = pd.read_csv("data/outputs/figure3/ml_results.csv")
    ml = ml[ml["split_type"] == "random_split"]
    ml_mean = _per_phenotype_mean(ml)
    return _paired_wilcoxon(ml_mean, _load_gapmind_random())


def test_fig5a_kofam_vs_gapmind_ceiling_dataset() -> tuple[float, int]:
    """Figure 5A: concordant-trained KOFAM vs GapMind step features (cross-dataset)."""
    kofam = pd.read_csv("data/outputs/figure5/figure5a_concordant_ml_results.csv")
    kofam = filter_by_minority(kofam, concordant_minority_counts())
    kofam = kofam[kofam["split_type"] == "dataset_split"]
    gm_raw = Path("data/outputs/figure5/figure5a_concordant_ml_results_gapmind_raw.csv")
    gm_path = (
        gm_raw if gm_raw.exists()
        else Path("data/outputs/figure5/figure5a_concordant_ml_results_gapmind.csv")
    )
    gm = pd.read_csv(gm_path)
    gm = filter_by_minority(gm, concordant_minority_counts())
    gm = gm[gm["split_type"] == "dataset_split"]
    return _paired_wilcoxon(_per_phenotype_mean(gm), _per_phenotype_mean(kofam))


def test_fig5c_random() -> tuple[float, int]:
    """Figure 5C random holdout: concordant ML vs GapMind on the full test."""
    ml = pd.read_csv(
        "data/outputs/figure5/figure5c_concordant_train_different_test.csv"
    )
    ml = ml[(ml["test_type"] == "full") & (ml["split_type"] == "random_split")]
    ml = filter_by_minority(ml, full_test_minority_counts())
    ml_mean = _per_phenotype_mean(ml)
    return _paired_wilcoxon(ml_mean, _load_gapmind_random())


def test_fig5c_dataset() -> tuple[float, int]:
    """Figure 5C cross-dataset: concordant ML vs GapMind on the full test."""
    ml = pd.read_csv(
        "data/outputs/figure5/figure5c_concordant_train_different_test.csv"
    )
    ml = ml[(ml["test_type"] == "full") & (ml["split_type"] == "dataset_split")]
    ml = filter_by_minority(ml, full_test_minority_counts())
    ml_mean = _per_phenotype_mean(ml)
    return _paired_wilcoxon(ml_mean, _load_gapmind_dataset())


def test_fig5d_fn_vs_fp_rescue() -> tuple[float, int]:
    """Figure 5D: per-phenotype FN rescue rate vs FP rescue rate (concordant model)."""
    per_sample = pd.read_csv("data/outputs/figure6/figure6_per_sample.tsv", sep="\t")
    disc = per_sample[per_sample["gapmind_pred"] != per_sample["y_true"]].copy()
    disc["error_type"] = np.where(disc["y_true"] == 1, "FN", "FP")
    disc["rescued"] = (disc["y_pred"] == disc["y_true"]).astype(int)
    rates = (
        disc.groupby(["phenotype", "error_type"])["rescued"]
        .mean()
        .unstack("error_type")
        .dropna()
    )
    return _paired_wilcoxon(rates["FN"], rates["FP"])


def _phenotype_aggregates() -> pd.DataFrame:
    """Per-phenotype mean confidence, fraction high-confidence, and cross-dataset BA."""
    per_sample = pd.read_csv("data/outputs/figure6/figure6_per_sample.tsv", sep="\t")
    high_conf = (per_sample["confidence"] >= 0.8).astype(float)
    per_sample["high_conf"] = high_conf
    agg = per_sample.groupby("phenotype").agg(
        mean_confidence=("confidence", "mean"),
        frac_high_conf=("high_conf", "mean"),
    )
    ml = pd.read_csv(
        "data/outputs/figure5/figure5c_concordant_train_different_test.csv"
    )
    ml = ml[(ml["test_type"] == "full") & (ml["split_type"] == "dataset_split")]
    ml = filter_by_minority(ml, full_test_minority_counts())
    agg["cross_dataset_ba"] = _per_phenotype_mean(ml)
    return agg.dropna(subset=["cross_dataset_ba"])


def test_fig6_spearman_confidence_vs_ba() -> tuple[float, int]:
    agg = _phenotype_aggregates()
    rho, p = spearmanr(agg["mean_confidence"], agg["cross_dataset_ba"])
    return float(p), len(agg)


def test_fig6_spearman_frac_high_vs_ba() -> tuple[float, int]:
    agg = _phenotype_aggregates()
    rho, p = spearmanr(agg["frac_high_conf"], agg["cross_dataset_ba"])
    return float(p), len(agg)


def test_fig6_spearman_novelty_vs_ba() -> tuple[float, int] | None:
    """Spearman between feature-space novelty and cross-dataset BA per phenotype."""
    novelty_path = Path("data/outputs/figure6/figure6_per_sample.tsv")
    sample = pd.read_csv(novelty_path, sep="\t")
    if "ood" not in sample.columns and "novelty" not in sample.columns:
        return None
    novelty_col = "ood" if "ood" in sample.columns else "novelty"
    agg = sample.groupby("phenotype")[novelty_col].mean()
    ml = pd.read_csv(
        "data/outputs/figure5/figure5c_concordant_train_different_test.csv"
    )
    ml = ml[(ml["test_type"] == "full") & (ml["split_type"] == "dataset_split")]
    ml = filter_by_minority(ml, full_test_minority_counts())
    ba = _per_phenotype_mean(ml)
    common = sorted(set(agg.index) & set(ba.index))
    rho, p = spearmanr(agg.loc[common], ba.loc[common])
    return float(p), len(common)


def test_fig6c_low_conf_vs_random() -> tuple[float, int]:
    """Figure 6C: low-confidence vs random selection paired by (phen, held-out, seed)."""
    df = pd.read_csv("data/outputs/figure6/figure6_prioritization.tsv", sep="\t")
    pivot = (
        df.pivot_table(
            index=["phenotype", "held_out_dataset", "seed"],
            columns="strategy",
            values="delta_balanced_accuracy",
        )
        .dropna(subset=["low_confidence", "random"])
    )
    res = wilcoxon(
        pivot["low_confidence"].to_numpy(),
        pivot["random"].to_numpy(),
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(res.pvalue), len(pivot)


def test_fig7d_combined_vs_filtered(split: str) -> tuple[float, int]:
    """Figure 7D: combined features vs phenotype-filtered features (per phenotype)."""
    df = pd.read_csv("data/outputs/figure7/figure7d_all_results.csv")
    df = df[df["split_type"] == split]
    summary = (
        df.groupby(["phenotype", "experiment"])["balanced_accuracy"]
        .mean()
        .unstack("experiment")
    )
    needed = {"combined", "phenotype_filtered"}
    if not needed.issubset(summary.columns):
        return float("nan"), 0
    summary = summary.dropna(subset=list(needed))
    return _paired_wilcoxon(summary["combined"], summary["phenotype_filtered"])


def benjamini_hochberg(pvals: Iterable[float]) -> list[float]:
    """Return BH-adjusted q-values for a list of p-values (NaNs propagate)."""
    arr = np.asarray(list(pvals), dtype=float)
    n = arr.size
    order = np.argsort(arr)
    ranked = arr[order]
    finite = ~np.isnan(ranked)
    adj = np.empty(n)
    adj.fill(np.nan)
    # Compute q for finite entries only
    finite_idx = np.where(finite)[0]
    m = finite_idx.size
    raw = ranked[finite]
    bh = raw * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    adj_finite = np.empty(m)
    adj_finite[:] = bh
    adj[finite_idx] = adj_finite
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(label: str, where: str, result: tuple[float, int] | None) -> None:
        if result is None:
            return
        p, n = result
        rows.append(
            {
                "test_id": label,
                "manuscript_location": where,
                "p_value": p,
                "n_paired": n,
            }
        )

    add(
        "T1_fig3a_ml_vs_gapmind_random",
        "Results §33 / Fig 3A",
        test_fig3a_ml_vs_gapmind_random(),
    )
    add(
        "T2_fig5a_kofam_vs_gapmind_ceiling",
        "Results §69 / Fig 5A",
        test_fig5a_kofam_vs_gapmind_ceiling_dataset(),
    )
    add(
        "T3_fig5c_concordant_vs_gapmind_random",
        "Results §78 / Fig 5C",
        test_fig5c_random(),
    )
    add(
        "T4_fig5d_fn_vs_fp_rescue",
        "Results §82 / Fig 5D",
        test_fig5d_fn_vs_fp_rescue(),
    )
    add(
        "T5_fig5c_concordant_vs_gapmind_dataset",
        "Results §84 / Fig 5C",
        test_fig5c_dataset(),
    )
    add(
        "T6a_spearman_conf_vs_ba",
        "Results §99",
        test_fig6_spearman_confidence_vs_ba(),
    )
    add(
        "T6b_spearman_frac_high_vs_ba",
        "Results §99",
        test_fig6_spearman_frac_high_vs_ba(),
    )
    add(
        "T6c_spearman_novelty_vs_ba",
        "Results §99",
        test_fig6_spearman_novelty_vs_ba(),
    )
    add(
        "T7_fig6c_low_conf_vs_random",
        "Results §102 / Fig 6C",
        test_fig6c_low_conf_vs_random(),
    )
    add(
        "T8a_fig7d_combined_vs_filtered_random",
        "Results §117 / Fig 7D",
        test_fig7d_combined_vs_filtered("random_split"),
    )
    add(
        "T8b_fig7d_combined_vs_filtered_dataset",
        "Results §117 / Fig 7D",
        test_fig7d_combined_vs_filtered("dataset_split"),
    )

    df = pd.DataFrame(rows)
    df["q_value_BH"] = benjamini_hochberg(df["p_value"].tolist())
    df = df.sort_values("p_value").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False, float_format="%.6g")
    print(df.to_string(index=False))
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
