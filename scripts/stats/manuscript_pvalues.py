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
    filter_by_minority,
    full_test_minority_counts,
)

OUTPUT_DIR = Path("data/outputs/stats")
OUTPUT_FILE = OUTPUT_DIR / "manuscript_pvalues.tsv"
PER_PHENOTYPE_FILE = OUTPUT_DIR / "per_phenotype_mcnemar.tsv"
PER_SAMPLE_FILE = Path("data/outputs/figure7/figure7_per_sample.tsv")

MECHFREE_PER_SAMPLE_FILE = Path("data/outputs/figure6/figure6c_mechfree_per_sample.tsv")
"""Mechanism-free arm of Figure 6C, produced on the concordant arm's terms by
``scripts/figure6/figure6c_mechfree_per_sample.py``."""
MECHFREE_PER_PHENOTYPE_FILE = OUTPUT_DIR / "per_phenotype_mcnemar_mechfree.tsv"

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
    df = pd.read_csv("data/outputs/figure3/gapmind_random_split_metrics.tsv", sep="\t")
    return _per_phenotype_mean(df)


def _load_gapmind_dataset() -> pd.Series:
    df = pd.read_csv("data/outputs/figure3/gapmind_dataset_split_metrics.tsv", sep="\t")
    # Same minority-class test filter as the ML side, so the comparison is
    # scored on matched cells.
    test_col = "test_dataset" if "test_dataset" in df.columns else None
    df = filter_by_minority(
        df, full_test_minority_counts(), test_dataset_column=test_col
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
        gm_raw
        if gm_raw.exists()
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
    per_sample = pd.read_csv("data/outputs/figure7/figure7_per_sample.tsv", sep="\t")
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
    per_sample = pd.read_csv("data/outputs/figure7/figure7_per_sample.tsv", sep="\t")
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
    novelty_path = Path("data/outputs/figure7/figure7_per_sample.tsv")
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


def test_fig7c_low_conf_vs_random() -> tuple[float, int]:
    """Figure 7C: low-confidence vs random selection paired by (phen, held-out, seed)."""
    df = pd.read_csv("data/outputs/figure7/figure7_prioritization.tsv", sep="\t")
    pivot = df.pivot_table(
        index=["phenotype", "held_out_dataset", "seed"],
        columns="strategy",
        values="delta_balanced_accuracy",
    ).dropna(subset=["low_confidence", "random"])
    res = wilcoxon(
        pivot["low_confidence"].to_numpy(),
        pivot["random"].to_numpy(),
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(res.pvalue), len(pivot)


def _figure6_metric_means(
    metric: str = "recall",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-phenotype mean cross-dataset metric for the Figure 6B/6C comparisons.

    Mirrors the Figure 6 panel computation in
    ``scripts.figure6.figure6b_aggregate_plot`` (same data sources and minority
    filtering) so the recomputed p-values match the plotted panels.

    Parameters
    ----------
    metric : str, optional
        Metric column to average, default ``"recall"`` (the Figure 6C panel
        metric). Pass ``"balanced_accuracy"`` for the Figure 6B comparison.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        ``(concordant, mechanism_free, gapmind)``, each a per-phenotype mean of
        ``metric`` indexed by phenotype.
    """
    from scripts.figure6.figure6b_aggregate_plot import (
        _gapmind_baseline,
        _load_long_form,
    )

    phenotypes = list(COMMON_PHENOTYPES)
    long_df = _load_long_form(Path("data/outputs/figure6"), phenotypes)
    ml_means = long_df.groupby(["phenotype", "config"])[metric].mean().unstack("config")
    gm_means = _gapmind_baseline(phenotypes).groupby("phenotype")[metric].mean()
    return ml_means["concordant"], ml_means["free_balanced"], gm_means


def _figure6_recall_means() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-phenotype mean cross-dataset recall for the Figure 6C comparison."""
    return _figure6_metric_means("recall")


def test_fig6b_concordant_vs_mechfree_ba() -> tuple[float, int]:
    """Figure 6B: concordant vs mechanism-free filter balanced accuracy (cross-dataset)."""
    concordant, mech_free, _gapmind = _figure6_metric_means("balanced_accuracy")
    return _paired_wilcoxon(concordant, mech_free)


def test_fig6c_concordant_recall_vs_gapmind() -> tuple[float, int]:
    """Figure 6C: concordant-trained ML recall vs GapMind recall (cross-dataset)."""
    concordant, _mech_free, gapmind = _figure6_recall_means()
    return _paired_wilcoxon(concordant, gapmind)


def test_fig6c_mechfree_recall_vs_gapmind() -> tuple[float, int]:
    """Figure 6C: mechanism-free filter recall vs GapMind recall (cross-dataset)."""
    _concordant, mech_free, gapmind = _figure6_recall_means()
    return _paired_wilcoxon(mech_free, gapmind)


def fig6c_concordant_recall_delta_ci(
    n_boot: int = 10000, seed: int = 42
) -> dict[str, float]:
    """Effect size and bootstrap CI for the concordant-ML minus GapMind recall delta.

    Cited in the main text alongside the Wilcoxon q-value of
    ``test_fig6c_concordant_recall_vs_gapmind``.

    Parameters
    ----------
    n_boot : int, optional
        Bootstrap resamples (default 10000).
    seed : int, optional
        Fixed RNG seed for reproducibility (default 42).

    Returns
    -------
    dict[str, float]
        ``mean_delta``, ``ci_low``, ``ci_high``, ``n_positive``, ``n``.
    """
    concordant, _mech_free, gapmind = _figure6_recall_means()
    common = sorted(set(concordant.index) & set(gapmind.index) & set(COMMON_PHENOTYPES))
    delta = (concordant.loc[common] - gapmind.loc[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_positive": int((delta > 0).sum()),
        "n": len(delta),
    }


def test_fig6d_combined_vs_filtered(split: str) -> tuple[float, int]:
    """Figure 6D: combined features vs phenotype-filtered features (per phenotype)."""
    from scripts.figure6.figure6d_plot import load_results

    df = load_results(Path("data/outputs/figure6/figure6d_all_results.csv"))
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


def per_phenotype_mcnemar(
    per_sample_file: Path = PER_SAMPLE_FILE,
    restrict_to: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Per-phenotype McNemar tests of an ML arm against GapMind.

    Each phenotype is tested at the sample level. Under leave-one-dataset-out
    every genome appears in exactly one held-out test set, so pooling the four
    splits yields one prediction pair per genome and no pseudo-replication.

    Sensitivity (true positives) and specificity (true negatives) are tested
    separately; together they decompose the balanced-accuracy comparison.

    Parameters
    ----------
    per_sample_file : Path, optional
        Per-genome prediction table. Defaults to the concordance-trained arm;
        pass the mechanism-free table to test that arm on identical terms.
    restrict_to : set[tuple[str, str]] | None, optional
        ``(phenotype, genome)`` pairs to keep, used to match two arms on the same
        evaluation set. A filter that discards more training data can leave a
        split unfittable, so one arm may cover fewer held-out genomes than the
        other.

    Returns
    -------
    pd.DataFrame
        One row per (phenotype, metric) with ``n``, ``ml``, ``gapmind``,
        ``delta``, McNemar exact ``p_value`` and BH ``q_value`` computed within
        each metric.

    Raises
    ------
    FileNotFoundError
        If the per-sample prediction table is absent.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    if not per_sample_file.exists():
        raise FileNotFoundError(f"per-sample predictions not found: {per_sample_file}")
    per_sample = filter_by_minority(
        pd.read_csv(per_sample_file, sep="\t"),
        full_test_minority_counts(),
        test_dataset_column="held_out_dataset",
    )
    if restrict_to is not None:
        keep = [
            (p, g) in restrict_to
            for p, g in zip(per_sample["phenotype"], per_sample["genome"])
        ]
        per_sample = per_sample.loc[keep]

    frames: list[pd.DataFrame] = []
    for cls, metric in ((1, "sensitivity"), (0, "specificity")):
        rows: list[dict[str, object]] = []
        for phenotype, group in per_sample[per_sample["y_true"] == cls].groupby(
            "phenotype"
        ):
            ml_ok = group["y_pred"] == cls
            gm_ok = group["gapmind_pred"] == cls
            both = int((ml_ok & gm_ok).sum())
            ml_only = int((ml_ok & ~gm_ok).sum())
            gm_only = int((~ml_ok & gm_ok).sum())
            neither = int((~ml_ok & ~gm_ok).sum())
            rows.append(
                {
                    "phenotype": phenotype,
                    "metric": metric,
                    "n": len(group),
                    "ml": float(ml_ok.mean()),
                    "gapmind": float(gm_ok.mean()),
                    "delta": float(ml_ok.mean() - gm_ok.mean()),
                    "ml_only": ml_only,
                    "gapmind_only": gm_only,
                    "p_value": float(
                        mcnemar(
                            [[both, ml_only], [gm_only, neither]], exact=True
                        ).pvalue
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        frame["q_value_BH"] = benjamini_hochberg(frame["p_value"].tolist())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(
        label: str,
        where: str,
        result: tuple[float, int] | None,
        adjust: bool = True,
    ) -> None:
        """Record one test. ``adjust=False`` keeps it out of the BH family.

        The BH family is the twelve primary tests described in
        ``sections/methods.tex``; every other comparison is reported unadjusted.
        """
        if result is None:
            return
        p, n = result
        rows.append(
            {
                "test_id": label,
                "manuscript_location": where,
                "p_value": p,
                "n_paired": n,
                "bh_family": adjust,
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
        "Results §108",
        test_fig6_spearman_confidence_vs_ba(),
    )
    add(
        "T6b_spearman_frac_high_vs_ba",
        "Results §108",
        test_fig6_spearman_frac_high_vs_ba(),
    )
    add(
        "T6c_spearman_novelty_vs_ba",
        "Not reported; skipped if novelty column is unavailable",
        test_fig6_spearman_novelty_vs_ba(),
    )
    add(
        "T7_fig7c_low_conf_vs_random",
        "Results §111 / Fig 7C",
        test_fig7c_low_conf_vs_random(),
    )
    add(
        "T11_fig6b_concordant_vs_mechfree_ba",
        "Results §86 / Fig 6B",
        test_fig6b_concordant_vs_mechfree_ba(),
        adjust=False,
    )
    add(
        "T9_fig6c_concordant_recall_vs_gapmind",
        "Results §90 / Fig 6C",
        test_fig6c_concordant_recall_vs_gapmind(),
    )
    add(
        "T10_fig6c_mechfree_recall_vs_gapmind",
        "Results §90 / Fig 6C",
        test_fig6c_mechfree_recall_vs_gapmind(),
    )
    add(
        "T8a_fig6d_combined_vs_filtered_random",
        "Results §95 / Fig 6D",
        test_fig6d_combined_vs_filtered("random_split"),
    )
    add(
        "T8b_fig6d_combined_vs_filtered_dataset",
        "Results §95 / Fig 6D",
        test_fig6d_combined_vs_filtered("dataset_split"),
    )

    df = pd.DataFrame(rows)
    family = df["bh_family"]
    df["q_value_BH"] = np.nan
    df.loc[family, "q_value_BH"] = benjamini_hochberg(
        df.loc[family, "p_value"].tolist()
    )
    df = df.sort_values("p_value").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False, float_format="%.6g")
    print(df.to_string(index=False))
    print(f"\nWrote {OUTPUT_FILE}")

    ci = fig6c_concordant_recall_delta_ci()
    print(
        "\nFig 6C concordant-ML recall delta (effect size): "
        f"mean={ci['mean_delta']:.3f}, 95% CI "
        f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], "
        f"{ci['n_positive']}/{ci['n']} phenotypes favour ML"
    )

    per_phenotype = per_phenotype_mcnemar()
    per_phenotype.to_csv(PER_PHENOTYPE_FILE, sep="\t", index=False, float_format="%.6g")
    print(f"\nPer-phenotype McNemar (concordant ML vs GapMind) -> {PER_PHENOTYPE_FILE}")
    for metric, frame in per_phenotype.groupby("metric"):
        better = int(((frame["delta"] > 0) & (frame["q_value_BH"] < 0.05)).sum())
        worse = int(((frame["delta"] < 0) & (frame["q_value_BH"] < 0.05)).sum())
        print(
            f"  {metric}: ML better on {better}/{len(frame)}, "
            f"worse on {worse}/{len(frame)} (BH q < 0.05)"
        )
    # Sensitivity and specificity average to balanced accuracy.
    balanced = per_phenotype.pivot(
        index="phenotype", columns="metric", values="delta"
    ).mean(axis=1)
    # Same test for the mechanism-free arm of Figure 6C.
    if MECHFREE_PER_SAMPLE_FILE.exists():
        # Match the mechanism-free arm to the concordant arm's evaluation set:
        # concordance filtering can leave a split unfittable (Alanine/ATLeaf),
        # so the mechanism-free arm otherwise covers more genomes.
        concordant_rows = filter_by_minority(
            pd.read_csv(PER_SAMPLE_FILE, sep="\t"),
            full_test_minority_counts(),
            test_dataset_column="held_out_dataset",
        )
        concordant_pairs = set(
            concordant_rows[["phenotype", "genome"]].itertuples(index=False, name=None)
        )
        mechfree = per_phenotype_mcnemar(
            MECHFREE_PER_SAMPLE_FILE, restrict_to=concordant_pairs
        )
        mechfree.to_csv(
            MECHFREE_PER_PHENOTYPE_FILE, sep="\t", index=False, float_format="%.6g"
        )
        print(
            "\nPer-phenotype McNemar (mechanism-free ML vs GapMind) -> "
            f"{MECHFREE_PER_PHENOTYPE_FILE}"
        )
        for metric, frame in mechfree.groupby("metric"):
            better = int(((frame["delta"] > 0) & (frame["q_value_BH"] < 0.05)).sum())
            worse = int(((frame["delta"] < 0) & (frame["q_value_BH"] < 0.05)).sum())
            print(
                f"  {metric}: ML better on {better}/{len(frame)}, "
                f"worse on {worse}/{len(frame)} (BH q < 0.05)"
            )
    else:
        print(
            f"\nMechanism-free per-sample table absent ({MECHFREE_PER_SAMPLE_FILE}); "
            "skipping its McNemar tests"
        )

    print(
        f"  implied balanced-accuracy shift: mean {balanced.mean():+.3f}, "
        f"positive for {int((balanced > 0).sum())}/{len(balanced)} phenotypes"
    )


if __name__ == "__main__":
    main()
