#!/usr/bin/env python3
"""Aggregate paired-boxplot view for Figure 6 Panel B.

Replaces the original per-phenotype grouped-bar visualisation. Each box shows
the distribution of per-phenotype mean cross-dataset balanced accuracy across
the 15 common phenotypes, for six training conditions:

1. Concordant samples (Figure 5C data).
2. Mechanism-free filter (``free_balanced``: ``y_soft`` with ``w_gapmind=0``).
3. Low-mechanism filter (``current``: original ``y_soft`` weights).
4. High-mechanism filter (``high_mech``).
5. Very-high-mechanism filter (``very_high_mech``).
6. Problematic samples removed (Figure 6C ``filtered`` condition).

All conditions are evaluated on the same full cross-dataset held-out test set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from scripts.minority_filter import filter_by_minority, full_test_minority_counts

CONDITION_ORDER: list[str] = [
    "Problematic removed",
    "Confidence (no GapMind)",
    "Confidence (w=0.3)",
    "Confidence (w=0.4)",
    "Confidence (w=0.5)",
    "Concordant",
]

# Distinct anchor colours for the two reference conditions (problematic-sample
# removal and concordant training); a perceptual single-hue gradient for the
# four confidence-filter variants, light-to-dark as GapMind weight increases.
CONDITION_COLORS: dict[str, str] = {
    "Problematic removed": "#8D6E63",
    "Confidence (no GapMind)": "#A6DBA0",
    "Confidence (w=0.3)": "#5AAE61",
    "Confidence (w=0.4)": "#1B7837",
    "Confidence (w=0.5)": "#00441B",
    "Concordant": "#2E86AB",
}

CONFIG_TO_LABEL: dict[str, str] = {
    "free_balanced": "Confidence (no GapMind)",
    "current": "Confidence (w=0.3)",
    "high_mech": "Confidence (w=0.4)",
    "very_high_mech": "Confidence (w=0.5)",
}


def load_panel_b_long(
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Assemble a long-form dataframe with one row per (condition, phenotype, repeat).

    Parameters
    ----------
    data_dir : Path
        Directory containing the Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include (in order).
    metric : str
        Metric column to extract, by default ``"balanced_accuracy"``.

    Returns
    -------
    pd.DataFrame
        Columns ``condition``, ``phenotype``, ``metric_value``.
    """
    full_minority = full_test_minority_counts()
    phenotype_set = set(phenotypes)

    frames: list[pd.DataFrame] = []

    # 1. Concordant-trained on full cross-dataset test (Figure 5C data).
    concordant = pd.read_csv(
        Path("data/outputs/figure5/figure5c_concordant_train_different_test.csv")
    )
    concordant = concordant[
        (concordant["split_type"] == "dataset_split")
        & (concordant["test_type"] == "full")
    ].copy()
    concordant = filter_by_minority(concordant, full_minority)
    concordant = concordant[concordant["phenotype"].isin(phenotype_set)]
    frames.append(
        pd.DataFrame(
            {
                "condition": "Concordant",
                "phenotype": concordant["phenotype"],
                "metric_value": concordant[metric],
            }
        )
    )

    # 2-5. Weight-sweep configs (Phase 2 results).
    sweep_file = data_dir / "figure6b_weight_sweep_combined.csv"
    sweep_df = pd.read_csv(sweep_file)
    sweep_df = sweep_df[sweep_df["split_type"] == "dataset_split"].copy()
    sweep_df = filter_by_minority(sweep_df, full_minority)
    sweep_df = sweep_df[sweep_df["phenotype"].isin(phenotype_set)]
    for config_name, label in CONFIG_TO_LABEL.items():
        sub = sweep_df[sweep_df["config"] == config_name]
        frames.append(
            pd.DataFrame(
                {
                    "condition": label,
                    "phenotype": sub["phenotype"],
                    "metric_value": sub[metric],
                }
            )
        )

    # 6. Problematic-sample removal (Figure 6C "filtered" condition).
    misclass = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    misclass = misclass[misclass["condition"] == "filtered"].copy()
    misclass = filter_by_minority(misclass, full_minority, key_column="split")
    misclass = misclass[misclass["phenotype"].isin(phenotype_set)]
    frames.append(
        pd.DataFrame(
            {
                "condition": "Problematic removed",
                "phenotype": misclass["phenotype"],
                "metric_value": misclass[metric],
            }
        )
    )

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["metric_value"])
    return long_df


def plot_aggregate_filter_comparison(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "balanced_accuracy",
    ylabel: str | None = None,
) -> pd.DataFrame:
    """Plot the new aggregate Panel B as paired boxplots across phenotypes.

    Each box shows the distribution of per-phenotype mean ``metric`` across
    the supplied phenotype set. Individual phenotype means are overlaid as
    small jittered points to expose the underlying spread.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    data_dir : Path
        Directory containing Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include.
    metric : str
        Metric column to plot, by default ``"balanced_accuracy"``.
    ylabel : str | None
        Y-axis label. Defaults to a human-readable label for the metric.

    Returns
    -------
    pd.DataFrame
        The per-condition per-phenotype mean used for the boxplot, useful for
        downstream summary tables.
    """
    long_df = load_panel_b_long(data_dir, phenotypes, metric=metric)

    pheno_means = (
        long_df.groupby(["condition", "phenotype"])["metric_value"]
        .mean()
        .reset_index()
    )
    conditions_present = [c for c in CONDITION_ORDER if c in pheno_means["condition"].unique()]

    palette = {c: CONDITION_COLORS[c] for c in conditions_present}

    sns.boxplot(
        data=pheno_means,
        x="condition",
        y="metric_value",
        order=conditions_present,
        ax=ax,
        palette=palette,
        width=0.6,
        fliersize=0,
        linewidth=1.0,
        boxprops={"alpha": 0.6, "edgecolor": "black"},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )
    sns.stripplot(
        data=pheno_means,
        x="condition",
        y="metric_value",
        order=conditions_present,
        ax=ax,
        palette=palette,
        size=3.2,
        alpha=0.85,
        jitter=0.18,
        edgecolor="black",
        linewidth=0.3,
    )

    if ylabel is None:
        ylabel = {
            "balanced_accuracy": "Cross-dataset balanced accuracy",
            "precision": "Precision",
            "recall": "Recall",
        }.get(metric, metric.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0.35, 0.95)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    return pheno_means


def best_panel_b_config(
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "balanced_accuracy",
) -> tuple[str, str, pd.Series]:
    """Identify the highest-performing weight-sweep config on the test set.

    Parameters
    ----------
    data_dir : Path
        Directory containing the weight-sweep CSV.
    phenotypes : list[str]
        Phenotypes to score over.
    metric : str
        Metric column to rank by.

    Returns
    -------
    tuple[str, str, pd.Series]
        ``(config_name, condition_label, per_phenotype_means)``.
    """
    full_minority = full_test_minority_counts()
    sweep = pd.read_csv(data_dir / "figure6b_weight_sweep_combined.csv")
    sweep = sweep[sweep["split_type"] == "dataset_split"].copy()
    sweep = filter_by_minority(sweep, full_minority)
    sweep = sweep[sweep["phenotype"].isin(set(phenotypes))]

    summary = (
        sweep.groupby(["config", "phenotype"])[metric]
        .mean()
        .groupby("config")
        .mean()
        .sort_values(ascending=False)
    )
    best_config = str(summary.idxmax())
    return best_config, CONFIG_TO_LABEL[best_config], summary


def plot_precision_recall_best_config(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    best_config_name: str | None = None,
) -> str:
    """Per-phenotype precision-recall scatter for the simplified Panel C.

    Shows GapMind, concordant-trained, and the best-performing weight-sweep
    config from Panel B. One point per phenotype per condition.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    data_dir : Path
        Directory containing the Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include (one scatter point per phenotype per condition).
    best_config_name : str | None
        Weight-sweep config name to plot. If ``None``, picked automatically by
        ``best_panel_b_config``.

    Returns
    -------
    str
        The weight-sweep config name actually used.
    """
    full_minority = full_test_minority_counts()
    phenotype_set = set(phenotypes)

    if best_config_name is None:
        best_config_name, _, _ = best_panel_b_config(data_dir, phenotypes)
    best_label = CONFIG_TO_LABEL.get(best_config_name, best_config_name)

    # Concordant.
    concordant = pd.read_csv(
        Path("data/outputs/figure5/figure5c_concordant_train_different_test.csv")
    )
    concordant = concordant[
        (concordant["split_type"] == "dataset_split")
        & (concordant["test_type"] == "full")
    ].copy()
    concordant = filter_by_minority(concordant, full_minority)
    concordant = concordant[concordant["phenotype"].isin(phenotype_set)]
    concordant_pr = (
        concordant.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )

    # Best weight-sweep config.
    sweep = pd.read_csv(data_dir / "figure6b_weight_sweep_combined.csv")
    sweep = sweep[
        (sweep["split_type"] == "dataset_split")
        & (sweep["config"] == best_config_name)
    ].copy()
    sweep = filter_by_minority(sweep, full_minority)
    sweep = sweep[sweep["phenotype"].isin(phenotype_set)]
    sweep_pr = (
        sweep.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )

    # GapMind baseline.
    gapmind = pd.read_csv(
        "data/outputs/figure3/gapmind_dataset_split_metrics.tsv", sep="\t"
    )
    if "key" in gapmind.columns or "test_dataset" in gapmind.columns:
        gm_test_col = "test_dataset" if "test_dataset" in gapmind.columns else None
        gapmind = filter_by_minority(
            gapmind, full_minority, test_dataset_column=gm_test_col
        )
    gapmind = gapmind[gapmind["phenotype"].isin(phenotype_set)]
    gapmind_pr = (
        gapmind.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )

    ax.scatter(
        gapmind_pr["recall"],
        gapmind_pr["precision"],
        s=42,
        alpha=0.85,
        facecolors="none",
        edgecolors="#8B5CF6",
        linewidths=1.2,
        label="GapMind",
        zorder=3,
    )
    ax.scatter(
        concordant_pr["recall"],
        concordant_pr["precision"],
        s=42,
        alpha=0.75,
        color=CONDITION_COLORS["Concordant"],
        edgecolors="black",
        linewidths=0.8,
        label="Concordant",
        zorder=3,
    )
    ax.scatter(
        sweep_pr["recall"],
        sweep_pr["precision"],
        s=42,
        alpha=0.75,
        color=CONDITION_COLORS[best_label],
        edgecolors="black",
        linewidths=0.8,
        label=best_label,
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=8, labelspacing=0.6)
    ax.set_aspect("equal")
    return best_config_name


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure6")
    # Quick smoke test: just load and print summary.
    df = pd.read_csv("data/outputs/figure6/figure6b_weight_sweep_combined.csv")
    print(df["config"].value_counts())
