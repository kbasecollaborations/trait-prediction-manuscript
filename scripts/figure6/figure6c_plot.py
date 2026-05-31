#!/usr/bin/env python3
"""Create Figure 6C: precision-recall scatter across training-set filtering conditions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_precision_recall_scatter(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
) -> None:
    """
    Plot precision vs recall scatter plot for three filtering conditions.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data_dir : Path
        Directory containing the data files.
    phenotypes : list[str]
        List of phenotypes in order.
    """
    from scripts.minority_filter import (
        filter_by_minority,
        full_test_minority_counts,
    )

    full_minority = full_test_minority_counts()

    # All three conditions are scored on the same full cross-dataset held-out
    # test set; only the training set changes.
    # 1. Concordant-trained model (Figure 5C data).
    concordant_df = pd.read_csv(
        Path("data/outputs/figure5/figure5c_concordant_train_different_test.csv")
    )
    concordant_df = concordant_df[
        (concordant_df["split_type"] == "dataset_split")
        & (concordant_df["test_type"] == "full")
    ].copy()
    concordant_df = filter_by_minority(concordant_df, full_minority)

    # 2. Y_soft-filtered training (Figure 6B data).
    ysoft_df = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")
    ysoft_df = ysoft_df[ysoft_df["split_type"] == "dataset_split"].copy()
    ysoft_df = filter_by_minority(ysoft_df, full_minority)

    # 3. Problematic-sample-removed training (Figure 6C "filtered" condition).
    misclass_df = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    misclass_df = misclass_df[misclass_df["condition"] == "filtered"].copy()
    misclass_df = filter_by_minority(
        misclass_df, full_minority, key_column="split"
    )

    concordant_summary = (
        concordant_df.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )
    ysoft_summary = (
        ysoft_df.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )
    misclass_summary = (
        misclass_df.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )
    gapmind_df = pd.read_csv(
        Path("data/outputs/figure3/gapmind_dataset_split_metrics.tsv"), sep="\t"
    )
    if "key" in gapmind_df.columns or "test_dataset" in gapmind_df.columns:
        gm_test_col = "test_dataset" if "test_dataset" in gapmind_df.columns else None
        gapmind_df = filter_by_minority(
            gapmind_df, full_minority, test_dataset_column=gm_test_col
        )
    gapmind_summary = (
        gapmind_df.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )

    ax.scatter(
        concordant_summary["recall"],
        concordant_summary["precision"],
        s=42,
        alpha=0.7,
        color="#2E86AB",
        edgecolors="black",
        linewidths=1,
        label="Concordant Samples",
        zorder=3,
    )
    ax.scatter(
        ysoft_summary["recall"],
        ysoft_summary["precision"],
        s=42,
        alpha=0.7,
        color="#06A77D",
        edgecolors="black",
        linewidths=1,
        label="Y_soft Filtered",
        zorder=3,
    )
    ax.scatter(
        misclass_summary["recall"],
        misclass_summary["precision"],
        s=42,
        alpha=0.7,
        color="#E63946",
        edgecolors="black",
        linewidths=1,
        label="Misclassified Removed",
        zorder=3,
    )
    ax.scatter(
        gapmind_summary["recall"],
        gapmind_summary["precision"],
        s=42,
        alpha=0.8,
        facecolors="none",
        edgecolors="#8B5CF6",
        linewidths=1,
        label="GapMind",
        zorder=3,
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=8,
        labelspacing=0.6,
        markerscale=0.7,
    )
    ax.set_aspect("equal")


def plot_gapmind_precision_recall_scatter(
    ax: Axes,
    phenotypes: list[str],
    gapmind_file: Path | None = None,
) -> None:
    """
    Plot GapMind precision vs recall scatter plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    phenotypes : list[str]
        List of phenotypes in order.
    gapmind_file : Path | None
        Path to GapMind metrics file. If None, uses default path.
    """
    if gapmind_file is None:
        gapmind_file = Path("data/outputs/figure3/gapmind_dataset_split_metrics.tsv")

    gapmind_df = pd.read_csv(gapmind_file, sep="\t")

    gapmind_summary = (
        gapmind_df.groupby("phenotype")[["precision", "recall"]]
        .mean()
        .reindex(phenotypes)
    )

    ax.scatter(
        gapmind_summary["recall"],
        gapmind_summary["precision"],
        s=200,
        alpha=0.7,
        color="#8B5CF6",
        edgecolors="black",
        linewidths=2,
        label="GapMind",
        zorder=3,
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, labelspacing=1.2)
    ax.set_aspect("equal")


def create_figure(
    data_file: Path,
    output_file: Path,
    phenotype_order: list[str] | None = None,
) -> None:
    """
    Create Figure 6C showing impact of filtering on precision, recall, and AUPRC.

    Parameters
    ----------
    data_file : Path
        Path to CSV file with results from figure6c_data.py.
    output_file : Path
        Path to save the output figure.
    phenotype_order : list[str] | None
        Order of phenotypes for x-axis. If None, uses alphabetical order.
    """
    df = pd.read_csv(data_file)

    if phenotype_order is None:
        phenotypes = sorted(df["phenotype"].unique())
    else:
        phenotypes = phenotype_order

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    plot_metric_comparison(
        axes[0],
        df,
        "precision",
        phenotypes,
        ylabel="Precision",
        title="A. Precision Before vs After Filtering",
    )

    plot_metric_comparison(
        axes[1],
        df,
        "recall",
        phenotypes,
        ylabel="Recall",
        title="B. Recall Before vs After Filtering",
    )

    plot_metric_comparison(
        axes[2],
        df,
        "auprc",
        phenotypes,
        ylabel="AUPRC (Average Precision)",
        title="C. AUPRC Before vs After Filtering",
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()

    print("\n" + "=" * 80)
    print("Summary: Average change after filtering")
    print("=" * 80)

    for metric in ["precision", "recall", "auprc"]:
        summary = df.groupby("condition")[metric].mean()
        if "full" in summary.index and "filtered" in summary.index:
            before = summary["full"]
            after = summary["filtered"]
            change = after - before
            pct_change = (change / before) * 100 if before != 0 else 0
            print(f"\n{metric.upper()}:")
            print(f"  Before: {before:.4f}")
            print(f"  After:  {after:.4f}")
            print(f"  Change: {change:+.4f} ({pct_change:+.2f}%)")


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure6")
    data_file = data_dir / "figure6c_dataset_split_results.csv"
    output_file = Path("figures/figure6c.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    create_figure(data_file, output_file)
