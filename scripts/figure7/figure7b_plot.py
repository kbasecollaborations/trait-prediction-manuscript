#!/usr/bin/env python3
"""
Create Figure 6B: Performance on confident samples only.

This figure compares ML performance on confident samples (y_soft < 0.4 OR > 0.6)
versus full data performance. Shows dataset split with random split as reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Set random seed for reproducible jitter
np.random.seed(42)


def plot_confident_samples_performance(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """
    Plot dataset split performance comparison across three filtering conditions.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 6B data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    from scripts.minority_filter import (
        concordant_minority_counts,
        filter_by_minority,
        full_test_minority_counts,
    )

    concordant_minority = concordant_minority_counts()
    full_minority = full_test_minority_counts()

    # Load data from three sources
    # 1. Concordant samples (Figure 5A) — test set is the concordant subset
    concordant_df = pd.read_csv(
        Path("data/outputs/figure5/figure5a_concordant_ml_results.csv")
    )
    concordant_df = concordant_df[concordant_df["split_type"] == "dataset_split"].copy()
    concordant_df = filter_by_minority(concordant_df, concordant_minority)

    # 2. Y_soft filtered samples (current Figure 6B) — test set is the full
    # held-out dataset
    ysoft_df = pd.read_csv(data_dir / "figure7b_confident_ml_results.csv")
    ysoft_df = ysoft_df[ysoft_df["split_type"] == "dataset_split"].copy()
    ysoft_df = filter_by_minority(ysoft_df, full_minority)

    # 3. Misclassified samples removed (Figure 6C filtered condition) — full test
    misclass_df = pd.read_csv(data_dir / "figure7c_dataset_split_results.csv")
    misclass_df = misclass_df[misclass_df["condition"] == "filtered"].copy()
    misclass_df = filter_by_minority(misclass_df, full_minority)

    # Load full dataset results from Figure 6C for comparison (to calculate samples removed)
    full_df = pd.read_csv(data_dir / "figure7c_dataset_split_results.csv")
    full_df = full_df[full_df["condition"] == "full"].copy()
    full_df = filter_by_minority(full_df, full_minority)

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(concordant_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.25  # Width of each bar

    # Calculate mean and std for balanced accuracy
    concordant_summary = (
        concordant_df.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std"])
        .reindex(phenotypes)
    )
    ysoft_summary = (
        ysoft_df.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std"])
        .reindex(phenotypes)
    )
    misclass_summary = (
        misclass_df.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std"])
        .reindex(phenotypes)
    )

    # Calculate total samples for each condition
    concordant_total = (
        concordant_df.groupby("phenotype")["n_concordant_total"]
        .mean()
        .reindex(phenotypes)
    )

    # Sum n_train, n_val, n_test columns first, then group by phenotype
    ysoft_df["total_samples"] = ysoft_df["n_train"] + ysoft_df["n_val"] + ysoft_df["n_test"]
    ysoft_total = (
        ysoft_df.groupby("phenotype")["total_samples"]
        .mean()
        .reindex(phenotypes)
    )

    full_df["total_samples"] = full_df["n_train"] + full_df["n_val"] + full_df["n_test"]
    full_total = (
        full_df.groupby("phenotype")["total_samples"]
        .mean()
        .reindex(phenotypes)
    )

    misclass_df["total_samples"] = misclass_df["n_train"] + misclass_df["n_val"] + misclass_df["n_test"]
    misclass_total = (
        misclass_df.groupby("phenotype")["total_samples"]
        .mean()
        .reindex(phenotypes)
    )

    # Calculate samples removed
    concordant_removed = full_total - concordant_total
    ysoft_removed = full_total - ysoft_total
    misclass_removed = full_total - misclass_total

    # Extract means and stds
    concordant_means = concordant_summary["mean"].values
    concordant_stds = concordant_summary["std"].values
    ysoft_means = ysoft_summary["mean"].values
    ysoft_stds = ysoft_summary["std"].values
    misclass_means = misclass_summary["mean"].values
    misclass_stds = misclass_summary["std"].values

    # Create grouped bars
    bars1 = ax.bar(
        x - width,
        concordant_means,
        width,
        yerr=concordant_stds,
        label="Concordant Samples",
        color="#2E86AB",
        alpha=0.7,
        capsize=3,
    )
    bars2 = ax.bar(
        x,
        ysoft_means,
        width,
        yerr=ysoft_stds,
        label="Y_soft Filtered",
        color="#06A77D",
        alpha=0.7,
        capsize=3,
    )
    bars3 = ax.bar(
        x + width,
        misclass_means,
        width,
        yerr=misclass_stds,
        label="Misclassified Removed",
        color="#E63946",
        alpha=0.7,
        capsize=3,
    )

    # Add text annotations showing samples removed
    for i, phenotype in enumerate(phenotypes):
        # Get y positions (top of bars + error bars)
        y1 = concordant_means[i] + concordant_stds[i] if not np.isnan(concordant_means[i]) else 0
        y2 = ysoft_means[i] + ysoft_stds[i] if not np.isnan(ysoft_means[i]) else 0
        y3 = misclass_means[i] + misclass_stds[i] if not np.isnan(misclass_means[i]) else 0

        # Add annotations if samples were removed
        if not np.isnan(concordant_removed.iloc[i]) and concordant_removed.iloc[i] > 0:
            concordant_ha = "left" if i == 0 else "center"
            ax.text(
                i - width,
                y1 + 0.02,
                f"-{int(concordant_removed.iloc[i])}",
                ha=concordant_ha,
                va="bottom",
                fontsize=6.5,
                color="#2E86AB",
                weight="bold",
            )

        if not np.isnan(ysoft_removed.iloc[i]) and ysoft_removed.iloc[i] > 0:
            ysoft_ha = "right" if i == len(phenotypes) - 1 else "center"
            ax.text(
                i,
                y2 + 0.02,
                f"-{int(ysoft_removed.iloc[i])}",
                ha=ysoft_ha,
                va="bottom",
                fontsize=6.5,
                color="#06A77D",
                weight="bold",
            )

        if not np.isnan(misclass_removed.iloc[i]) and misclass_removed.iloc[i] > 0:
            misclass_ha = "right" if i == len(phenotypes) - 1 else "center"
            ax.text(
                i + width,
                y3 + 0.02,
                f"-{int(misclass_removed.iloc[i])}",
                ha=misclass_ha,
                va="bottom",
                fontsize=6.5,
                color="#E63946",
                weight="bold",
            )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        phenotypes,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.tick_params(axis="x", which="major", pad=1)
    ax.set_xlim(float(x[0] - width * 1.5), float(x[-1] + width * 1.5))
    # Headroom above 1.0 leaves room for the `-N samples removed` annotations
    # printed on top of each bar.
    ax.set_ylim(0, 1.18)

    # Legend sits just above the headroom region so it does not collide with
    # the per-bar annotations.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """
    Create Figure 6B plot.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    # Load data from all three sources to determine common phenotypes
    concordant_df = pd.read_csv(
        Path("data/outputs/figure5/figure5a_concordant_ml_results.csv")
    )
    ysoft_df = pd.read_csv(data_dir / "figure7b_confident_ml_results.csv")
    misclass_df = pd.read_csv(data_dir / "figure7c_dataset_split_results.csv")

    # Get phenotypes from each dataset (dataset split only)
    concordant_phenotypes = set(
        concordant_df[concordant_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    ysoft_phenotypes = set(
        ysoft_df[ysoft_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    misclass_phenotypes = set(
        misclass_df[misclass_df["condition"] == "filtered"]["phenotype"].unique()
    )

    # Use intersection to ensure consistent x-axis across all three datasets
    print("Determining common phenotypes...")
    print(f" - Concordant samples: {len(concordant_phenotypes)}")
    print(f" - Y_soft filtered: {len(ysoft_phenotypes)}")
    print(f" - Misclassified removed: {len(misclass_phenotypes)}")
    common_phenotypes = sorted(
        concordant_phenotypes.intersection(ysoft_phenotypes).intersection(misclass_phenotypes)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot
    plot_confident_samples_performance(ax, data_dir, common_phenotypes)

    # Adjust layout
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure7")
    output_file = Path("figures/figure7b.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
