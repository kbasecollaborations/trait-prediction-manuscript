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
    # Load data from three sources
    # 1. Concordant samples (Figure 5A)
    concordant_df = pd.read_csv(
        Path("data/outputs/figure5/figure5a_concordant_ml_results.csv")
    )
    concordant_df = concordant_df[concordant_df["split_type"] == "dataset_split"].copy()

    # 2. Y_soft filtered samples (current Figure 6B)
    ysoft_df = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")
    ysoft_df = ysoft_df[ysoft_df["split_type"] == "dataset_split"].copy()

    # 3. Misclassified samples removed (Figure 6C filtered condition)
    misclass_df = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    misclass_df = misclass_df[misclass_df["condition"] == "filtered"].copy()

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(concordant_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.25  # Width of each bar

    # Calculate mean and std for each condition
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

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)

    # Add legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
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
    ysoft_df = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")
    misclass_df = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")

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
    data_dir = Path("data/outputs/figure6")
    output_file = Path("figures/figure6b.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
