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
    Plot dataset split performance on confident samples with random split reference.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 6B data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load data (confident samples only)
    ml_df = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")
    dataset_df = ml_df[ml_df["split_type"] == "dataset_split"].copy()
    random_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(dataset_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Prepare data for box plot - combining all datasets
    box_data = []
    for phenotype in phenotypes:
        phenotype_data = dataset_df[dataset_df["phenotype"] == phenotype][
            "balanced_accuracy"
        ].values
        box_data.append(phenotype_data)

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="#2E86AB", alpha=0.7, linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Calculate mean random split performance
    random_means = random_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()

    # Plot random split mean as reference lines
    for phenotype in phenotypes:
        if phenotype in random_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [random_means[phenotype], random_means[phenotype]],
                color="#06A77D",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Create legend handles
    legend_handles = [
        Patch(facecolor="#2E86AB", alpha=0.7, label="Dataset Split"),
        Line2D(
            [0],
            [0],
            color="#06A77D",
            linewidth=2,
            alpha=0.7,
            label="Random Split (mean)",
        ),
    ]

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy\n(confident samples only)")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(legend_handles),
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
    # Load data to determine phenotypes
    ml_df = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")

    # Get phenotypes from dataset split
    dataset_phenotypes = set(
        ml_df[ml_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    random_phenotypes = set(
        ml_df[ml_df["split_type"] == "random_split"]["phenotype"].unique()
    )

    # Use intersection to ensure consistent x-axis
    print("Determining common phenotypes...")
    print(f" - Dataset split phenotypes: {len(dataset_phenotypes)}")
    print(f" - Random split phenotypes: {len(random_phenotypes)}")
    common_phenotypes = sorted(dataset_phenotypes.intersection(random_phenotypes))
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
