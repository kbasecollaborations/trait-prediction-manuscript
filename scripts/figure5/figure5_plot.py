#!/usr/bin/env python3
"""
Create Figure 5: Performance on GapMind-concordant samples.

This figure shows:
- Panel A: Dataset split performance (concordant samples) with random split reference
- Panel B: Shared stable features between individual datasets (concordant samples)
- Panel C: Performance of concordant-trained models on discordant vs full test sets (random split)
- Panel D: Performance of concordant-trained models on discordant vs full test sets (dataset split)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Set random seed for reproducible jitter
np.random.seed(42)


def extract_test_dataset(key: str) -> str:
    """Extract test dataset from key string.

    Parameters
    ----------
    key : str
        Key string like "Mannose_train(atleaf+marine+pmi),test(lit)"

    Returns
    -------
    str
        Test dataset name (e.g., "lit")
    """
    # Extract test dataset from "test(dataset)" pattern
    test_part = key.split("test(")[1].split(")")[0]
    return test_part


def plot_dataset_split_performance(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """Plot dataset split performance with random split reference (concordant samples).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 5 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load data
    ml_df = pd.read_csv(data_dir / "figure5a_concordant_ml_results.csv")
    dataset_df = ml_df[ml_df["split_type"] == "dataset_split"].copy()
    random_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Extract test dataset from key
    dataset_df["test_dataset"] = dataset_df["key"].apply(extract_test_dataset)

    # Get dataset colors
    dataset_colors = get_dataset_colors()

    # Get unique phenotypes and test datasets
    if phenotypes is None:
        phenotypes = sorted(dataset_df["phenotype"].unique())
    test_datasets = sorted(dataset_df["test_dataset"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.2
    offsets = np.linspace(
        -width * (len(test_datasets) - 1) / 2,
        width * (len(test_datasets) - 1) / 2,
        len(test_datasets),
    )

    # Plot dataset split results
    for idx, test_dataset in enumerate(test_datasets):
        test_data = dataset_df[dataset_df["test_dataset"] == test_dataset]

        # Plot individual points
        for phenotype in phenotypes:
            phenotype_data = test_data[test_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            # Add jitter
            x_jitter = x_pos + np.random.normal(0, 0.02, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[test_dataset],
                alpha=0.6,
                s=60,
                zorder=2,
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
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dataset_colors[dataset],
            markersize=8,
            alpha=0.8,
            label=f"Test: {format_dataset_names([dataset])[0]}",
            linestyle="None",
        )
        for dataset in test_datasets
    ]

    # Add random split reference to legend
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#06A77D",
            linewidth=2,
            alpha=0.7,
            label="Random Split (mean)",
        )
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)
    ax.set_ylim(0, 1.05)

    # Add subplot label
    ax.text(
        -0.08,
        1.05,
        "(A)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(legend_handles),
        frameon=False,
    )


def create_feature_comparison_plot(
    ax: Axes, data_dir: Path, phenotypes: list[str]
) -> None:
    """Create grouped + stacked bar plot comparing features between datasets (concordant samples).

    Shows the number of features in common (intersection) and unique features
    between combined dataset and individual datasets.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data_dir : Path
        Directory containing the figure 5B data files.
    phenotypes : list[str]
        List of phenotypes in alphabetical order.
    """
    # Load data
    data_file = data_dir / "figure5b_feature_comparison_summary.csv"
    df = pd.read_csv(data_file)

    # Datasets and their colors (using consistent color scheme)
    datasets = ["atleaf", "lit", "marine"]
    dataset_color_map = get_dataset_colors()
    dataset_colors = [dataset_color_map[d] for d in datasets]
    dataset_display_names = format_dataset_names(datasets)

    # Prepare data for plotting
    x_pos = np.arange(len(phenotypes))
    bar_width = 0.8 / len(datasets)

    # Calculate offset to center the grouped bars at each x position
    bar_group_center = bar_width * (len(datasets) - 1) / 2

    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df["test_dataset"] == dataset].set_index("phenotype")

        # Align with phenotypes order
        common = []
        unique_individual = []

        for phenotype in phenotypes:
            if phenotype in dataset_df.index:
                row = dataset_df.loc[phenotype]
                common.append(row["n_intersection"])
                unique_individual.append(row["n_unique_to_individual"])
            else:
                common.append(0)
                unique_individual.append(0)

        # Calculate positions for grouped bars (centered at each x position)
        positions = x_pos - bar_group_center + i * bar_width

        # Create stacked bars with dataset color and patterns
        # Bottom layer: common features (solid, alpha=0.8)
        ax.bar(
            positions,
            common,
            bar_width,
            color=dataset_colors[i],
            alpha=0.8,
        )
        # Top layer: unique to individual (hatched //, alpha=0.4)
        ax.bar(
            positions,
            unique_individual,
            bar_width,
            bottom=common,
            color=dataset_colors[i],
            alpha=0.4,
            hatch="//",
        )

    # Create custom legend with both datasets and feature types
    # Dataset legend entries
    dataset_handles = [
        Rectangle((0, 0), 1, 1, fc=dataset_colors[i], alpha=0.8)
        for i in range(len(datasets))
    ]

    # Feature type legend entries
    feature_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            fc="gray",
            alpha=0.8,
            label="Common",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fc="gray",
            alpha=0.4,
            hatch="//",
            label="Unique to Individual",
        ),
    ]

    # Add dataset legend (positioned left side of top)
    legend1 = ax.legend(
        dataset_handles,
        dataset_display_names,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.20),
        ncol=len(datasets),
        frameon=False,
    )
    ax.add_artist(legend1)

    # Add feature type legend (positioned right side of top)
    ax.legend(
        handles=feature_handles,
        title="Stable features",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        ncol=2,
        frameon=False,
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Customize plot
    ax.set_ylabel("Number of Stable Features\n(concordant samples)", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")

    # Set x-axis limits to match other panels
    ax.set_xlim(-0.5, len(phenotypes) - 0.5)

    # Set y-axis limits
    ax.set_ylim(0, 10)

    # Add subplot label
    ax.text(
        -0.08,
        1.05,
        "(B)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def plot_concordant_train_performance(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """Plot performance of models trained on concordant, tested on discordant vs full.

    Only uses random_split data for visualization. Also plots mean random split
    accuracy from Figure 3A as a reference line.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 5C data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load data
    ml_df = pd.read_csv(data_dir / "figure5c_concordant_train_different_test.csv")

    # Filter to only random_split
    ml_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Load Figure 3 random split results for reference line (full data, not concordant)
    fig3_data_dir = Path("data/outputs/figure3")
    fig3_ml_df = pd.read_csv(fig3_data_dir / "ml_results.csv")
    fig3_random_df = fig3_ml_df[fig3_ml_df["split_type"] == "random_split"].copy()

    # Calculate mean random split performance for each phenotype
    fig3_random_means = (
        fig3_random_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()
    )

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(ml_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Prepare data for box plot - one box per test type
    test_types = ["full", "discordant"]
    test_type_labels = ["Full", "Discordant"]
    colors = ["#2E86AB", "#A23B72"]  # Blue for full, purple for discordant

    # Create grouped box plots
    box_data_full = []
    box_data_disc = []

    for phenotype in phenotypes:
        full_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["test_type"] == "full")
        ]["balanced_accuracy"].values
        disc_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["test_type"] == "discordant")
        ]["balanced_accuracy"].values

        box_data_full.append(full_data)
        box_data_disc.append(disc_data)

    # Box plot width and positions
    width = 0.35
    positions_full = x - width / 2
    positions_disc = x + width / 2

    # Create box plots
    bp1 = ax.boxplot(
        box_data_full,
        positions=positions_full,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=colors[0], alpha=0.7, linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    bp2 = ax.boxplot(
        box_data_disc,
        positions=positions_disc,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=colors[1], alpha=0.7, linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Plot Figure 3 random split mean as reference dotted line
    for phenotype in phenotypes:
        if phenotype in fig3_random_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [fig3_random_means[phenotype], fig3_random_means[phenotype]],
                color="#06A77D",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Create legend handles
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors[0], alpha=0.7, label="Test: Full"),
        Patch(facecolor=colors[1], alpha=0.7, label="Test: Discordant"),
        Line2D(
            [0],
            [0],
            color="#06A77D",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label="Random Split (Fig 3A)",
        ),
    ]

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Add subplot label
    ax.text(
        -0.08,
        1.05,
        "(C)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
    )


def plot_dataset_split_train_performance(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """Plot performance of dataset split models trained on concordant, tested on discordant vs full.

    Only uses dataset_split data for visualization. Also plots GapMind
    accuracy from Figure 3B as a reference line (calculated only on test sets).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 5C data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load data
    ml_df = pd.read_csv(data_dir / "figure5c_concordant_train_different_test.csv")

    # Filter to only dataset_split
    ml_df = ml_df[ml_df["split_type"] == "dataset_split"].copy()

    # Load GapMind results for dataset split test sets (from Figure 3)
    fig3_data_dir = Path("data/outputs/figure3")
    gapmind_df = pd.read_csv(
        fig3_data_dir / "gapmind_dataset_split_metrics.tsv", sep="\t"
    )

    # Calculate mean GapMind performance for each phenotype (across dataset splits)
    gapmind_means = (
        gapmind_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()
    )

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(ml_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Prepare data for box plot - one box per test type
    test_types = ["full", "discordant"]
    test_type_labels = ["Full", "Discordant"]
    colors = ["#2E86AB", "#A23B72"]  # Blue for full, purple for discordant

    # Create grouped box plots
    box_data_full = []
    box_data_disc = []

    for phenotype in phenotypes:
        full_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["test_type"] == "full")
        ]["balanced_accuracy"].values
        disc_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["test_type"] == "discordant")
        ]["balanced_accuracy"].values

        box_data_full.append(full_data)
        box_data_disc.append(disc_data)

    # Box plot width and positions
    width = 0.35
    positions_full = x - width / 2
    positions_disc = x + width / 2

    # Create box plots
    bp1 = ax.boxplot(
        box_data_full,
        positions=positions_full,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=colors[0], alpha=0.7, linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    bp2 = ax.boxplot(
        box_data_disc,
        positions=positions_disc,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=colors[1], alpha=0.7, linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Plot GapMind as reference dotted line
    for phenotype in phenotypes:
        if phenotype in gapmind_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [gapmind_means[phenotype], gapmind_means[phenotype]],
                color="#F18F01",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Create legend handles
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors[0], alpha=0.7, label="Test: Full"),
        Patch(facecolor=colors[1], alpha=0.7, label="Test: Discordant"),
        Line2D(
            [0],
            [0],
            color="#F18F01",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label="GapMind (Fig 3B)",
        ),
    ]

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Add subplot label
    ax.text(
        -0.08,
        1.05,
        "(D)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 5 with four subplots.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    # Load all data to determine common phenotypes
    ml_df = pd.read_csv(data_dir / "figure5a_concordant_ml_results.csv")
    feature_comp_df = pd.read_csv(data_dir / "figure5b_feature_comparison_summary.csv")
    test_df = pd.read_csv(data_dir / "figure5c_concordant_train_different_test.csv")

    # Get phenotypes from each dataset
    dataset_phenotypes = set(
        ml_df[ml_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    feature_phenotypes = set(feature_comp_df["phenotype"].unique())
    # Filter test_df to only random_split for phenotype determination (panel C)
    test_phenotypes_random = set(
        test_df[test_df["split_type"] == "random_split"]["phenotype"].unique()
    )
    # Filter test_df to only dataset_split for phenotype determination (panel D)
    test_phenotypes_dataset = set(
        test_df[test_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )

    # Use intersection to ensure consistent x-axis
    print("Determining common phenotypes across all analyses...")
    print(f" - Dataset split phenotypes: {len(dataset_phenotypes)}")
    print(f" - Feature comparison phenotypes: {len(feature_phenotypes)}")
    print(f" - Test type phenotypes (random): {len(test_phenotypes_random)}")
    print(f" - Test type phenotypes (dataset): {len(test_phenotypes_dataset)}")
    common_phenotypes = sorted(
        dataset_phenotypes.intersection(feature_phenotypes)
        .intersection(test_phenotypes_random)
        .intersection(test_phenotypes_dataset)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Create figure with 4 subplots arranged vertically
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Plot each subplot with common phenotypes
    plot_dataset_split_performance(axes[0], data_dir, common_phenotypes)
    create_feature_comparison_plot(axes[1], data_dir, common_phenotypes)
    plot_concordant_train_performance(axes[2], data_dir, common_phenotypes)
    plot_dataset_split_train_performance(axes[3], data_dir, common_phenotypes)

    # Manually align x-axes to ensure consistency
    x_pos = np.arange(len(common_phenotypes))
    for ax in axes:
        ax.set_xlim(-0.5, len(common_phenotypes) - 0.5)
        ax.set_xticks(x_pos)

    # Remove x-axis labels from all but bottom plot
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("Phenotype")

    # Only show x-tick labels on bottom plot
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticklabels([])
    axes[3].set_xticklabels(common_phenotypes, rotation=45, ha="right")

    # Adjust layout with more space between subplots
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure5")
    output_file = Path("figures/figure5.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
