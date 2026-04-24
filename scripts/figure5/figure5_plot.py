#!/usr/bin/env python3
"""
Create Figure 5: Performance on GapMind-concordant samples.

This figure shows:
- Panel A: Dataset split performance (concordant samples) with random split reference
- Panel B: Shared stable features between individual datasets (concordant samples)
- Panel C: Performance of concordant-trained models on discordant test sets (random split vs dataset split)
"""

import json
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

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


def _format_p_value(p_value: float | None) -> str:
    """Format a p-value compactly for panel annotations.

    Parameters
    ----------
    p_value : float | None
        P-value to format.

    Returns
    -------
    str
        Compact string representation for manuscript figures.
    """
    if p_value is None:
        return "n/a"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def _paired_panel_pvalue(
    left_values: pd.Series,
    right_values: pd.Series,
    phenotypes: list[str],
) -> tuple[float | None, int]:
    """Compute a paired phenotype-level Wilcoxon p-value.

    Phenotype means are compared pairwise to avoid cluttering the figure with
    per-phenotype significance annotations.

    Parameters
    ----------
    left_values : pd.Series
        Phenotype-level values for the first condition. The series index must be
        phenotype names.
    right_values : pd.Series
        Phenotype-level values for the second condition. The series index must be
        phenotype names.
    phenotypes : list[str]
        Phenotypes displayed in the panel. Only overlapping phenotypes from this
        plotted list are used in the paired comparison.

    Returns
    -------
    tuple[float | None, int]
        The Wilcoxon p-value and the number of paired phenotypes used. Returns
        ``None`` when too few overlapping pairs are available or the test is not
        numerically well-defined.
    """
    paired_phenotypes = [
        phenotype
        for phenotype in phenotypes
        if phenotype in left_values.index and phenotype in right_values.index
    ]
    n_pairs = len(paired_phenotypes)
    if n_pairs < 2:
        return None, n_pairs

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            test_result = wilcoxon(
                left_values.loc[paired_phenotypes],
                right_values.loc[paired_phenotypes],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
    except ValueError:
        return None, n_pairs

    p_value = float(test_result.pvalue)
    if not np.isfinite(p_value):
        return None, n_pairs
    return p_value, n_pairs


def _add_panel_pvalue_annotation(
    ax: Axes,
    p_value: float | None,
    n_pairs: int,
    position: Literal["top_right", "bottom_right"] = "top_right",
) -> None:
    """Add a compact panel-level p-value annotation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    p_value : float | None
        Paired comparison p-value.
    n_pairs : int
        Number of phenotype pairs used in the comparison.
    position : Literal["top_right", "bottom_right"]
        Anchor location of the annotation within the axes. ``"top_right"``
        places the text at the upper-right corner; ``"bottom_right"`` places
        it at the lower-right corner.
    """
    if position == "top_right":
        x, y, ha, va = 0.98, 0.97, "right", "top"
    else:
        x, y, ha, va = 0.98, 0.05, "right", "bottom"

    ax.text(
        x,
        y,
        f"Phenotype means\npaired Wilcoxon, n={n_pairs}, p={_format_p_value(p_value)}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9
        ),
    )


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

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(dataset_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Alternating phenotype background bands for visual grouping
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Prepare data for box plot - combining all datasets
    box_data = []
    for phenotype in phenotypes:
        phenotype_data = dataset_df[dataset_df["phenotype"] == phenotype][
            "balanced_accuracy"
        ].values
        box_data.append(phenotype_data)

    # Create box plot
    ax.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#2E86AB", alpha=0.65, linewidth=1.1),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
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
                linewidth=1.6,
                alpha=0.65,
                zorder=1,
            )

    # Create legend handles
    from matplotlib.patches import Patch

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

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)
    ax.set_ylim(0, 1.05)

    dataset_means = dataset_df.groupby("phenotype")["balanced_accuracy"].mean()
    random_mean_series = pd.Series(random_means)
    p_value, n_pairs = _paired_panel_pvalue(
        dataset_means,
        random_mean_series,
        phenotypes,
    )
    _add_panel_pvalue_annotation(ax, p_value, n_pairs, position="bottom_right")

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

    # Alternating phenotype background bands for visual grouping
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

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

    # Create separate legend entries for datasets and feature types
    dataset_handles = [
        Rectangle((0, 0), 1, 1, fc=dataset_colors[i], alpha=0.8)
        for i in range(len(datasets))
    ]
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
    legend1 = ax.legend(
        dataset_handles,
        dataset_display_names,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.13),
        ncol=len(datasets),
        frameon=False,
        fontsize=8,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=feature_handles,
        title="Stable features",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.13),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    # Customize plot
    ax.set_ylabel("Stable features (n)", fontsize=10)
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
    """Plot performance of models trained on concordant, tested on discordant only.

    Shows both random split and dataset split results in grouped box plots.

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

    # Filter to only discordant test type
    ml_df = ml_df[ml_df["test_type"] == "discordant"].copy()

    # Get unique phenotypes
    if phenotypes is None:
        phenotypes = sorted(ml_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Alternating phenotype background bands for visual grouping
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Create grouped box plots
    box_data_random = []
    box_data_dataset = []

    for phenotype in phenotypes:
        random_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["split_type"] == "random_split")
        ]["balanced_accuracy"].values
        dataset_data = ml_df[
            (ml_df["phenotype"] == phenotype) & (ml_df["split_type"] == "dataset_split")
        ]["balanced_accuracy"].values

        box_data_random.append(random_data)
        box_data_dataset.append(dataset_data)

    # Box plot width and positions
    width = 0.35
    positions_random = x - width / 2
    positions_dataset = x + width / 2

    # Create box plots
    # Random split: green (#06A77D) to match figure3 panel A
    ax.boxplot(
        box_data_random,
        positions=positions_random,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#06A77D", alpha=0.65, linewidth=1.1),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
    )

    # Dataset split: blue (#2E86AB) as complementary color
    ax.boxplot(
        box_data_dataset,
        positions=positions_dataset,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#2E86AB", alpha=0.65, linewidth=1.1),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
    )

    # Create legend handles
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#06A77D", alpha=0.7, label="Random Split"),
        Patch(facecolor="#2E86AB", alpha=0.7, label="Dataset Split"),
    ]

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    random_means = (
        ml_df[ml_df["split_type"] == "random_split"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )
    dataset_means = (
        ml_df[ml_df["split_type"] == "dataset_split"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )
    p_value, n_pairs = _paired_panel_pvalue(
        dataset_means,
        random_means,
        phenotypes,
    )
    _add_panel_pvalue_annotation(ax, p_value, n_pairs)

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
        ncol=len(legend_handles),
        frameon=False,
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 5 with three subplots.

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
    # Filter test_df to only discordant test type for phenotype determination (panel C)
    test_phenotypes = set(
        test_df[test_df["test_type"] == "discordant"]["phenotype"].unique()
    )

    # Use intersection to ensure consistent x-axis
    print("Determining common phenotypes across all analyses...")
    print(f" - Dataset split phenotypes: {len(dataset_phenotypes)}")
    print(f" - Feature comparison phenotypes: {len(feature_phenotypes)}")
    print(f" - Test type phenotypes (discordant): {len(test_phenotypes)}")
    common_phenotypes = sorted(
        dataset_phenotypes.intersection(feature_phenotypes).intersection(
            test_phenotypes
        )
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Create figure with 3 subplots arranged vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Plot each subplot with common phenotypes
    plot_dataset_split_performance(axes[0], data_dir, common_phenotypes)
    create_feature_comparison_plot(axes[1], data_dir, common_phenotypes)
    plot_concordant_train_performance(axes[2], data_dir, common_phenotypes)

    # Manually align x-axes to ensure consistency
    x_pos = np.arange(len(common_phenotypes))
    for ax in axes:
        ax.set_xlim(-0.5, len(common_phenotypes) - 0.5)
        ax.set_xticks(x_pos)

    # Remove x-axis labels from all but bottom plot
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Phenotype")

    # Only show x-tick labels on bottom plot
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticklabels(common_phenotypes, rotation=45, ha="right")

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
