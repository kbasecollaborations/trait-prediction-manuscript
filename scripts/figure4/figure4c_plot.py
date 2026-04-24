#!/usr/bin/env python3

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from scripts.figure4.style import (
    ANNOTATION_FONT_SIZE,
    AXIS_LABEL_SIZE,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    TICK_LABEL_SIZE,
)
from scripts.visualization import format_dataset_names, get_dataset_colors


def create_feature_stability_plot(ax: Axes, phenotypes: list[str]) -> None:
    """Create bar plot showing feature stability for different phenotypes.

    Feature stability is defined as the number of features that appear in
    more than 70% of bootstrap samples.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    phenotypes : list[str]
        List of phenotypes in alphabetical order.
    """
    # Load data
    data_file = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    with open(data_file) as f:
        feature_data = json.load(f)

    # Calculate feature counts (stability) in the order of phenotypes
    feature_counts = [len(feature_data.get(p, [])) for p in phenotypes]

    # Create bar plot aligned with bottom subplot
    # Need to match the bottom subplot's bar grouping
    x_pos = np.arange(len(phenotypes))
    n_datasets = 3  # atleaf, lit, marine
    bar_width_bottom = 0.8 / n_datasets  # Width used in bottom subplot

    # Center the bars at the same position as the bottom subplot's grouped bars
    bar_center_offset = bar_width_bottom * (n_datasets - 1) / 2
    bar_width = 0.5  # Narrower bar for better appearance

    bars = ax.bar(
        x_pos + bar_center_offset,
        feature_counts,
        bar_width,
        color="#2E86AB",
        alpha=0.8,
    )

    # Add alternating background colors for x-axis categories (matching Figure 3)
    # Center backgrounds around the bar positions
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            center = i + bar_center_offset
            ax.axvspan(center - 0.5, center + 0.5, color="gray", alpha=0.1, zorder=0)

    # Customize plot
    ax.set_ylabel("Stable Features\n(Combined)", fontsize=AXIS_LABEL_SIZE, labelpad=1)
    # Set ticks to match bottom subplot
    ax.set_xticks(x_pos + bar_center_offset)
    ax.set_xticklabels([])  # No labels on top plot
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=1)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # Set x-axis limits to remove extra spacing
    ax.set_xlim(bar_center_offset - 0.5, len(phenotypes) - 1 + bar_center_offset + 0.5)

    # Set y-axis limits
    ax.set_ylim(0, 10)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, feature_counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )


def create_feature_comparison_plot(ax: Axes, phenotypes: list[str]) -> None:
    """Create grouped + stacked bar plot comparing features between datasets.

    Shows the number of features in common (intersection) and unique features
    between combined dataset and individual datasets.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    phenotypes : list[str]
        List of phenotypes in alphabetical order.
    """
    # Load data
    data_file = Path("data/outputs/figure4/feature_comparison_summary.csv")
    df = pd.read_csv(data_file)

    # Datasets and their colors (using consistent color scheme)
    datasets = ["atleaf", "lit", "marine"]
    dataset_color_map = get_dataset_colors()
    dataset_colors = [dataset_color_map[d] for d in datasets]
    dataset_display_names = format_dataset_names(datasets)

    # Hatching patterns for different feature types (similar to Figure 1C)
    patterns = {
        "common": "",  # Solid (like positive counts in Figure 1C)
        "unique_individual": "//",  # Diagonal hatching (like negative counts)
        "unique_combined": "xx",  # Cross-hatching for distinction
    }

    # Prepare data for plotting (matching Figure 1C style)
    x_pos = np.arange(len(phenotypes))
    bar_width = 0.8 / len(datasets)  # Match Figure 1C bar width calculation

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

        # Calculate positions for grouped bars (matching Figure 1C style)
        positions = x_pos + i * bar_width

        # Create stacked bars with dataset color and patterns (matching Figure 1C)
        # Bottom layer: common features (solid, alpha=0.8)
        p1 = ax.bar(
            positions,
            common,
            bar_width,
            color=dataset_colors[i],
            alpha=0.8,
        )
        # Top layer: unique to individual (hatched //, alpha=0.4)
        p2 = ax.bar(
            positions,
            unique_individual,
            bar_width,
            bottom=common,
            color=dataset_colors[i],
            alpha=0.4,
            hatch="//",
        )

    # Create custom legend with both datasets and feature types (matching Figure 1C)
    # Dataset legend entries
    dataset_handles = [
        Rectangle((0, 0), 1, 1, fc=dataset_colors[i], alpha=0.8)
        for i in range(len(datasets))
    ]

    # Feature type legend entries (matching Figure 1C style)
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
        bbox_to_anchor=(0.0, 1.37),
        ncol=len(datasets),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    ax.add_artist(legend1)

    # Add feature type legend (positioned right side of top)
    ax.legend(
        handles=feature_handles,
        title="Stable features",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.37),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    # Calculate the center offset for grouped bars
    bar_group_center = bar_width * (len(datasets) - 1) / 2

    # Add alternating background colors for x-axis categories (matching Figure 3)
    # Center backgrounds around the grouped bar positions
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            center = i + bar_group_center
            ax.axvspan(center - 0.5, center + 0.5, color="gray", alpha=0.1, zorder=0)

    # Customize plot (matching Figure 1C)
    ax.set_ylabel(
        "Stable Features\n(Per Dataset)", fontsize=AXIS_LABEL_SIZE, labelpad=1
    )
    ax.set_xlabel("Phenotype", fontsize=AXIS_LABEL_SIZE)
    # Center x-tick labels in the middle of the group of bars (matching Figure 1C)
    ax.set_xticks(x_pos + bar_group_center)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=1)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # Set x-axis limits to remove extra spacing
    ax.set_xlim(bar_group_center - 0.5, len(phenotypes) - 1 + bar_group_center + 0.5)

    # Set y-axis limits
    ax.set_ylim(0, 10)


def create_panel_c_plots(ax_top: Axes, ax_bottom: Axes) -> None:
    """Create both subplots for panel C.

    Parameters
    ----------
    ax_top : Axes
        Top subplot for feature stability plot.
    ax_bottom : Axes
        Bottom subplot for feature comparison plot.
    """
    # Get unique phenotypes from both datasets and sort alphabetically
    data_file1 = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    data_file2 = Path("data/outputs/figure4/feature_comparison_summary.csv")

    with open(data_file1) as f:
        feature_data = json.load(f)

    df = pd.read_csv(data_file2)

    # Get all unique phenotypes and sort alphabetically
    phenotypes_set = set(feature_data.keys()) | set(df["phenotype"].unique())
    phenotypes = sorted(list(phenotypes_set))

    print("Creating Figure 4C (feature stability and comparison)...")
    create_feature_stability_plot(ax_top, phenotypes)
    create_feature_comparison_plot(ax_bottom, phenotypes)
