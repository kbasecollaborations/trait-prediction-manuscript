#!/usr/bin/env python3

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


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

    # Create bar plot
    x_pos = np.arange(len(phenotypes))
    bars = ax.bar(x_pos, feature_counts, color="#2E86AB", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Customize plot
    ax.set_ylabel("Number of Stable Features", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])  # No labels on top plot
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, feature_counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=7,
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

    # Datasets and their colors
    datasets = ["atleaf", "lit", "marine"]
    dataset_colors = {
        "atleaf": "#E63946",
        "lit": "#457B9D",
        "marine": "#2A9D8F"
    }

    # Hatching patterns for different feature types
    patterns = {
        "common": "///",
        "unique_individual": "\\\\\\",
        "unique_combined": ""
    }

    # Prepare data for plotting
    x_pos = np.arange(len(phenotypes))
    width = 0.25  # Width of each bar

    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df["test_dataset"] == dataset].set_index("phenotype")

        # Align with phenotypes order
        common = []
        unique_individual = []
        unique_combined = []

        for phenotype in phenotypes:
            if phenotype in dataset_df.index:
                row = dataset_df.loc[phenotype]
                common.append(row["n_intersection"])
                unique_individual.append(row["n_unique_to_individual"])
                unique_combined.append(row["n_unique_to_combined"])
            else:
                common.append(0)
                unique_individual.append(0)
                unique_combined.append(0)

        # Calculate positions for grouped bars
        positions = x_pos + (i - 1) * width

        # Create stacked bars with dataset color and patterns
        p1 = ax.bar(
            positions,
            common,
            width,
            label=f"{dataset.upper()}" if i == 0 else "",
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.5,
            hatch=patterns["common"],
            alpha=0.8
        )
        p2 = ax.bar(
            positions,
            unique_individual,
            width,
            bottom=common,
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.5,
            hatch=patterns["unique_individual"],
            alpha=0.8
        )

        # Stack unique_combined on top
        bottom = np.array(common) + np.array(unique_individual)
        p3 = ax.bar(
            positions,
            unique_combined,
            width,
            bottom=bottom,
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.5,
            hatch=patterns["unique_combined"],
            alpha=0.8
        )

    # Create custom legend with both datasets and feature types
    from matplotlib.patches import Patch

    # Dataset legend entries
    dataset_handles = [
        Patch(facecolor=dataset_colors[d], edgecolor="black", label=d.upper())
        for d in datasets
    ]

    # Feature type legend entries
    feature_handles = [
        Patch(facecolor="gray", edgecolor="black", hatch=patterns["common"], label="Common"),
        Patch(facecolor="gray", edgecolor="black", hatch=patterns["unique_individual"], label="Unique to Individual"),
        Patch(facecolor="gray", edgecolor="black", hatch=patterns["unique_combined"], label="Unique to Combined"),
    ]

    # Combine legends
    first_legend = ax.legend(
        handles=dataset_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        title="Dataset"
    )
    ax.add_artist(first_legend)

    ax.legend(
        handles=feature_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        title="Feature Type"
    )

    # Customize plot
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.set_xlabel("Phenotype", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)


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
