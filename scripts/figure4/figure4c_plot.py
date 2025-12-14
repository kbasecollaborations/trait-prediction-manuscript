#!/usr/bin/env python3

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def create_feature_stability_plot(ax: Axes) -> None:
    """Create bar plot showing feature stability for different phenotypes.

    Feature stability is defined as the number of features that appear in
    more than 70% of bootstrap samples.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    """
    # Load data
    data_file = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    with open(data_file) as f:
        feature_data = json.load(f)

    # Calculate feature counts (stability)
    phenotypes = list(feature_data.keys())
    feature_counts = [len(features) for features in feature_data.values()]

    # Create bar plot
    x_pos = np.arange(len(phenotypes))
    bars = ax.bar(x_pos, feature_counts, color="#2E86AB", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Customize plot
    ax.set_ylabel("Number of Stable Features", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=9)
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


def create_feature_comparison_plot(ax: Axes) -> None:
    """Create grouped + stacked bar plot comparing features between datasets.

    Shows the number of features in common (intersection) and unique features
    between combined dataset and individual datasets.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    """
    # Load data
    data_file = Path("data/outputs/figure4/feature_comparison_summary.csv")
    df = pd.read_csv(data_file)

    # Get unique phenotypes and datasets
    phenotypes = df["phenotype"].unique()
    datasets = ["atleaf", "lit", "marine"]

    # Prepare data for plotting
    x_pos = np.arange(len(phenotypes))
    width = 0.25  # Width of each bar

    # Colors for stacks
    color_common = "#06A77D"
    color_unique_individual = "#F77F00"
    color_unique_combined = "#D62828"

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

        # Create stacked bars
        p1 = ax.bar(
            positions,
            common,
            width,
            label="Common" if i == 0 else "",
            color=color_common,
            edgecolor="black",
            linewidth=0.5,
        )
        p2 = ax.bar(
            positions,
            unique_individual,
            width,
            bottom=common,
            label="Unique to Individual" if i == 0 else "",
            color=color_unique_individual,
            edgecolor="black",
            linewidth=0.5,
        )

        # Stack unique_combined on top
        bottom = np.array(common) + np.array(unique_individual)
        p3 = ax.bar(
            positions,
            unique_combined,
            width,
            bottom=bottom,
            label="Unique to Combined" if i == 0 else "",
            color=color_unique_combined,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add dataset labels
        for j, pos in enumerate(positions):
            if j == len(positions) // 2:  # Label in the middle
                ax.text(
                    pos,
                    -1.5,
                    dataset.upper(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    style="italic",
                )

    # Customize plot
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.set_xlabel("Phenotype", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add legend
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, edgecolor="black")


def create_panel_c_plots(ax_top: Axes, ax_bottom: Axes) -> None:
    """Create both subplots for panel C.

    Parameters
    ----------
    ax_top : Axes
        Top subplot for feature stability plot.
    ax_bottom : Axes
        Bottom subplot for feature comparison plot.
    """
    print("Creating Figure 4C (feature stability and comparison)...")
    create_feature_stability_plot(ax_top)
    create_feature_comparison_plot(ax_bottom)
