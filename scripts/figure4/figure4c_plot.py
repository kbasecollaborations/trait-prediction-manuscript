#!/usr/bin/env python3
"""Draw Figure 4C: stable SHAP feature clusters per phenotype.

The top subplot counts the distinct redundancy clusters among the stable
features of the all-datasets-combined model; the bottom subplot splits the
per-dataset counts into clusters shared with the cross-dataset combined-training
model and clusters unique to the individual dataset.

Reads ``data/outputs/figure4/all_datasets_combined_shap_features.json``,
``data/outputs/figure4/feature_comparison_summary.csv`` and
``data/outputs/clustering/ko_clusters_shap_hclust.json``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

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
    data_file = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    with open(data_file) as f:
        feature_data = json.load(f)

    # Redundancy-cluster mapping (shap.utils.hclust): the combined-model count is
    # the number of distinct cluster identities rather than raw KOs, falling back
    # to KO counts when the cluster JSON is absent.
    cluster_file = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")
    cluster_mapping: dict[str, dict[str, int]] = {}
    if cluster_file.exists():
        with open(cluster_file) as f:
            cluster_mapping = json.load(f)

    def _cluster_count(phenotype: str) -> int:
        kos = feature_data.get(phenotype, [])
        ko_to_cluster = cluster_mapping.get(phenotype)
        if not kos:
            return 0
        if ko_to_cluster is None:
            return len(kos)
        identities: set[str] = set()
        for ko in kos:
            identities.add(
                f"c:{ko_to_cluster[ko]}" if ko in ko_to_cluster else f"singleton:{ko}"
            )
        return len(identities)

    feature_counts = [_cluster_count(p) for p in phenotypes]

    # Align bars with the bottom subplot's grouped-bar centers.
    x_pos = np.arange(len(phenotypes))
    n_datasets = 3  # atleaf, lit, marine
    bar_width_bottom = 0.8 / n_datasets  # width used in the bottom subplot

    bar_center_offset = bar_width_bottom * (n_datasets - 1) / 2
    bar_width = 0.5

    bars = ax.bar(
        x_pos + bar_center_offset,
        feature_counts,
        bar_width,
        # Pooled over all datasets.
        color="#595959",
        alpha=0.8,
    )

    # Alternating background bands centered on the bar positions (matching Figure 3).
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            center = i + bar_center_offset
            ax.axvspan(center - 0.5, center + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_ylabel(
        "Stable feature\nclusters (Combined)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1,
    )
    # Match tick positions to the bottom subplot.
    ax.set_xticks(x_pos + bar_center_offset)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=1)

    ax.set_xlim(bar_center_offset - 0.5, len(phenotypes) - 1 + bar_center_offset + 0.5)

    ax.set_ylim(0, 10)
    # Cluster counts are whole numbers, so fractional y-ticks are meaningless.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

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
    data_file = Path("data/outputs/figure4/feature_comparison_summary.csv")
    df = pd.read_csv(data_file)

    datasets = ["atleaf", "lit", "marine"]
    dataset_color_map = get_dataset_colors()
    dataset_colors = [dataset_color_map[d] for d in datasets]
    dataset_display_names = format_dataset_names(datasets)

    # Hatching patterns for feature types (matching Figure 1C).
    patterns = {
        "common": "",  # solid
        "unique_individual": "//",  # diagonal hatching
        "unique_combined": "xx",  # cross-hatching
    }

    # Cluster-level counts (redundancy clusters from shap.utils.hclust), falling
    # back to raw KO counts when those columns are absent.
    common_col = (
        "n_intersection_clusters"
        if "n_intersection_clusters" in df.columns
        else "n_intersection"
    )
    unique_col = (
        "n_unique_to_individual_clusters"
        if "n_unique_to_individual_clusters" in df.columns
        else "n_unique_to_individual"
    )

    x_pos = np.arange(len(phenotypes))
    bar_width = 0.8 / len(datasets)  # Match Figure 1C bar width calculation

    for i, dataset in enumerate(datasets):
        dataset_df = df[df["test_dataset"] == dataset].set_index("phenotype")

        common = []
        unique_individual = []

        for phenotype in phenotypes:
            if phenotype in dataset_df.index:
                row = dataset_df.loc[phenotype]
                common.append(row[common_col])
                unique_individual.append(row[unique_col])
            else:
                common.append(0)
                unique_individual.append(0)

        positions = x_pos + i * bar_width

        p1 = ax.bar(
            positions,
            common,
            bar_width,
            color=dataset_colors[i],
            alpha=0.8,
        )
        p2 = ax.bar(
            positions,
            unique_individual,
            bar_width,
            bottom=common,
            color=dataset_colors[i],
            alpha=0.4,
            hatch="//",
        )

    # Separate legends for datasets and feature types (matching Figure 1C).
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

    ax.legend(
        handles=feature_handles,
        title="Stable feature clusters",
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

    bar_group_center = bar_width * (len(datasets) - 1) / 2

    # Alternating background bands centered on the grouped bars (matching Figure 3).
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            center = i + bar_group_center
            ax.axvspan(center - 0.5, center + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_ylabel(
        "Stable feature\nclusters (Per Dataset)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1,
    )
    ax.set_xlabel("Phenotype", fontsize=AXIS_LABEL_SIZE)
    # Center x-tick labels under each group of bars (matching Figure 1C).
    ax.set_xticks(x_pos + bar_group_center)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=1)

    ax.set_xlim(bar_group_center - 0.5, len(phenotypes) - 1 + bar_group_center + 0.5)
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def create_panel_c_plots(ax_top: Axes, ax_bottom: Axes) -> None:
    """Create both subplots for panel C.

    Parameters
    ----------
    ax_top : Axes
        Top subplot for feature stability plot.
    ax_bottom : Axes
        Bottom subplot for feature comparison plot.
    """
    data_file1 = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    data_file2 = Path("data/outputs/figure4/feature_comparison_summary.csv")

    with open(data_file1) as f:
        feature_data = json.load(f)

    df = pd.read_csv(data_file2)

    phenotypes_set = set(feature_data.keys()) | set(df["phenotype"].unique())
    phenotypes = sorted(list(phenotypes_set))

    print("Creating Figure 4C (feature stability and comparison)...")
    create_feature_stability_plot(ax_top, phenotypes)
    create_feature_comparison_plot(ax_bottom, phenotypes)
