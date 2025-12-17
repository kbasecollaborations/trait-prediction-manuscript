#!/usr/bin/env python3
"""
Create Figure 6: Comprehensive analysis of model performance and GapMind predictions.

This figure includes:
- Panel A: GapMind misclassification patterns (3 subplots)
- Panel B: Performance on confident samples
- Panel C: Impact of filtering problematic samples (3 metrics)
- Panel D: Combined vs phenotype-filtered features (2 split types)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from scripts.figure6.figure6a_plot import create_misclassification_plots
from scripts.figure6.figure6b_plot import plot_confident_samples_performance
from scripts.figure6.figure6c_plot import (
    plot_gapmind_precision_recall_scatter,
    plot_precision_recall_scatter,
)
from scripts.figure6.figure6d_plot import (
    plot_balanced_accuracy_scatter,
    plot_precision_recall_scatter_by_feature_type,
    plot_split_comparison,
)

plt.style.use(["science", "nature"])


def create_figure6(output_file: Path) -> None:
    """Create Figure 6 with all panels arranged vertically.

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    # Data directories
    data_dir = Path("data/outputs/figure6")

    # Load data to determine common phenotypes
    print("Loading data files...")

    # Figure 6B data
    ml_df_6b = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")

    # Figure 6C data
    df_6c = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")

    # Figure 6D data
    df_6d = pd.read_csv(data_dir / "figure6d_all_results.csv")

    # Get phenotypes from each dataset
    phenotypes_6b_dataset = set(
        ml_df_6b[ml_df_6b["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    phenotypes_6b_random = set(
        ml_df_6b[ml_df_6b["split_type"] == "random_split"]["phenotype"].unique()
    )
    phenotypes_6c = set(df_6c["phenotype"].unique())
    phenotypes_6d = set(df_6d["phenotype"].unique())

    # Use intersection to ensure consistent x-axis across all panels
    print("Determining common phenotypes...")
    print(f" - Figure 6B dataset split: {len(phenotypes_6b_dataset)}")
    print(f" - Figure 6B random split: {len(phenotypes_6b_random)}")
    print(f" - Figure 6C: {len(phenotypes_6c)}")
    print(f" - Figure 6D: {len(phenotypes_6d)}")

    common_phenotypes = sorted(
        phenotypes_6b_dataset.intersection(phenotypes_6b_random)
        .intersection(phenotypes_6c)
        .intersection(phenotypes_6d)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Create figure with complex layout using GridSpec
    # Panel A: 3 plots horizontal (row 0)
    # Panel B: Performance comparison (row 1)
    # Panel C: Precision-recall scatter - 2 side-by-side (row 2)
    # Panel D: Feature comparison scatter - 2 side-by-side (row 3)
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(
        4, 2, figure=fig, height_ratios=[1, 1, 1.5, 1.5], hspace=0.3, wspace=0.4
    )

    # Panel A: GapMind misclassification patterns (3 subplots horizontal)
    # Create a nested GridSpec for Panel A with 3 columns
    # Adjust width ratios: give less space to first two plots, more to third plot with labels
    print("\nCreating Panel A: GapMind misclassification patterns...")
    gs_a = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, :], wspace=0.35, hspace=0, width_ratios=[1, 1, 1.5]
    )
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_a3 = fig.add_subplot(gs_a[0, 2])
    create_misclassification_plots(ax_a1, ax_a2, ax_a3)

    # Panel B: Performance on confident samples (spans all 2 columns)
    print("Creating Panel B: Performance on confident samples...")
    ax_b = fig.add_subplot(gs[1, :])
    plot_confident_samples_performance(ax_b, data_dir, common_phenotypes)

    # Panel C: Precision-recall scatter plots (2 side-by-side: ML and GapMind)
    print("Creating Panel C: Precision-recall scatter plots...")
    gs_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.3)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    plot_precision_recall_scatter(ax_c1, data_dir, common_phenotypes)
    plot_gapmind_precision_recall_scatter(ax_c2, common_phenotypes)

    # Panel D: Combined vs phenotype-filtered features (2 scatter plots side-by-side)
    print("Creating Panel D: Combined vs phenotype-filtered features...")
    gs_d = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :], wspace=0.3)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])
    plot_balanced_accuracy_scatter(ax_d1, df_6d, common_phenotypes)
    plot_precision_recall_scatter_by_feature_type(ax_d2, df_6d, common_phenotypes)

    # Collect all axes in order
    all_axes = [ax_a1, ax_a2, ax_a3, ax_b, ax_c1, ax_c2, ax_d1, ax_d2]

    # Add main panel labels
    panel_labels_map = {
        ax_a1: "(A)",  # First misclassification plot
        ax_b: "(B)",  # Confident samples
        ax_c1: "(C)",  # Precision-recall scatter (ML)
        ax_d1: "(D)",  # First feature comparison
    }

    for ax, label in panel_labels_map.items():
        # Use different x-position for Panel A due to nested GridSpec
        x_pos = -0.30 if ax == ax_a1 else -0.08
        ax.text(
            x_pos,
            1.05,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="right",
            fontsize=14,
        )

    # Remove x-tick labels from Panel B (phenotype bar plot)
    ax_b.set_xlabel("")
    ax_b.set_xticklabels([])

    # Adjust x-axis limits for phenotype bar plot (Panel B only)
    x_pos = np.arange(len(common_phenotypes))
    ax_b.set_xlim(-0.5, len(common_phenotypes) - 0.5)
    ax_b.set_xticks(x_pos)

    # Save figure
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure6.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure6(output_file)
