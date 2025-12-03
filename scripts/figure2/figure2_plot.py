#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.patches import Rectangle

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_color_list,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_gapmind_comparison(ax: plt.Axes, data_dir: Path) -> None:
    """Plot comparison of strict vs loose GapMind confidence thresholds.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the GapMind metrics files.
    """
    # Load data
    strict_df = pd.read_csv(data_dir / "gapmind_strict_metrics.tsv", sep="\t")
    loose_df = pd.read_csv(data_dir / "gapmind_loose_metrics.tsv", sep="\t")

    # Add confidence level column
    strict_df["confidence"] = "Strict"
    loose_df["confidence"] = "Loose"

    # Combine dataframes
    df = pd.concat([strict_df, loose_df], ignore_index=True)

    # Get unique phenotypes (sorted for consistency)
    phenotypes = sorted(df["phenotype"].unique())

    # Set up positions for grouped bars
    x = np.arange(len(phenotypes))
    width = 0.35

    # Get data for each confidence level
    strict_data = strict_df.set_index("phenotype").reindex(phenotypes)
    loose_data = loose_df.set_index("phenotype").reindex(phenotypes)

    # Define colors for strict and loose (reversed)
    color_strict = "#A23B72"  # Purple
    color_loose = "#2E86AB"  # Blue

    # Plot bars
    ax.bar(
        x - width / 2,
        strict_data["balanced_accuracy"],
        width,
        label="Strict",
        color=color_strict,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        loose_data["balanced_accuracy"],
        width,
        label="Loose",
        color=color_loose,
        alpha=0.8,
    )

    # Add horizontal line at mean balanced accuracy
    mean_ba = df["balanced_accuracy"].mean()
    ax.axhline(
        y=mean_ba,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Mean ({mean_ba:.2f})",
    )

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")
    ax.set_ylim(0, 1)

    # Add subplot label
    ax.text(
        -0.05,
        1.05,
        "(A)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Add legend at the top in a single row
    ax.legend(
        title="GapMind Confidence",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )


def plot_baseline_comparison(axes: np.ndarray, data_dir: Path) -> None:
    """Plot comparison of nearest neighbor model against null models across split types.

    Parameters
    ----------
    axes : np.ndarray
        Array of matplotlib axes objects (2 subplots for 2 split types).
    data_dir : Path
        Directory containing the baseline metrics files.
    """
    # Load data (only Random and Out-of-Clade)
    random_df = pd.read_csv(data_dir / "random_split_baselines.tsv", sep="\t")
    phylo_df = pd.read_csv(data_dir / "out_of_clade_split_baselines.tsv", sep="\t")

    # Add split type column
    random_df["split"] = "Random"
    phylo_df["split"] = "Out-of-Clade"

    # Combine dataframes
    df = pd.concat([random_df, phylo_df], ignore_index=True)

    # Define split order and model order
    split_order = ["Random", "Out-of-Clade"]
    model_order = ["identity", "bernoulli", "nearest_neighbor"]
    model_labels = ["Identity", "Bernoulli", "Nearest Neighbor"]

    # Define colors for each model
    colors = {
        "identity": "#808080",  # Gray
        "bernoulli": "#F18F01",  # Orange
        "nearest_neighbor": "#06A77D",  # Green
    }

    # Get unique phenotypes across all splits
    phenotypes = sorted(df["phenotype"].unique())

    # Define markers for each model
    markers = {
        "identity": "o",  # Circle
        "bernoulli": "s",  # Square
        "nearest_neighbor": "^",  # Triangle
    }

    # Create legend handles for top placement
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[model],
            color="w",
            markerfacecolor=colors[model],
            markersize=8,
            alpha=0.8,
            label=label,
            linestyle="None",
        )
        for model, label in zip(model_order, model_labels)
    ]

    # Set random seed for reproducible jitter
    np.random.seed(42)

    # Plot each split type in a separate subplot
    for idx, split_type in enumerate(split_order):
        ax = axes[idx]
        split_df = df[df["split"] == split_type]

        # Set up positions
        x = np.arange(len(phenotypes))
        jitter_width = 0.08
        offset_scale = 0.15

        # Plot strip plots for each model
        for i, (model, label) in enumerate(zip(model_order, model_labels)):
            model_df = split_df[split_df["model"] == model]

            # Calculate offset for each model
            offset = (i - 1) * offset_scale

            # Plot points for each phenotype
            for j, phenotype in enumerate(phenotypes):
                phenotype_data = model_df[model_df["phenotype"] == phenotype][
                    "balanced_accuracy"
                ].values

                if len(phenotype_data) > 0:
                    # Add jitter
                    jitter = np.random.uniform(
                        -jitter_width, jitter_width, size=len(phenotype_data)
                    )
                    x_positions = x[j] + offset + jitter

                    # Plot strip
                    ax.scatter(
                        x_positions,
                        phenotype_data,
                        color=colors[model],
                        marker=markers[model],
                        alpha=0.6,
                        s=30,
                        edgecolors="none",
                    )

                    # Calculate and plot mean line
                    mean_val = phenotype_data.mean()
                    ax.plot(
                        [x[j] + offset - jitter_width * 1.5, x[j] + offset + jitter_width * 1.5],
                        [mean_val, mean_val],
                        color=colors[model],
                        linewidth=2,
                        alpha=0.9,
                    )

        # Formatting
        ax.set_title(split_type, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_xticks(x)

        # Only show x-tick labels on the bottom subplot
        if idx == len(split_order) - 1:
            ax.set_xticklabels(phenotypes, rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis="x", which="both", top=False, bottom=True)
        ax.tick_params(axis="y")
        ax.set_ylim(0, 1)

    # Add legend at the top in a single row (above the first subplot)
    axes[0].legend(
        handles=legend_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
    )

    # Add subplot label to the first subplot
    axes[0].text(
        -0.08,
        1.05,
        "(B)",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Add x-label only to the bottom subplot
    axes[1].set_xlabel("Phenotype")


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 2 with multiple subplots.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    # Create figure with nested gridspec for better control over spacing
    fig = plt.figure(figsize=(12, 14))

    # Top level gridspec: 2 rows (A and B section)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.6)

    # Nested gridspec for B section: 2 rows with less spacing
    gs_b = gs_main[1].subgridspec(2, 1, hspace=0.25)

    # Create subplots with shared x-axis for B section
    ax_a = fig.add_subplot(gs_main[0])
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)

    # Plot subplot A: GapMind comparison
    plot_gapmind_comparison(ax_a, data_dir)

    # Plot subplot B: Baseline comparison (2 rows)
    plot_baseline_comparison(np.array([ax_b1, ax_b2]), data_dir)

    # Adjust layout
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure2")
    output_file = Path("figures/figure2.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
