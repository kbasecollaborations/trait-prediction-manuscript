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
    loose_df["confidence"] = "Permissive"

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
        label="Permissive",
        color=color_loose,
        alpha=0.8,
    )

    # Add horizontal lines for mean balanced accuracy of each confidence level
    mean_strict = strict_df["balanced_accuracy"].mean()
    mean_loose = loose_df["balanced_accuracy"].mean()

    ax.axhline(
        y=mean_strict,
        color=color_strict,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Strict Mean ({mean_strict:.2f})",
    )
    ax.axhline(
        y=mean_loose,
        color=color_loose,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Permissive Mean ({mean_loose:.2f})",
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
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
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
    # Load data (only Random split and Out-of-Clade split)
    random_df = pd.read_csv(data_dir / "random_split_baselines.tsv", sep="\t")
    phylo_df = pd.read_csv(data_dir / "out_of_clade_split_baselines.tsv", sep="\t")

    # Add split type column
    random_df["split"] = "Random split"
    phylo_df["split"] = "Out-of-Clade split"

    # Combine dataframes
    df = pd.concat([random_df, phylo_df], ignore_index=True)

    # Define split order and model order
    split_order = ["Random split", "Out-of-Clade split"]
    model_order = ["identity", "bernoulli", "nearest_neighbor"]
    model_labels = {
        "identity": "Identity",
        "bernoulli": "Bernoulli",
        "nearest_neighbor": "Nearest Neighbor",
    }

    # Rename models for display
    df["model"] = df["model"].map(model_labels)

    # Define colors and markers for each model
    palette = {
        "Identity": "#A0A0A0",  # Light gray
        "Bernoulli": "#707070",  # Medium gray
        "Nearest Neighbor": "#000000",  # Black
    }

    markers = {
        "Identity": "o",  # Circle
        "Bernoulli": "s",  # Square
        "Nearest Neighbor": "^",  # Triangle
    }

    # Get unique phenotypes across all splits
    phenotypes = sorted(df["phenotype"].unique())

    # Plot each split type in a separate subplot
    for idx, split_type in enumerate(split_order):
        ax = axes[idx]
        split_df = df[df["split"] == split_type]

        # Plot each model separately with correct markers
        for model_name in model_labels.values():
            model_df = split_df[split_df["model"] == model_name]

            # Get x positions for each phenotype
            x_positions = []
            y_values = []
            for _, row in model_df.iterrows():
                x_pos = phenotypes.index(row["phenotype"])
                # Add jitter
                x_jitter = x_pos + np.random.uniform(-0.15, 0.15)
                x_positions.append(x_jitter)
                y_values.append(row["balanced_accuracy"])

            # Plot with correct marker for this model
            ax.scatter(
                x_positions,
                y_values,
                marker=markers[model_name],
                color=palette[model_name],
                alpha=0.6,
                s=50,
                label=model_name,
            )

        # Add mean lines for each model and phenotype
        for phenotype in phenotypes:
            phenotype_data = split_df[split_df["phenotype"] == phenotype]
            x_pos = phenotypes.index(phenotype)

            for model_name in model_labels.values():
                model_data = phenotype_data[phenotype_data["model"] == model_name][
                    "balanced_accuracy"
                ]
                if len(model_data) > 0:
                    mean_val = model_data.mean()
                    ax.plot(
                        [x_pos - 0.15, x_pos + 0.15],
                        [mean_val, mean_val],
                        color=palette[model_name],
                        linewidth=2,
                        alpha=0.9,
                    )

        # Formatting
        ax.set_title(split_type, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

        # Set x-ticks and labels
        ax.set_xticks(range(len(phenotypes)))
        ax.set_xticklabels(phenotypes)
        ax.set_xlim(-0.5, len(phenotypes) - 0.5)

        # Only show x-tick labels on the bottom subplot
        if idx == len(split_order) - 1:
            ax.set_xlabel("Phenotype")
            ax.tick_params(
                axis="x", which="both", top=False, bottom=True, labelbottom=True
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.tick_params(
                axis="x", which="both", top=False, bottom=True, labelbottom=False
            )

    # Create custom legend with markers
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[model],
            color="w",
            markerfacecolor=palette[model],
            markersize=8,
            alpha=0.8,
            label=model,
            linestyle="None",
        )
        for model in model_labels.values()
    ]

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
