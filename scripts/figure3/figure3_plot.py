#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

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


def plot_within_dataset_performance(
    ax: plt.Axes, data_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot within-dataset cross-validation performance.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load CV results
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    # Get dataset colors
    dataset_colors = get_dataset_colors()

    # Get unique phenotypes (sorted for consistency)
    if phenotypes is None:
        phenotypes = sorted(cv_df["phenotype"].unique())
    datasets = sorted(cv_df["dataset"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(datasets) - 1) / 2, width * (len(datasets) - 1) / 2, len(datasets)
    )

    # Plot stripplot for each dataset
    for idx, dataset in enumerate(datasets):
        dataset_data = cv_df[cv_df["dataset"] == dataset]

        # Plot individual points
        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            # Add jitter to x position
            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

            # Plot mean as horizontal line
            mean_val = phenotype_data["balanced_accuracy"].mean()
            ax.plot(
                [x_pos - 0.08, x_pos + 0.08],
                [mean_val, mean_val],
                color=dataset_colors[dataset],
                linewidth=3,
                alpha=0.9,
                zorder=3,
            )

    # Create legend handles
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dataset_colors[dataset],
            markersize=8,
            alpha=0.8,
            label=format_dataset_names([dataset])[0],
            linestyle="None",
        )
        for dataset in datasets
    ]

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Within-Dataset Performance (CV)", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Add horizontal line at 0.5 (random performance)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(datasets),
        frameon=False,
    )


def plot_cross_dataset_performance(
    ax: plt.Axes, data_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot cross-dataset test performance (train != test dataset).

    Shows models trained on AtLeaf and Marine datasets tested on Literature dataset.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load test results and CV results for comparison
    test_df = pd.read_csv(data_dir / "intra_vs_inter" / "test_results.csv")
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    # Filter for cross-dataset tests (train_dataset != test_dataset)
    # Only show test on Literature dataset, trained on AtLeaf and Marine
    cross_df = test_df[
        (test_df["train_dataset"] != test_df["test_dataset"])
        & (test_df["test_dataset"] == "lit")
        & (test_df["train_dataset"].isin(["atleaf", "marine"]))
    ].copy()

    # Get dataset colors
    dataset_colors = get_dataset_colors()

    # Get unique phenotypes and datasets
    if phenotypes is None:
        phenotypes = sorted(cross_df["phenotype"].unique())
    train_datasets = sorted(cross_df["train_dataset"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(train_datasets) - 1) / 2,
        width * (len(train_datasets) - 1) / 2,
        len(train_datasets),
    )

    # Plot cross-dataset test results
    for idx, train_dataset in enumerate(train_datasets):
        dataset_data = cross_df[cross_df["train_dataset"] == train_dataset]

        # Plot individual points
        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            # Add jitter
            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[train_dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

            # Plot mean as horizontal line
            if len(phenotype_data) > 0:
                mean_val = phenotype_data["balanced_accuracy"].mean()
                ax.plot(
                    [x_pos - 0.08, x_pos + 0.08],
                    [mean_val, mean_val],
                    color=dataset_colors[train_dataset],
                    linewidth=3,
                    alpha=0.9,
                    zorder=3,
                )

    # Calculate mean within-dataset CV performance for Literature (test dataset)
    cv_lit = (
        cv_df[cv_df["dataset"] == "lit"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )

    # Plot within-dataset performance as red reference lines
    for phenotype in phenotypes:
        if phenotype in cv_lit.index:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.4, x_pos + 0.4],
                [cv_lit[phenotype], cv_lit[phenotype]],
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Create legend handles
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dataset_colors[dataset],
            markersize=8,
            alpha=0.8,
            label=f"{format_dataset_names([dataset])[0]} → Literature",
            linestyle="None",
        )
        for dataset in train_datasets
    ]

    # Add within-dataset reference to legend
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Literature (intra-dataset)",
        )
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        "Cross-Dataset Test Performance (Test: Literature)", fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Add horizontal line at 0.5 (random performance)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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

    # Add legend
    ax.legend(
        handles=legend_handles,
        title="Train → Test",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(legend_handles),
        frameon=False,
    )


def plot_phylogeny_independent_performance(
    ax: plt.Axes, data_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot phylogenetically independent test performance.

    Shows cross-dataset test performance with phylogenetic control (in-clade filtering).
    Tests on Literature dataset, trained on AtLeaf dataset only.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load phylogeny-independent test results and CV results for reference
    phylo_df = pd.read_csv(data_dir / "phylo_indep" / "test_results.tsv", sep="\t")
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    # Filter for "full" test type (cross-dataset with phylogenetic filtering)
    # Based on the notebook, "full" means testing on full test dataset but with phylogenetic control
    phylo_full = phylo_df[phylo_df["test_type"] == "full"].copy()

    # Filter for test on Literature, train on AtLeaf (phylo_indep only has atleaf->lit based on notebook)
    phylo_cross = phylo_full[
        (phylo_full["train_dataset"] != phylo_full["test_dataset"])
        & (phylo_full["test_dataset"] == "lit")
        & (phylo_full["train_dataset"] == "atleaf")
    ].copy()

    # Get dataset colors
    dataset_colors = get_dataset_colors()

    # Get unique phenotypes and training datasets
    if phenotypes is None:
        phenotypes = sorted(phylo_cross["phenotype"].unique())
    train_datasets = sorted(phylo_cross["train_dataset"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(train_datasets) - 1) / 2,
        width * (len(train_datasets) - 1) / 2,
        len(train_datasets),
    )

    # Plot phylo-independent test results
    for idx, train_dataset in enumerate(train_datasets):
        dataset_data = phylo_cross[phylo_cross["train_dataset"] == train_dataset]

        # Plot individual points
        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            # Add jitter
            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[train_dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

            # Plot mean as horizontal line
            if len(phenotype_data) > 0:
                mean_val = phenotype_data["balanced_accuracy"].mean()
                ax.plot(
                    [x_pos - 0.08, x_pos + 0.08],
                    [mean_val, mean_val],
                    color=dataset_colors[train_dataset],
                    linewidth=3,
                    alpha=0.9,
                    zorder=3,
                )

    # Calculate mean within-dataset CV performance for Literature (test dataset)
    cv_lit = (
        cv_df[cv_df["dataset"] == "lit"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )

    # Plot within-dataset performance as red reference lines
    for phenotype in phenotypes:
        if phenotype in cv_lit.index:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.4, x_pos + 0.4],
                [cv_lit[phenotype], cv_lit[phenotype]],
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Create legend handles
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dataset_colors[dataset],
            markersize=8,
            alpha=0.8,
            label=f"{format_dataset_names([dataset])[0]} → Literature (phylo-filtered)",
            linestyle="None",
        )
        for dataset in train_datasets
    ]

    # Add within-dataset reference to legend
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Literature (intra-dataset)",
        )
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        "Phylogeny-Independent Test Performance (AtLeaf → Literature)",
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Add horizontal line at 0.5 (random performance)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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
        title="Train → Test",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(legend_handles),
        frameon=False,
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 3 with three subplots showing generalization failures.

    Demonstrates progressive performance degradation:
    (A) Within-dataset CV: High performance when train/test are from same dataset
    (B) Cross-dataset: Performance drop when testing on different dataset
    (C) Phylogeny-independent: Further drop with phylogenetic control

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    # Load all data to determine common phenotypes
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")
    test_df = pd.read_csv(data_dir / "intra_vs_inter" / "test_results.csv")
    phylo_df = pd.read_csv(data_dir / "phylo_indep" / "test_results.tsv", sep="\t")

    # Get phenotypes from each dataset
    cv_phenotypes = set(cv_df["phenotype"].unique())
    test_phenotypes = set(
        test_df[
            (test_df["train_dataset"] != test_df["test_dataset"])
            & (test_df["test_dataset"] == "lit")
            & (test_df["train_dataset"].isin(["atleaf", "marine"]))
        ]["phenotype"].unique()
    )
    phylo_phenotypes = set(
        phylo_df[
            (phylo_df["test_type"] == "full")
            & (phylo_df["train_dataset"] != phylo_df["test_dataset"])
            & (phylo_df["test_dataset"] == "lit")
            & (phylo_df["train_dataset"] == "atleaf")
        ]["phenotype"].unique()
    )

    # Use intersection of all phenotypes to ensure consistent x-axis
    common_phenotypes = sorted(
        cv_phenotypes.intersection(test_phenotypes).intersection(phylo_phenotypes)
    )

    # Create figure with 3 subplots arranged vertically with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # Plot each subplot with common phenotypes
    plot_within_dataset_performance(axes[0], data_dir, common_phenotypes)
    plot_cross_dataset_performance(axes[1], data_dir, common_phenotypes)
    plot_phylogeny_independent_performance(axes[2], data_dir, common_phenotypes)

    # Remove x-axis labels from top and middle plots
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")

    # Only show x-tick labels on bottom plot
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].tick_params(axis="x", labelbottom=False)

    # Adjust layout with more space between subplots
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure3")
    output_file = Path("figures/figure3.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
