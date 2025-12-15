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


def plot_random_split_vs_gapmind(
    ax: plt.Axes, data_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot random split performance as box plot with GapMind baseline.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load random split results
    ml_df = pd.read_csv(data_dir / "ml_results.csv")
    random_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Load GapMind results for random split test sets
    gapmind_df = pd.read_csv(data_dir / "gapmind_random_split_metrics.tsv", sep="\t")

    # Get unique phenotypes (sorted for consistency)
    if phenotypes is None:
        phenotypes = sorted(random_df["phenotype"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Prepare data for box plot
    box_data = [
        random_df[random_df["phenotype"] == phenotype]["balanced_accuracy"].values
        for phenotype in phenotypes
    ]

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="#06A77D", alpha=0.7, linewidth=1.5),
        medianprops=dict(color="#044d29", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Plot GapMind results as horizontal lines (mean across random splits for each phenotype)
    gapmind_color = "#A23B72"  # Purple
    gapmind_means = gapmind_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()

    for phenotype in phenotypes:
        if phenotype in gapmind_means:
            x_pos = phenotypes.index(phenotype)
            ax.plot(
                [x_pos - 0.4, x_pos + 0.4],
                [gapmind_means[phenotype], gapmind_means[phenotype]],
                color=gapmind_color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                zorder=3,
            )

    # Create legend handles
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#06A77D", alpha=0.7, label="Random Split"),
        Line2D(
            [0],
            [0],
            color=gapmind_color,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="GapMind",
        ),
    ]

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)
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
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )


def plot_dataset_split_performance(
    ax: plt.Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """Plot dataset split performance with baselines.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load data
    ml_df = pd.read_csv(data_dir / "ml_results.csv")
    dataset_df = ml_df[ml_df["split_type"] == "dataset_split"].copy()
    random_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Load GapMind results for dataset split test sets
    gapmind_df = pd.read_csv(data_dir / "gapmind_dataset_split_metrics.tsv", sep="\t")

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

    # Plot GapMind results as dashed lines (mean across dataset splits for each phenotype)
    gapmind_color = "#A23B72"  # Purple
    gapmind_means = gapmind_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()

    for phenotype in phenotypes:
        if phenotype in gapmind_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [gapmind_means[phenotype], gapmind_means[phenotype]],
                color=gapmind_color,
                linestyle="--",
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

    # Add GapMind reference to legend
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=gapmind_color,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="GapMind",
        )
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)
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
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(legend_handles),
        frameon=False,
    )


def plot_phylogeny_independent_difference(
    ax: plt.Axes, data_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot difference between full and in-clade phylogeny-independent performance.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    # Load phylogeny-independent results
    phylo_df = pd.read_csv(data_dir / "figure3c_results.csv")

    # Extract test dataset from train_test_config
    phylo_df["test_dataset"] = phylo_df["train_test_config"].apply(extract_test_dataset)

    # Pivot to get full and in-clade side by side
    phylo_pivot = phylo_df.pivot_table(
        index=["phenotype", "train_test_config", "test_dataset"],
        columns="test_type",
        values="balanced_accuracy",
    ).reset_index()

    # Calculate difference (full - in-clade)
    phylo_pivot["difference"] = phylo_pivot["full"] - phylo_pivot["in-clade"]

    # Get dataset colors
    dataset_colors = get_dataset_colors()

    # Get unique phenotypes and test datasets
    if phenotypes is None:
        phenotypes = sorted(phylo_pivot["phenotype"].unique())
    test_datasets = sorted(phylo_pivot["test_dataset"].unique())

    # Set up positions
    x = np.arange(len(phenotypes))

    # Calculate mean difference across all datasets and phenotypes
    mean_difference = phylo_pivot["difference"].mean()

    # Plot difference for each test dataset
    for idx, test_dataset in enumerate(test_datasets):
        test_data = phylo_pivot[phylo_pivot["test_dataset"] == test_dataset]

        # Plot individual difference points
        for phenotype in phenotypes:
            phenotype_data = test_data[test_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)]

            # Plot vertical line connecting dot to x-axis for each point
            for diff_val in phenotype_data["difference"]:
                ax.plot(
                    [x_pos, x_pos],
                    [0, diff_val],
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.3,
                    zorder=1,
                )

            # Add small jitter for visibility when multiple datasets overlap
            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["difference"],
                color=dataset_colors[test_dataset],
                alpha=0.6,
                s=60,
                zorder=2,
            )

    # Add horizontal line at mean difference
    ax.axhline(
        y=mean_difference,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
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
            label=f"Test: {format_dataset_names([dataset])[0]}",
            linestyle="None",
        )
        for dataset in test_datasets
    ]

    # Add mean difference line to legend
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean Difference ({mean_difference:.3f})",
        )
    )

    # Add alternating background colors for x-axis categories
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy Difference\n(Full - In-Clade)")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(-0.5, 0.5)

    # Add horizontal line at 0 (no difference)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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
    """Create Figure 3 with three subplots.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    # Load all data to determine common phenotypes
    ml_df = pd.read_csv(data_dir / "ml_results.csv")
    phylo_df = pd.read_csv(data_dir / "figure3c_results.csv")
    gapmind_random_df = pd.read_csv(data_dir / "gapmind_random_split_metrics.tsv", sep="\t")
    gapmind_dataset_df = pd.read_csv(data_dir / "gapmind_dataset_split_metrics.tsv", sep="\t")

    # Get phenotypes from each dataset
    random_phenotypes = set(
        ml_df[ml_df["split_type"] == "random_split"]["phenotype"].unique()
    )
    dataset_phenotypes = set(
        ml_df[ml_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    phylo_phenotypes = set(phylo_df["phenotype"].unique())
    gapmind_random_phenotypes = set(gapmind_random_df["phenotype"].unique())
    gapmind_dataset_phenotypes = set(gapmind_dataset_df["phenotype"].unique())

    # Use intersection of all phenotypes to ensure consistent x-axis
    print("Determining common phenotypes across all analyses...")
    print(f" - Random split phenotypes: {len(random_phenotypes)}")
    print(f" - Dataset split phenotypes: {len(dataset_phenotypes)}")
    print(f" - Phylo-independent phenotypes: {len(phylo_phenotypes)}")
    print(f" - GapMind random split phenotypes: {len(gapmind_random_phenotypes)}")
    print(f" - GapMind dataset split phenotypes: {len(gapmind_dataset_phenotypes)}")
    common_phenotypes = sorted(
        random_phenotypes.intersection(dataset_phenotypes)
        .intersection(phylo_phenotypes)
        .intersection(gapmind_random_phenotypes)
        .intersection(gapmind_dataset_phenotypes)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Create figure with 3 subplots arranged vertically with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot each subplot with common phenotypes
    # Both Panel A and B now use test-set-specific GapMind metrics
    plot_random_split_vs_gapmind(axes[0], data_dir, common_phenotypes)
    plot_dataset_split_performance(axes[1], data_dir, common_phenotypes)
    plot_phylogeny_independent_difference(axes[2], data_dir, common_phenotypes)

    # Remove x-axis labels from all but bottom plot
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
