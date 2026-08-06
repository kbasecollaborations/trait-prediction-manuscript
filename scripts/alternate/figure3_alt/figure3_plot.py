#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Reproducible jitter
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
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    dataset_colors = get_dataset_colors()

    if phenotypes is None:
        phenotypes = sorted(cv_df["phenotype"].unique())
    datasets = sorted(cv_df["dataset"].unique())

    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(datasets) - 1) / 2, width * (len(datasets) - 1) / 2, len(datasets)
    )

    for idx, dataset in enumerate(datasets):
        dataset_data = cv_df[cv_df["dataset"] == dataset]

        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

            mean_val = phenotype_data["balanced_accuracy"].mean()
            ax.plot(
                [x_pos - 0.08, x_pos + 0.08],
                [mean_val, mean_val],
                color=dataset_colors[dataset],
                linewidth=3,
                alpha=0.9,
                zorder=3,
            )

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

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Within-Dataset Performance (CV)", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Random-performance reference at 0.5
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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

    ax.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(datasets),
        frameon=False,
    )


def plot_cross_dataset_performance(
    ax: plt.Axes, data_dir: Path, gapmind_dir: Path, phenotypes: list[str] | None = None
) -> None:
    """Plot cross-dataset test performance (train != test dataset).

    Shows models trained on AtLeaf and Marine datasets tested on Literature dataset,
    with GapMind predictions for comparison.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    gapmind_dir : Path
        Directory containing the GapMind metrics files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    test_df = pd.read_csv(data_dir / "intra_vs_inter" / "test_results.csv")
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    gapmind_df = pd.read_csv(gapmind_dir / "gapmind_loose_metrics.tsv", sep="\t")

    cross_df = test_df[
        (test_df["train_dataset"] != test_df["test_dataset"])
        & (test_df["test_dataset"] == "lit")
        & (test_df["train_dataset"].isin(["atleaf", "marine"]))
    ].copy()

    dataset_colors = get_dataset_colors()

    if phenotypes is None:
        phenotypes = sorted(cross_df["phenotype"].unique())
    train_datasets = sorted(cross_df["train_dataset"].unique())

    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(train_datasets) - 1) / 2,
        width * (len(train_datasets) - 1) / 2,
        len(train_datasets),
    )

    for idx, train_dataset in enumerate(train_datasets):
        dataset_data = cross_df[cross_df["train_dataset"] == train_dataset]

        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[train_dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

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

    # Within-dataset CV reference for Literature (the test dataset)
    cv_lit = (
        cv_df[cv_df["dataset"] == "lit"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )

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

    gapmind_color = "#A23B72"  # same purple as strict in figure2
    gapmind_dict = gapmind_df.set_index("phenotype")["balanced_accuracy"].to_dict()

    for phenotype in phenotypes:
        if phenotype in gapmind_dict:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.4, x_pos + 0.4],
                [gapmind_dict[phenotype], gapmind_dict[phenotype]],
                color=gapmind_color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

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

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        "Cross-Dataset Test Performance (Test: Literature)", fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Random-performance reference at 0.5
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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

    Cross-dataset test performance with phylogenetic control (in-clade
    filtering), trained on Marine and tested on Literature.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 3 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    phylo_df = pd.read_csv(data_dir / "phylo_indep" / "test_results.tsv", sep="\t")
    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")

    # "full" = full test dataset with phylogenetic control
    phylo_full = phylo_df[phylo_df["test_type"] == "full"].copy()

    phylo_cross = phylo_full[
        (phylo_full["train_dataset"] != phylo_full["test_dataset"])
        & (phylo_full["test_dataset"] == "lit")
        & (phylo_full["train_dataset"] == "marine")
    ].copy()

    dataset_colors = get_dataset_colors()

    if phenotypes is None:
        phenotypes = sorted(phylo_cross["phenotype"].unique())
    train_datasets = sorted(phylo_cross["train_dataset"].unique())

    x = np.arange(len(phenotypes))
    width = 0.25
    offsets = np.linspace(
        -width * (len(train_datasets) - 1) / 2,
        width * (len(train_datasets) - 1) / 2,
        len(train_datasets),
    )

    for idx, train_dataset in enumerate(train_datasets):
        dataset_data = phylo_cross[phylo_cross["train_dataset"] == train_dataset]

        for phenotype in phenotypes:
            phenotype_data = dataset_data[dataset_data["phenotype"] == phenotype]
            x_pos = x[phenotypes.index(phenotype)] + offsets[idx]

            x_jitter = x_pos + np.random.normal(0, 0.03, len(phenotype_data))

            ax.scatter(
                x_jitter,
                phenotype_data["balanced_accuracy"],
                color=dataset_colors[train_dataset],
                alpha=0.4,
                s=20,
                zorder=2,
            )

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

    # Within-dataset CV reference for Literature (the test dataset)
    cv_lit = (
        cv_df[cv_df["dataset"] == "lit"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )

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

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        "Phylogeny-Independent Test Performance (Marine → Literature)",
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.set_ylim(0, 1.05)

    # Random-performance reference at 0.5
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

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

    ax.legend(
        handles=legend_handles,
        title="Train → Test",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(legend_handles),
        frameon=False,
    )


def create_figure(
    data_dir: Path, output_file: Path, gapmind_dir: Path | None = None
) -> None:
    """Create Figure 3 with three subplots.

    (A) within-dataset CV, (B) cross-dataset test (with GapMind),
    (C) cross-dataset test with phylogenetic control.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    gapmind_dir : Path | None
        Directory containing GapMind metrics. If None, uses data/outputs/figure2.
    """
    if gapmind_dir is None:
        gapmind_dir = Path("data/outputs/figure2")

    cv_df = pd.read_csv(data_dir / "intra_vs_inter" / "cv_results.csv")
    test_df = pd.read_csv(data_dir / "intra_vs_inter" / "test_results.csv")
    phylo_df = pd.read_csv(data_dir / "phylo_indep" / "test_results.tsv", sep="\t")
    gapmind_df = pd.read_csv(gapmind_dir / "gapmind_loose_metrics.tsv", sep="\t")

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
            & (phylo_df["train_dataset"] == "marine")
        ]["phenotype"].unique()
    )
    gapmind_phenotypes = set(gapmind_df["phenotype"].unique())

    print("Determining common phenotypes across all analyses...")
    print(f" - Test phenotypes: {len(test_phenotypes)}")
    print(f" - Phylo-independent phenotypes: {len(phylo_phenotypes)}")
    print(f" - GapMind phenotypes: {len(gapmind_phenotypes)}")
    common_phenotypes = sorted(
        cv_phenotypes.intersection(test_phenotypes)
        .intersection(phylo_phenotypes)
        .intersection(gapmind_phenotypes)
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    plot_within_dataset_performance(axes[0], data_dir, common_phenotypes)
    plot_cross_dataset_performance(axes[1], data_dir, gapmind_dir, common_phenotypes)
    plot_phylogeny_independent_performance(axes[2], data_dir, common_phenotypes)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].tick_params(axis="x", labelbottom=False)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure3_alt")
    output_file = Path("figures/figure3_alt.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
