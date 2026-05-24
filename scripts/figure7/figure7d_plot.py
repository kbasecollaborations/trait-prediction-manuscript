#!/usr/bin/env python3
"""
Create Figure 6D: Comparing combined vs phenotype-filtered features.

This figure shows the performance difference between:
1. Combined features (GapMind + KOFAM + RAST)
2. Phenotype-filtered features (GapMind features specific to each phenotype)

Performance is shown for both random split and dataset split approaches.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import wilcoxon

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def _format_p_value(p_value: float | None) -> str:
    """Format a p-value compactly for panel annotations.

    Parameters
    ----------
    p_value : float | None
        P-value to format.

    Returns
    -------
    str
        Compact string representation for manuscript figures.
    """
    if p_value is None:
        return "n/a"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def _paired_panel_pvalue(
    left_values: pd.Series,
    right_values: pd.Series,
    phenotypes: list[str],
) -> tuple[float | None, int]:
    """Compute a paired phenotype-level Wilcoxon p-value.

    Parameters
    ----------
    left_values : pd.Series
        Phenotype-level values for the first condition. The series index must be
        phenotype names.
    right_values : pd.Series
        Phenotype-level values for the second condition. The series index must
        be phenotype names.
    phenotypes : list[str]
        Phenotypes displayed in the panel. Only overlapping phenotypes from this
        plotted list are used in the paired comparison.

    Returns
    -------
    tuple[float | None, int]
        The Wilcoxon p-value and the number of paired phenotypes used. Returns
        ``None`` when too few pairs are available or the test is not numerically
        well-defined.
    """
    paired_phenotypes = [
        phenotype
        for phenotype in phenotypes
        if phenotype in left_values.index and phenotype in right_values.index
    ]
    n_pairs = len(paired_phenotypes)
    if n_pairs < 2:
        return None, n_pairs

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            test_result = wilcoxon(
                left_values.loc[paired_phenotypes],
                right_values.loc[paired_phenotypes],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
    except ValueError:
        return None, n_pairs

    p_value = float(test_result.pvalue)
    if not np.isfinite(p_value):
        return None, n_pairs
    return p_value, n_pairs


def _add_pvalue_text(
    ax: Axes,
    random_p_value: float | None,
    random_n: int,
    dataset_p_value: float | None,
    dataset_n: int,
) -> None:
    """Add compact paired-test annotations for split-specific comparisons.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    random_p_value : float | None
        Paired Wilcoxon p-value for random split comparisons.
    random_n : int
        Number of random split phenotype pairs.
    dataset_p_value : float | None
        Paired Wilcoxon p-value for dataset split comparisons.
    dataset_n : int
        Number of dataset split phenotype pairs.
    """
    ax.text(
        0.02,
        1.04,
        (
            "Wilcoxon p: "
            f"random={_format_p_value(random_p_value)}; "
            f"dataset={_format_p_value(dataset_p_value)} "
            f"(n={min(random_n, dataset_n)})"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )


def _plot_feature_precision_recall_inset(
    ax: Axes,
    data: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """Plot a compact precision-recall inset for feature-set comparisons.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with all data.
    phenotypes : list[str]
        List of phenotypes to include.
    """
    inset_ax = ax.inset_axes([0.08, 0.62, 0.32, 0.32])
    summary = (
        data.groupby(["phenotype", "experiment", "split_type"])[["precision", "recall"]]
        .mean()
        .reset_index()
    )
    summary = summary[summary["phenotype"].isin(phenotypes)]

    colors = {"combined": "#2E86AB", "phenotype_filtered": "#E63946"}
    markers = {"random_split": "o", "dataset_split": "s"}

    for experiment in ["combined", "phenotype_filtered"]:
        for split_type in ["random_split", "dataset_split"]:
            subset = summary[
                (summary["experiment"] == experiment)
                & (summary["split_type"] == split_type)
            ]
            inset_ax.scatter(
                subset["recall"],
                subset["precision"],
                s=12,
                alpha=0.7,
                color=colors[experiment],
                edgecolors="black",
                linewidths=0.4,
                marker=markers[split_type],
                zorder=3,
            )

    inset_ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=0.8, zorder=1)
    inset_ax.set_xlim(0, 1.05)
    inset_ax.set_ylim(0, 1.05)
    inset_ax.set_xticks([0, 1.0])
    inset_ax.set_yticks([0, 1.0])
    inset_ax.tick_params(labelsize=6, pad=1)
    inset_ax.set_title("Precision-recall", fontsize=8, pad=2)
    inset_ax.set_aspect("equal")


def plot_split_comparison(
    ax: Axes,
    data: pd.DataFrame,
    split_type: str,
    phenotypes: list[str],
    title: str,
) -> None:
    """
    Plot comparison for a single split type.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe filtered for this split type.
    split_type : str
        Name of split type (for filtering).
    phenotypes : list[str]
        List of phenotypes in order.
    title : str
        Subplot title.
    """
    # Filter data for this split type
    split_data = data[data["split_type"] == split_type].copy()

    # Calculate mean and std for each phenotype and experiment
    summary = (
        split_data.groupby(["phenotype", "experiment"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    x = np.arange(len(phenotypes))
    width = 0.35

    # Separate combined and filtered data
    combined_data = summary[summary["experiment"] == "combined"].set_index("phenotype")
    filtered_data = summary[summary["experiment"] == "phenotype_filtered"].set_index(
        "phenotype"
    )

    # Align with phenotypes order
    combined_means = [
        combined_data.loc[p, "mean"] if p in combined_data.index else np.nan
        for p in phenotypes
    ]
    combined_stds = [
        combined_data.loc[p, "std"] if p in combined_data.index else 0
        for p in phenotypes
    ]
    filtered_means = [
        filtered_data.loc[p, "mean"] if p in filtered_data.index else np.nan
        for p in phenotypes
    ]
    filtered_stds = [
        filtered_data.loc[p, "std"] if p in filtered_data.index else 0
        for p in phenotypes
    ]

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        combined_means,
        width,
        yerr=combined_stds,
        label="Combined Features\n(GapMind + KOFAM + RAST)",
        color="#2E86AB",
        alpha=0.7,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        filtered_means,
        width,
        yerr=filtered_stds,
        label="Phenotype-Filtered\n(GapMind only)",
        color="#E63946",
        alpha=0.7,
        capsize=3,
    )

    # Add alternating background colors
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Phenotype")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)

    # Add feature count annotation
    # Get average feature counts for this split type
    combined_n_features = (
        split_data[split_data["experiment"] == "combined"]["n_features"]
        .mean()
    )
    filtered_n_features = (
        split_data[split_data["experiment"] == "phenotype_filtered"]["n_features"]
        .mean()
    )

    # Add text annotation for feature counts
    if not np.isnan(combined_n_features):
        ax.text(
            0.02,
            0.98,
            f"Combined: ~{int(combined_n_features)} features",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            color="#2E86AB",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    if not np.isnan(filtered_n_features):
        ax.text(
            0.02,
            0.90,
            f"Filtered: ~{int(filtered_n_features)} features (avg)",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            color="#E63946",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def plot_balanced_accuracy_scatter(
    ax: Axes,
    data: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """
    Plot scatter plot of Combined vs Filtered balanced accuracy.

    Shapes indicate split type (circle=random, square=dataset).
    Points above the diagonal indicate filtered features outperform combined.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with all data.
    phenotypes : list[str]
        List of phenotypes to include.
    """
    # Calculate mean balanced accuracy for each phenotype, experiment, and split_type
    summary = (
        data.groupby(["phenotype", "experiment", "split_type"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )

    # Define markers for split type (no fill colors, just shape difference)
    markers = {"random_split": "o", "dataset_split": "s"}
    split_labels = {"random_split": "Random Split", "dataset_split": "Dataset Split"}
    random_p_value = None
    random_n = 0
    dataset_p_value = None
    dataset_n = 0

    # Plot combined vs filtered for each phenotype and split type
    for split_type in ["random_split", "dataset_split"]:
        split_data = summary[summary["split_type"] == split_type]

        combined = split_data[split_data["experiment"] == "combined"].set_index(
            "phenotype"
        )["balanced_accuracy"]
        filtered = split_data[
            split_data["experiment"] == "phenotype_filtered"
        ].set_index("phenotype")["balanced_accuracy"]

        # Get common phenotypes
        common = [p for p in phenotypes if p in combined.index and p in filtered.index]

        combined_vals = [combined.loc[p] for p in common]
        filtered_vals = [filtered.loc[p] for p in common]

        ax.scatter(
            combined_vals,
            filtered_vals,
            s=42,
            alpha=0.7,
            facecolors="white",
            edgecolors="black",
            linewidths=1,
            marker=markers[split_type],
            label=split_labels[split_type],
            zorder=3,
        )

        p_value, n_pairs = _paired_panel_pvalue(combined, filtered, common)
        if split_type == "random_split":
            random_p_value = p_value
            random_n = n_pairs
        else:
            dataset_p_value = p_value
            dataset_n = n_pairs

    # Add diagonal line (y = x)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    # Formatting
    ax.set_xlabel("Combined Features\n(Balanced Accuracy)")
    ax.set_ylabel("Phenotype-Filtered\n(Balanced Accuracy)")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=8,
        labelspacing=0.6,
        markerscale=0.7,
    )
    ax.set_aspect("equal")
    _add_pvalue_text(ax, random_p_value, random_n, dataset_p_value, dataset_n)
    _plot_feature_precision_recall_inset(ax, data, phenotypes)


def plot_precision_recall_scatter_by_feature_type(
    ax: Axes,
    data: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """
    Plot precision vs recall scatter plot.

    Shapes indicate split type (circle=random, square=dataset).
    Colors indicate filter type (blue=combined, red=phenotype-filtered).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with all data.
    phenotypes : list[str]
        List of phenotypes to include.
    """
    # Calculate mean precision and recall for each phenotype, experiment, and split_type
    summary = (
        data.groupby(["phenotype", "experiment", "split_type"])[["precision", "recall"]]
        .mean()
        .reset_index()
    )

    # Filter to phenotypes of interest
    summary = summary[summary["phenotype"].isin(phenotypes)]

    # Define colors for filter type and markers for split type
    colors = {"combined": "#2E86AB", "phenotype_filtered": "#E63946"}
    markers = {"random_split": "o", "dataset_split": "s"}
    filter_labels = {
        "combined": "Combined Features",
        "phenotype_filtered": "Phenotype-Filtered",
    }
    split_labels = {"random_split": "Random Split", "dataset_split": "Dataset Split"}

    # Plot each combination of experiment and split type
    for experiment in ["combined", "phenotype_filtered"]:
        for split_type in ["random_split", "dataset_split"]:
            subset = summary[
                (summary["experiment"] == experiment)
                & (summary["split_type"] == split_type)
            ]

            # Create combined label for legend
            label = f"{filter_labels[experiment]} ({split_labels[split_type]})"

            ax.scatter(
                subset["recall"],
                subset["precision"],
                s=200,
                alpha=0.7,
                color=colors[experiment],
                edgecolors="black",
                linewidths=2,
                marker=markers[split_type],
                label=label,
                zorder=3,
            )

    # Add diagonal line (precision = recall)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    # Formatting
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=7, labelspacing=1.2)
    ax.set_aspect("equal")


def create_figure(
    data_file: Path,
    output_file: Path,
    phenotype_order: list[str] | None = None,
) -> None:
    """
    Create Figure 6D showing combined vs phenotype-filtered features comparison.

    Parameters
    ----------
    data_file : Path
        Path to CSV file with results from figure7d_data.py.
    output_file : Path
        Path to save the output figure.
    phenotype_order : list[str] | None
        Order of phenotypes for x-axis. If None, uses alphabetical order.
    """
    # Load data
    df = pd.read_csv(data_file)

    # Apply the manuscript's minority-class-in-test filter (Methods). Fig 6D
    # evaluates on the full held-out test set; dataset_split rows whose
    # held-out test minority count falls below the threshold are dropped.
    from scripts.minority_filter import (
        filter_by_minority,
        full_test_minority_counts,
    )

    df = filter_by_minority(df, full_test_minority_counts())

    # Get unique phenotypes
    if phenotype_order is None:
        phenotypes = sorted(df["phenotype"].unique())
    else:
        # Filter to only phenotypes that are in the data
        phenotypes = [p for p in phenotype_order if p in df["phenotype"].values]

    # Create figure with 2 subplots (one for each split type)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot random split
    plot_split_comparison(
        axes[0],
        df,
        "random_split",
        phenotypes,
        title="A. Random Split: Combined vs Phenotype-Filtered Features",
    )

    # Plot dataset split
    plot_split_comparison(
        axes[1],
        df,
        "dataset_split",
        phenotypes,
        title="B. Dataset Split: Combined vs Phenotype-Filtered Features",
    )

    # Adjust layout
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary: Combined vs Phenotype-Filtered Features")
    print("=" * 80)

    for split_type in ["random_split", "dataset_split"]:
        split_data = df[df["split_type"] == split_type]

        print(f"\n{split_type.replace('_', ' ').title()}:")

        for experiment in ["combined", "phenotype_filtered"]:
            exp_data = split_data[split_data["experiment"] == experiment]
            if len(exp_data) > 0:
                mean_ba = exp_data["balanced_accuracy"].mean()
                std_ba = exp_data["balanced_accuracy"].std()
                mean_features = exp_data["n_features"].mean()
                print(f"  {experiment.replace('_', ' ').title()}:")
                print(f"    Balanced Accuracy: {mean_ba:.4f} ± {std_ba:.4f}")
                print(f"    Features: ~{int(mean_features)}")

        # Calculate difference
        combined_mean = split_data[split_data["experiment"] == "combined"][
            "balanced_accuracy"
        ].mean()
        filtered_mean = split_data[split_data["experiment"] == "phenotype_filtered"][
            "balanced_accuracy"
        ].mean()

        if not np.isnan(combined_mean) and not np.isnan(filtered_mean):
            diff = filtered_mean - combined_mean
            pct_diff = (diff / combined_mean) * 100 if combined_mean != 0 else 0
            print(f"  Difference (Filtered - Combined): {diff:+.4f} ({pct_diff:+.2f}%)")

    # Print per-phenotype comparison
    print("\n" + "=" * 80)
    print("Per-Phenotype Comparison (Dataset Split)")
    print("=" * 80)

    dataset_split_data = df[df["split_type"] == "dataset_split"]
    phenotype_summary = (
        dataset_split_data.groupby(["phenotype", "experiment"])["balanced_accuracy"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    if "combined" in phenotype_summary.columns and "phenotype_filtered" in phenotype_summary.columns:
        phenotype_summary["difference"] = (
            phenotype_summary["phenotype_filtered"] - phenotype_summary["combined"]
        )
        phenotype_summary = phenotype_summary.sort_values("difference", ascending=False)
        print(phenotype_summary.round(4))


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure7")
    data_file = data_dir / "figure7d_all_results.csv"
    output_file = Path("figures/figure7d.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Define phenotype order (matching common phenotypes from figure3)
    phenotype_order = [
        "Alanine",
        "Arginine",
        "Histidine",
        "Serine",
        "Fructose",
        "Galactose",
        "Maltose",
        "Mannose",
        "Sucrose",
        "m-Inositol",
        "Mannitol",
        "Glycerol",
        "Galacturonic-Acid",
        "Cellobiose",
    ]

    create_figure(data_file, output_file, phenotype_order)
