#!/usr/bin/env python3
"""
Create Figure 6C: Impact of filtering problematic samples on precision/recall/AUPRC.

This figure shows how removing GapMind-misclassified samples (identified in Figure 6A)
affects ML model performance metrics across all phenotypes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.axes import Axes

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_metric_comparison(
    ax: Axes,
    data: pd.DataFrame,
    metric: str,
    phenotypes: list[str],
    ylabel: str,
    title: str,
) -> None:
    """
    Plot before/after comparison for a single metric.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with columns: phenotype, condition, metric.
    metric : str
        Name of the metric column to plot.
    phenotypes : list[str]
        List of phenotypes in order.
    ylabel : str
        Y-axis label.
    title : str
        Subplot title.
    """
    # Calculate mean and std for each phenotype and condition
    summary = (
        data.groupby(["phenotype", "condition"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    x = np.arange(len(phenotypes))
    width = 0.35

    # Separate full and filtered data
    full_data = summary[summary["condition"] == "full"].set_index("phenotype")
    filtered_data = summary[summary["condition"] == "filtered"].set_index("phenotype")

    # Align with phenotypes order
    full_means = [full_data.loc[p, "mean"] if p in full_data.index else 0 for p in phenotypes]
    full_stds = [full_data.loc[p, "std"] if p in full_data.index else 0 for p in phenotypes]
    filtered_means = [
        filtered_data.loc[p, "mean"] if p in filtered_data.index else 0 for p in phenotypes
    ]
    filtered_stds = [
        filtered_data.loc[p, "std"] if p in filtered_data.index else 0 for p in phenotypes
    ]

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        full_means,
        width,
        yerr=full_stds,
        label="Before Filtering",
        color="#E63946",
        alpha=0.7,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        filtered_means,
        width,
        yerr=filtered_stds,
        label="After Filtering",
        color="#06A77D",
        alpha=0.7,
        capsize=3,
    )

    # Add alternating background colors
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    # Formatting
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Phenotype")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)

    # Add horizontal line at baseline (if appropriate)
    if metric == "auprc":
        # Calculate baseline (proportion of positive class)
        # This is approximate; actual baseline varies by phenotype
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)


def create_figure(
    data_file: Path,
    output_file: Path,
    phenotype_order: list[str] | None = None,
) -> None:
    """
    Create Figure 6C showing impact of filtering on precision, recall, and AUPRC.

    Parameters
    ----------
    data_file : Path
        Path to CSV file with results from figure6c_data.py.
    output_file : Path
        Path to save the output figure.
    phenotype_order : list[str] | None
        Order of phenotypes for x-axis. If None, uses alphabetical order.
    """
    # Load data
    df = pd.read_csv(data_file)

    # Get unique phenotypes
    if phenotype_order is None:
        phenotypes = sorted(df["phenotype"].unique())
    else:
        phenotypes = phenotype_order

    # Create figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Plot precision
    plot_metric_comparison(
        axes[0],
        df,
        "precision",
        phenotypes,
        ylabel="Precision",
        title="A. Precision Before vs After Filtering",
    )

    # Plot recall
    plot_metric_comparison(
        axes[1],
        df,
        "recall",
        phenotypes,
        ylabel="Recall",
        title="B. Recall Before vs After Filtering",
    )

    # Plot AUPRC
    plot_metric_comparison(
        axes[2],
        df,
        "auprc",
        phenotypes,
        ylabel="AUPRC (Average Precision)",
        title="C. AUPRC Before vs After Filtering",
    )

    # Adjust layout
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary: Average change after filtering")
    print("=" * 80)

    for metric in ["precision", "recall", "auprc"]:
        summary = df.groupby("condition")[metric].mean()
        if "full" in summary.index and "filtered" in summary.index:
            before = summary["full"]
            after = summary["filtered"]
            change = after - before
            pct_change = (change / before) * 100 if before != 0 else 0
            print(f"\n{metric.upper()}:")
            print(f"  Before: {before:.4f}")
            print(f"  After:  {after:.4f}")
            print(f"  Change: {change:+.4f} ({pct_change:+.2f}%)")


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure6")
    data_file = data_dir / "figure6c_dataset_split_results.csv"
    output_file = Path("figures/figure6c.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    create_figure(data_file, output_file)
