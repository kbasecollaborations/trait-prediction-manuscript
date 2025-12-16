#!/usr/bin/env python3
"""
Generate plots for Figure 7: Data requirements for model performance.

Creates visualizations showing how training data size affects model performance
across different split types and training data quality (full vs concordant).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


# Feature type to use: "gapmind", "kofam", or "rast"
# Change this line to match the feature type used in figure7_data.py
FEATURE_TYPE = "gapmind"


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for plotting by formatting labels and names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results dataframe.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe ready for plotting.
    """
    plot_data = df.copy()

    # Map split types to readable names
    plot_data["split_type"] = plot_data["split_type"].map(
        {
            "random_split": "Random Split",
            "dataset_split": "Dataset Split",
            "phylo_ooc": "Out-of-Clade",
        }
    )

    # Map training types to readable names
    plot_data["training_type"] = plot_data["training_type"].map(
        {
            "concordant": "Concordant",
            "full": "Full",
        }
    )

    # Map test subsets to readable names
    plot_data["test_subset"] = plot_data["test_subset"].map(
        {
            "full": "Full Test",
            "concordant": "Concordant Test",
            "discordant": "Discordant Test",
        }
    )

    return plot_data


def plot_performance_vs_sample_size(
    df: pd.DataFrame, output_file: Path, test_subset: str = "Full Test"
) -> None:
    """
    Plot model performance vs training sample size.

    Creates a faceted plot showing performance (balanced accuracy) vs number
    of training samples, faceted by phenotype (rows) and split type (columns).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared plot data.
    output_file : Path
        Path to save the output figure.
    test_subset : str, optional
        Which test subset to plot (Full Test, Concordant Test, Discordant Test),
        by default "Full Test".
    """
    # Filter to selected test subset
    plot_data = df[df["test_subset"] == test_subset].copy()

    # Get unique phenotypes and split types
    phenotypes = sorted(plot_data["phenotype"].unique())
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]

    # Create figure
    fig, axes = plt.subplots(
        nrows=len(phenotypes),
        ncols=len(split_types),
        figsize=(12, 6),
        sharex=True,
        sharey=True,
    )

    # Define colors and markers for training types
    colors = {"Full": "#1f77b4", "Concordant": "#ff7f0e"}
    markers = {"Full": "o", "Concordant": "s"}

    for row_idx, phenotype in enumerate(phenotypes):
        for col_idx, split_type in enumerate(split_types):
            ax = axes[row_idx, col_idx]

            # Filter data for this subplot
            subset = plot_data[
                (plot_data["phenotype"] == phenotype)
                & (plot_data["split_type"] == split_type)
            ].copy()

            if len(subset) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot for each training type
            for training_type in ["Full", "Concordant"]:
                train_subset = subset[subset["training_type"] == training_type].copy()

                if len(train_subset) == 0:
                    continue

                # Plot individual points
                ax.scatter(
                    train_subset["n_train_samples"],
                    train_subset["balanced_accuracy"],
                    color=colors[training_type],
                    marker=markers[training_type],
                    alpha=0.6,
                    s=40,
                    label=training_type,
                    edgecolors="black",
                    linewidths=0.5,
                )

                # Calculate and plot mean line
                mean_data = (
                    train_subset.groupby("n_train_samples")["balanced_accuracy"]
                    .mean()
                    .reset_index()
                )
                ax.plot(
                    mean_data["n_train_samples"],
                    mean_data["balanced_accuracy"],
                    color=colors[training_type],
                    linestyle="-",
                    linewidth=2,
                    alpha=0.8,
                )

            # Add reference line at 0.95
            ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.3, linewidth=1)

            # Set labels and title
            if row_idx == len(phenotypes) - 1:
                ax.set_xlabel("Number of Training Samples")
            if col_idx == 0:
                ax.set_ylabel("Balanced Accuracy")

            # Add subplot title
            if row_idx == 0:
                ax.set_title(split_type, fontweight="bold")

            # Add phenotype label on the left
            if col_idx == 0:
                ax.text(
                    -0.3,
                    0.5,
                    phenotype,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )

            # Set y-axis limits
            ax.set_ylim(0, 1.05)

            # Add legend only to top-right subplot
            if row_idx == 0 and col_idx == len(split_types) - 1:
                ax.legend(title="Training Type", loc="lower right", frameon=False)

            # Format x-axis
            ax.tick_params(axis="both", which="both")

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


def plot_all_test_subsets(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create separate plots for each test subset.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared plot data.
    output_dir : Path
        Directory to save output figures.
    """
    test_subsets = ["Full Test", "Concordant Test", "Discordant Test"]

    for test_subset in test_subsets:
        subset_name = test_subset.lower().replace(" ", "_")
        output_file = output_dir / f"figure7_{FEATURE_TYPE}_{subset_name}.pdf"
        plot_performance_vs_sample_size(df, output_file, test_subset=test_subset)


def plot_combined_test_subsets(df: pd.DataFrame, output_file: Path) -> None:
    """
    Plot all test subsets together for comparison.

    Creates separate figures for each phenotype showing all test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared plot data.
    output_file : Path
        Path to save the output figure.
    """
    phenotypes = sorted(df["phenotype"].unique())
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]
    test_subsets = ["Full Test", "Concordant Test", "Discordant Test"]

    for phenotype in phenotypes:
        fig, axes = plt.subplots(
            nrows=len(test_subsets),
            ncols=len(split_types),
            figsize=(12, 12),
            sharex=True,
            sharey=True,
        )

        # Define colors and markers
        colors = {"Full": "#1f77b4", "Concordant": "#ff7f0e"}
        markers = {"Full": "o", "Concordant": "s"}

        for row_idx, test_subset in enumerate(test_subsets):
            for col_idx, split_type in enumerate(split_types):
                ax = axes[row_idx, col_idx]

                # Filter data
                subset = df[
                    (df["phenotype"] == phenotype)
                    & (df["split_type"] == split_type)
                    & (df["test_subset"] == test_subset)
                ].copy()

                if len(subset) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Plot for each training type
                for training_type in ["Full", "Concordant"]:
                    train_subset = subset[subset["training_type"] == training_type].copy()

                    if len(train_subset) == 0:
                        continue

                    # Plot individual points
                    ax.scatter(
                        train_subset["n_train_samples"],
                        train_subset["balanced_accuracy"],
                        color=colors[training_type],
                        marker=markers[training_type],
                        alpha=0.6,
                        s=40,
                        label=training_type,
                        edgecolors="black",
                        linewidths=0.5,
                    )

                    # Calculate and plot mean line
                    mean_data = (
                        train_subset.groupby("n_train_samples")["balanced_accuracy"]
                        .mean()
                        .reset_index()
                    )
                    ax.plot(
                        mean_data["n_train_samples"],
                        mean_data["balanced_accuracy"],
                        color=colors[training_type],
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                    )

                # Add reference line
                ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.3, linewidth=1)

                # Set labels
                if row_idx == len(test_subsets) - 1:
                    ax.set_xlabel("Number of Training Samples")
                if col_idx == 0:
                    ax.set_ylabel("Balanced Accuracy")

                # Add titles
                if row_idx == 0:
                    ax.set_title(split_type, fontweight="bold")

                # Add test subset label on the left
                if col_idx == 0:
                    ax.text(
                        -0.3,
                        0.5,
                        test_subset,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                    )

                # Set y-axis limits
                ax.set_ylim(0, 1.05)

                # Add legend only to top-right subplot
                if row_idx == 0 and col_idx == len(split_types) - 1:
                    ax.legend(title="Training Type", loc="lower right", frameon=False)

                ax.tick_params(axis="both", which="both")

        # Add overall title
        fig.suptitle(
            f"{phenotype}: Performance vs Training Sample Size",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        phenotype_file = output_file.parent / f"figure7_{FEATURE_TYPE}_{phenotype.lower()}_all_tests.pdf"
        fig.savefig(phenotype_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {phenotype_file}")
        plt.close()


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics about the data.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared plot data.
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nOverall statistics:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Phenotypes: {sorted(df['phenotype'].unique())}")
    print(f"  Split types: {sorted(df['split_type'].unique())}")
    print(f"  Training types: {sorted(df['training_type'].unique())}")
    print(f"  Test subsets: {sorted(df['test_subset'].unique())}")

    print("\nMean balanced accuracy by configuration:")
    summary = (
        df[df["test_subset"] == "Full Test"]
        .groupby(["phenotype", "split_type", "training_type", "sample_size"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(summary)

    print("\nPerformance at maximum sample size (full):")
    full_data = df[
        (df["test_subset"] == "Full Test") & (df["sample_size"] == "full")
    ].copy()
    full_summary = (
        full_data.groupby(["phenotype", "split_type", "training_type"])["balanced_accuracy"]
        .agg(["mean", "std"])
        .round(3)
    )
    print(full_summary)


def main() -> None:
    """Main function to generate Figure 7 plots."""
    # Load data
    data_file = Path(f"data/outputs/figure7/figure7_data_requirements_{FEATURE_TYPE}.csv")
    df = pd.read_csv(data_file)

    print(f"Loaded {len(df)} rows from {data_file}")
    print(f"Feature type: {FEATURE_TYPE.upper()}")

    # Prepare data
    plot_data = prepare_plot_data(df)

    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plots
    print("\nGenerating plots...")

    # Main plot: Full test set only
    print("  Creating main plot (Full Test)...")
    plot_performance_vs_sample_size(
        plot_data, output_dir / f"figure7_{FEATURE_TYPE}.pdf", test_subset="Full Test"
    )

    # Individual plots for each test subset
    print("  Creating plots for each test subset...")
    plot_all_test_subsets(plot_data, output_dir)

    # Combined plots showing all test subsets per phenotype
    print("  Creating combined plots per phenotype...")
    plot_combined_test_subsets(plot_data, output_dir / f"figure7_{FEATURE_TYPE}_combined.pdf")

    # Print summary statistics
    print_summary_statistics(plot_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
