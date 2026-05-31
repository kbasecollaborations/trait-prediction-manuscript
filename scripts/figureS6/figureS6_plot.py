#!/usr/bin/env python3
"""Plot Figure S6: model performance versus training sample size."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


# Feature type to use: "gapmind", "kofam", or "rast"
# Change this line to match the feature type used in figure_s6_data.py
FEATURE_TYPE = "kofam"

# Sample sizes used in figure_s6_data.py
SAMPLE_SIZES = [50, 100, 200, 500, "full"]

# Manuscript figure routing: only the Histidine combined-test grid appears in
# the manuscript (as figure_s6.pdf). All other variants — the Full/Concordant/
# Discordant test-subset 2x3 grids and the Galactose combined-test grid — go to
# figures/alternate/ to keep the manuscript figures directory uncluttered.
MANUSCRIPT_PHENOTYPE = "Histidine"
MANUSCRIPT_FIGURE_NAME = "figure_s6.pdf"


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
    Lines connect points with the same key and repeat across sample sizes.

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
    plot_data = df[df["test_subset"] == test_subset].copy()

    phenotypes = sorted(plot_data["phenotype"].unique())
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]

    fig, axes = plt.subplots(
        nrows=len(phenotypes),
        ncols=len(split_types),
        figsize=(12, 6),
        sharex=True,
        sharey=True,
    )

    # Get unique sample sizes for x-axis labeling
    raw_sizes = plot_data["sample_size"].unique()
    unique_sample_sizes = sorted(
        raw_sizes,
        key=lambda x: float("inf") if str(x) == "full" else float(x),
    )

    colors = {"Full": "#1f77b4", "Concordant": "#ff7f0e"}
    markers = {"Full": "o", "Concordant": "s"}

    for row_idx, phenotype in enumerate(phenotypes):
        for col_idx, split_type in enumerate(split_types):
            ax = axes[row_idx, col_idx]

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

            for training_type in ["Full", "Concordant"]:
                train_subset = subset[subset["training_type"] == training_type].copy()

                if len(train_subset) == 0:
                    continue

                # Get unique (key, repeat) combinations for line connections
                train_subset["key_repeat"] = (
                    train_subset["key"] + "_" + train_subset["repeat"].astype(str)
                )
                unique_key_repeats = train_subset["key_repeat"].unique()

                # Plot lines connecting points with the same key and repeat
                for key_repeat in unique_key_repeats:
                    line_data = train_subset[
                        train_subset["key_repeat"] == key_repeat
                    ].copy()

                    # Sort by n_train_samples for proper line connection
                    line_data = line_data.sort_values("n_train_samples")

                    ax.plot(
                        line_data["n_train_samples"],
                        line_data["balanced_accuracy"],
                        color=colors[training_type],
                        linestyle="-",
                        linewidth=1,
                        alpha=0.4,
                    )

                # Plot all points (only add label once for legend)
                ax.scatter(
                    train_subset["n_train_samples"],
                    train_subset["balanced_accuracy"],
                    color=colors[training_type],
                    marker=markers[training_type],
                    alpha=0.7,
                    s=40,
                    label=training_type
                    if row_idx == 0 and col_idx == 0
                    else None,
                    edgecolors="black",
                    linewidths=0.5,
                )

            if row_idx == len(phenotypes) - 1:
                ax.set_xlabel("Number of Training Samples")
            if col_idx == 0:
                ax.set_ylabel("Balanced Accuracy")

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

            ax.set_ylim(0, 1.05)

            # Set x-axis ticks to match sample sizes
            x_ticks = []
            x_labels = []
            for sample_size in unique_sample_sizes:
                size_data = subset[subset["sample_size"] == sample_size]
                if len(size_data) > 0:
                    mean_n_samples = size_data["n_train_samples"].mean()
                    x_ticks.append(mean_n_samples)
                    x_labels.append(
                        str(sample_size) if sample_size != "full" else "full"
                    )

            if x_ticks:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")

            ax.tick_params(axis="both", which="both")

    # Add figure-level legend at the top in one row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        title="Training Type",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        output_file = output_dir / f"figure_s6_{subset_name}.pdf"
        plot_performance_vs_sample_size(df, output_file, test_subset=test_subset)


def plot_combined_test_subsets(
    df: pd.DataFrame,
    manuscript_dir: Path,
    alternate_dir: Path,
) -> None:
    """
    Plot all test subsets together for comparison.

    Creates one combined-test-subset grid per phenotype. The
    ``MANUSCRIPT_PHENOTYPE`` grid is saved as ``manuscript_dir /
    MANUSCRIPT_FIGURE_NAME`` (the manuscript Figure 7); the remaining
    phenotypes are saved under ``alternate_dir`` to keep the manuscript
    figures directory uncluttered.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared plot data.
    manuscript_dir : Path
        Directory for the manuscript Figure 7 PDF.
    alternate_dir : Path
        Directory for non-manuscript phenotype grids.
    """
    phenotypes = sorted(df["phenotype"].unique())
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]
    test_subsets = ["Full Test", "Concordant Test", "Discordant Test"]

    # Get unique sample sizes for x-axis labeling
    raw_sizes = df["sample_size"].unique()
    unique_sample_sizes = sorted(
        raw_sizes,
        key=lambda x: float("inf") if str(x) == "full" else float(x),
    )

    colors = {"Full": "#1f77b4", "Concordant": "#ff7f0e"}
    markers = {"Full": "o", "Concordant": "s"}

    for phenotype in phenotypes:
        fig, axes = plt.subplots(
            nrows=len(test_subsets),
            ncols=len(split_types),
            figsize=(12, 12),
            sharex=True,
            sharey=True,
        )

        for row_idx, test_subset in enumerate(test_subsets):
            for col_idx, split_type in enumerate(split_types):
                ax = axes[row_idx, col_idx]

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

                for training_type in ["Full", "Concordant"]:
                    train_subset = subset[
                        subset["training_type"] == training_type
                    ].copy()

                    if len(train_subset) == 0:
                        continue

                    # Get unique (key, repeat) combinations for line connections
                    train_subset["key_repeat"] = (
                        train_subset["key"] + "_" + train_subset["repeat"].astype(str)
                    )
                    unique_key_repeats = train_subset["key_repeat"].unique()

                    # Plot lines connecting points with the same key and repeat
                    for key_repeat in unique_key_repeats:
                        line_data = train_subset[
                            train_subset["key_repeat"] == key_repeat
                        ].copy()

                        # Sort by n_train_samples for proper line connection
                        line_data = line_data.sort_values("n_train_samples")

                        ax.plot(
                            line_data["n_train_samples"],
                            line_data["balanced_accuracy"],
                            color=colors[training_type],
                            linestyle="-",
                            linewidth=1,
                            alpha=0.4,
                        )

                    # Plot all points (only add label once for legend)
                    ax.scatter(
                        train_subset["n_train_samples"],
                        train_subset["balanced_accuracy"],
                        color=colors[training_type],
                        marker=markers[training_type],
                        alpha=0.7,
                        s=40,
                        label=training_type
                        if row_idx == 0 and col_idx == 0
                        else None,
                        edgecolors="black",
                        linewidths=0.5,
                    )

                if row_idx == len(test_subsets) - 1:
                    ax.set_xlabel("Number of Training Samples")
                if col_idx == 0:
                    ax.set_ylabel("Balanced Accuracy")

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

                # Set x-axis ticks to match sample sizes
                x_ticks = []
                x_labels = []
                for sample_size in unique_sample_sizes:
                    size_data = subset[subset["sample_size"] == sample_size]
                    if len(size_data) > 0:
                        mean_n_samples = size_data["n_train_samples"].mean()
                        x_ticks.append(mean_n_samples)
                        x_labels.append(
                            str(sample_size) if sample_size != "full" else "full"
                        )

                if x_ticks:
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")

                ax.tick_params(axis="both", which="both")

        # Add figure-level legend at the top in one row
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            title="Training Type",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if phenotype == MANUSCRIPT_PHENOTYPE:
            phenotype_file = manuscript_dir / MANUSCRIPT_FIGURE_NAME
        else:
            phenotype_file = (
                alternate_dir / f"figure_s6_{phenotype.lower()}_all_tests.pdf"
            )
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
        .groupby(["phenotype", "split_type", "training_type", "sample_size"])[
            "balanced_accuracy"
        ]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(summary)

    print("\nPerformance at maximum sample size (full):")
    full_data = df[
        (df["test_subset"] == "Full Test") & (df["sample_size"] == "full")
    ].copy()
    full_summary = (
        full_data.groupby(["phenotype", "split_type", "training_type"])[
            "balanced_accuracy"
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    print(full_summary)


def main() -> None:
    """Generate Figure S6 plots."""
    data_file = Path(
        f"data/outputs/figureS6/figure_s6_data_requirements_{FEATURE_TYPE}.csv"
    )
    df = pd.read_csv(data_file)

    print(f"Loaded {len(df)} rows from {data_file}")
    print(f"Feature type: {FEATURE_TYPE.upper()}")

    plot_data = prepare_plot_data(df)

    # Create output directories. The manuscript figure (figure_s6.pdf) lives in
    # figures/; all other variants are routed to figures/alternate/.
    output_dir = Path("figures")
    alternate_dir = output_dir / "alternate"
    output_dir.mkdir(parents=True, exist_ok=True)
    alternate_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    # Auxiliary main plot (Full Test only, 2x3 grid for both phenotypes): alternate.
    print("  Creating auxiliary Full Test plot (alternate)...")
    plot_performance_vs_sample_size(
        plot_data, alternate_dir / "figure_s6_full_test_2x3.pdf", test_subset="Full Test"
    )

    # Per-test-subset plots: alternate.
    print("  Creating per-test-subset plots (alternate)...")
    plot_all_test_subsets(plot_data, alternate_dir)

    # Combined-test-subset plots per phenotype. The MANUSCRIPT_PHENOTYPE goes to
    # figures/figure_s6.pdf; the rest land in figures/alternate/.
    print("  Creating combined-test-subset plots per phenotype...")
    plot_combined_test_subsets(
        plot_data,
        manuscript_dir=output_dir,
        alternate_dir=alternate_dir,
    )

    print_summary_statistics(plot_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
