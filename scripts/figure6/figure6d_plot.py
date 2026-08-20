#!/usr/bin/env python3
"""Create Figure 6D: combined vs phenotype-filtered feature performance."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import wilcoxon

from scripts.create_data_splits import COMMON_PHENOTYPES
from scripts.minority_filter import filter_by_minority, full_test_minority_counts
from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def load_results(data_file: Path) -> pd.DataFrame:
    """Load the Figure 6D results table with the manuscript's reporting filter applied.

    Figure 6D evaluates on the full held-out test set, so the minority-class
    rule stated in ``sections/methods.tex`` applies: ``dataset_split`` cells
    whose held-out test set carries fewer than ten minority-class samples are
    dropped. The standalone panel and the combined Figure 6 composer both read
    through here.

    Parameters
    ----------
    data_file : Path
        Path to the CSV written by ``figure6d_data.py``.

    Returns
    -------
    pd.DataFrame
        Results table with sub-threshold ``dataset_split`` rows removed.
    """
    return filter_by_minority(pd.read_csv(data_file), full_test_minority_counts())


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
    dataset_p_value: float | None,
) -> None:
    """Add compact paired-test annotations for split-specific comparisons.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    random_p_value : float | None
        Paired Wilcoxon p-value for random split comparisons.
    dataset_p_value : float | None
        Paired Wilcoxon p-value for dataset split comparisons.
    """
    ax.text(
        0.03,
        0.97,
        (
            "Wilcoxon p: "
            f"random={_format_p_value(random_p_value)}; "
            f"dataset={_format_p_value(dataset_p_value)}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )


def plot_split_comparison(
    ax: Axes,
    data: pd.DataFrame,
    split_type: str,
    phenotypes: list[str],
    title: str,
) -> None:
    """Plot the combined vs phenotype-filtered comparison for a single split type.

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
    split_data = data[data["split_type"] == split_type].copy()

    summary = (
        split_data.groupby(["phenotype", "experiment"])["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    x = np.arange(len(phenotypes))
    width = 0.35

    combined_data = summary[summary["experiment"] == "combined"].set_index("phenotype")
    filtered_data = summary[summary["experiment"] == "phenotype_filtered"].set_index(
        "phenotype"
    )

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

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Phenotype")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)

    combined_n_features = split_data[split_data["experiment"] == "combined"][
        "n_features"
    ].mean()
    filtered_n_features = split_data[split_data["experiment"] == "phenotype_filtered"][
        "n_features"
    ].mean()

    if not np.isnan(combined_n_features):
        ax.text(
            0.02,
            0.98,
            f"Combined: ~{int(combined_n_features)} features",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
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
            fontsize=10,
            color="#E63946",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def plot_balanced_accuracy_scatter(
    ax: Axes,
    data: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """Scatter combined against phenotype-filtered balanced accuracy.

    Shapes indicate split type (circle = random, square = dataset). Points
    above the diagonal indicate filtered features outperform combined.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with all data.
    phenotypes : list[str]
        List of phenotypes to include.
    """
    summary = (
        data.groupby(["phenotype", "experiment", "split_type"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )

    markers = {"random_split": "o", "dataset_split": "s"}
    split_labels = {"random_split": "Random Split", "dataset_split": "Dataset Split"}
    # Split type is a manuscript-wide category, so it takes the same colours
    # here as in figures 3, 5C and S6B rather than being colour-free.
    split_colors = {"random_split": "#57BA64", "dataset_split": "#2E86AB"}
    random_p_value = None
    dataset_p_value = None

    for split_type in ["random_split", "dataset_split"]:
        split_data = summary[summary["split_type"] == split_type]

        combined = split_data[split_data["experiment"] == "combined"].set_index(
            "phenotype"
        )["balanced_accuracy"]
        filtered = split_data[
            split_data["experiment"] == "phenotype_filtered"
        ].set_index("phenotype")["balanced_accuracy"]

        common = [p for p in phenotypes if p in combined.index and p in filtered.index]

        combined_vals = [combined.loc[p] for p in common]
        filtered_vals = [filtered.loc[p] for p in common]

        ax.scatter(
            combined_vals,
            filtered_vals,
            s=42,
            alpha=0.75,
            facecolors=split_colors[split_type],
            edgecolors="black",
            linewidths=0.7,
            marker=markers[split_type],
            label=split_labels[split_type],
            zorder=3,
        )

        p_value, _ = _paired_panel_pvalue(combined, filtered, common)
        if split_type == "random_split":
            random_p_value = p_value
        else:
            dataset_p_value = p_value

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    ax.set_xlabel("Combined Features\n(Balanced Accuracy)")
    ax.set_ylabel("Phenotype-Filtered\n(Balanced Accuracy)")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10,
        labelspacing=0.6,
        markerscale=0.7,
    )
    ax.set_aspect("equal")
    _add_pvalue_text(ax, random_p_value, dataset_p_value)


def plot_precision_recall_scatter_by_feature_type(
    ax: Axes,
    data: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """Scatter precision against recall.

    Shapes indicate split type (circle = random, square = dataset) and colors
    indicate feature set (blue = combined, red = phenotype-filtered).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : pd.DataFrame
        Results dataframe with all data.
    phenotypes : list[str]
        List of phenotypes to include.
    """
    summary = (
        data.groupby(["phenotype", "experiment", "split_type"])[["precision", "recall"]]
        .mean()
        .reset_index()
    )
    summary = summary[summary["phenotype"].isin(phenotypes)]

    colors = {"combined": "#2E86AB", "phenotype_filtered": "#E63946"}
    markers = {"random_split": "o", "dataset_split": "s"}
    filter_labels = {
        "combined": "Combined Features",
        "phenotype_filtered": "Phenotype-Filtered",
    }
    split_labels = {"random_split": "Random Split", "dataset_split": "Dataset Split"}

    for experiment in ["combined", "phenotype_filtered"]:
        for split_type in ["random_split", "dataset_split"]:
            subset = summary[
                (summary["experiment"] == experiment)
                & (summary["split_type"] == split_type)
            ]

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

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=10, labelspacing=1.2)
    ax.set_aspect("equal")


def create_figure(
    data_file: Path,
    output_file: Path,
    phenotype_order: list[str] | None = None,
) -> None:
    """Create Figure 6D comparing combined and phenotype-filtered features.

    Parameters
    ----------
    data_file : Path
        Path to CSV file with results from figure6d_data.py.
    output_file : Path
        Path to save the output figure.
    phenotype_order : list[str] | None
        Order of phenotypes for x-axis. If None, uses alphabetical order.
    """
    df = load_results(data_file)

    if phenotype_order is None:
        phenotypes = sorted(df["phenotype"].unique())
    else:
        phenotypes = [p for p in phenotype_order if p in df["phenotype"].values]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    plot_split_comparison(
        axes[0],
        df,
        "random_split",
        phenotypes,
        title="A. Random Split: Combined vs Phenotype-Filtered Features",
    )

    plot_split_comparison(
        axes[1],
        df,
        "dataset_split",
        phenotypes,
        title="B. Dataset Split: Combined vs Phenotype-Filtered Features",
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()

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

    print("\n" + "=" * 80)
    print("Per-Phenotype Comparison (Dataset Split)")
    print("=" * 80)

    dataset_split_data = df[df["split_type"] == "dataset_split"]
    phenotype_summary = (
        dataset_split_data.groupby(["phenotype", "experiment"])["balanced_accuracy"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    if (
        "combined" in phenotype_summary.columns
        and "phenotype_filtered" in phenotype_summary.columns
    ):
        phenotype_summary["difference"] = (
            phenotype_summary["phenotype_filtered"] - phenotype_summary["combined"]
        )
        phenotype_summary = phenotype_summary.sort_values("difference", ascending=False)
        print(phenotype_summary.round(4))


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure6")
    data_file = data_dir / "figure6d_all_results.csv"
    output_file = Path("figures/figure6d.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read from the split-generation module so the order cannot drift out of
    # sync with the phenotypes the splits were built for.
    phenotype_order = list(COMMON_PHENOTYPES)

    create_figure(data_file, output_file, phenotype_order)
