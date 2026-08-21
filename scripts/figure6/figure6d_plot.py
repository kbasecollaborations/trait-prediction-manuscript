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
    dropped. Figure 6 composition and ``scripts/stats/manuscript_pvalues.py``
    both read the table through here.

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
        frameon=False,
        fontsize=10,
        labelspacing=0.6,
        markerscale=0.7,
    )
    ax.set_aspect("equal")
    _add_pvalue_text(ax, random_p_value, dataset_p_value)
