#!/usr/bin/env python3
"""Create Figure 5: performance on GapMind-concordant samples.

Reads ``data/outputs/figure5/`` and the GapMind baseline metrics under
``data/outputs/figure3/``; writes ``figures/figure5.pdf``.

Run with::

    uv run python -m scripts.figure5.figure5_plot
"""

import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

from scripts.minority_filter import (
    concordant_minority_counts as _concordant_minority_counts,
)
from scripts.minority_filter import (
    discordant_minority_counts as _discordant_minority_counts,
)
from scripts.minority_filter import (
    filter_by_minority as _filter_by_minority,
)
from scripts.minority_filter import (
    full_test_minority_counts as _full_test_minority_counts,
)
from scripts.visualization import (
    configure_plot_style,
    hide_categorical_minor_ticks,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Fixed seed for reproducibility.
np.random.seed(42)


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
    """Compute a paired phenotype-level Wilcoxon p-value on phenotype means.

    Parameters
    ----------
    left_values : pd.Series
        First-condition values, indexed by phenotype.
    right_values : pd.Series
        Second-condition values, indexed by phenotype.
    phenotypes : list[str]
        Phenotypes displayed in the panel; only these are paired.

    Returns
    -------
    tuple[float | None, int]
        Wilcoxon p-value and number of paired phenotypes. The p-value is
        ``None`` with fewer than two pairs or when the test is not
        numerically well-defined.
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


def _add_panel_pvalue_annotation(
    ax: Axes,
    p_value: float | None,
    n_pairs: int,
    position: Literal["top_right", "bottom_right"] = "top_right",
) -> None:
    """Add a compact panel-level p-value annotation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    p_value : float | None
        Paired comparison p-value.
    n_pairs : int
        Number of phenotype pairs used in the comparison.
    position : Literal["top_right", "bottom_right"]
        Anchor corner of the annotation within the axes.
    """
    if position == "top_right":
        x, y, ha, va = 0.98, 0.97, "right", "top"
    else:
        x, y, ha, va = 0.98, 0.05, "right", "bottom"

    ax.text(
        x,
        y,
        f"Phenotype means\npaired Wilcoxon, n={n_pairs}, p={_format_p_value(p_value)}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9
        ),
    )


def plot_dataset_split_performance(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str] | None = None,
) -> None:
    """Plot dataset split performance with random split reference (concordant samples).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the figure 5 data files.
    phenotypes : list[str] | None
        List of phenotypes to plot in order. If None, uses all available.
    """
    ml_df = pd.read_csv(data_dir / "figure5a_concordant_ml_results.csv")

    # Methods filter: drop dataset-split cells whose held-out concordant test set
    # has fewer than MIN_MINORITY_TEST_SAMPLES minority-class samples.
    # Random-split rows pass through unchanged.
    minority_counts = _concordant_minority_counts()
    ml_df = _filter_by_minority(ml_df, minority_counts)

    dataset_df = ml_df[ml_df["split_type"] == "dataset_split"].copy()
    random_df = ml_df[ml_df["split_type"] == "random_split"].copy()

    # Optional GapMind-feature ceiling: cross-dataset BA when concordant models
    # are trained on the GapMind step features directly. Raw features are
    # preferred over the correlation-filtered ones because the 0.95 filter
    # consolidates step features duplicated across phenotype prefixes and
    # depletes some per-phenotype feature spaces (e.g., amino-acid pathways).
    gapmind_raw_file = data_dir / "figure5a_concordant_ml_results_gapmind_raw.csv"
    gapmind_file = data_dir / "figure5a_concordant_ml_results_gapmind.csv"
    if gapmind_raw_file.exists():
        chosen = gapmind_raw_file
    elif gapmind_file.exists():
        chosen = gapmind_file
    else:
        chosen = None
    if chosen is not None:
        gapmind_df = pd.read_csv(chosen)
        gapmind_df = _filter_by_minority(gapmind_df, minority_counts)
        gapmind_dataset_df = gapmind_df[
            gapmind_df["split_type"] == "dataset_split"
        ].copy()
        gapmind_means = (
            gapmind_dataset_df.groupby("phenotype")["balanced_accuracy"]
            .mean()
            .to_dict()
        )
    else:
        gapmind_means = {}

    if phenotypes is None:
        phenotypes = sorted(dataset_df["phenotype"].unique())

    x = np.arange(len(phenotypes))

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    box_data = []
    for phenotype in phenotypes:
        phenotype_data = dataset_df[dataset_df["phenotype"] == phenotype][
            "balanced_accuracy"
        ].values
        box_data.append(phenotype_data)

    ax.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#2E86AB", alpha=0.65, linewidth=1.1),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
    )

    random_means = random_df.groupby("phenotype")["balanced_accuracy"].mean().to_dict()

    # Random-split mean, one reference line per phenotype.
    for phenotype in phenotypes:
        if phenotype in random_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [random_means[phenotype], random_means[phenotype]],
                color="#57BA64",
                linestyle="-",
                linewidth=1.6,
                alpha=0.65,
                zorder=1,
            )

    # GapMind-feature ceiling, one reference line per phenotype: BA reached by
    # the same pipeline trained on GapMind step features under concordant
    # training and cross-dataset evaluation.
    for phenotype in phenotypes:
        if phenotype in gapmind_means:
            x_pos = x[phenotypes.index(phenotype)]
            ax.plot(
                [x_pos - 0.45, x_pos + 0.45],
                [gapmind_means[phenotype], gapmind_means[phenotype]],
                color="#D7263D",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
                zorder=2,
            )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2E86AB", alpha=0.7, label="Dataset Split (KOFAM)"),
        Line2D(
            [0],
            [0],
            color="#57BA64",
            linewidth=2,
            alpha=0.7,
            label="Random Split mean (KOFAM)",
        ),
    ]
    if gapmind_means:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#D7263D",
                linewidth=2,
                linestyle="--",
                alpha=0.9,
                label="Dataset Split mean (GapMind)",
            )
        )

    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)
    ax.set_ylim(0, 1.05)

    dataset_means = dataset_df.groupby("phenotype")["balanced_accuracy"].mean()
    # With the GapMind ceiling available the paired comparison is KOFAM
    # cross-dataset vs GapMind cross-dataset; otherwise it is KOFAM
    # cross-dataset vs KOFAM random.
    if gapmind_means:
        reference_series = pd.Series(gapmind_means)
    else:
        reference_series = pd.Series(random_means)
    p_value, n_pairs = _paired_panel_pvalue(
        dataset_means,
        reference_series,
        phenotypes,
    )
    _add_panel_pvalue_annotation(ax, p_value, n_pairs, position="bottom_right")

    ax.text(
        -0.08,
        1.05,
        "A",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10,
    )


def create_feature_comparison_plot(
    ax: Axes, data_dir: Path, phenotypes: list[str]
) -> None:
    """Plot grouped, stacked bars of common and unique features per dataset.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data_dir : Path
        Directory containing the figure 5B data files.
    phenotypes : list[str]
        List of phenotypes in alphabetical order.
    """
    data_file = data_dir / "figure5b_feature_comparison_summary.csv"
    df = pd.read_csv(data_file)

    # Cluster-level counts (redundancy clusters from shap.utils.hclust), falling
    # back to raw KO counts when those columns are absent.
    common_col = (
        "n_intersection_clusters"
        if "n_intersection_clusters" in df.columns
        else "n_intersection"
    )
    unique_col = (
        "n_unique_to_individual_clusters"
        if "n_unique_to_individual_clusters" in df.columns
        else "n_unique_to_individual"
    )

    datasets = ["atleaf", "lit", "marine"]
    dataset_color_map = get_dataset_colors()
    dataset_colors = [dataset_color_map[d] for d in datasets]
    dataset_display_names = format_dataset_names(datasets)

    x_pos = np.arange(len(phenotypes))
    bar_width = 0.8 / len(datasets)
    bar_group_center = bar_width * (len(datasets) - 1) / 2

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    for i, dataset in enumerate(datasets):
        dataset_df = df[df["test_dataset"] == dataset].set_index("phenotype")

        common = []
        unique_individual = []

        for phenotype in phenotypes:
            if phenotype in dataset_df.index:
                row = dataset_df.loc[phenotype]
                common.append(row[common_col])
                unique_individual.append(row[unique_col])
            else:
                common.append(0)
                unique_individual.append(0)

        positions = x_pos - bar_group_center + i * bar_width

        ax.bar(
            positions,
            common,
            bar_width,
            color=dataset_colors[i],
            alpha=0.8,
        )
        ax.bar(
            positions,
            unique_individual,
            bar_width,
            bottom=common,
            color=dataset_colors[i],
            alpha=0.4,
            hatch="//",
        )

    dataset_handles = [
        Rectangle((0, 0), 1, 1, fc=dataset_colors[i], alpha=0.8)
        for i in range(len(datasets))
    ]
    feature_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            fc="gray",
            alpha=0.8,
            label="Common",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fc="gray",
            alpha=0.4,
            hatch="//",
            label="Unique to Individual",
        ),
    ]
    legend1 = ax.legend(
        dataset_handles,
        dataset_display_names,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.30),
        ncol=len(datasets),
        frameon=False,
        fontsize=10,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=feature_handles,
        title="Stable feature clusters",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.30),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    ax.set_ylabel("Stable feature clusters (n)", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")
    ax.set_xlim(-0.5, len(phenotypes) - 0.5)
    ax.set_ylim(0, 10)

    ax.text(
        -0.08,
        1.05,
        "B",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def _load_gapmind_phenotype_means(
    metrics_path: Path,
    minority_counts: dict | None = None,
) -> pd.Series:
    """Load GapMind permissive-threshold balanced accuracy aggregated to phenotype means.

    Parameters
    ----------
    metrics_path : Path
        TSV file with per-phenotype, per-split rows containing
        ``balanced_accuracy`` and ``phenotype`` columns.
    minority_counts : dict | None, optional
        When provided, exclude per-(phenotype, held-out-dataset) cells below the
        minority-class threshold before aggregating, so the GapMind baseline is
        filtered on the same cells as the ML side.

    Returns
    -------
    pd.Series
        Phenotype-mean balanced accuracy indexed by phenotype.
    """
    df = pd.read_csv(metrics_path, sep="\t")
    if minority_counts is not None:
        test_col = "test_dataset" if "test_dataset" in df.columns else None
        df = _filter_by_minority(df, minority_counts, test_dataset_column=test_col)
    return df.groupby("phenotype")["balanced_accuracy"].mean()


def _panel_scatter_axes(ax: Axes, lo: float = 0.0, hi: float = 1.02) -> None:
    """Apply shared scatter formatting (axis range, diagonal, grid)."""
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)


def plot_ml_vs_gapmind_full_test(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    panel_label: str = "C",
) -> None:
    """Plot panel C: ML vs GapMind balanced accuracy on the full test set.

    Two series share the scatter axes: random holdout (green) and cross-dataset
    (blue). Each point is one phenotype mean; points above the dashed y=x
    diagonal mark ML outperforming GapMind.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing figure 5 data files.
    phenotypes : list[str]
        Phenotypes to include in the comparison.
    panel_label : str
        Subplot annotation label (default ``"C"``).
    """
    ml_df = pd.read_csv(data_dir / "figure5c_concordant_train_different_test.csv")
    ml_df = ml_df[ml_df["test_type"] == "full"].copy()

    full_minority = _full_test_minority_counts()
    ml_df = _filter_by_minority(ml_df, full_minority)

    ml_random_mean = (
        ml_df[ml_df["split_type"] == "random_split"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )
    ml_dataset_mean = (
        ml_df[ml_df["split_type"] == "dataset_split"]
        .groupby("phenotype")["balanced_accuracy"]
        .mean()
    )

    gm_random_mean = _load_gapmind_phenotype_means(
        Path("data/outputs/figure3/gapmind_random_split_metrics.tsv")
    )
    gm_dataset_mean = _load_gapmind_phenotype_means(
        Path("data/outputs/figure3/gapmind_dataset_split_metrics.tsv"),
        minority_counts=full_minority,
    )

    common_r = sorted(
        set(ml_random_mean.index) & set(gm_random_mean.index) & set(phenotypes)
    )
    common_d = sorted(
        set(ml_dataset_mean.index) & set(gm_dataset_mean.index) & set(phenotypes)
    )

    ax.scatter(
        gm_random_mean.loc[common_r].values,
        ml_random_mean.loc[common_r].values,
        c="#57BA64",
        alpha=0.85,
        s=70,
        edgecolors="black",
        linewidths=0.7,
        label="Random holdout",
        zorder=3,
    )
    ax.scatter(
        gm_dataset_mean.loc[common_d].values,
        ml_dataset_mean.loc[common_d].values,
        c="#2E86AB",
        alpha=0.85,
        s=70,
        edgecolors="black",
        linewidths=0.7,
        marker="s",
        label="Cross-dataset",
        zorder=3,
    )

    _panel_scatter_axes(ax, lo=-0.05, hi=1.05)
    ax.set_xlabel("GapMind Balanced Accuracy")
    ax.set_ylabel("Concordant-ML Balanced Accuracy")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    p_r, n_r = _paired_panel_pvalue(ml_random_mean, gm_random_mean, common_r)
    p_d, n_d = _paired_panel_pvalue(ml_dataset_mean, gm_dataset_mean, common_d)

    ax.text(
        0.02,
        0.98,
        f"Random: paired Wilcoxon p={_format_p_value(p_r)}, n={n_r}\n"
        f"Cross-dataset: paired Wilcoxon p={_format_p_value(p_d)}, n={n_d}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    ax.text(
        -0.12,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def plot_ml_vs_gapmind_test_subsets(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    panel_label: str = "D",
) -> None:
    """Plot panel D: cross-dataset ML vs GapMind BA split by test subset.

    Shares panel C's axes. The concordant (purple) and discordant (orange)
    cross-dataset subsets sit at GapMind BA exactly 1 and 0 by construction, so
    each series collapses to a vertical strip.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing figure 5 data files.
    phenotypes : list[str]
        Phenotypes to include in the comparison.
    panel_label : str
        Subplot annotation label (default ``"D"``).
    """
    concordant_df = pd.read_csv(data_dir / "figure5a_concordant_ml_results.csv")
    concordant_df = _filter_by_minority(concordant_df, _concordant_minority_counts())
    concordant_df = concordant_df[concordant_df["split_type"] == "dataset_split"]
    ml_concordant = concordant_df.groupby("phenotype")["balanced_accuracy"].mean()

    discordant_df = pd.read_csv(
        data_dir / "figure5c_concordant_train_different_test.csv"
    )
    discordant_df = discordant_df[discordant_df["test_type"] == "discordant"]
    discordant_df = _filter_by_minority(discordant_df, _discordant_minority_counts())
    discordant_df = discordant_df[discordant_df["split_type"] == "dataset_split"]
    ml_discordant = discordant_df.groupby("phenotype")["balanced_accuracy"].mean()

    concordant_phens = sorted(set(ml_concordant.index) & set(phenotypes))
    discordant_phens = sorted(set(ml_discordant.index) & set(phenotypes))

    # GapMind BA is exactly 1.0 on the concordant subset and 0.0 on the
    # discordant subset by construction.
    gm_concordant_x = np.full(len(concordant_phens), 1.0)
    gm_discordant_x = np.full(len(discordant_phens), 0.0)

    ax.scatter(
        gm_concordant_x,
        ml_concordant.loc[concordant_phens].values,
        c="#6A4C93",
        alpha=0.85,
        s=70,
        edgecolors="black",
        linewidths=0.7,
        label="Concordant test subset",
        zorder=3,
    )
    ax.scatter(
        gm_discordant_x,
        ml_discordant.loc[discordant_phens].values,
        c="#E89149",
        alpha=0.85,
        s=70,
        edgecolors="black",
        linewidths=0.7,
        marker="s",
        label="Discordant test subset",
        zorder=3,
    )

    _panel_scatter_axes(ax, lo=-0.05, hi=1.05)
    ax.set_xlabel("GapMind Balanced Accuracy")
    ax.set_ylabel("Concordant-ML Balanced Accuracy")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    median_conc = ml_concordant.loc[concordant_phens].median()
    median_disc = ml_discordant.loc[discordant_phens].median()
    ax.text(
        0.02,
        0.98,
        f"Median ML BA, concordant: {median_conc:.2f}\n"
        f"Median ML BA, discordant: {median_disc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    ax.text(
        -0.12,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 5 with four subplots laid out in three rows.

    Row 0 is panel A (per-phenotype concordant cross-dataset BA), row 1 is
    panel B (stable feature clusters, carrying the phenotype tick labels), and
    row 2 holds panels C and D, the ML vs GapMind scatters.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    ml_df = pd.read_csv(data_dir / "figure5a_concordant_ml_results.csv")
    feature_comp_df = pd.read_csv(data_dir / "figure5b_feature_comparison_summary.csv")
    test_df = pd.read_csv(data_dir / "figure5c_concordant_train_different_test.csv")

    dataset_phenotypes = set(
        ml_df[ml_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    feature_phenotypes = set(feature_comp_df["phenotype"].unique())
    test_phenotypes = set(
        test_df[test_df["test_type"] == "discordant"]["phenotype"].unique()
    )

    print("Determining common phenotypes across all analyses...")
    print(f" - Dataset split phenotypes: {len(dataset_phenotypes)}")
    print(f" - Feature comparison phenotypes: {len(feature_phenotypes)}")
    print(f" - Test type phenotypes (discordant): {len(test_phenotypes)}")
    common_phenotypes = sorted(
        dataset_phenotypes.intersection(feature_phenotypes).intersection(
            test_phenotypes
        )
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    # Panels C and D also cover phenotypes absent from the feature-intersection
    # panels but present in the per-phenotype BA tables.
    full_test_phenotypes = sorted(
        set(test_df[test_df["test_type"] == "full"]["phenotype"].unique())
    )

    fig = plt.figure(figsize=(12, 10))
    # Panels A and B share the per-phenotype x-axis (top gridspec); panels C and
    # D are scatters with a different x-axis (bottom gridspec).
    gs_top = fig.add_gridspec(
        nrows=2,
        ncols=1,
        top=0.97,
        bottom=0.57,
        hspace=0.25,
    )
    gs_bot = fig.add_gridspec(
        nrows=1,
        ncols=2,
        top=0.40,
        bottom=0.04,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs_top[0])
    ax_b = fig.add_subplot(gs_top[1])
    ax_c = fig.add_subplot(gs_bot[0])
    ax_d = fig.add_subplot(gs_bot[1])

    plot_dataset_split_performance(ax_a, data_dir, common_phenotypes)
    create_feature_comparison_plot(ax_b, data_dir, common_phenotypes)
    plot_ml_vs_gapmind_full_test(ax_c, data_dir, full_test_phenotypes, panel_label="C")
    plot_ml_vs_gapmind_test_subsets(
        ax_d, data_dir, full_test_phenotypes, panel_label="D"
    )

    x_pos = np.arange(len(common_phenotypes))
    for ax in (ax_a, ax_b):
        ax.set_xlim(-0.5, len(common_phenotypes) - 0.5)
        ax.set_xticks(x_pos)

    # Phenotype tick labels live on panel B.
    ax_a.set_xlabel("")
    ax_a.set_xticklabels([])
    ax_b.set_xlabel("Phenotype", labelpad=-18)
    ax_b.set_xticklabels(common_phenotypes, rotation=45, ha="right")
    ax_b.tick_params(axis="x", labelbottom=True)

    hide_categorical_minor_ticks(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure5")
    output_file = Path("figures/figure5.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
