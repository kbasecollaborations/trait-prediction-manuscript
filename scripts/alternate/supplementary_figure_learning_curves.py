#!/usr/bin/env python3
"""Shared plotting utilities for the supplementary learning-curve figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from matplotlib.lines import Line2D

from scripts.create_data_splits import COMMON_PHENOTYPES
from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


FEATURE_TYPE = "kofam"
DATA_FILE = Path(f"data/outputs/figureS7/figureS7_learning_curves_{FEATURE_TYPE}.csv")
OUTPUT_DIR = Path("figures")


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    """Map internal result labels to manuscript display labels.

    Parameters
    ----------
    df : pd.DataFrame
        Raw learning-curve results.

    Returns
    -------
    pd.DataFrame
        Copy of the input with normalized display labels.
    """
    out = df.copy()
    out["split_type"] = out["split_type"].map(
        {
            "random_split": "Random Split",
            "dataset_split": "Dataset Split",
            "phylo_ooc": "Out-of-Clade",
        }
    )
    out["training_type"] = out["training_type"].map(
        {"full": "Full", "concordant": "Concordant"}
    )
    out["test_subset"] = out["test_subset"].map(
        {
            "full": "Full Test",
            "concordant": "Concordant Test",
            "discordant": "Discordant Test",
        }
    )
    return out


def normalize_sample_size_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sample-size labels to a consistent string representation.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``sample_size`` stored as lowercase strings.
    """
    out = df.copy()
    out["sample_size"] = out["sample_size"].astype(str).str.strip().str.lower()
    return out


def get_sample_size_order(df: pd.DataFrame) -> list[str]:
    """Return the canonical display order for sample sizes present in the data.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results with normalized sample-size labels.

    Returns
    -------
    list[str]
        Ordered sample-size labels.
    """
    present = set(df["sample_size"])
    numeric = sorted((s for s in present if s != "full"), key=int)
    return numeric + (["full"] if "full" in present else [])


def estimate_saturation_sizes(
    df: pd.DataFrame, split_type: str = "Dataset Split"
) -> dict[str, str]:
    """Estimate the smallest sample size that reaches 90% of full performance.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results with normalized sample-size labels.

    Returns
    -------
    dict[str, str]
        Mapping from phenotype to the earliest sample-size label that reaches
        at least 90% of the full split-specific performance.
    """
    sub = df[
        (df["training_type"] == "Concordant")
        & (df["split_type"] == split_type)
        & (df["test_subset"] == "Full Test")
    ].copy()

    phenotypes = [p for p in COMMON_PHENOTYPES if p in sub["phenotype"].unique()]
    # Candidate sizes come from the data, not a literal: the grid gained 25 and
    # dropped 500, and a hardcoded list would skip the smallest size and fall
    # back to a label that no longer exists.
    candidate_sizes = [s for s in get_sample_size_order(sub) if s != "full"]
    saturation_sizes: dict[str, str] = {}

    for phenotype in phenotypes:
        phenotype_df = sub[sub["phenotype"] == phenotype]
        full_perf = phenotype_df.loc[
            phenotype_df["sample_size"] == "full", "balanced_accuracy"
        ].mean()
        threshold = 0.90 * full_perf

        for sample_size in candidate_sizes:
            sample_perf = phenotype_df.loc[
                phenotype_df["sample_size"] == sample_size, "balanced_accuracy"
            ].mean()
            if pd.notna(sample_perf) and sample_perf >= threshold:
                saturation_sizes[phenotype] = sample_size
                break
        else:
            saturation_sizes[phenotype] = "full"

    return saturation_sizes


def load_plot_data() -> pd.DataFrame:
    """Load and normalize the shared learning-curve results used by S8--S10.

    Applies the minority-class-in-test filter (Methods): rows whose held-out
    test set has fewer than 10 minority-class samples are dropped, using the
    test-set definition matching each row's ``test_subset``.

    Returns
    -------
    pd.DataFrame
        Prepared plotting data.
    """
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")

    from scripts.minority_filter import (
        concordant_minority_counts,
        discordant_minority_counts,
        filter_by_minority,
        full_test_minority_counts,
    )

    if "test_subset" in df.columns:
        full_counts = full_test_minority_counts()
        conc_counts = concordant_minority_counts()
        disc_counts = discordant_minority_counts()
        subset_counts = {
            "full": full_counts,
            "concordant": conc_counts,
            "discordant": disc_counts,
        }
        filtered_parts: list[pd.DataFrame] = []
        for subset, sub_df in df.groupby("test_subset"):
            counts = subset_counts.get(subset, full_counts)
            filtered_parts.append(filter_by_minority(sub_df, counts))
        df = pd.concat(filtered_parts, ignore_index=True)
    else:
        df = filter_by_minority(df, full_test_minority_counts())
    print(f"After minority-class filter: {len(df)} rows")
    return normalize_sample_size_labels(prepare_results(df))


def ensure_output_dir() -> Path:
    """Create the shared figure output directory if needed.

    Returns
    -------
    Path
        Output directory path.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def plot_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    """Plot the supplementary heatmap figure.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    sub = df[
        (df["training_type"] == "Concordant")
        & (df["split_type"] == "Dataset Split")
        & (df["test_subset"] == "Full Test")
    ].copy()

    pivot = (
        sub.groupby(["phenotype", "sample_size"])["balanced_accuracy"]
        .mean()
        .unstack("sample_size")
    )
    size_order = get_sample_size_order(sub)
    pivot = pivot[size_order]
    pivot = pivot.reindex([p for p in COMMON_PHENOTYPES if p in pivot.index])
    col_labels = [size if size != "full" else "Full" for size in size_order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0.4, vmax=1.0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Training Sample Size")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if value >= 0.72 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Balanced Accuracy")

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close(fig)


#: Training-type colours, matching ``scripts/figureS6/figureS6_plot.py``.
TRAINING_COLORS: dict[str, str] = {"Full": "#1f77b4", "Concordant": "#ff7f0e"}
#: Test-subset line styles for the multi-phenotype grid.
TEST_SUBSET_STYLES: dict[str, tuple[str, str]] = {
    "Full Test": ("-", "o"),
    "Concordant Test": ("--", "s"),
}
#: Evaluation blocks stacked in the multi-phenotype grid, as
#: ``(panel letter, internal split label, display title)``.
GRID_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("A", "Dataset Split", "Cross-dataset evaluation"),
    ("B", "Random Split", "Random-holdout evaluation"),
)


def build_training_size_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate learning-curve results for the multi-phenotype grid.

    Averages balanced accuracy over repeats and over the splits available for
    each phenotype, for the four training-type by test-subset series.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results from :func:`load_plot_data`.

    Returns
    -------
    pd.DataFrame
        Long-form means indexed by split type, phenotype, training type, test
        subset, and sample size, with an ``n_measurements`` column recording
        how many rows each mean summarises.
    """
    sub = df[df["test_subset"].isin(TEST_SUBSET_STYLES)].copy()
    grouped = sub.groupby(
        ["split_type", "phenotype", "training_type", "test_subset", "sample_size"],
        dropna=False,
    )["balanced_accuracy"]
    return grouped.agg(mean="mean", n_measurements="size").reset_index()


def plot_training_size_grid(df: pd.DataFrame, output_file: Path) -> None:
    """Plot per-phenotype learning curves by training type and test subset.

    Two stacked blocks of 15 panels, one panel per shared phenotype. Within a
    panel, colour distinguishes full from concordant training and line style
    distinguishes the full from the concordant test subset, so the panel shows
    whether concordance-trained models plateau earlier than full-data models.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results from :func:`load_plot_data`.
    output_file : Path
        Where to save the figure.

    Raises
    ------
    ValueError
        If none of the expected evaluation blocks are present in the data.
    """
    means = build_training_size_grid(df)
    blocks = [b for b in GRID_BLOCKS if b[1] in set(means["split_type"])]
    if not blocks:
        raise ValueError(
            f"No expected split types in data; found {sorted(set(means['split_type']))}"
        )

    size_order = get_sample_size_order(means)
    x_positions = np.arange(len(size_order))
    x_labels = [s if s != "full" else "All" for s in size_order]
    n_rows, n_cols = 5, 3

    fig = plt.figure(figsize=(12, 15))
    outer = fig.add_gridspec(len(blocks), 1, hspace=0.30, top=0.93, bottom=0.05)

    for block_idx, (letter, split_label, block_title) in enumerate(blocks):
        block = means[means["split_type"] == split_label]
        inner = outer[block_idx].subgridspec(n_rows, n_cols, hspace=0.45, wspace=0.18)
        phenotypes = [p for p in COMMON_PHENOTYPES if p in set(block["phenotype"])]

        for pos, phenotype in enumerate(phenotypes):
            ax = fig.add_subplot(inner[pos // n_cols, pos % n_cols])
            panel = block[block["phenotype"] == phenotype]

            for training_type, color in TRAINING_COLORS.items():
                for test_subset, (linestyle, marker) in TEST_SUBSET_STYLES.items():
                    series = panel[
                        (panel["training_type"] == training_type)
                        & (panel["test_subset"] == test_subset)
                    ].set_index("sample_size")["mean"]
                    if series.empty:
                        continue
                    values = [series.get(size, np.nan) for size in size_order]
                    ax.plot(
                        x_positions,
                        values,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=3,
                        linewidth=1.1,
                        markeredgewidth=0,
                    )

            ax.axhline(0.5, ls=":", color="grey", lw=0.6)
            ax.set_title(phenotype, fontsize=9, pad=3)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(-0.4, len(size_order) - 0.6)
            ax.set_xticks(x_positions)
            ax.grid(axis="y", color="0.92", linewidth=0.4)
            ax.tick_params(labelsize=7)

            on_bottom_row = pos >= len(phenotypes) - n_cols
            ax.set_xticklabels(x_labels if on_bottom_row else [])
            if pos % n_cols == 0:
                ax.set_ylabel("Balanced accuracy", fontsize=8)
            else:
                ax.set_yticklabels([])
            if on_bottom_row and pos % n_cols == 1:
                ax.set_xlabel("Training samples", fontsize=8)

        block_ax = fig.add_subplot(outer[block_idx], frame_on=False)
        block_ax.set_xticks([])
        block_ax.set_yticks([])
        block_ax.set_title(
            f"{letter}. {block_title}",
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=22,
        )

    handles = [
        Line2D(
            [],
            [],
            color=TRAINING_COLORS[training_type],
            linestyle=TEST_SUBSET_STYLES[test_subset][0],
            marker=TEST_SUBSET_STYLES[test_subset][1],
            markersize=4,
            linewidth=1.2,
            label=f"{training_type} training, {test_subset.lower()}",
        )
        for training_type in TRAINING_COLORS
        for test_subset in TEST_SUBSET_STYLES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved multi-phenotype training-size grid to {output_file}")
    plt.close(fig)


def plot_learning_curves(df: pd.DataFrame, output_file: Path) -> None:
    """Plot the supplementary learning-curve grid.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    sub = df[
        (df["training_type"] == "Concordant") & (df["test_subset"] == "Full Test")
    ].copy()

    phenotypes = [p for p in COMMON_PHENOTYPES if p in sub["phenotype"].unique()]
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]
    split_colors = {
        "Random Split": "#06A77D",
        "Dataset Split": "#2E86AB",
        "Out-of-Clade": "#DE8F05",
    }
    sample_size_order = get_sample_size_order(sub)
    x_positions = np.arange(len(sample_size_order))
    x_labels = [size if size != "full" else "Full" for size in sample_size_order]

    n_cols = 3
    n_rows = int(np.ceil(len(phenotypes) / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for phenotype_index, phenotype in enumerate(phenotypes):
        ax = axes[phenotype_index]
        for split_type in split_types:
            color = split_colors[split_type]
            subset = sub[
                (sub["phenotype"] == phenotype) & (sub["split_type"] == split_type)
            ]

            if subset.empty:
                continue

            summary = (
                subset.groupby("sample_size")
                .agg(mean=("balanced_accuracy", "mean"))
                .reindex(sample_size_order)
            )
            valid_summary = summary["mean"].notna()
            ax.plot(
                x_positions[valid_summary.to_numpy()],
                summary.loc[valid_summary, "mean"],
                color=color,
                marker="o",
                markersize=3.5,
                lw=1.4,
                label=split_type,
            )

        ax.set_title(phenotype, fontsize=10, pad=5)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls=":", color="grey", lw=0.5)
        ax.grid(axis="y", color="0.9", linewidth=0.4)
        ax.set_xticks(x_positions)

    for axis_index, ax in enumerate(axes):
        if axis_index < len(phenotypes):
            if axis_index // n_cols == n_rows - 1:
                ax.set_xticklabels(x_labels)
            else:
                ax.tick_params(labelbottom=False)
        else:
            ax.set_visible(False)

    fig.supxlabel("Training Sample Size", y=0.02)
    fig.supylabel("Balanced Accuracy", x=0.01)
    handles = [
        Line2D(
            [0],
            [0],
            color=split_colors[split_type],
            marker="o",
            markersize=3.5,
            lw=1.4,
            label=split_type,
        )
        for split_type in split_types
    ]
    labels = split_types
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved learning curves to {output_file}")
    plt.close(fig)


def plot_saturation_summary(df: pd.DataFrame, output_file: Path) -> None:
    """Plot the supplementary saturation summary.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    random_saturation_sizes = estimate_saturation_sizes(df, split_type="Random Split")
    dataset_saturation_sizes = estimate_saturation_sizes(df, split_type="Dataset Split")
    order_map = {"50": 50, "100": 100, "200": 200, "500": 500}
    sorted_phenotypes = sorted(
        dataset_saturation_sizes,
        key=lambda phenotype: (
            order_map[dataset_saturation_sizes[phenotype]],
            order_map[random_saturation_sizes[phenotype]],
            phenotype,
        ),
    )
    random_sizes = [
        order_map[random_saturation_sizes[phenotype]] for phenotype in sorted_phenotypes
    ]
    dataset_sizes = [
        order_map[dataset_saturation_sizes[phenotype]]
        for phenotype in sorted_phenotypes
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y_positions = np.arange(len(sorted_phenotypes))
    offset = 0.16

    for y_position, random_size, dataset_size in zip(
        y_positions, random_sizes, dataset_sizes, strict=False
    ):
        ax.plot(
            [random_size, dataset_size],
            [y_position - offset, y_position + offset],
            color="0.7",
            linewidth=1.0,
            zorder=1,
        )

    ax.scatter(
        random_sizes,
        y_positions - offset,
        color="#06A77D",
        s=44,
        edgecolors="black",
        linewidths=0.4,
        label="Intra-dataset",
        zorder=3,
    )
    ax.scatter(
        dataset_sizes,
        y_positions + offset,
        color="#2E86AB",
        marker="s",
        s=44,
        edgecolors="black",
        linewidths=0.4,
        label="Cross-dataset",
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_phenotypes)
    ax.set_xlabel("Estimated Saturation Sample Size")
    ax.invert_yaxis()
    observed_sizes = sorted(set(random_sizes + dataset_sizes))
    ax.set_xlim(min(observed_sizes) - 15, max(observed_sizes) + 25)
    ax.set_xticks(observed_sizes)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved saturation summary to {output_file}")
    plt.close(fig)
