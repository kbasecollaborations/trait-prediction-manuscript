#!/usr/bin/env python3
"""Plot Figure 5D: full held-out test BA (top) and discordance-category
composition (bottom), per phenotype.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (matplotlib style registration)
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()
np.random.seed(42)


DATA_FILE: Path = Path("data/outputs/figure5/figure5d_full_test.tsv")
OUTPUT_FILE: Path = Path("figures/figure_s8.pdf")

DATASET_ORDER: list[str] = ["atleaf", "lit", "marine", "pmi"]

CATEGORY_COLORS: dict[str, str] = {
    "concordant": "#2E86AB",
    "fp_discordant": "#E76F51",
    "fn_discordant": "#F4A261",
}
CATEGORY_LABELS: dict[str, str] = {
    "concordant": "GapMind concordant",
    "fp_discordant": "GapMind FP-discordant",
    "fn_discordant": "GapMind FN-discordant",
}


def plot_full_test_balanced_accuracy(
    ax: Axes,
    df: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """Plot per-phenotype balanced accuracy on the full held-out test set.

    Each phenotype gets a boxplot over the four held-out datasets, overlaid with
    one coloured strip point per held-out dataset.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Figure 5D data table.
    phenotypes : list[str]
        Ordered phenotype names defining the x-axis.
    """
    x = np.arange(len(phenotypes))
    color_map = get_dataset_colors()

    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    box_data: list[np.ndarray] = []
    for phenotype in phenotypes:
        values = df.loc[df["phenotype"] == phenotype, "balanced_accuracy_full"].values
        box_data.append(values)

    ax.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="lightgray", alpha=0.5, linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        zorder=1,
    )

    for i, phenotype in enumerate(phenotypes):
        sub = df[df["phenotype"] == phenotype]
        for dataset in DATASET_ORDER:
            row = sub[sub["held_out_dataset"] == dataset]
            if row.empty:
                continue
            value = float(row["balanced_accuracy_full"].iloc[0])
            jitter = np.random.uniform(-0.18, 0.18)
            ax.scatter(
                x[i] + jitter,
                value,
                color=color_map[dataset],
                s=42,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xlabel("")
    ax.set_ylabel("Balanced accuracy (full held-out test)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.5, len(phenotypes) - 0.5)
    ax.set_xticks(x)
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labelbottom=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[dataset],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=label,
        )
        for dataset, label in zip(
            DATASET_ORDER, format_dataset_names(DATASET_ORDER), strict=True
        )
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(DATASET_ORDER),
        frameon=False,
        title="Held-out dataset",
        title_fontsize=9,
        fontsize=9,
    )

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


def plot_test_composition(
    ax: Axes,
    df: pd.DataFrame,
    phenotypes: list[str],
) -> None:
    """Plot per-phenotype stacked bar of test-set discordance composition.

    Fractions sum counts across the four held-out datasets, so one bar per
    phenotype reflects the overall cross-dataset test composition.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Figure 5D data table.
    phenotypes : list[str]
        Ordered phenotype names defining the x-axis.
    """
    x = np.arange(len(phenotypes))
    fractions: dict[str, list[float]] = {
        "concordant": [],
        "fp_discordant": [],
        "fn_discordant": [],
    }
    for phenotype in phenotypes:
        sub = df[df["phenotype"] == phenotype]
        n_total = float(sub["n_test_full"].sum())
        if n_total == 0:
            for cat in fractions:
                fractions[cat].append(0.0)
            continue
        fractions["concordant"].append(float(sub["n_test_concordant"].sum()) / n_total)
        fractions["fp_discordant"].append(
            float(sub["n_test_FP_discordant"].sum()) / n_total
        )
        fractions["fn_discordant"].append(
            float(sub["n_test_FN_discordant"].sum()) / n_total
        )

    bottom = np.zeros(len(phenotypes))
    for cat in ["concordant", "fp_discordant", "fn_discordant"]:
        values = np.array(fractions[cat])
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=CATEGORY_COLORS[cat],
            edgecolor="black",
            linewidth=0.4,
            label=CATEGORY_LABELS[cat],
            width=0.7,
        )
        bottom = bottom + values

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Fraction of held-out test samples")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(phenotypes) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)

    legend_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[cat],
            edgecolor="black",
            linewidth=0.4,
            label=CATEGORY_LABELS[cat],
        )
        for cat in ["concordant", "fp_discordant", "fn_discordant"]
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

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


from scripts.minority_filter import (
    full_test_minority_counts as _full_test_minority_counts,
)


def create_figure(data_file: Path, output_file: Path) -> None:
    """Build and persist Figure 5D.

    Parameters
    ----------
    data_file : Path
        Path to the Figure 5D data TSV.
    output_file : Path
        Destination PDF path.
    """
    df = pd.read_csv(data_file, sep="\t")

    # Minority-class-test-samples filter (Methods): exclude (phenotype, held-out
    # dataset) cells with fewer than 10 minority-class samples in the full test.
    full_minority = _full_test_minority_counts()
    keep = df.apply(
        lambda r: full_minority.get((r["phenotype"], r["held_out_dataset"]), 0) >= 10,
        axis=1,
    )
    df = df.loc[keep].copy()

    phenotypes = sorted(df["phenotype"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    plot_full_test_balanced_accuracy(axes[0], df, phenotypes)
    plot_test_composition(axes[1], df, phenotypes)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def report_summary(data_file: Path) -> None:
    """Print median balanced accuracies for the full and per-subset evaluations.

    Parameters
    ----------
    data_file : Path
        Path to the Figure 5D data TSV.
    """
    df = pd.read_csv(data_file, sep="\t")
    full_minority = _full_test_minority_counts()
    keep = df.apply(
        lambda r: full_minority.get((r["phenotype"], r["held_out_dataset"]), 0) >= 10,
        axis=1,
    )
    df = df.loc[keep].copy()
    print("\nMedian balanced accuracy across (phenotype, held-out dataset) pairs:")
    print(
        f"  full held-out test set:   {np.nanmedian(df['balanced_accuracy_full']):.3f}"
    )
    print(
        f"  concordant subset:        "
        f"{np.nanmedian(df['balanced_accuracy_concordant_subset']):.3f}"
    )
    print(
        f"  FP-discordant subset:     "
        f"{np.nanmedian(df['balanced_accuracy_FP_subset']):.3f}"
    )
    print(
        f"  FN-discordant subset:     "
        f"{np.nanmedian(df['balanced_accuracy_FN_subset']):.3f}"
    )


def main() -> None:
    """Build Figure 5D and print the summary statistics."""
    create_figure(DATA_FILE, OUTPUT_FILE)
    report_summary(DATA_FILE)


if __name__ == "__main__":
    main()
