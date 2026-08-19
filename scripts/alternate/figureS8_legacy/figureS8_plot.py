#!/usr/bin/env python3
"""Render Supplementary Figure S8: faceted stacked-bar grid of per-phenotype x
per-dataset concordance counts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  side-effect: registers plot styles
from matplotlib.patches import Patch

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
)

INPUT_FILE: Final[Path] = Path("data/outputs/concordance_counts/concordance_counts.tsv")
OUTPUT_FILE: Final[Path] = Path("figures/figure_s8.pdf")

# Match Figure 4 quadrant palette: concordant green, FP red, FN yellow, excluded grey.
CATEGORY_COLORS: Final[dict[str, str]] = {
    "n_concordant": "#c8e6c9",
    "n_discordant_FP": "#ffcdd2",
    "n_discordant_FN": "#fff9c4",
    "n_excluded": "#e0e0e0",
}
CATEGORY_EDGES: Final[dict[str, str]] = {
    "n_concordant": "#388e3c",
    "n_discordant_FP": "#d32f2f",
    "n_discordant_FN": "#f57c00",
    "n_excluded": "#9e9e9e",
}
CATEGORY_LABELS: Final[dict[str, str]] = {
    "n_concordant": "Concordant",
    "n_discordant_FP": "Discordant (FP)",
    "n_discordant_FN": "Discordant (FN)",
    "n_excluded": "Excluded",
}

CATEGORY_ORDER: Final[tuple[str, ...]] = (
    "n_concordant",
    "n_discordant_FP",
    "n_discordant_FN",
    "n_excluded",
)


def prepare_plot_frame(counts: pd.DataFrame) -> pd.DataFrame:
    """Add an ``n_excluded`` column that pools both exclusion reasons.

    Parameters
    ----------
    counts : pd.DataFrame
        Long-form counts table from ``figureS8_data``.

    Returns
    -------
    pd.DataFrame
        Copy of ``counts`` with an added ``n_excluded`` column.
    """
    df = counts.copy()
    df["n_excluded"] = df["n_excluded_no_gapmind"] + df["n_excluded_no_phenotype"]
    return df


def plot_concordance_grid(
    counts: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot the faceted stacked-bar grid of concordance counts.

    Parameters
    ----------
    counts : pd.DataFrame
        Long-form counts table containing the categorical columns named in
        ``CATEGORY_ORDER`` plus ``phenotype`` and ``dataset``.
    output_file : Path
        Destination PDF path.
    """
    plt.style.use(["science", "nature"])
    configure_plot_style()

    df = prepare_plot_frame(counts)
    phenotypes = sorted(df["phenotype"].unique())
    datasets = sorted(df["dataset"].unique())
    dataset_titles = format_dataset_names(datasets)

    n_rows = len(phenotypes)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 12),
        sharex=False,
        sharey=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )

    # Determine per-dataset x-limit so bars are comparable within a column.
    per_dataset_max: dict[str, int] = {}
    for dataset in datasets:
        sub = df[df["dataset"] == dataset]
        per_dataset_max[dataset] = int(sub[list(CATEGORY_ORDER)].sum(axis=1).max() or 1)

    for i, phenotype in enumerate(phenotypes):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            row = df[(df["phenotype"] == phenotype) & (df["dataset"] == dataset)]
            if row.empty:
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            row_vals = {cat: int(row.iloc[0][cat]) for cat in CATEGORY_ORDER}

            left = 0.0
            for cat in CATEGORY_ORDER:
                value = row_vals[cat]
                if value == 0:
                    continue
                ax.barh(
                    0,
                    value,
                    left=left,
                    color=CATEGORY_COLORS[cat],
                    edgecolor=CATEGORY_EDGES[cat],
                    linewidth=0.4,
                    height=0.7,
                )
                # Annotate counts only for segments wide enough to read.
                if value / per_dataset_max[dataset] >= 0.08:
                    ax.text(
                        left + value / 2,
                        0,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#212121",
                    )
                left += value

            ax.set_xlim(0, per_dataset_max[dataset])
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)

            if i == 0:
                ax.set_title(
                    dataset_titles[j],
                    fontsize=11,
                    fontweight="bold",
                    pad=6,
                )
            if j == 0:
                ax.set_ylabel(
                    phenotype,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    labelpad=8,
                )
            if i != n_rows - 1:
                ax.set_xticklabels([])

    legend_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[cat],
            edgecolor=CATEGORY_EDGES[cat],
            label=CATEGORY_LABELS[cat],
        )
        for cat in CATEGORY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Per-phenotype concordance with GapMind predictions, by dataset",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("Number of genomes", fontsize=10, y=0.025)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close(fig)


def main() -> None:
    """Load the counts table and render the figure."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Counts table not found at {INPUT_FILE}; run figureS12_data first."
        )
    counts = pd.read_csv(INPUT_FILE, sep="\t")
    plot_concordance_grid(counts, OUTPUT_FILE)


if __name__ == "__main__":
    main()
