#!/usr/bin/env python3
"""Plot the supplementary pangenome completeness audit figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  # Registers the matplotlib styles below.
import seaborn as sns

from scripts.visualization import (
    configure_plot_style,
    hide_categorical_minor_ticks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Default paths (relative to project root)
DEFAULT_INPUT = Path("data/outputs/pangenome_completeness/pangenome_completeness.tsv")
DEFAULT_OUTPUT = Path("figures/figure_s4.pdf")


def load_completeness_data(input_path: Path) -> pd.DataFrame:
    """Load pangenome completeness TSV into a DataFrame.

    Parameters
    ----------
    input_path : Path
        Path to the completeness results TSV file.

    Returns
    -------
    pd.DataFrame
        Full completeness table with columns: genome_id, species,
        core_genes_expected, core_genes_present, completeness_pct, status,
        error_message.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Completeness results not found: {input_path}")
    return pd.read_csv(input_path, sep="\t")


def plot_status_breakdown(ax: plt.Axes, df: pd.DataFrame, label: str = "(A)") -> None:
    """Bar chart of genome counts by status.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    df : pd.DataFrame
        Full completeness DataFrame (must have 'status' column).
    label : str
        Subplot label (e.g. "(A)").
    """
    status_order = ["success", "no_species", "no_core_genes", "error"]
    counts = df["status"].value_counts().reindex(status_order, fill_value=0)
    colors = {
        "success": "#2E7D5B",
        "no_species": "#8C8C8C",
        "no_core_genes": "#BFBFBF",
        "error": "#E0E0E0",
    }
    x = np.arange(len(status_order))
    bars = ax.bar(
        x,
        counts.values,
        color=[colors[s] for s in status_order],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Success", "No species", "No core genes", "Error"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Number of genomes")
    ax.set_xlabel("")
    for bar, val in zip(bars, counts.values, strict=False):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def plot_completeness_distribution(
    ax: plt.Axes, df_success: pd.DataFrame, label: str = "(B)"
) -> None:
    """Histogram of completeness (successful genomes only).

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    df_success : pd.DataFrame
        Rows with status == "success"; must have 'completeness_pct'.
    label : str
        Subplot label (e.g. "(B)").
    """
    bins = [0, 50, 70, 80, 90, 100]
    ax.hist(
        df_success["completeness_pct"],
        bins=bins,
        color="#2E7D5B",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("Completeness (percentage)")
    ax.set_ylabel("Number of genomes")
    ax.set_xticks([0, 50, 70, 80, 90, 100])
    ax.set_xticklabels(["0", "50", "70", "80", "90", "100"])
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def create_figure(input_path: Path, output_path: Path) -> None:
    """Build the two-panel quality-control figure and save to PDF.

    Parameters
    ----------
    input_path : Path
        Path to completeness TSV.
    output_path : Path
        Path for output PDF (e.g. figures/figure_s4.pdf).
    """
    df = load_completeness_data(input_path)
    df_success = df[df["status"] == "success"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_status_breakdown(axes[0], df, label="(A)")
    plot_completeness_distribution(axes[1], df_success, label="(B)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hide_categorical_minor_ticks(fig)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse args, create figure, exit.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="Visualize pangenome completeness results (supplementary figure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to completeness results TSV.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    args = parser.parse_args(argv)

    try:
        create_figure(args.input, args.output)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
