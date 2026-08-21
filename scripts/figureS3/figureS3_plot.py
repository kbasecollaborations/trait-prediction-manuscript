#!/usr/bin/env python3
"""
Render Figure S3: in-clade vs out-of-clade balanced accuracy.

Two stacked panels (in-clade on top, out-of-clade below) show balanced accuracy for
the 15 shared phenotypes, with the four leave-one-dataset-out combinations drawn as
coloured strips, one colour per held-out dataset. Excluded combinations are dropped
from the plot and only counted on stdout.

Reads data/outputs/figureS3/figureS3_data.tsv, writes figures/figure_s3.pdf.

Run with::

    uv run python -m scripts.figureS3.figureS3_plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns
from matplotlib.lines import Line2D

from scripts.visualization import (
    configure_plot_style,
    hide_categorical_minor_ticks,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

np.random.seed(42)


DATA_FILE: Path = Path("data/outputs/figureS3/figureS3_data.tsv")
OUTPUT_FILE: Path = Path("figures/figure_s3.pdf")

PANEL_TITLES: dict[str, str] = {
    "in_clade": "In-clade",
    "out_of_clade": "Out-of-clade",
}
PANEL_LABELS: dict[str, str] = {
    "in_clade": "(A)",
    "out_of_clade": "(B)",
}


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    split_type: str,
    phenotypes: list[str],
    test_datasets: list[str],
    annotate_n: bool = True,
) -> None:
    """
    Plot one panel (in-clade or out-of-clade) on the given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Long-form results DataFrame containing only retained rows.
    split_type : str
        Either ``"in_clade"`` or ``"out_of_clade"``; selects which subset of
        ``df`` to plot and which panel label / title to use.
    phenotypes : list[str]
        Ordered list of phenotypes for the x axis (shared across panels).
    test_datasets : list[str]
        Ordered list of held-out dataset names; defines the colour groups.
    annotate_n : bool, optional
        If ``True``, annotate each strip with the test-subset size, by
        default ``True``.
    """
    panel_df = df[df["split_type"] == split_type]
    dataset_colors = get_dataset_colors()

    n_datasets = len(test_datasets)
    width = 0.18
    offsets = np.linspace(
        -width * (n_datasets - 1) / 2,
        width * (n_datasets - 1) / 2,
        n_datasets,
    )

    x = np.arange(len(phenotypes))

    # Alternating background bands help track phenotype columns.
    for i in range(len(phenotypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.1, zorder=0)

    for ds_idx, dataset in enumerate(test_datasets):
        ds_df = panel_df[panel_df["test_dataset"] == dataset]
        for ph_idx, phenotype in enumerate(phenotypes):
            sub = ds_df[ds_df["phenotype"] == phenotype]
            if sub.empty:
                continue
            x_pos = x[ph_idx] + offsets[ds_idx]
            jitter = np.random.normal(0, 0.015, size=len(sub))
            ax.scatter(
                x_pos + jitter,
                sub["balanced_accuracy"],
                color=dataset_colors[dataset],
                alpha=0.85,
                s=60,
                zorder=2,
                edgecolor="white",
                linewidth=0.5,
            )
            if annotate_n:
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"{int(row['n_test'])}",
                        (x_pos, row["balanced_accuracy"]),
                        textcoords="offset points",
                        # Alternating label heights keep adjacent datasets at
                        # similar accuracy from colliding.
                        xytext=(0, 6 if ds_idx % 2 == 0 else 15),
                        ha="center",
                        # Included at \textwidth (scale ~0.55), so this prints
                        # at roughly 4.5 pt.
                        fontsize=8,
                        color="dimgray",
                        zorder=3,
                    )

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)

    ax.set_title(PANEL_TITLES[split_type], loc="center", pad=4, fontsize=12)
    ax.text(
        -0.06,
        1.05,
        PANEL_LABELS[split_type],
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def build_legend_handles(test_datasets: list[str]) -> list[Line2D]:
    """
    Build legend handles - one marker per held-out dataset.

    Parameters
    ----------
    test_datasets : list[str]
        Ordered list of held-out dataset names.

    Returns
    -------
    list[Line2D]
        Legend handles suitable for ``ax.legend(handles=...)``.
    """
    dataset_colors = get_dataset_colors()
    display_names = format_dataset_names(test_datasets)
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dataset_colors[dataset],
            markersize=8,
            label=f"Test: {display_name}",
            linestyle="None",
        )
        for dataset, display_name in zip(test_datasets, display_names)
    ]


def create_figure(data_file: Path, output_file: Path) -> None:
    """
    Build and save Figure S3.

    Parameters
    ----------
    data_file : Path
        Path to the TSV produced by :mod:`scripts.figureS3.figureS3_data`.
    output_file : Path
        Where to write the PDF.
    """
    df = pd.read_csv(data_file, sep="\t")

    retained = df[~df["excluded"]].copy()
    excluded = df[df["excluded"]].copy()

    if retained.empty:
        raise RuntimeError(f"No retained rows in {data_file}; nothing to plot.")

    phenotypes = sorted(df["phenotype"].unique())
    test_datasets = sorted(df["test_dataset"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    plot_panel(axes[0], retained, "in_clade", phenotypes, test_datasets)
    plot_panel(axes[1], retained, "out_of_clade", phenotypes, test_datasets)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("Phenotype")
    axes[0].tick_params(axis="x", labelbottom=False)

    legend_handles = build_legend_handles(test_datasets)
    axes[0].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_handles),
        frameon=False,
    )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hide_categorical_minor_ticks(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_in = int((retained["split_type"] == "in_clade").sum())
    n_out = int((retained["split_type"] == "out_of_clade").sum())
    print(f"Saved plot to {output_file}")
    print(f"Retained: in_clade={n_in}, out_of_clade={n_out}")
    if not excluded.empty:
        print(f"Excluded: {len(excluded)} combinations")
        print(excluded["exclusion_reason"].value_counts())
    mean_in = retained.loc[
        retained["split_type"] == "in_clade", "balanced_accuracy"
    ].mean()
    mean_out = retained.loc[
        retained["split_type"] == "out_of_clade", "balanced_accuracy"
    ].mean()
    print(
        f"Mean balanced accuracy: in_clade={mean_in:.3f}, out_of_clade={mean_out:.3f}"
    )


if __name__ == "__main__":
    create_figure(DATA_FILE, OUTPUT_FILE)
