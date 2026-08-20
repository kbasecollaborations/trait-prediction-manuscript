#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns
from matplotlib.patches import Rectangle

from scripts.visualization import (
    configure_plot_style,
    hide_categorical_minor_ticks,
    format_dataset_names,
    get_dataset_color_list,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_data(df: pd.DataFrame, output_file: Path) -> None:
    """Plot stacked bar chart of positive and negative genome counts per phenotype, hued by dataset."""

    phenotypes = sorted(df["phenotype"].unique())
    datasets = sorted(df["dataset"].unique())
    dataset_display_names = format_dataset_names(datasets)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.text(
        -0.05,
        1.05,
        "B",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

    bar_width = 0.8 / len(datasets)
    x = np.arange(len(phenotypes))

    dataset_colors = get_dataset_color_list(datasets)

    for i, dataset in enumerate(datasets):
        df_dataset = df[df["dataset"] == dataset].set_index("phenotype")
        df_dataset = df_dataset.reindex(phenotypes, fill_value=0)

        positions = x + i * bar_width

        positive = df_dataset["positive_count"].values
        ax.bar(
            positions,
            positive,
            bar_width,
            label=f"{dataset}",
            color=dataset_colors[i],
            alpha=0.8,
        )

        negative = df_dataset["negative_count"].values
        ax.bar(
            positions,
            negative,
            bar_width,
            bottom=positive,
            color=dataset_colors[i],
            alpha=0.4,
            hatch="//",
        )

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Number of Genomes")
    # Center the tick under each group of dataset bars.
    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")

    dataset_handles = [
        Rectangle((0, 0), 1, 1, fc=color, alpha=0.8) for color in dataset_colors
    ]
    legend1 = ax.legend(
        dataset_handles,
        dataset_display_names,
        title="Dataset",
        loc="upper right",
        bbox_to_anchor=(0.48, 1.15),
        ncol=len(datasets),
        frameon=False,
    )
    ax.add_artist(legend1)

    status_handles = [
        Rectangle((0, 0), 1, 1, fc="gray", alpha=0.8, label="Positive"),
        Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4, hatch="//", label="Negative"),
    ]
    ax.legend(
        handles=status_handles,
        title="Phenotype",
        loc="upper left",
        bbox_to_anchor=(0.58, 1.15),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    hide_categorical_minor_ticks(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/outputs/figure1/figure1b_data.csv")
    output_file = Path("figures/figure1b.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_data(df, output_file)
