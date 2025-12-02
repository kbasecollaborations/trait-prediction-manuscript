#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

plt.style.use(["science", "nature"])
sns.set_context("paper")


def plot_data(df: pd.DataFrame, output_file: Path) -> None:
    """Plot stacked bar chart of positive and negative genome counts per phenotype, hued by dataset."""

    # Get unique phenotypes and datasets
    phenotypes = sorted(df["phenotype"].unique())
    datasets = sorted(df["dataset"].unique())

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set bar width and positions
    bar_width = 0.8 / len(datasets)
    x = np.arange(len(phenotypes))

    # Define colors for each dataset
    dataset_colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    # Plot bars for each dataset
    for i, dataset in enumerate(datasets):
        df_dataset = df[df["dataset"] == dataset].set_index("phenotype")
        df_dataset = df_dataset.reindex(phenotypes, fill_value=0)

        positions = x + i * bar_width

        # Plot positive counts (bottom of stack)
        positive = df_dataset["positive_count"].values
        ax.bar(
            positions,
            positive,
            bar_width,
            label=f"{dataset}",
            color=dataset_colors[i],
            alpha=0.8,
        )

        # Plot negative counts (top of stack)
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

    # Customize the plot
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Number of Genomes")
    # Center x-tick labels in the middle of the group of bars
    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    # Only show ticks at the bottom where labels are
    ax.tick_params(axis="x", which="both", top=False, bottom=True)

    # Create legend
    # Add dataset legend
    dataset_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=dataset_colors[i], alpha=0.8)
        for i in range(len(datasets))
    ]
    legend1 = ax.legend(
        dataset_handles,
        datasets,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        ncol=len(datasets),
        frameon=False,
    )
    ax.add_artist(legend1)

    # Add status legend (positive vs negative)
    status_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.8, label="Positive"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4, hatch="//", label="Negative"),
    ]
    ax.legend(
        handles=status_handles,
        title="Phenotype",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.15),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/outputs/figure1/figure1c_data.csv")
    output_file = Path("figures/figure1c.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_data(df, output_file)
