#!/usr/bin/env python3
"""Create Figure 6: hybrid composition of filtering and feature-selection diagnostics.

Layout (after the hybrid revision):

- Panel A (row 1, two columns): condensed problematic-sample summary on the left
  and the ranked top-20 misclassified microbe diagnostic on the right.
- Panel B (row 2, full width): per-phenotype grouped balanced-accuracy bars across
  three filtering strategies, with sample-removal annotations.
- Panel C and D (row 3, two columns): combined ML+GapMind precision-recall
  scatter on the left and combined-vs-phenotype-filtered balanced-accuracy
  scatter on the right.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from scripts.figure6.figure6a_plot import (
    plot_microbe_misclassification_ranking,
    plot_problematic_sample_summary,
)
from scripts.figure6.figure6b_plot import plot_confident_samples_performance
from scripts.figure6.figure6c_plot import plot_precision_recall_scatter
from scripts.figure6.figure6d_plot import plot_balanced_accuracy_scatter
from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def create_figure6(output_file: Path) -> None:
    """Create the hybrid four-panel Figure 6.

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    data_dir = Path("data/outputs/figure6")

    print("Loading data files...")
    ml_df_6b = pd.read_csv(data_dir / "figure6b_confident_ml_results.csv")
    df_6c = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    df_6d = pd.read_csv(data_dir / "figure6d_all_results.csv")

    phenotypes_6b_dataset = set(
        ml_df_6b[ml_df_6b["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    phenotypes_6b_random = set(
        ml_df_6b[ml_df_6b["split_type"] == "random_split"]["phenotype"].unique()
    )
    phenotypes_6c = set(df_6c["phenotype"].unique())
    phenotypes_6d = set(df_6d["phenotype"].unique())

    common_phenotypes = sorted(
        phenotypes_6b_dataset.intersection(phenotypes_6b_random)
        .intersection(phenotypes_6c)
        .intersection(phenotypes_6d)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    fig = plt.figure(figsize=(9, 9.5))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.05, 0.9, 2.25],
        hspace=0.5,
    )

    # --- Panel A: condensed summary (left) + ranked microbe diagnostic (right) ---
    print("Creating Panel A...")
    gs_a = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 0], width_ratios=[1.0, 1.18], wspace=0.45
    )
    ax_a_left = fig.add_subplot(gs_a[0, 0])
    ax_a_right = fig.add_subplot(gs_a[0, 1])
    plot_problematic_sample_summary(ax_a_left)
    plot_microbe_misclassification_ranking(ax_a_right)

    # --- Panel B: per-phenotype grouped filtering comparison ---
    print("Creating Panel B...")
    ax_b = fig.add_subplot(gs[1, 0])
    plot_confident_samples_performance(ax_b, data_dir, common_phenotypes)
    ax_b.set_xlabel("")

    # --- Panel C (left) and Panel D (right): compact lower diagnostic block ---
    print("Creating Panels C and D...")
    gs_cd = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.14)
    ax_c = fig.add_subplot(gs_cd[0, 0])
    ax_d = fig.add_subplot(gs_cd[0, 1])
    plot_precision_recall_scatter(ax_c, data_dir, common_phenotypes)
    plot_balanced_accuracy_scatter(ax_d, df_6d, common_phenotypes)

    panel_label_axes: dict[str, plt.Axes] = {
        "(A)": ax_a_left,
        "(B)": ax_b,
        "(C)": ax_c,
        "(D)": ax_d,
    }
    for label, ax in panel_label_axes.items():
        if ax is ax_a_left:
            x_pos = -0.18
            y_pos = 1.05
        elif ax is ax_b:
            x_pos = -0.08
            y_pos = 1.05
        elif ax is ax_d:
            x_pos = -0.08
            y_pos = 1.02
        else:
            x_pos = -0.20
            y_pos = 1.02
        ax.text(
            x_pos,
            y_pos,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="right",
            fontsize=14,
        )

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    output_file = Path("figures/figure6.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure6(output_file)
