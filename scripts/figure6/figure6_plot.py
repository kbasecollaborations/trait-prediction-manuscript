#!/usr/bin/env python3
"""Compose the four-panel Figure 6 from the per-panel diagnostic plots."""

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
from scripts.figure6.figure6b_aggregate_plot import (
    best_panel_b_config,
    plot_gapmind_delta_forest,
    plot_metric_sweep,
)
from scripts.figure6.figure6d_plot import (
    load_results,
    plot_balanced_accuracy_scatter,
)
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
    sweep_df = pd.read_csv(data_dir / "figure6b_weight_sweep_combined.csv")
    df_7c = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    df_7d = load_results(data_dir / "figure6d_all_results.csv")

    phenotypes_sweep = set(
        sweep_df[sweep_df["split_type"] == "dataset_split"]["phenotype"].unique()
    )
    phenotypes_7c = set(df_7c["phenotype"].unique())
    phenotypes_7d = set(df_7d["phenotype"].unique())

    common_phenotypes = sorted(
        phenotypes_sweep.intersection(phenotypes_7c).intersection(phenotypes_7d)
    )
    print(f" - Common phenotypes: {len(common_phenotypes)}")

    fig = plt.figure(figsize=(9, 11.5))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.05, 1.55, 2.25],
        hspace=0.55,
    )

    print("Creating Panel A...")
    gs_a = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 0], width_ratios=[1.0, 1.18], wspace=0.45
    )
    ax_a_left = fig.add_subplot(gs_a[0, 0])
    ax_a_right = fig.add_subplot(gs_a[0, 1])
    plot_problematic_sample_summary(ax_a_left)
    plot_microbe_misclassification_ranking(ax_a_right)

    print("Creating Panel B...")
    ax_b = fig.add_subplot(gs[1, 0])
    plot_metric_sweep(ax_b, data_dir, common_phenotypes)

    print("Creating Panels C and D...")
    best_config_name, best_label, summary = best_panel_b_config(
        data_dir, common_phenotypes
    )
    print(
        f" - Top-BA confidence config: {best_config_name} ({best_label}); "
        f"mech-free gap: {summary['free_balanced'] - summary.max():+.3f} BA"
    )
    gs_cd = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.32)
    ax_c = fig.add_subplot(gs_cd[0, 0])
    ax_d = fig.add_subplot(gs_cd[0, 1])
    plot_gapmind_delta_forest(ax_c, data_dir, common_phenotypes)
    plot_balanced_accuracy_scatter(ax_d, df_7d, common_phenotypes)

    panel_label_axes: dict[str, plt.Axes] = {
        "A": ax_a_left,
        "B": ax_b,
        "C": ax_c,
        "D": ax_d,
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
