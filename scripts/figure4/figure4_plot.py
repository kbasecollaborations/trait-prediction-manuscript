#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

from scripts.figure4.figure4a_quadrant import create_quadrant_plot
from scripts.figure4.figure4b_plot import create_confusion_matrix_plots
from scripts.figure4.figure4c_plot import create_panel_c_plots
from scripts.figure4.style import PANEL_LABEL_SIZE
from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def create_figure4(output_file: Path) -> None:
    """Create Figure 4 with quadrant plot A, confusion matrix plots B, and feature plots C.

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    # Keep the physical figure size closer to the final manuscript scale so
    # labels remain legible after LaTeX includes the PDF at \textwidth.
    fig = plt.figure(figsize=(12.5, 10.4))

    import matplotlib.gridspec as gridspec

    # Main grid: tighten the top row so panels A/B do not carry excess empty
    # space relative to panel C. Explicit top/left/right/bottom remove the
    # default matplotlib margins so panel A's content reaches close to the
    # figure edges instead of leaving a wide whitespace band.
    main_gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.28, 0.82],
        hspace=0.18,
        top=0.94,
        bottom=0.06,
        left=0.04,
        right=0.99,
    )

    # Top row grid: allocate slightly more width to panel B to reduce empty
    # margins around panel A while preserving readability of the bar labels.
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=main_gs[0, :],
        width_ratios=[1.08, 0.92],
        wspace=0.12,
    )

    # Left side: Figure 4A (quadrant plot)
    ax_quadrant = fig.add_subplot(top_gs[0, 0])

    # Right side: Figure 4B (2 vertically stacked subplots).
    # Keep the per-phenotype plot slightly shorter than before so the stacked
    # bars do not dominate the top row.
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=top_gs[0, 1],
        height_ratios=[1.45, 1],
        hspace=0.68,
    )

    ax_b1 = fig.add_subplot(right_gs[0, 0])  # Top subplot
    ax_b2 = fig.add_subplot(right_gs[1, 0])  # Bottom subplot

    # Bottom row: Figure 4C (2 vertically stacked subplots with shared x-axis)
    bottom_panel_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=main_gs[1, :], hspace=0.56
    )

    ax_c1 = fig.add_subplot(bottom_panel_gs[0, 0])  # Top subplot
    ax_c2 = fig.add_subplot(bottom_panel_gs[1, 0], sharex=ax_c1)  # Bottom subplot, shared x-axis

    print("Creating Figure 4A (quadrant plot)...")
    create_quadrant_plot(ax_quadrant)

    print("\nCreating Figure 4B (confusion matrix plots)...")
    create_confusion_matrix_plots(ax_b1, ax_b2)

    print("\nCreating Figure 4C (feature stability and comparison)...")
    create_panel_c_plots(ax_c1, ax_c2)

    # Add panel labels. A and B share an explicit figure-level y so they
    # render at the same vertical position regardless of subplot heights.
    panel_label_y = 0.98
    fig.text(
        0.005, panel_label_y, "A",
        fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left",
    )
    b1_bbox = ax_b1.get_position()
    fig.text(
        b1_bbox.x0 - 0.025, panel_label_y, "B",
        fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left",
    )
    c1_bbox = ax_c1.get_position()
    fig.text(
        c1_bbox.x0 - 0.085,
        c1_bbox.y1 + 0.01,
        "C",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure4.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure4(output_file)
