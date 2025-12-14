#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots

from scripts.figure4.figure4a_quadrant import create_quadrant_plot
from scripts.figure4.figure4b_plot import create_misclassification_plots
from scripts.figure4.figure4c_plot import create_panel_c_plots

plt.style.use(["science", "nature"])


def create_figure4(output_file: Path) -> None:
    """Create Figure 4 with quadrant plot (A), misclassification plots (B), and feature plots (C).

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    # Create figure with three rows
    fig = plt.figure(figsize=(20, 18))

    # Create grid: 2 rows, top row has 2 columns, bottom row spans full width
    import matplotlib.gridspec as gridspec

    # Main grid: 2 rows
    main_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.3)

    # Top row grid: 2 columns for panels A and B
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_gs[0, :], width_ratios=[1, 1.2], wspace=0.15
    )

    # Left side: Figure 4A (quadrant plot)
    ax_quadrant = fig.add_subplot(top_gs[0, 0])

    # Right side: Figure 4B (3 subplots)
    # Create nested gridspec for right side
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=top_gs[0, 1],
        height_ratios=[1, 1.2], hspace=0.25, wspace=0.3
    )

    ax_b1 = fig.add_subplot(right_gs[0, 0])  # Top left
    ax_b2 = fig.add_subplot(right_gs[0, 1])  # Top right

    # Create a nested gridspec for the bottom row to center and narrow the third plot
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=right_gs[1, :],
        width_ratios=[0.2, 1, 0.2], wspace=0
    )
    ax_b3 = fig.add_subplot(bottom_gs[0, 1])  # Center column only

    # Bottom row: Figure 4C (2 vertically stacked subplots)
    bottom_panel_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=main_gs[1, :], hspace=0.15
    )

    ax_c1 = fig.add_subplot(bottom_panel_gs[0, 0])  # Top subplot
    ax_c2 = fig.add_subplot(bottom_panel_gs[1, 0])  # Bottom subplot

    # Create the plots
    print("Creating Figure 4A (quadrant plot)...")
    create_quadrant_plot(ax_quadrant)

    print("\nCreating Figure 4B (misclassification plots)...")
    create_misclassification_plots(ax_b1, ax_b2, ax_b3)

    print("\nCreating Figure 4C (feature stability and comparison)...")
    create_panel_c_plots(ax_c1, ax_c2)

    # Add panel labels
    ax_quadrant.text(
        -0.1, 1.05, "(A)", transform=ax_quadrant.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="right"
    )
    ax_b1.text(
        -0.15, 1.05, "(B)", transform=ax_b1.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="right"
    )
    ax_c1.text(
        -0.05, 1.05, "(C)", transform=ax_c1.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="right"
    )

    # Save figure
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure4.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure4(output_file)
