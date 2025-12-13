#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots

from scripts.figure4.figure4a_quadrant import create_quadrant_plot
from scripts.figure4.figure4b_plot import create_misclassification_plots

plt.style.use(["science", "nature"])


def create_figure4(output_file: Path) -> None:
    """Create Figure 4 combining quadrant plot (A) and misclassification plots (B).

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    # Create figure with two columns
    fig = plt.figure(figsize=(20, 12))

    # Create grid: 2 columns, first column spans full height for 4A
    # Second column has grid for 4B subplots
    import matplotlib.gridspec as gridspec

    # Main grid: 2 columns
    main_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.15)

    # Left side: Figure 4A (quadrant plot)
    ax_quadrant = fig.add_subplot(main_gs[0, 0])

    # Right side: Figure 4B (3 subplots)
    # Create nested gridspec for right side
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=main_gs[0, 1],
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

    # Create the plots
    print("Creating Figure 4A (quadrant plot)...")
    create_quadrant_plot(ax_quadrant)

    print("\nCreating Figure 4B (misclassification plots)...")
    create_misclassification_plots(ax_b1, ax_b2, ax_b3)

    # Save figure
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure4.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure4(output_file)
