#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots

from scripts.figure6.figure6a_plot import create_misclassification_plots

plt.style.use(["science", "nature"])


def create_figure6(output_file: Path) -> None:
    """Create Figure 6 with misclassification plots (A).

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    import matplotlib.gridspec as gridspec

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right

    # Create a nested gridspec for the bottom row to center and narrow the third plot
    bottom_gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[0.2, 1, 0.2],
        wspace=0,
        left=gs[1, :].get_position(fig).x0,
        right=gs[1, :].get_position(fig).x1,
        bottom=gs[1, :].get_position(fig).y0,
        top=gs[1, :].get_position(fig).y1,
    )
    ax3 = fig.add_subplot(bottom_gs[0, 1])  # Center column only

    # Create the plots
    print("Creating Figure 6A (misclassification plots)...")
    create_misclassification_plots(ax1, ax2, ax3)

    # Add panel labels
    ax1.text(
        -0.15, 1.05, "(A)", transform=ax1.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="right"
    )

    # Save figure
    gs.tight_layout(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure6.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure6(output_file)
