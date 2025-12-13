#!/usr/bin/env python3

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import FancyBboxPatch

# Apply publication-ready style
plt.style.use(["science", "nature"])


def create_quadrant_plot(ax: plt.Axes) -> None:
    """Create the quadrant plot showing concordant/discordant classifications.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)  # Taller aspect ratio
    ax.set_aspect("equal")
    ax.axis("off")

    # Define colors
    concordant_color = "#c8e6c9"  # Light green for concordant
    discordant_fp_color = "#ffcdd2"  # Light red for false positive
    discordant_fn_color = "#fff9c4"  # Light yellow for false negative
    concordant_tn_color = "#e3f2fd"  # Light blue for true negative

    # Adjust vertical positions for taller aspect ratio
    y_offset = 2.5
    quad_height = 5.0
    quad_width = 4.3

    # Draw the four quadrants
    # Top-left: GapMind+ / Experiment+ (True Positive - Concordant)
    quad_tl = FancyBboxPatch(
        (0.5, 5.2 + y_offset),
        quad_width,
        quad_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=concordant_color,
        edgecolor="#388e3c",
        linewidth=2.5,
    )
    ax.add_patch(quad_tl)

    # Top-right: GapMind- / Experiment+ (False Negative - Discordant)
    quad_tr = FancyBboxPatch(
        (5.2, 5.2 + y_offset),
        quad_width,
        quad_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=discordant_fn_color,
        edgecolor="#f57c00",
        linewidth=2.5,
    )
    ax.add_patch(quad_tr)

    # Bottom-left: GapMind+ / Experiment- (False Positive - Discordant)
    quad_bl = FancyBboxPatch(
        (0.5, 0.5),
        quad_width,
        quad_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=discordant_fp_color,
        edgecolor="#d32f2f",
        linewidth=2.5,
    )
    ax.add_patch(quad_bl)

    # Bottom-right: GapMind- / Experiment- (True Negative - Concordant)
    quad_br = FancyBboxPatch(
        (5.2, 0.5),
        quad_width,
        quad_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=concordant_tn_color,
        edgecolor="#1976d2",
        linewidth=2.5,
    )
    ax.add_patch(quad_br)

    # Font size multiplier
    fs = 1.3

    # Add axis labels (adjusted for taller layout)
    ax.text(
        5,
        13.5,
        "GapMind Prediction",
        ha="center",
        va="bottom",
        fontsize=14 * fs,
        fontweight="bold",
    )
    ax.text(
        2.65,
        13.0,
        "Pathway Present (+)",
        ha="center",
        va="bottom",
        fontsize=12 * fs,
        fontweight="bold",
        color="#2e7d32",
    )
    ax.text(
        7.35,
        13.0,
        "Pathway Absent (−)",
        ha="center",
        va="bottom",
        fontsize=12 * fs,
        fontweight="bold",
        color="#c62828",
    )

    ax.text(
        -0.3,
        7,
        "Experimental\nOutcome",
        ha="center",
        va="center",
        fontsize=14 * fs,
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        0.15,
        9.7,
        "Growth (+)",
        ha="center",
        va="center",
        fontsize=12 * fs,
        fontweight="bold",
        color="#2e7d32",
        rotation=90,
    )
    ax.text(
        0.15,
        3.0,
        "No Growth (−)",
        ha="center",
        va="center",
        fontsize=12 * fs,
        fontweight="bold",
        color="#c62828",
        rotation=90,
    )

    # Top-left quadrant content (True Positive - Concordant)
    y_tl = 7.7 + y_offset
    ax.text(
        2.65,
        y_tl + 1.7,
        "CONCORDANT",
        ha="center",
        va="center",
        fontsize=11 * fs,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.text(
        2.65,
        y_tl + 1.15,
        "True Positive",
        ha="center",
        va="center",
        fontsize=10 * fs,
        fontstyle="italic",
        color="#2e7d32",
    )
    ax.text(
        2.65,
        y_tl + 0.4,
        "Genes present and functional",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#1b5e20",
    )
    ax.text(
        2.65,
        y_tl + 0.0,
        "Pathway correctly annotated",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#1b5e20",
    )
    ax.text(
        2.65,
        y_tl - 0.4,
        "Experiment accurately measured",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#1b5e20",
    )
    ax.text(
        2.65,
        y_tl - 1.15,
        "High-quality training samples",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#388e3c", alpha=0.8),
    )

    # Top-right quadrant content (False Negative - Discordant: GapMind-, Experiment+)
    y_tr = 7.7 + y_offset
    ax.text(
        7.35,
        y_tr + 1.7,
        "DISCORDANT",
        ha="center",
        va="center",
        fontsize=11 * fs,
        fontweight="bold",
        color="#e65100",
    )
    ax.text(
        7.35,
        y_tr + 1.15,
        "False Negative",
        ha="center",
        va="center",
        fontsize=10 * fs,
        fontstyle="italic",
        color="#f57c00",
    )

    # Causes for FN (genes absent but growth observed)
    ax.text(
        7.35,
        y_tr + 0.6,
        "Possible Causes:",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#e65100",
    )
    ax.text(
        5.45,
        y_tr + 0.15,
        "• Annotation: homolog not detected",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        5.45,
        y_tr - 0.2,
        "• Biology: alternative pathway",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        5.45,
        y_tr - 0.55,
        "• Biology: promiscuous enzymes",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        5.45,
        y_tr - 0.9,
        "• Measurement: false positive",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        5.45,
        y_tr - 1.25,
        "• Media: growth on base media",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )

    ax.text(
        7.35,
        y_tr - 2.0,
        "Growth via unknown\nmechanism",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#e65100",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#f57c00", alpha=0.8),
    )

    # Bottom-left quadrant content (False Positive - Discordant: GapMind+, Experiment-)
    y_bl = 2.65
    ax.text(
        2.65,
        y_bl + 2.0,
        "DISCORDANT",
        ha="center",
        va="center",
        fontsize=11 * fs,
        fontweight="bold",
        color="#b71c1c",
    )
    ax.text(
        2.65,
        y_bl + 1.45,
        "False Positive",
        ha="center",
        va="center",
        fontsize=10 * fs,
        fontstyle="italic",
        color="#c62828",
    )

    # Causes for FP (genes present but no growth)
    ax.text(
        2.65,
        y_bl + 0.9,
        "Possible Causes:",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#b71c1c",
    )
    ax.text(
        0.75,
        y_bl + 0.45,
        "• Regulation: pathway repressed",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        0.75,
        y_bl + 0.1,
        "• Measurement: insufficient time",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        0.75,
        y_bl - 0.25,
        "• Media: missing cofactors",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        0.75,
        y_bl - 0.6,
        "• Annotation: false positive call",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )
    ax.text(
        0.75,
        y_bl - 0.95,
        "• Biology: incomplete pathway",
        ha="left",
        va="center",
        fontsize=8 * fs,
        color="#424242",
    )

    ax.text(
        2.65,
        y_bl - 1.7,
        "Genes present but\nnot functional",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d32f2f", alpha=0.8),
    )

    # Bottom-right quadrant content (True Negative - Concordant)
    y_br = 2.65
    ax.text(
        7.35,
        y_br + 2.0,
        "CONCORDANT",
        ha="center",
        va="center",
        fontsize=11 * fs,
        fontweight="bold",
        color="#0d47a1",
    )
    ax.text(
        7.35,
        y_br + 1.45,
        "True Negative",
        ha="center",
        va="center",
        fontsize=10 * fs,
        fontstyle="italic",
        color="#1976d2",
    )
    ax.text(
        7.35,
        y_br + 0.7,
        "Genes absent",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#0d47a1",
    )
    ax.text(
        7.35,
        y_br + 0.3,
        "No alternative pathway",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#0d47a1",
    )
    ax.text(
        7.35,
        y_br - 0.1,
        "Experiment accurately measured",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        color="#0d47a1",
    )
    ax.text(
        7.35,
        y_br - 0.85,
        "High-quality training samples",
        ha="center",
        va="center",
        fontsize=8.5 * fs,
        fontweight="bold",
        color="#0d47a1",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1976d2", alpha=0.8),
    )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 12))
    create_quadrant_plot(ax)
    plt.tight_layout()

    # Save output
    output_file = Path("figures/figure4a.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()
