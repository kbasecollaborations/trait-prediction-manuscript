#!/usr/bin/env python3
"""
Generate Supplementary Figure S10.
"""

from scripts.supplementary_figure_learning_curves import (
    ensure_output_dir,
    load_plot_data,
    plot_saturation_summary,
)


def main() -> None:
    """Generate Supplementary Figure S10."""
    output_dir = ensure_output_dir()
    plot_data = load_plot_data()
    plot_saturation_summary(plot_data, output_dir / "figure_s10.pdf")


if __name__ == "__main__":
    main()
