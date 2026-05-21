#!/usr/bin/env python3
"""
Generate Supplementary Figure S6: cross-dataset performance heatmap by training
sample size across all 15 shared phenotypes.

Reads the consolidated learning-curve results produced by
``scripts.figureS6.figureS6_data`` and renders the heatmap defined in the
shared ``supplementary_figure_learning_curves`` module.
"""

from scripts.supplementary_figure_learning_curves import (
    ensure_output_dir,
    load_plot_data,
    plot_heatmap,
)


def main() -> None:
    """Generate Supplementary Figure S6."""
    output_dir = ensure_output_dir()
    plot_data = load_plot_data()
    plot_heatmap(plot_data, output_dir / "figure_s6.pdf")


if __name__ == "__main__":
    main()
