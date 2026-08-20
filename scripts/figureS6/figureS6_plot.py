#!/usr/bin/env python3
"""Generate Supplementary Figure S6: per-phenotype training-set-size curves.

Replaces the histidine-only grid with all 15 shared phenotypes, comparing full
against concordant training on both the full and the concordant test subset,
under cross-dataset and random-holdout evaluation.

Reads the consolidated KOFAM learning curves written by
``scripts.figureS6.figureS6_data`` and writes
``figures/figure_s6.pdf``.

Run:
    uv run python -m scripts.figureS6.figureS6_plot
"""

from pathlib import Path

import pandas as pd

from scripts.figureS6.figureS6_grid import (
    GRID_BLOCKS,
    TEST_SUBSET_STYLES,
    build_training_size_grid,
    ensure_output_dir,
    load_plot_data,
    plot_training_size_grid,
)
from scripts.create_data_splits import COMMON_PHENOTYPES

FIGURE_NAME = "figure_s6.pdf"


def report_coverage(df: pd.DataFrame) -> None:
    """Print per-block phenotype and series coverage.

    The minority-class-in-test filter legitimately removes phenotype--split
    cells, so coverage is reported rather than asserted. Silent gaps in the
    figure are the failure this guards against.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results from :func:`load_plot_data`.
    """
    means = build_training_size_grid(df)
    expected_series = 2 * len(TEST_SUBSET_STYLES)

    for _, split_label, block_title in GRID_BLOCKS:
        block = means[means["split_type"] == split_label]
        if block.empty:
            print(f"  {block_title}: ABSENT")
            continue
        present = set(block["phenotype"])
        missing = [p for p in COMMON_PHENOTYPES if p not in present]
        counts = block.groupby("phenotype")["test_subset"].size()
        print(
            f"  {block_title}: {len(present)}/{len(COMMON_PHENOTYPES)} phenotypes, "
            f"{len(block)} series-points"
        )
        if missing:
            print(f"    dropped entirely: {', '.join(missing)}")
        thin = counts[counts < expected_series]
        if len(thin):
            print(f"    partial series ({len(thin)}): {', '.join(thin.index)}")


def _self_check() -> None:
    """Verify the grid aggregation on a synthetic frame.

    Raises
    ------
    AssertionError
        If aggregation drops series, mislabels them, or miscomputes the mean.
    """
    rows = [
        {
            "split_type": "Dataset Split",
            "phenotype": "Glucose",
            "training_type": training_type,
            "test_subset": test_subset,
            "sample_size": "25",
            "balanced_accuracy": value,
        }
        for training_type, base in (("Full", 0.6), ("Concordant", 0.8))
        for test_subset, bump in (("Full Test", 0.0), ("Concordant Test", 0.1))
        # Two repeats per series; the mean must be the midpoint.
        for value in (base + bump - 0.05, base + bump + 0.05)
    ]
    # Discordant rows must be excluded from the grid entirely.
    rows.append(
        {
            "split_type": "Dataset Split",
            "phenotype": "Glucose",
            "training_type": "Full",
            "test_subset": "Discordant Test",
            "sample_size": "25",
            "balanced_accuracy": 0.1,
        }
    )
    out = build_training_size_grid(pd.DataFrame(rows))

    assert len(out) == 4, f"expected 4 series, got {len(out)}"
    assert "Discordant Test" not in set(out["test_subset"]), "discordant leaked in"
    assert (out["n_measurements"] == 2).all(), "repeats not aggregated"
    lookup = {
        (r.training_type, r.test_subset): round(r.mean, 3) for r in out.itertuples()
    }
    assert lookup[("Full", "Full Test")] == 0.6, lookup
    assert lookup[("Full", "Concordant Test")] == 0.7, lookup
    assert lookup[("Concordant", "Full Test")] == 0.8, lookup
    assert lookup[("Concordant", "Concordant Test")] == 0.9, lookup


def main() -> None:
    """Generate Supplementary Figure S6."""
    _self_check()
    output_dir = ensure_output_dir()
    plot_data = load_plot_data()
    print("Coverage:")
    report_coverage(plot_data)
    plot_training_size_grid(plot_data, Path(output_dir) / FIGURE_NAME)


if __name__ == "__main__":
    main()
