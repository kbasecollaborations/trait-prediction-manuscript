#!/usr/bin/env python3
"""
Plot Supplementary Figure S15: selective prediction / applicability domain for
concordant-trained models on the full held-out cross-dataset test set.

Two panels:

    (A) Risk-coverage curve -- balanced accuracy on the retained subset as a
        function of coverage, where the least-confident genomes are abstained
        on first. Confidence is the model's own ``max(p, 1 - p)``.
    (B) Balanced accuracy split by GapMind-ML agreement -- when the curated
        mechanistic call and the ML prediction coincide versus disagree. Both
        signals are computable without the experimental outcome of a new genome.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (matplotlib style registration)
import seaborn as sns
from matplotlib.axes import Axes

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

DATA_DIR: Path = Path("data/outputs/figureS15")
RISK_FILE: Path = DATA_DIR / "figureS15_risk_coverage.tsv"
AGREEMENT_FILE: Path = DATA_DIR / "figureS15_agreement.tsv"
OUTPUT_FILE: Path = Path("figures/figure8.pdf")

CURVE_COLOR: str = "#2E86AB"
AGREE_COLOR: str = "#2E86AB"
DISAGREE_COLOR: str = "#E76F51"

AGREEMENT_LABELS: dict[str, str] = {
    "gapmind_ml_agree": "GapMind-ML\nagree",
    "gapmind_ml_disagree": "GapMind-ML\ndisagree",
}


def _panel_label(ax: Axes, label: str) -> None:
    """
    Draw a bold panel label in the upper-left corner of an axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    label : str
        Panel label text, e.g. ``"(A)"``.
    """
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def plot_risk_coverage(ax: Axes, risk: pd.DataFrame) -> None:
    """
    Plot the balanced-accuracy risk-coverage curve.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    risk : pd.DataFrame
        Contents of ``figureS15_risk_coverage.tsv``.
    """
    risk = risk.sort_values("coverage")
    ax.plot(
        risk["coverage"],
        risk["balanced_accuracy"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=CURVE_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.4,
        zorder=3,
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)

    full = risk[np.isclose(risk["coverage"], 1.0)].iloc[0]
    half = risk.iloc[(risk["coverage"] - 0.5).abs().argmin()]
    for point, va in [(full, "top"), (half, "bottom")]:
        ax.annotate(
            f"{point['balanced_accuracy']:.2f}",
            xy=(point["coverage"], point["balanced_accuracy"]),
            xytext=(0, -12 if va == "top" else 10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=CURVE_COLOR,
        )

    ax.set_xlabel("Coverage (fraction of genomes the model commits to)")
    ax.set_ylabel("Balanced accuracy (retained subset)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.45, 1.0)
    ax.text(
        0.98,
        0.52,
        "random (0.5)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    _panel_label(ax, "(A)")


def plot_agreement(ax: Axes, agreement: pd.DataFrame) -> None:
    """
    Plot balanced accuracy split by GapMind-ML agreement.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    agreement : pd.DataFrame
        Contents of ``figureS15_agreement.tsv``.
    """
    indexed = agreement.set_index("subset")
    order = ["gapmind_ml_agree", "gapmind_ml_disagree"]
    colors = [AGREE_COLOR, DISAGREE_COLOR]
    x = np.arange(len(order))

    bars = ax.bar(
        x,
        [indexed.loc[s, "balanced_accuracy"] for s in order],
        width=0.6,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)

    if "all" in indexed.index:
        ax.axhline(
            float(indexed.loc["all", "balanced_accuracy"]),
            linestyle=":",
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.text(
            len(order) - 0.5,
            float(indexed.loc["all", "balanced_accuracy"]) + 0.012,
            "full test set",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    for bar, subset in zip(bars, order, strict=True):
        ba = float(indexed.loc[subset, "balanced_accuracy"])
        coverage = float(indexed.loc[subset, "coverage"])
        n = int(indexed.loc[subset, "n_samples"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ba + 0.015,
            f"BA = {ba:.2f}\n{coverage * 100:.0f}\\% of genomes\n(n = {n})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGREEMENT_LABELS[s] for s in order])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    _panel_label(ax, "(B)")


def create_figure(output_file: Path) -> None:
    """
    Build and persist Supplementary Figure S15.

    Parameters
    ----------
    output_file : Path
        Destination PDF path.
    """
    risk = pd.read_csv(RISK_FILE, sep="\t")
    agreement = pd.read_csv(AGREEMENT_FILE, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_risk_coverage(axes[0], risk)
    plot_agreement(axes[1], agreement)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def main() -> None:
    """Build Supplementary Figure S15."""
    create_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
