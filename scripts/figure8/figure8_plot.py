#!/usr/bin/env python3
"""
Plot Figure 8: diagnostic and applicability-domain toolkit for genome-based
phenotype prediction.

Four panels span two levels of reliability assessment --- per genome and
per phenotype:

    (A) Risk-coverage curves for three phenotypes spanning strong, medium,
        and weak cross-dataset generalization (m-Inositol, Histidine,
        Glucose). Each mini-plot compares the concordant-trained model with
        the full-data model: balanced accuracy on the retained subset as the
        least-confident genomes are abstained on first.
    (B) Reliability diagram: mean predicted probability of growth against the
        empirically observed growth fraction within each confidence bin, per
        model. The expected calibration error (ECE) is reported in the legend.
    (C) Label-free prioritization: the gain in cross-dataset balanced accuracy
        after adding selected held-out genome labels, one bar per selection
        strategy (low confidence, diversity, random, high novelty) at the
        labelling budget.
    (D) Per-phenotype gain from randomly versus selectively (low-confidence)
        added labels, for the three Panel-A archetypes, showing that selective
        acquisition helps most on the weak generaliser.
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

DATA_DIR = Path("data/outputs/figure8")
RISK_FILE = DATA_DIR / "figure8_risk_coverage_by_phenotype.tsv"
CALIB_FILE = DATA_DIR / "figure8_calibration.tsv"
PRIOR_FILE = DATA_DIR / "figure8_prioritization.tsv"
OUTPUT_FILE = Path("figures/figure8.pdf")

# Colour palette: seaborn ``colorblind`` (matches ``visualization.get_dataset_colors``).
# Index 0 (blue ``#0173b2``) is reused across panels as the primary accent so
# Figure 8 sits visually next to Figures 3 and 5; index 3 (vermillion
# ``#d55e00``) is the secondary accent.
_PALETTE = sns.color_palette("colorblind", n_colors=6)
PRIMARY_COLOR: str = "#%02x%02x%02x" % tuple(int(255 * v) for v in _PALETTE[0])
ACCENT_COLOR: str = "#%02x%02x%02x" % tuple(int(255 * v) for v in _PALETTE[3])

PANEL_A_PHENOTYPES = ["m-Inositol", "Histidine", "Glucose"]
MODEL_LABELS = {"concordant": "Concordant", "full_data": "Full-data"}
MODEL_COLORS = {"concordant": PRIMARY_COLOR, "full_data": ACCENT_COLOR}
STRATEGY_LABELS = {
    "low_confidence": "Low confidence",
    "high_ood": "High novelty",
    "diversity": "Diversity",
    "random": "Random",
}


def _panel_label(ax: Axes, label: str, x: float = -0.08) -> None:
    """
    Draw a bold panel label in the upper-left corner of an axes.

    Position and size match the convention used in Figures 3 and 5
    (``(-0.08, 1.05)`` in axes coordinates, ``fontsize=14``, bold).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    label : str
        Panel label text, e.g. ``"(A)"``.
    x : float, optional
        Horizontal position in axes coordinates.
    """
    ax.text(
        x,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        fontsize=14,
    )


def plot_risk_coverage(axes: list[Axes], risk: pd.DataFrame) -> None:
    """
    Plot three mini risk-coverage plots, one per Panel-A phenotype.

    Within each phenotype the concordant-trained and full-data models are
    compared, showing balanced accuracy on the retained subset as coverage
    rises (least-confident genomes abstained on first).

    Parameters
    ----------
    axes : list[Axes]
        Three matplotlib axes, one per phenotype in
        :data:`PANEL_A_PHENOTYPES`.
    risk : pd.DataFrame
        Contents of ``figure8_risk_coverage_by_phenotype.tsv``.
    """
    for ax, phen in zip(axes, PANEL_A_PHENOTYPES, strict=True):
        ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.7, zorder=1)
        for model in ("concordant", "full_data"):
            sub = risk[(risk.phenotype == phen) & (risk.model == model)].sort_values(
                "coverage"
            )
            ax.plot(
                sub.coverage,
                sub.balanced_accuracy,
                marker="o",
                ms=3.5,
                lw=1.5,
                color=MODEL_COLORS[model],
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=MODEL_LABELS[model],
                zorder=3,
            )
        ax.set_title(phen, fontsize=12)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.40, 1.0)
        ax.set_xlabel("Coverage")
    axes[0].set_ylabel("Balanced accuracy\n(retained subset)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")


def plot_calibration(ax: Axes, calib: pd.DataFrame) -> None:
    """
    Plot a reliability diagram of predicted confidence vs empirical accuracy.

    One curve per model is drawn, with the model's expected calibration error
    (ECE) reported in the legend. The dashed diagonal marks perfect
    calibration.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    calib : pd.DataFrame
        Contents of ``figure8_calibration.tsv``.
    """
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, alpha=0.7, label="Perfect")
    for model in ("concordant", "full_data"):
        sub = calib[calib.model == model].sort_values("mean_pred")
        ece = float(sub["ece"].iloc[0])
        ax.plot(
            sub.mean_pred,
            sub.frac_pos,
            marker="o",
            ms=4,
            lw=1.5,
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} (ECE = {ece:.2f})",
            zorder=3,
        )
    ax.set_xlabel("Mean predicted P(growth)")
    ax.set_ylabel("Observed growth fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _panel_label(ax, "(B)", x=-0.22)


def plot_prioritization(ax: Axes, prior: pd.DataFrame) -> None:
    """
    Plot balanced-accuracy gain per label-free selection strategy at the budget.

    One bar per strategy shows the mean cross-dataset balanced-accuracy gain after
    adding the selected held-out labels, with standard-error-of-the-mean error bars
    and jittered per-run points. The strategies are ordered by mean gain.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    prior : pd.DataFrame
        Contents of ``figure8_prioritization.tsv``.
    """
    palette = sns.color_palette("colorblind", n_colors=6)

    def _hex(i: int) -> str:
        return "#%02x%02x%02x" % tuple(int(255 * v) for v in palette[i])

    colors = {
        "low_confidence": PRIMARY_COLOR,
        "high_ood": _hex(2),
        "diversity": _hex(4),
        "random": "0.6",
    }
    budget = int(prior["n_added"].max())
    final = prior[prior["n_added"] == budget]
    order = ["low_confidence", "diversity", "random", "high_ood"]
    rng = np.random.default_rng(42)

    ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
    for x, strat in enumerate(order):
        vals = final.loc[final.strategy == strat, "delta_balanced_accuracy"].to_numpy()
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(
            x, mean, width=0.62, color=colors[strat], edgecolor="black",
            linewidth=0.7, alpha=0.9, zorder=2,
        )
        ax.errorbar(x, mean, yerr=sem, color="black", lw=0.8, capsize=3, zorder=4)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter, vals, s=10, color="black",
            alpha=0.35, linewidth=0, zorder=3,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in order], rotation=15, ha="right")
    ax.set_ylabel("$\\Delta$ cross-dataset\nbalanced accuracy")
    ax.set_xlabel(f"Selection strategy ({budget} labels added)")
    _panel_label(ax, "(C)")


def plot_phenotype_priority(ax: Axes, prior: pd.DataFrame) -> None:
    """
    Plot per-phenotype gain from randomly vs selectively added labels.

    Grouped bars compare the cross-dataset balanced-accuracy gain from adding 25
    randomly chosen labels against 25 low-confidence-selected labels, for the
    three Panel-A archetype phenotypes (strong to weak generaliser). Error bars
    are the standard error of the mean across runs.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    prior : pd.DataFrame
        Contents of ``figure8_prioritization.tsv`` (per-run rows).
    """
    budget = int(prior["n_added"].max())
    final = prior[prior["n_added"] == budget]
    strategies = [("random", "Random", "0.6"),
                  ("low_confidence", "Low-confidence", PRIMARY_COLOR)]
    width = 0.38
    x = np.arange(len(PANEL_A_PHENOTYPES))

    ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
    for i, (strat, label, color) in enumerate(strategies):
        means, sems = [], []
        for phen in PANEL_A_PHENOTYPES:
            vals = final.loc[
                (final.phenotype == phen) & (final.strategy == strat),
                "delta_balanced_accuracy",
            ].to_numpy()
            means.append(float(np.mean(vals)))
            sems.append(
                float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, means, width=width, color=color, edgecolor="black",
            linewidth=0.7, alpha=0.9, label=label, zorder=2,
        )
        ax.errorbar(
            x + offset, means, yerr=sems, fmt="none", color="black",
            lw=0.8, capsize=3, zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_A_PHENOTYPES)
    ax.set_ylabel(f"$\\Delta$ balanced accuracy\nfrom {budget} added labels")
    ax.set_xlabel("Phenotype (strong $\\rightarrow$ weak generaliser)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _panel_label(ax, "(D)")


def create_figure(output_file: Path) -> None:
    """
    Build and persist the redesigned Figure 8.

    Parameters
    ----------
    output_file : Path
        Destination PDF path.
    """
    risk = pd.read_csv(RISK_FILE, sep="\t")
    calib = pd.read_csv(CALIB_FILE, sep="\t")
    prior = pd.read_csv(PRIOR_FILE, sep="\t")

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.42, wspace=0.32)
    ax_a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1:])
    ax_d = fig.add_subplot(gs[2, :])
    plot_risk_coverage(ax_a, risk)
    _panel_label(ax_a[0], "(A)", x=-0.22)
    plot_calibration(ax_b, calib)
    plot_prioritization(ax_c, prior)
    plot_phenotype_priority(ax_d, prior)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def main() -> None:
    """Build Figure 8."""
    create_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
