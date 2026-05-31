#!/usr/bin/env python3
"""Plot Figure 7: risk-coverage, calibration, and label-acquisition diagnostics."""

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

DATA_DIR = Path("data/outputs/figure7")
RISK_FILE = DATA_DIR / "figure7_risk_coverage_by_phenotype.tsv"
CALIB_FILE = DATA_DIR / "figure7_calibration.tsv"
PRIOR_FILE = DATA_DIR / "figure7_prioritization.tsv"
OUTPUT_FILE = Path("figures/figure7.pdf")

# Colour palette: seaborn ``colorblind`` (matches ``visualization.get_dataset_colors``).
# Index 0 (blue) is the primary accent; index 3 (vermillion) the secondary.
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

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    label : str
        Panel label text, e.g. ``"A"``.
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

    Parameters
    ----------
    axes : list[Axes]
        Three matplotlib axes, one per phenotype in
        :data:`PANEL_A_PHENOTYPES`.
    risk : pd.DataFrame
        Contents of ``figure7_risk_coverage_by_phenotype.tsv``.
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
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        fontsize=9,
    )


def plot_calibration(ax: Axes, calib: pd.DataFrame) -> None:
    """
    Plot a reliability diagram of predicted confidence vs empirical accuracy.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    calib : pd.DataFrame
        Contents of ``figure7_calibration.tsv``.
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
    _panel_label(ax, "B", x=-0.22)


def plot_prioritization(ax: Axes, prior: pd.DataFrame) -> None:
    """
    Plot balanced-accuracy gain per label-free selection strategy as violins.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    prior : pd.DataFrame
        Contents of ``figure7_prioritization.tsv``.
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
    data_by_strategy = [
        final.loc[final.strategy == s, "delta_balanced_accuracy"].to_numpy()
        for s in order
    ]
    parts = ax.violinplot(
        data_by_strategy,
        positions=range(len(order)),
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[order[i]])
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(0.65)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        bar = parts.get(key)
        if bar is not None:
            bar.set_color("black")
            bar.set_linewidth(0.8)
    for x, vals in enumerate(data_by_strategy):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter, vals, s=10, color="black",
            alpha=0.45, linewidth=0, zorder=4,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in order], rotation=15, ha="right")
    ax.set_ylabel("$\\Delta$ cross-dataset\nbalanced accuracy")
    ax.set_xlabel(f"Selection strategy ({budget} labels added)")
    _panel_label(ax, "C")


def plot_phenotype_priority(ax: Axes, prior: pd.DataFrame) -> None:
    """
    Plot per-phenotype gain from randomly vs selectively added labels (paired).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    prior : pd.DataFrame
        Contents of ``figure7_prioritization.tsv`` (per-run rows).
    """
    budget = int(prior["n_added"].max())
    final = prior[prior["n_added"] == budget]
    strategies: list[tuple[str, str, str]] = [
        ("random", "Random", "0.6"),
        ("low_confidence", "Low-confidence", PRIMARY_COLOR),
    ]
    box_width = 0.32
    rng = np.random.default_rng(42)

    ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
    for p_idx, phen in enumerate(PANEL_A_PHENOTYPES):
        paired = (
            final[final.phenotype == phen]
            .pivot_table(
                index=["held_out_dataset", "seed"],
                columns="strategy",
                values="delta_balanced_accuracy",
            )
            .dropna(subset=[s[0] for s in strategies])
        )
        if paired.empty:
            continue

        # Two box positions flanking the phenotype centre.
        positions = [p_idx - box_width / 1.6, p_idx + box_width / 1.6]
        jitters = [rng.uniform(-0.05, 0.05, size=len(paired)) for _ in strategies]

        # Thin connecting lines first so boxes/points sit on top.
        for run_idx in range(len(paired)):
            xs = [positions[i] + jitters[i][run_idx] for i in range(2)]
            ys = [paired.iloc[run_idx][s[0]] for s in strategies]
            ax.plot(xs, ys, color="gray", alpha=0.45, linewidth=0.6, zorder=2)

        # Boxes and overlaid points per strategy.
        for i, (strat, _label, color) in enumerate(strategies):
            vals = paired[strat].to_numpy()
            ax.boxplot(
                vals,
                positions=[positions[i]],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.7, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.0),
                whiskerprops=dict(color="black", linewidth=0.7),
                capprops=dict(color="black", linewidth=0.7),
                showfliers=False,
                zorder=3,
            )
            ax.scatter(
                np.full(len(vals), positions[i]) + jitters[i],
                vals,
                s=14, color=color, edgecolors="black", linewidth=0.4, zorder=4,
            )

    ax.set_xticks(range(len(PANEL_A_PHENOTYPES)))
    ax.set_xticklabels(PANEL_A_PHENOTYPES)
    ax.set_xlim(-0.6, len(PANEL_A_PHENOTYPES) - 0.4)
    ax.set_ylabel(f"$\\Delta$ balanced accuracy\nfrom {budget} added labels")
    ax.set_xlabel("Phenotype (strong $\\rightarrow$ weak generaliser)")

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.7,
            alpha=0.7, label=label,
        )
        for _strat, label, color in strategies
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper left")
    _panel_label(ax, "D")


def create_figure(output_file: Path) -> None:
    """
    Build and persist Figure 7.

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
    _panel_label(ax_a[0], "A", x=-0.22)
    plot_calibration(ax_b, calib)
    plot_prioritization(ax_c, prior)
    plot_phenotype_priority(ax_d, prior)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def main() -> None:
    """Build Figure 7."""
    create_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
