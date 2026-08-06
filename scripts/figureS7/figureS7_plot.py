#!/usr/bin/env python3
"""C vs F vs FB cross-dataset: accuracy gain versus feature-stability cost.

Composite figure for Supplementary Text S11. Strategies (leave-one-dataset-out,
tested on each held-out manuscript dataset's full labels):
    C   concordant samples of the other 3 manuscript datasets
    F   full other 3 manuscript datasets (no filter)
    FB  full other 3 plus all BacDive

Panels: A cross-dataset balanced accuracy, B cross-seed feature stability
(top-10 importance Jaccard), C the accuracy-versus-stability trade-off.

Reads ``data/outputs/bacdive/head_to_head.csv`` and ``strategy_stability.csv``,
written by the BacDive analysis subsystem deposited with the manuscript.

Run:
    uv run python -m scripts.figureS7.figureS7_plot
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from scipy.stats import wilcoxon

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

HEAD_TO_HEAD = Path("data/outputs/bacdive/head_to_head.csv")
STRATEGY_STABILITY = Path("data/outputs/bacdive/strategy_stability.csv")

ORDER = ["C", "F", "FB"]
_PAL = sns.color_palette("colorblind", n_colors=4)
COLORS = {"C": _PAL[0], "F": (0.55, 0.55, 0.55), "FB": _PAL[1]}
LABELS = {
    "C": "Concordant\n(curated)",
    "F": "Full\nmanuscript",
    "FB": "Full +\nBacDive",
}


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    """Place a bold panel letter at the top-left of an axis."""
    ax.text(
        -0.14,
        1.08,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=14,
        va="top",
        ha="right",
    )


def _accuracy_cells() -> pd.DataFrame:
    """Per-cell (phenotype x held-out) mean balanced accuracy for C/F/FB.

    Returns
    -------
    pandas.DataFrame
        Indexed by (phenotype, held_out) with one column per strategy.
    """
    df = pd.read_csv(HEAD_TO_HEAD)
    df = df[
        (df["status"] == "ok")
        & (df["analysis"] == "strategy")
        & (df["strategy"].isin(ORDER))
    ]
    return (
        df.groupby(["phenotype", "held_out", "strategy"])["balanced_accuracy"]
        .mean()
        .unstack("strategy")[ORDER]
    )


def _stability_cells() -> pd.DataFrame:
    """Per-cell cross-seed top-feature Jaccard for C/F/FB.

    For each (phenotype, held_out, strategy), the mean pairwise Jaccard of the
    top-feature sets across the five random seeds.

    Returns
    -------
    pandas.DataFrame
        Indexed by (phenotype, held_out) with one column per strategy.
    """
    df = pd.read_csv(STRATEGY_STABILITY)
    df = df[df["status"] == "ok"].copy()
    df["feats"] = (
        df["top_features"].fillna("").apply(lambda s: set(s.split(";")) if s else set())
    )

    def _mean_jaccard(sets: list[set[str]]) -> float:
        pairs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
        return float(np.mean(pairs)) if pairs else np.nan

    rows = []
    for (ph, ho, st), g in df.groupby(["phenotype", "held_out", "strategy"]):
        rows.append((ph, ho, st, _mean_jaccard(list(g["feats"]))))
    cells = pd.DataFrame(rows, columns=["phenotype", "held_out", "strategy", "jac"])
    return cells.pivot_table(
        index=["phenotype", "held_out"], columns="strategy", values="jac"
    )[ORDER]


def _stars(p: float) -> str:
    """Significance label for a p-value."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _bracket(
    ax: plt.Axes, x1: float, x2: float, y: float, label: str, dy: float = 0.012
) -> None:
    """Draw a significance bracket between two bars."""
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + dy, label, ha="center", va="bottom", fontsize=9)


def _paired_p(cells: pd.DataFrame, a: str, b: str) -> float:
    """Paired Wilcoxon p-value between two strategy columns over shared cells."""
    s = cells[[a, b]].dropna()
    return float(wilcoxon(s[a], s[b]).pvalue)


def plot_figure(output_file: Path) -> None:
    """Render and save the three-panel composite figure."""
    acc = _accuracy_cells()
    stab = _stability_cells()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    ax_a, ax_b, ax_c = axes
    x = np.arange(len(ORDER))
    colors = [COLORS[s] for s in ORDER]

    means_a = acc[ORDER].mean()
    sems_a = acc[ORDER].sem()
    ax_a.bar(
        x,
        means_a,
        yerr=sems_a,
        color=colors,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2},
    )
    ax_a.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.6)
    p_cf_a, p_ffb_a, p_cfb_a = (
        _paired_p(acc, "C", "F"),
        _paired_p(acc, "F", "FB"),
        _paired_p(acc, "C", "FB"),
    )
    top = float(means_a.max() + sems_a.max())
    _bracket(ax_a, 1, 2, top + 0.012, _stars(p_ffb_a))
    _bracket(ax_a, 0, 2, top + 0.052, f"{_stars(p_cfb_a)} (p={p_cfb_a:.2f})")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LABELS[s] for s in ORDER])
    ax_a.set_ylabel("Cross-dataset balanced accuracy")
    ax_a.set_ylim(0, 0.9)
    _panel_letter(ax_a, "A")

    means_b = stab[ORDER].mean()
    sems_b = stab[ORDER].sem()
    ax_b.bar(
        x,
        means_b,
        yerr=sems_b,
        color=colors,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.6,
    )
    for i, s in enumerate(ORDER):
        vals = stab[s].dropna().to_numpy()
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
        ax_b.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=8,
            color="black",
            alpha=0.25,
            zorder=3,
        )
    p_cfb_b = _paired_p(stab, "C", "FB")
    _bracket(ax_b, 0, 2, 0.86, _stars(p_cfb_b), dy=0.025)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([LABELS[s] for s in ORDER])
    ax_b.set_ylabel("Cross-seed feature stability\n(top-10 Jaccard)")
    ax_b.set_ylim(0, 0.98)
    _panel_letter(ax_b, "B")

    for s in ORDER:
        ax_c.errorbar(
            stab[s].mean(),
            acc[s].mean(),
            xerr=stab[s].sem(),
            yerr=acc[s].sem(),
            marker="o",
            ms=11,
            color=COLORS[s],
            capsize=3,
            elinewidth=1.2,
            mec="black",
            mew=0.5,
            lw=0,
            zorder=4,
        )
    ax_c.plot(
        [stab[s].mean() for s in ORDER],
        [acc[s].mean() for s in ORDER],
        color="black",
        lw=0.8,
        ls=":",
        alpha=0.6,
        zorder=0,
    )
    point_labels = {
        "C": ("Concordant\n(curated)", 0.0, 0.022, "center", "bottom"),
        "F": ("Full\nmanuscript", 0.0, -0.022, "center", "top"),
        "FB": ("Full +\nBacDive", 0.0, -0.024, "center", "top"),
    }
    for s, (txt, dx, dy, ha, va) in point_labels.items():
        ax_c.text(
            stab[s].mean() + dx,
            acc[s].mean() + dy,
            txt,
            fontsize=8,
            ha=ha,
            va=va,
            color=COLORS[s],
            fontweight="bold",
        )
    ax_c.set_xlabel("Feature stability (top-10 Jaccard)")
    ax_c.set_ylabel("Cross-dataset balanced accuracy")
    ax_c.set_xlim(0.22, 0.42)
    ax_c.set_ylim(0.645, 0.755)
    _panel_letter(ax_c, "C")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    print(
        f"  Panel A p-values: C-F={p_cf_a:.3f}  F-FB={p_ffb_a:.4f}  C-FB={p_cfb_a:.3f}"
    )
    print(f"  Panel B p-value:  C-FB={p_cfb_b:.2e}")
    plt.close(fig)


if __name__ == "__main__":
    plot_figure(Path("figures/figure_s7.pdf"))
