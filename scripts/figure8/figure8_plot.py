#!/usr/bin/env python3
"""
Plot Figure 8: diagnostic and applicability-domain toolkit for genome-based
phenotype prediction.

Four panels span two levels of reliability assessment --- per genome and
per phenotype:

    (A) Risk-coverage curve -- balanced accuracy on the retained subset as a
        function of coverage, where the least-confident genomes are abstained
        on first. Confidence is the model's own ``max(p, 1 - p)``.
    (B) Balanced accuracy split by GapMind-ML agreement -- when the curated
        mechanistic call and the ML prediction coincide versus disagree. The
        disagreement subset prioritises experimental follow-up. Both
        per-genome signals are computable without the experimental outcome of
        a new genome.
    (C) Per-phenotype diagnostic for the value of concordance filtering --
        the shortcut gap of the full-data model (random-split balanced
        accuracy minus cross-dataset balanced accuracy) plotted against the
        concordance benefit (concordant cross-dataset BA minus full-data
        cross-dataset BA). A positive relationship indicates that phenotypes
        whose full-data models rely on dataset- or phylogeny-specific
        shortcuts are also those that gain most from concordance filtering,
        providing an a-priori signal that filtering will pay off.
    (D) Retrospective active-learning pilot -- simulated improvement after
        adding 25 selected held-out genome labels for the best- and
        worst-generalising phenotypes from panel C, comparing random selection
        with GapMind-ML-disagreement-guided selection.
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

DATA_DIR: Path = Path("data/outputs/figure8")
RISK_FILE: Path = DATA_DIR / "figure8_risk_coverage.tsv"
AGREEMENT_FILE: Path = DATA_DIR / "figure8_agreement.tsv"
ML_RESULTS_FILE: Path = Path("data/outputs/figure3/ml_results.csv")
FULL_TEST_FILE: Path = Path("data/outputs/figure5/figure5d_full_test.tsv")
ACTIVE_LEARNING_FILE: Path = Path(
    "data/outputs/active_learning_pilot/active_learning_pilot_detailed.tsv"
)
OUTPUT_FILE: Path = Path("figures/figure8.pdf")

# Colour palette: seaborn ``colorblind`` (matches ``visualization.get_dataset_colors``).
# Index 0 (blue ``#0173b2``) is reused across panels as the primary accent so
# Figure 8 sits visually next to Figures 3 and 5; index 3 (vermillion
# ``#d55e00``) is the discordance highlight, mirroring the same role red plays
# in Figure 5D's discordance categories.
_PALETTE = sns.color_palette("colorblind", n_colors=6)
PRIMARY_COLOR: str = "#%02x%02x%02x" % tuple(int(255 * v) for v in _PALETTE[0])
ACCENT_COLOR: str = "#%02x%02x%02x" % tuple(int(255 * v) for v in _PALETTE[3])

CURVE_COLOR: str = PRIMARY_COLOR
AGREE_COLOR: str = PRIMARY_COLOR
DISAGREE_COLOR: str = ACCENT_COLOR
DIAGNOSTIC_COLOR: str = PRIMARY_COLOR
RANDOM_COLOR: str = "0.72"
GUIDED_COLOR: str = ACCENT_COLOR

AGREEMENT_LABELS: dict[str, str] = {
    "gapmind_ml_agree": "GapMind-ML\nagree",
    "gapmind_ml_disagree": "GapMind-ML\ndisagree",
}

# Hand-tuned label offsets (in points) for Panel C phenotypes whose default
# positions overlap or run off-axis. Phenotypes not listed here use the default.
_PANEL_C_LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Galacturonic-Acid": (6, -10),
    "Mannose": (6, 6),
    "Fructose": (-6, -10),
    "Cellobiose": (6, -3),
    "Alanine": (6, -3),
    "Serine": (6, 4),
    "m-Inositol": (6, 4),
    "Maltose": (-58, 4),
    "Glycerol": (-44, 4),
    "Arginine": (6, 4),
    "Histidine": (6, -3),
    "Mannitol": (6, 4),
    "Galactose": (6, -3),
    "Sucrose": (-44, 4),
    "Glucose": (-44, 4),
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


def plot_risk_coverage(ax: Axes, risk: pd.DataFrame) -> None:
    """
    Plot the balanced-accuracy risk-coverage curve.

    A shaded band between the curve and the random-baseline line conveys the
    "value above chance" of every coverage level at a glance.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    risk : pd.DataFrame
        Contents of ``figure8_risk_coverage.tsv``.
    """
    risk = risk.sort_values("coverage")
    coverage = risk["coverage"].to_numpy()
    ba = risk["balanced_accuracy"].to_numpy()

    # Shaded band: curve vs. random baseline (0.5).
    ax.fill_between(
        coverage,
        0.5,
        ba,
        where=ba >= 0.5,
        color=CURVE_COLOR,
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )

    # Random-chance reference.
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7, zorder=2)

    ax.plot(
        coverage,
        ba,
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=CURVE_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.4,
        zorder=3,
    )

    full = risk[np.isclose(risk["coverage"], 1.0)].iloc[0]
    half = risk.iloc[(risk["coverage"] - 0.5).abs().argmin()]
    # Push annotations away from the curve so labels never sit on the marker.
    for point, va, dy in [(full, "top", -14), (half, "bottom", 12)]:
        ax.annotate(
            f"BA = {point['balanced_accuracy']:.2f}",
            xy=(point["coverage"], point["balanced_accuracy"]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=CURVE_COLOR,
        )

    ax.set_xlabel("Coverage (fraction of genomes the model commits to)")
    ax.set_ylabel("Balanced accuracy\n(retained subset)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.45, 1.0)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=2)
    ax.text(
        0.98,
        0.51,
        "random (0.5)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    _panel_label(ax, "(A)", x=-0.14)


def plot_agreement(ax: Axes, agreement: pd.DataFrame) -> None:
    """
    Plot balanced accuracy split by GapMind-ML agreement.

    Each bar carries a compact two-line stat block: the headline balanced
    accuracy on top, and the coverage / sample-size pair underneath.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    agreement : pd.DataFrame
        Contents of ``figure8_agreement.tsv``.
    """
    indexed = agreement.set_index("subset")
    order = ["gapmind_ml_agree", "gapmind_ml_disagree"]
    colors = [AGREE_COLOR, DISAGREE_COLOR]
    x = np.arange(len(order))

    bars = ax.bar(
        x,
        [indexed.loc[s, "balanced_accuracy"] for s in order],
        width=0.55,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
        zorder=2,
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7, zorder=1)

    if "all" in indexed.index:
        full_ba = float(indexed.loc["all", "balanced_accuracy"])
        ax.axhline(
            full_ba,
            linestyle=":",
            color="black",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.text(
            -0.55,
            full_ba - 0.015,
            f"full test set ({full_ba:.2f})",
            ha="left",
            va="top",
            fontsize=8,
            color="black",
        )

    for bar, subset in zip(bars, order, strict=True):
        ba = float(indexed.loc[subset, "balanced_accuracy"])
        coverage = float(indexed.loc[subset, "coverage"])
        n = int(indexed.loc[subset, "n_samples"])
        # Headline BA, larger and bold.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ba + 0.055,
            f"BA = {ba:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        # Coverage / n sub-label, smaller and de-emphasised.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ba + 0.015,
            f"{coverage * 100:.0f}\\% of genomes, $n$ = {n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.25",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGREEMENT_LABELS[s] for s in order])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    _panel_label(ax, "(B)")


def compute_phenotype_diagnostic(
    ml_results_file: Path, full_test_file: Path
) -> pd.DataFrame:
    """
    Build the per-phenotype shortcut-gap vs concordance-benefit table.

    The shortcut gap of the full-data (unfiltered) model is the average
    random-split balanced accuracy minus the average cross-dataset balanced
    accuracy. A large gap means the full-data model relies on dataset- or
    phylogeny-specific patterns that do not transfer. The concordance benefit
    is the average cross-dataset balanced accuracy of the concordant-trained
    model on the full natural-composition held-out test set minus the
    cross-dataset balanced accuracy of the full-data model.

    Parameters
    ----------
    ml_results_file : Path
        Path to the Figure 3 ``ml_results.csv`` (full-data KOFAM model).
    full_test_file : Path
        Path to ``figure5d_full_test.tsv`` (concordant-trained model on the
        full held-out cross-dataset test set).

    Returns
    -------
    pd.DataFrame
        One row per phenotype with columns ``phenotype``, ``random_ba``,
        ``cross_ba_full``, ``cross_ba_concordant``, ``shortcut_gap``,
        ``concordance_benefit``.
    """
    ml = pd.read_csv(ml_results_file)
    ml = ml[ml["split_type"].isin(["random_split", "dataset_split"])]
    pivot = (
        ml.groupby(["phenotype", "split_type"])["balanced_accuracy"]
        .mean()
        .unstack("split_type")
        .rename(
            columns={"random_split": "random_ba", "dataset_split": "cross_ba_full"}
        )
    )

    full_test = pd.read_csv(full_test_file, sep="\t")
    concordant_cross = (
        full_test.groupby("phenotype")["balanced_accuracy_full"]
        .mean()
        .rename("cross_ba_concordant")
    )

    diagnostic = pivot.join(concordant_cross, how="inner").reset_index()
    diagnostic["shortcut_gap"] = (
        diagnostic["random_ba"] - diagnostic["cross_ba_full"]
    )
    diagnostic["concordance_benefit"] = (
        diagnostic["cross_ba_concordant"] - diagnostic["cross_ba_full"]
    )
    return diagnostic


def plot_diagnostic(ax: Axes, diagnostic: pd.DataFrame) -> None:
    """
    Plot the per-phenotype shortcut-gap vs concordance-benefit scatter.

    A faint linear-fit guide visualises the positive monotone trend reported
    in the Spearman annotation. Per-point labels are placed with hand-tuned
    offsets to avoid the dense Fructose / Mannose / Galacturonic-Acid cluster.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    diagnostic : pd.DataFrame
        Output of :func:`compute_phenotype_diagnostic`.
    """
    from scipy.stats import spearmanr

    x = diagnostic["shortcut_gap"].to_numpy()
    y = diagnostic["concordance_benefit"].to_numpy()

    # Faint linear-fit guide for the trend.
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        linestyle="-",
        color=DIAGNOSTIC_COLOR,
        linewidth=1.2,
        alpha=0.35,
        zorder=2,
    )

    # Zero reference lines.
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)

    ax.scatter(
        x,
        y,
        s=64,
        color=DIAGNOSTIC_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    # Thin dark outline for better separation against the fit line.
    ax.scatter(
        x,
        y,
        s=64,
        facecolor="none",
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    for _, row in diagnostic.iterrows():
        offset = _PANEL_C_LABEL_OFFSETS.get(row["phenotype"], (5, 4))
        ax.annotate(
            row["phenotype"],
            xy=(row["shortcut_gap"], row["concordance_benefit"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            zorder=6,
        )

    rho, p_value = spearmanr(x, y)
    ax.text(
        0.98,
        0.05,
        (
            f"Spearman $\\rho$ = {rho:.2f}\n"
            f"$p$ = {p_value:.2g}\n"
            f"$n$ = {len(diagnostic)}"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    ax.set_xlabel(
        "Full-data shortcut gap (random $-$ cross-dataset BA)"
    )
    ax.set_ylabel(
        "Concordance benefit\n(concordant $-$ full-data cross-dataset BA)"
    )
    # Pad axes so labels don't run off the panel.
    ax.set_xlim(x.min() - 0.03, x.max() + 0.04)
    ax.set_ylim(y.min() - 0.03, y.max() + 0.04)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=2)
    _panel_label(ax, "(C)")


def plot_active_learning(ax: Axes, active_learning: pd.DataFrame) -> None:
    """
    Plot the retrospective active-learning pilot for the best/worst phenotypes.

    The panel compares random selection with GapMind-ML-disagreement-guided
    selection. Thin paired lines connect runs sharing phenotype, held-out
    dataset, and seed, so the effect is read as a paired change rather than as
    independent bars.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    active_learning : pd.DataFrame
        Detailed output from ``scripts.figure8.figure8d_active_learning``.
    """
    strategies = ["random", "disagreement"]
    strategy_labels = {"random": "Random", "disagreement": "Guided"}
    colors = {"random": RANDOM_COLOR, "disagreement": GUIDED_COLOR}
    phenotype_order = ["m-Inositol", "Glucose"]
    phenotype_labels = {
        "m-Inositol": "m-Inositol\n(best transfer)",
        "Glucose": "Glucose\n(worst transfer)",
    }
    offsets = {"random": -0.18, "disagreement": 0.18}
    bar_width = 0.32

    data = active_learning[
        active_learning["phenotype"].isin(phenotype_order)
        & active_learning["strategy"].isin(strategies)
    ].copy()
    data["run_id"] = (
        data["phenotype"].astype(str)
        + "|"
        + data["held_out_dataset"].astype(str)
        + "|"
        + data["seed"].astype(str)
    )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    rng = np.random.default_rng(42)
    for x_base, phenotype in enumerate(phenotype_order):
        phenotype_data = data[data["phenotype"] == phenotype]
        paired = phenotype_data.pivot_table(
            index="run_id",
            columns="strategy",
            values="delta_balanced_accuracy",
        ).dropna(subset=strategies)
        for _, row in paired.iterrows():
            ax.plot(
                [x_base + offsets["random"], x_base + offsets["disagreement"]],
                [row["random"], row["disagreement"]],
                color="0.80",
                linewidth=0.7,
                alpha=0.8,
                zorder=2,
            )

        for strategy in strategies:
            values = phenotype_data.loc[
                phenotype_data["strategy"] == strategy,
                "delta_balanced_accuracy",
            ].to_numpy()
            x = x_base + offsets[strategy]
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            ax.bar(
                x,
                mean,
                width=bar_width,
                color=colors[strategy],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
                label=strategy_labels[strategy] if x_base == 0 else None,
            )
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                color="black",
                linewidth=0.8,
                capsize=3,
                zorder=4,
            )
            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=24,
                color=colors[strategy],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
                zorder=5,
            )

        random_mean = paired["random"].mean()
        guided_mean = paired["disagreement"].mean()
        paired_delta = guided_mean - random_mean
        if abs(paired_delta) < 0.005:
            paired_delta = 0.0
        ax.text(
            x_base,
            max(random_mean, guided_mean) + 0.045,
            f"$\\Delta$ = {paired_delta:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=GUIDED_COLOR,
        )

    ax.set_xticks(np.arange(len(phenotype_order)))
    ax.set_xticklabels([phenotype_labels[p] for p in phenotype_order])
    ax.set_ylabel("$\\Delta$ balanced accuracy after 25 labels")
    ax.set_xlabel("Phenotypes selected from panel C")
    ax.set_ylim(-0.08, 0.45)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", which="minor", length=2)
    _panel_label(ax, "(D)")


def create_figure(output_file: Path) -> None:
    """
    Build and persist Figure 8.

    Parameters
    ----------
    output_file : Path
        Destination PDF path.
    """
    risk = pd.read_csv(RISK_FILE, sep="\t")
    agreement = pd.read_csv(AGREEMENT_FILE, sep="\t")
    diagnostic = compute_phenotype_diagnostic(ML_RESULTS_FILE, FULL_TEST_FILE)
    active_learning = pd.read_csv(ACTIVE_LEARNING_FILE, sep="\t")

    fig = plt.figure(figsize=(12, 10.2))
    # Row 1 (A, B) gets slightly less height than row 2 (C), so the wider
    # scatter has room to breathe. Row 3 (D) shows the paired active-learning
    # pilot and uses the full width for readable run-level overlays.
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.95, 1.1, 0.85],
        hspace=0.30,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])
    plot_risk_coverage(ax_a, risk)
    plot_agreement(ax_b, agreement)
    plot_diagnostic(ax_c, diagnostic)
    plot_active_learning(ax_d, active_learning)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def main() -> None:
    """Build Figure 8."""
    create_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
