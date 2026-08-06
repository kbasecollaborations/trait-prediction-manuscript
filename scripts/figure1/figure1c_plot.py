#!/usr/bin/env python3
"""Figure 1C: levers for cross-dataset generalisation.

Five analysis variants on a shared balanced-accuracy ladder, each with method
columns (Training data, Features, Evaluation) and a one-line Finding. Capsules
span the per-phenotype interquartile range on a 0.40-1.00 axis with a dashed
chance line at 0.50, a median tick, and the per-phenotype values as a strip.

Rows 1-2 come from data/outputs/figure3/ml_results.csv (KOFAM), the feature rows
from data/outputs/figure6/figure6d_*_results.csv, and concordance from
data/outputs/figure5/figure5d_full_test.tsv (balanced_accuracy_full). The four
cross-dataset variants use the full held-out test and apply the
<10-minority-test exclusion; random holdout is the in-distribution split.
Sized for inclusion at \\textwidth.

Run with:
    uv run python -m scripts.figure1.figure1c_plot
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from scripts.minority_filter import filter_by_minority, full_test_minority_counts

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "nature"])
except Exception:
    pass
plt.rcParams.update({"text.usetex": False, "svg.fonttype": "none"})

PDF_OUT = Path("figures/figure1c.pdf")
PNG_OUT = Path("data/outputs/figure1c_options/figure1c_final.png")

GREY = "#8A8A8A"
AMBER = "#D9A441"
GREEN = "#1B7837"
STEEL = "#4E79A7"
LGREEN = "#5FA874"
KEYBAND = "#E6F0E9"
INK = "#1A1A1A"
# Capsules share one neutral colour; the verdict is carried by the accent colour
# of the row label and finding.
CAP_COLOR = "#6E8299"
DOT_COLOR = "#3F5A78"  # per-phenotype strip
ORANGE = "#C0561E"

BA_LO, BA_HI = 0.40, 1.00
GX = 0.075  # global x-shift centring the table within the full-width canvas
X0, X1 = 0.455 + GX, 0.645 + GX  # BA axis maps into this figure-fraction x-range

FS_PANEL = 12
FS_HEAD = 7.6
FS_LABEL = 8.0
FS_COL = 6.5
FS_TICK = 6.5
FS_VAL = 7.0
FS_MED = 6.3
FS_BADGE = 10.5
FS_RATIO = 6.0
FS_FIND = 6.7
FS_NOTE = 6.2

X_LABEL = 0.004 + GX
X_TRAIN = 0.150 + GX
X_FEAT = 0.252 + GX
X_EVAL = 0.358 + GX
X_FIND = 0.680 + GX

CAP_H = 0.045
# Median label above the capsule and IQR label below; loose offsets let one row's
# lower label crowd the next row's upper label.
VAL_DY = 0.046
MED_DY = 0.042

# Table rules and BA-axis positions (figure fraction). RULE_R stops just past the
# Finding text so the table has no empty ruled column on the right.
RULE_L, RULE_R = 0.0 + GX, 0.850 + GX
TOP_RULE = 0.925
MID_RULE = 0.855
BOT_RULE = 0.020
BA_AXIS_Y = 0.110  # horizontal BA x-axis baseline
GRID_TOP = 0.845

YS = [0.775, 0.633, 0.490, 0.347, 0.205]


def bax(ba: float) -> float:
    """Map a balanced-accuracy value to the panel x-coordinate."""
    return X0 + (ba - BA_LO) / (BA_HI - BA_LO) * (X1 - X0)


OUTPUTS = Path("data/outputs")


def _rung(
    frame: pd.DataFrame, test_dataset_column: str | None = None
) -> dict[str, float | str]:
    """Summarise one ladder rung as a per-phenotype balanced-accuracy distribution.

    The <10-minority-test exclusion is applied to every rung.

    Parameters
    ----------
    frame
        Result rows carrying ``phenotype`` and ``balanced_accuracy`` columns.
    test_dataset_column
        Column naming the held-out dataset. When ``None`` it is parsed from the
        ``key`` column.

    Returns
    -------
    dict[str, float | str]
        ``q1``, ``q3``, ``median``, the per-phenotype ``values`` for the strip,
        and the pre-rounded ``txt`` and ``med_txt`` labels.

    Raises
    ------
    ValueError
        If the frame yields no phenotypes.
    """
    frame = filter_by_minority(
        frame, full_test_minority_counts(), test_dataset_column=test_dataset_column
    )
    per_phenotype = frame.groupby("phenotype")["balanced_accuracy"].mean()
    if per_phenotype.empty:
        raise ValueError("no phenotypes survived aggregation for this rung")
    q1, med, q3 = per_phenotype.quantile([0.25, 0.5, 0.75])
    return {
        "q1": q1,
        "q3": q3,
        "median": med,
        "values": per_phenotype.to_numpy(),
        "txt": f"med {med:.2f}",
        "med_txt": f"IQR {q1:.2f}-{q3:.2f}",
    }


def load_rungs() -> dict[str, dict[str, float | str]]:
    """Derive all five ladder rungs from the current result files.

    Returns
    -------
    dict[str, dict[str, float | str]]
        Keyed ``random``, ``cross_dataset``, ``concordant``, ``combined``, ``filtered``.
    """
    ml = pd.read_csv(OUTPUTS / "figure3/ml_results.csv")
    # Concordant-trained models evaluated on the full held-out test, so the rung is
    # comparable with the other cross-dataset rows; mean 0.68, as reported in results.tex.
    concordant = pd.read_csv(
        OUTPUTS / "figure5/figure5d_full_test.tsv", sep="\t"
    ).rename(columns={"balanced_accuracy_full": "balanced_accuracy"})
    combined = pd.read_csv(OUTPUTS / "figure6/figure6d_combined_features_results.csv")
    filtered = pd.read_csv(OUTPUTS / "figure6/figure6d_phenotype_filtered_results.csv")
    return {
        "random": _rung(ml[ml.split_type == "random_split"]),
        "cross_dataset": _rung(ml[ml.split_type == "dataset_split"]),
        "concordant": _rung(concordant, test_dataset_column="held_out_dataset"),
        "combined": _rung(combined[combined.split_type == "dataset_split"]),
        "filtered": _rung(filtered[filtered.split_type == "dataset_split"]),
    }


def gapmind_selection_enrichment() -> float:
    """How strongly the combined-feature model over-selects curated GapMind features.

    Returns
    -------
    float
        Share of top-10 selected features that are GapMind terms, divided by the
        GapMind share of the combined feature matrix.
    """
    matrix = pd.read_csv(
        "data/processed/features_reduced/combined_datasets/gapmind_kofam_rast.tsv",
        sep="\t",
        index_col=0,
        nrows=1,
    )
    is_gapmind = [
        not (re.fullmatch(r"K\d{5}", c) or c.startswith("SSO:")) for c in matrix.columns
    ]
    matrix_share = sum(is_gapmind) / len(is_gapmind)

    results = pd.read_csv(OUTPUTS / "figure6/figure6d_combined_features_results.csv")
    results = results[results.split_type == "dataset_split"]
    picked = [
        feature
        for row in results.features.dropna()
        for feature in ast.literal_eval(row)
    ]
    picked_share = sum(
        not (re.fullmatch(r"K\d{5}", f) or f.startswith("SSO:")) for f in picked
    ) / len(picked)
    return picked_share / matrix_share


def phenotype_filtered_feature_count() -> float:
    """Typical size of a single phenotype's curated GapMind feature block.

    Returns
    -------
    float
        Median across phenotypes of the mean number of features used by the
        phenotype-filtered cross-dataset models.
    """
    results = pd.read_csv(OUTPUTS / "figure6/figure6d_phenotype_filtered_results.csv")
    results = results[results.split_type == "dataset_split"]
    return float(results.groupby("phenotype")["n_features"].mean().median())


RUNGS = load_rungs()
GAPMIND_ENRICHMENT = gapmind_selection_enrichment()
FILTERED_N_FEATURES = phenotype_filtered_feature_count()

ROWS = [
    dict(
        y=YS[0],
        label="Random\nholdout",
        train=("full", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("in-distribution", INK, "normal"),
        kind="range_median",
        **RUNGS["random"],
        finding="Optimistic; inflated by\nrelated genomes in test",
    ),
    dict(
        y=YS[1],
        label="Cross-dataset\n(full data)",
        train=("full", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median",
        **RUNGS["cross_dataset"],
        neg=True,
        finding="Overfits; transferable\nfeatures for only 6/15",
    ),
    dict(
        y=YS[2],
        label="Concordance\nfiltering",
        train=("concordant", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median",
        **RUNGS["concordant"],
        key=True,
        finding="Recovers pathway genes;\ntransferable for 12/15",
    ),
    dict(
        y=YS[3],
        label="Feature\ncombination",
        train=("full", INK, "normal"),
        feat=("combined\n(~17k)", STEEL, "bold"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median",
        **RUNGS["combined"],
        finding=f"Favours curated ({GAPMIND_ENRICHMENT:.1f}$\\times$), yet\nloses to curated alone",
    ),
    dict(
        y=YS[4],
        label="Feature\nfiltering",
        train=("full", INK, "normal"),
        feat=(f"filtered GapMind\n($\\sim${FILTERED_N_FEATURES:.0f})", STEEL, "bold"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median",
        **RUNGS["filtered"],
        finding="Highest median, but capped\nby the curated pathway",
    ),
]


def render(fig) -> None:
    """Draw the panel onto ``fig``."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel letter sits at the far left, aligned with panels A and B, while the
    # table itself is centred via GX.
    ax.text(
        0.004, 0.995, "C", fontsize=FS_PANEL, fontweight="normal", va="top", ha="left"
    )

    # Table rules (booktabs style): top, header underline, bottom.
    ax.plot([RULE_L, RULE_R], [TOP_RULE, TOP_RULE], color=INK, lw=1.1, zorder=1)
    ax.plot([RULE_L, RULE_R], [MID_RULE, MID_RULE], color=INK, lw=0.6, zorder=1)
    ax.plot([RULE_L, RULE_R], [BOT_RULE, BOT_RULE], color=INK, lw=1.1, zorder=1)

    hy = 0.884
    ax.text(
        X_LABEL,
        hy,
        "Approach",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        X_TRAIN,
        hy,
        "Training\ndata",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=0.95,
    )
    ax.text(
        X_FEAT,
        hy,
        "Features",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        X_EVAL,
        hy,
        "Evaluation",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        (X0 + X1) / 2,
        hy,
        "Balanced accuracy",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(X_FIND, hy, "Finding", fontsize=FS_HEAD, fontweight="bold", va="center")

    # Balanced-accuracy scale: gridlines, dashed chance line at 0.50, and the
    # x-axis baseline with ticks.
    for t in (0.4, 0.6, 0.8, 1.0):
        ax.plot(
            [bax(t), bax(t)], [BA_AXIS_Y, GRID_TOP], color="#E2E2E2", lw=0.6, zorder=0
        )
    ax.plot(
        [bax(0.5), bax(0.5)],
        [BA_AXIS_Y, GRID_TOP],
        color="#9A9A9A",
        lw=0.9,
        ls=(0, (4, 3)),
        zorder=1,
    )
    ax.plot(
        [bax(0.4), bax(1.0)], [BA_AXIS_Y, BA_AXIS_Y], color="#333333", lw=0.9, zorder=2
    )
    for t in (0.4, 0.6, 0.8, 1.0):
        ax.plot(
            [bax(t), bax(t)],
            [BA_AXIS_Y, BA_AXIS_Y - 0.016],
            color="#333333",
            lw=0.8,
            zorder=2,
        )
        ax.text(
            bax(t),
            BA_AXIS_Y - 0.048,
            f"{t:.1f}",
            fontsize=FS_TICK,
            color="#444",
            ha="center",
            va="center",
        )
    ax.text(
        bax(0.5),
        BA_AXIS_Y - 0.048,
        "chance",
        fontsize=FS_NOTE,
        color="#8A8A8A",
        style="italic",
        ha="center",
        va="center",
    )

    for r in ROWS:
        y = r["y"]
        key = r.get("key", False)
        # Verdict accent: green = concordance, orange = cross-dataset collapse,
        # ink = neutral.
        accent = GREEN if key else (ORANGE if r.get("neg") else INK)

        for i, line in enumerate(r["label"].split("\n")):
            ax.text(
                X_LABEL,
                y + 0.028 - i * 0.050,
                line,
                fontsize=FS_LABEL,
                fontweight="bold",
                color=INK,
                va="center",
            )

        for xc, (txt, col, weight) in (
            (X_TRAIN, r["train"]),
            (X_FEAT, r["feat"]),
            (X_EVAL, r["evalu"]),
        ):
            lines = txt.split("\n")
            n = len(lines)
            for i, line in enumerate(lines):
                yy = y + (n - 1) * 0.024 - i * 0.048
                ax.text(
                    xc,
                    yy,
                    line,
                    fontsize=FS_COL,
                    color=col,
                    fontweight=weight,
                    ha="center",
                    va="center",
                )

        # Capsule spans the interquartile range; per-phenotype values are drawn
        # on top as a strip.
        x_lo, x_hi = bax(r["q1"]), bax(r["q3"])
        ax.add_patch(
            FancyBboxPatch(
                (x_lo, y - CAP_H / 2),
                x_hi - x_lo,
                CAP_H,
                boxstyle="round,pad=0,rounding_size=0.010",
                facecolor=CAP_COLOR,
                edgecolor="none",
                alpha=(0.45 if key else 0.36),
                zorder=2,
            )
        )
        ax.plot(
            [bax(v) for v in r["values"]],
            [y] * len(r["values"]),
            ls="none",
            marker="o",
            ms=2.8,
            mfc=DOT_COLOR,
            mec="white",
            mew=0.4,
            alpha=0.9,
            zorder=3,
        )
        xm = bax(r["median"])
        ax.plot([xm, xm], [y - CAP_H / 2, y + CAP_H / 2], color=INK, lw=1.5, zorder=4)
        ax.text(
            xm,
            y + VAL_DY,
            r["txt"],
            fontsize=FS_VAL,
            fontweight="bold",
            color="#555",
            ha="center",
            va="center",
            zorder=3,
        )
        ax.text(
            (x_lo + x_hi) / 2,
            y - MED_DY,
            r["med_txt"],
            fontsize=FS_MED,
            fontweight="normal",
            color="#777",
            ha="center",
            va="center",
            zorder=3,
        )

        ax.text(
            X_FIND,
            y,
            r["finding"],
            fontsize=FS_FIND,
            color=accent,
            fontweight=("bold" if key else "normal"),
            va="center",
            linespacing=1.1,
        )


def main() -> None:
    """Render Figure 1C to PDF (manuscript) and PNG (review)."""
    fig = plt.figure(figsize=(8.2, 3.0))
    render(fig)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", PDF_OUT, "and", PNG_OUT)


if __name__ == "__main__":
    main()
