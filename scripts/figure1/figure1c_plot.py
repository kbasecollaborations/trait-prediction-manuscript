#!/usr/bin/env python3
"""Figure 1C: levers for cross-dataset generalisation.

Five analysis variants sit on a shared balanced-accuracy interval ladder
(capsules positioned by value on a 0.40-1.00 axis with a dashed chance line at
0.50; each capsule carries a median tick and value). Plain-text method columns
(Training data, Features, Evaluation) make explicit what each variant changes,
and the Mechanistic-features / Generalises columns carry the features axis.

Row 5 (concordance filtering) is evaluated on the concordant subset (its
applicability domain), labelled accordingly; rows 2-4 are the full cross-dataset
held-out test. Numbers: rows 1-2 from data/outputs/figure3/ml_results.csv (full
KOFAM), feature rows from data/outputs/figure6/figure6d_*_results.csv, and
concordance (concordant subset) from scripts/figure5_diagnostic/ba_table.csv.

Sized for inclusion at \\textwidth so fonts print near their set size and
Figure 1 (panels A-C) fits on one page.

Run with:
    uv run python -m scripts.figure1.figure1c_plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

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
# Capsules share one neutral colour (position on the axis carries the
# performance); the verdict is signalled by the accent colour of the row
# label and finding: green = concordance (the win), orange = cross-dataset
# full-data (the collapse), ink = neutral.
CAP_COLOR = "#6E8299"
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
VAL_DY = 0.055
MED_DY = 0.050

# Table rules and BA-axis positions (figure fraction).
# RULE_R stops just past the Finding text so the table has no empty ruled
# column on the right; the panel letter sits above TOP_RULE (see hy below).
RULE_L, RULE_R = 0.0 + GX, 0.850 + GX
TOP_RULE = 0.925
MID_RULE = 0.855
BOT_RULE = 0.020
BA_AXIS_Y = 0.110      # horizontal BA x-axis baseline
GRID_TOP = 0.845

YS = [0.775, 0.633, 0.490, 0.347, 0.205]


def bax(ba: float) -> float:
    """Map a balanced-accuracy value to the panel x-coordinate."""
    return X0 + (ba - BA_LO) / (BA_HI - BA_LO) * (X1 - X0)


ROWS = [
    dict(
        y=YS[0], label="Random\nholdout",
        train=("full", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("in-distribution", INK, "normal"),
        kind="range_median", lo=0.70, hi=0.90, median=0.85, txt="0.70-0.90",
        med_txt="med 0.85",
        finding="Optimistic; inflated by\nrelated genomes in test",
    ),
    dict(
        y=YS[1], label="Cross-dataset\n(full data)",
        train=("full", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median", lo=0.48, hi=0.84, median=0.62, txt="0.48-0.84",
        med_txt="med 0.62", neg=True,
        finding="Overfits; transferable\nfeatures for only 6/15",
    ),
    dict(
        y=YS[2], label="Concordance\nfiltering",
        train=("concordant", INK, "normal"),
        feat=("KOFAM", INK, "normal"),
        evalu=("cross-dataset\n(concordant)", INK, "normal"),
        kind="range_median", lo=0.51, hi=0.98, median=0.83, txt="0.51-0.98",
        med_txt="med 0.83",
        key=True,
        finding="Recovers pathway genes;\ntransferable for 12/15",
    ),
    dict(
        y=YS[3], label="Feature\ncombination",
        train=("full", INK, "normal"),
        feat=("combined\n(~17k)", STEEL, "bold"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median", lo=0.55, hi=0.85, median=0.66, txt="0.55-0.85",
        med_txt="med 0.66",
        finding="Comprehensive features\ndo not generalise better",
    ),
    dict(
        y=YS[4], label="Feature\nfiltering",
        train=("full", INK, "normal"),
        feat=("filtered GapMind\n(~32)", STEEL, "bold"),
        evalu=("cross-dataset", INK, "normal"),
        kind="range_median", lo=0.56, hi=0.85, median=0.73, txt="0.56-0.85",
        med_txt="med 0.73",
        finding="Curbs overfit; higher\nmedian, capped ceiling",
    ),
]


def render(fig) -> None:
    """Draw the panel onto ``fig``."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel letter stays at the far left (aligned with panels A and B); the
    # table itself is centred via GX.
    ax.text(0.004, 0.995, "C", fontsize=FS_PANEL, fontweight="normal", va="top", ha="left")

    # Table rules (booktabs style): top, header underline, bottom.
    ax.plot([RULE_L, RULE_R], [TOP_RULE, TOP_RULE], color=INK, lw=1.1, zorder=1)
    ax.plot([RULE_L, RULE_R], [MID_RULE, MID_RULE], color=INK, lw=0.6, zorder=1)
    ax.plot([RULE_L, RULE_R], [BOT_RULE, BOT_RULE], color=INK, lw=1.1, zorder=1)

    # Column headers.
    hy = 0.884
    ax.text(X_LABEL, hy, "Approach", fontsize=FS_HEAD, fontweight="bold", ha="left", va="center")
    ax.text(X_TRAIN, hy, "Training\ndata", fontsize=FS_HEAD, fontweight="bold",
            ha="center", va="center", linespacing=0.95)
    ax.text(X_FEAT, hy, "Features", fontsize=FS_HEAD, fontweight="bold", ha="center", va="center")
    ax.text(X_EVAL, hy, "Evaluation", fontsize=FS_HEAD, fontweight="bold", ha="center", va="center")
    ax.text((X0 + X1) / 2, hy, "Balanced accuracy", fontsize=FS_HEAD, fontweight="bold",
            ha="center", va="center")
    ax.text(X_FIND, hy, "Finding", fontsize=FS_HEAD, fontweight="bold", va="center")

    # Balanced-accuracy scale: light gridlines up the rows, dashed chance line,
    # and a horizontal x-axis baseline with ticks and labels at the bottom.
    for t in (0.4, 0.6, 0.8, 1.0):
        ax.plot([bax(t), bax(t)], [BA_AXIS_Y, GRID_TOP], color="#E2E2E2", lw=0.6, zorder=0)
    ax.plot([bax(0.5), bax(0.5)], [BA_AXIS_Y, GRID_TOP], color="#9A9A9A", lw=0.9,
            ls=(0, (4, 3)), zorder=1)
    ax.plot([bax(0.4), bax(1.0)], [BA_AXIS_Y, BA_AXIS_Y], color="#333333", lw=0.9, zorder=2)
    for t in (0.4, 0.6, 0.8, 1.0):
        ax.plot([bax(t), bax(t)], [BA_AXIS_Y, BA_AXIS_Y - 0.016], color="#333333", lw=0.8, zorder=2)
        ax.text(bax(t), BA_AXIS_Y - 0.048, f"{t:.1f}", fontsize=FS_TICK, color="#444",
                ha="center", va="center")
    ax.text(bax(0.5), BA_AXIS_Y - 0.048, "chance", fontsize=FS_NOTE, color="#8A8A8A",
            style="italic", ha="center", va="center")

    for r in ROWS:
        y = r["y"]
        key = r.get("key", False)
        # Verdict accent: green = concordance (win), orange = cross-dataset
        # collapse, ink = neutral. Applied to the row label and finding.
        accent = GREEN if key else (ORANGE if r.get("neg") else INK)

        for i, line in enumerate(r["label"].split("\n")):
            ax.text(X_LABEL, y + 0.028 - i * 0.050, line, fontsize=FS_LABEL,
                    fontweight="bold", color=INK, va="center")

        for xc, (txt, col, weight) in (
            (X_TRAIN, r["train"]),
            (X_FEAT, r["feat"]),
            (X_EVAL, r["evalu"]),
        ):
            lines = txt.split("\n")
            n = len(lines)
            for i, line in enumerate(lines):
                yy = y + (n - 1) * 0.024 - i * 0.048
                ax.text(xc, yy, line, fontsize=FS_COL, color=col, fontweight=weight,
                        ha="center", va="center")

        x_lo, x_hi = bax(r["lo"]), bax(r["hi"])
        ax.add_patch(FancyBboxPatch(
            (x_lo, y - CAP_H / 2), x_hi - x_lo, CAP_H,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor=CAP_COLOR, edgecolor="none",
            alpha=(0.95 if key else 0.9), zorder=2))
        ax.text((x_lo + x_hi) / 2, y + VAL_DY, r["txt"], fontsize=FS_VAL,
                fontweight="bold", color="#555",
                ha="center", va="center", zorder=3)
        xm = bax(r["median"])
        ax.plot([xm, xm], [y - CAP_H / 2, y + CAP_H / 2], color="white", lw=1.6, zorder=3)
        ax.text(xm, y - MED_DY, r["med_txt"], fontsize=FS_MED,
                fontweight="normal", color="#777", ha="center", va="center", zorder=3)

        ax.text(X_FIND, y, r["finding"], fontsize=FS_FIND,
                color=accent,
                fontweight=("bold" if key else "normal"), va="center", linespacing=1.1)


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
