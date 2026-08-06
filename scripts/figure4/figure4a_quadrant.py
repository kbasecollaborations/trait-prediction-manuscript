#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers matplotlib styles)
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.style.use(["science", "nature"])


COL_TP = "#3a8a83"  # teal
COL_FN = "#c47a3d"  # amber
COL_FP = "#a85060"  # rose
COL_TN = "#5876a0"  # muted blue

BG_TP = "#e8f1ef"
BG_FN = "#fbf3e9"
BG_FP = "#f3e6e9"
BG_TN = "#e7ecf3"

CAT_ANNOTATION = "#6f5fb0"  # purple
CAT_BIOLOGY = "#3d6e3f"  # dark green
CAT_MEASUREMENT = "#6c7a89"  # slate gray
CAT_MEDIA = "#8a6d3a"  # brown
CAT_REGULATION = "#9c8a35"  # olive yellow

TXT_DARK = "#1f1f1f"
TXT_BODY = "#333333"
TXT_ITALIC = "#6f6f6f"

# Font sizes tuned for the composite Figure 4 slot.
FS_LETTER = 14
FS_BOX_TITLE = 9
FS_BOX_SUB = 6.5
FS_SECTION = 7.5
FS_BODY = 7
FS_BULLET = 7
FS_CORNER = 6.5
FS_TOP_TITLE = 11
FS_TOP_TITLE_ITALIC = 9
FS_COL_SUB = 9
FS_ROW_LABEL = 8
FS_OUTER_LABEL = 9
FS_OUTER_ITALIC = 8


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    border: str,
    bg: str,
) -> None:
    """Draw a rounded rectangle quadrant box."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=bg,
        edgecolor=border,
        linewidth=1.6,
        joinstyle="round",
    )
    ax.add_patch(patch)


def _draw_header(
    ax: plt.Axes,
    bx: float,
    by: float,
    bh: float,
    letter: str,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    """Draw a quadrant header: large letter + colored caps title + italic subtitle."""
    letter_y = by + bh - 0.35
    ax.text(
        bx + 0.18,
        letter_y,
        letter,
        fontsize=FS_LETTER,
        fontweight="bold",
        color=TXT_DARK,
        ha="left",
        va="center",
    )
    ax.text(
        bx + 0.58,
        letter_y + 0.02,
        title,
        fontsize=FS_BOX_TITLE,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
    )
    ax.text(
        bx + 0.58,
        letter_y - 0.27,
        subtitle,
        fontsize=FS_BOX_SUB,
        fontstyle="italic",
        color=TXT_ITALIC,
        ha="left",
        va="center",
    )


def _draw_section_heading(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=FS_SECTION,
        fontweight="bold",
        color=TXT_DARK,
        ha="left",
        va="center",
    )


def _draw_body_line(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=FS_BODY,
        color=TXT_BODY,
        ha="left",
        va="center",
    )


def _draw_bullet_row(
    ax: plt.Axes,
    x: float,
    y: float,
    cat_color: str,
    category: str,
    description: str,
) -> None:
    """Draw a "possible causes" bullet row: colored square + category + description."""
    sq_size = 0.11
    ax.add_patch(
        Rectangle(
            (x, y - sq_size / 2),
            sq_size,
            sq_size,
            facecolor=cat_color,
            edgecolor="none",
        )
    )
    ax.text(
        x + 0.27,
        y,
        category,
        fontsize=FS_BULLET,
        fontweight="bold",
        color=cat_color,
        ha="left",
        va="center",
    )
    ax.text(
        x + 2.05,
        y,
        description,
        fontsize=FS_BULLET,
        color=TXT_BODY,
        ha="left",
        va="center",
    )


def _draw_corner_note(
    ax: plt.Axes,
    bx: float,
    by: float,
    bw: float,
    text: str,
    color: str,
) -> None:
    """Draw the bottom-right italic colored note in a quadrant."""
    ax.text(
        bx + bw - 0.2,
        by + 0.25,
        text,
        fontsize=FS_CORNER,
        fontstyle="italic",
        fontweight="bold",
        color=color,
        ha="right",
        va="center",
    )


def _draw_bracket(
    ax: plt.Axes,
    x: float,
    y_bot: float,
    y_top: float,
    serif_len: float = 0.13,
    color: str = "#2a2a2a",
    lw: float = 0.7,
) -> None:
    """Draw a [-style bracket along the left edge of a row."""
    ax.plot([x, x], [y_bot, y_top], color=color, lw=lw, solid_capstyle="butt")
    ax.plot(
        [x, x + serif_len], [y_top, y_top], color=color, lw=lw, solid_capstyle="butt"
    )
    ax.plot(
        [x, x + serif_len], [y_bot, y_bot], color=color, lw=lw, solid_capstyle="butt"
    )


def create_quadrant_plot(ax: plt.Axes) -> None:
    """Create the quadrant plot showing concordant/discordant classifications.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    """
    box_w = 5.8
    box_h = 4.55
    h_gap = 0.32
    v_gap = 0.40

    left_x = 0.55
    right_x = left_x + box_w + h_gap
    bottom_y = 0.0
    top_y = bottom_y + box_h + v_gap

    total_right = right_x + box_w

    # Keep the data range tight to the drawn content so matplotlib's auto
    # aspect stretches the layout across the entire panel A slot.
    ax.set_xlim(-0.18, total_right + 0.08)
    ax.set_ylim(-0.18, 10.40)
    ax.set_aspect("auto")
    ax.margins(0, 0)
    ax.axis("off")

    _draw_box(ax, left_x, top_y, box_w, box_h, COL_TP, BG_TP)
    _draw_box(ax, right_x, top_y, box_w, box_h, COL_FN, BG_FN)
    _draw_box(ax, left_x, bottom_y, box_w, box_h, COL_FP, BG_FP)
    _draw_box(ax, right_x, bottom_y, box_w, box_h, COL_TN, BG_TN)

    _draw_header(ax, left_x, top_y, box_h, "a", "TRUE POSITIVE", "concordant", COL_TP)
    _draw_header(ax, right_x, top_y, box_h, "b", "FALSE NEGATIVE", "discordant", COL_FN)
    _draw_header(
        ax, left_x, bottom_y, box_h, "c", "FALSE POSITIVE", "discordant", COL_FP
    )
    _draw_header(
        ax, right_x, bottom_y, box_h, "d", "TRUE NEGATIVE", "concordant", COL_TN
    )

    # Box a (true positive).
    bx, by = left_x, top_y
    cx = bx + 0.35
    sec_y = by + box_h - 1.35
    _draw_section_heading(ax, cx, sec_y, "BIOLOGICAL STATE")
    _draw_body_line(ax, cx, sec_y - 0.35, "Genes present and functional")
    _draw_body_line(ax, cx, sec_y - 0.62, "Pathway correctly annotated as present")
    _draw_body_line(ax, cx, sec_y - 0.89, "Experiment accurately measures growth")
    _draw_section_heading(ax, cx, sec_y - 1.35, "INTERPRETATION")
    _draw_body_line(ax, cx, sec_y - 1.70, "Prediction and phenotype agree:")
    _draw_body_line(ax, cx, sec_y - 1.97, "gene-to-function mapping is supported.")
    _draw_corner_note(ax, bx, by, box_w, "High-quality training sample", COL_TP)

    # Box b (false negative).
    bx, by = right_x, top_y
    cx = bx + 0.35
    sec_y = by + box_h - 1.35
    _draw_section_heading(ax, cx, sec_y, "POSSIBLE CAUSES")
    bullets_b = [
        (CAT_ANNOTATION, "ANNOTATION", "homolog not detected"),
        (CAT_BIOLOGY, "BIOLOGY", "alternative pathway present"),
        (CAT_BIOLOGY, "BIOLOGY", "promiscuous enzyme activity"),
        (CAT_MEASUREMENT, "MEASUREMENT", "false-positive growth call"),
        (CAT_MEDIA, "MEDIA", "growth on base media (carryover)"),
    ]
    for i, (cc, cat, desc) in enumerate(bullets_b):
        _draw_bullet_row(ax, cx, sec_y - 0.42 - i * 0.36, cc, cat, desc)
    _draw_corner_note(ax, bx, by, box_w, "Growth via unknown mechanism", COL_FN)

    # Box c (false positive).
    bx, by = left_x, bottom_y
    cx = bx + 0.35
    sec_y = by + box_h - 1.35
    _draw_section_heading(ax, cx, sec_y, "POSSIBLE CAUSES")
    bullets_c = [
        (CAT_REGULATION, "REGULATION", "pathway repressed under condition"),
        (CAT_MEDIA, "MEDIA", "insufficient incubation time"),
        (CAT_MEDIA, "MEDIA", "missing cofactors / inducers"),
        (CAT_ANNOTATION, "ANNOTATION", "false-positive homology call"),
        (CAT_BIOLOGY, "BIOLOGY", "pseudogene or non-functional allele"),
    ]
    for i, (cc, cat, desc) in enumerate(bullets_c):
        _draw_bullet_row(ax, cx, sec_y - 0.42 - i * 0.36, cc, cat, desc)
    _draw_corner_note(ax, bx, by, box_w, "Genes present but not functional", COL_FP)

    # Box d (true negative).
    bx, by = right_x, bottom_y
    cx = bx + 0.35
    sec_y = by + box_h - 1.35
    _draw_section_heading(ax, cx, sec_y, "BIOLOGICAL STATE")
    _draw_body_line(ax, cx, sec_y - 0.35, "Genes absent from genome")
    _draw_body_line(ax, cx, sec_y - 0.62, "Pathway correctly annotated as absent")
    _draw_body_line(ax, cx, sec_y - 0.89, "No alternative pathway or leaky activity")
    _draw_body_line(ax, cx, sec_y - 1.16, "Experiment accurately measures no-growth")
    _draw_section_heading(ax, cx, sec_y - 1.62, "INTERPRETATION")
    _draw_body_line(ax, cx, sec_y - 1.97, "Absence of capability confirmed by both")
    _draw_body_line(ax, cx, sec_y - 2.24, "genome content and phenotype.")
    _draw_corner_note(ax, bx, by, box_w, "High-quality training sample", COL_TN)

    title_y = 10.22
    subtitle_y = 9.78

    title_center = (left_x + total_right) / 2.0
    # One text object with the parenthetical as math-italic, so the label
    # centres as a single block on title_center.
    ax.text(
        title_center,
        title_y,
        r"GapMind prediction $\mathit{(in\ silico)}$",
        fontsize=FS_TOP_TITLE,
        fontweight="bold",
        color=TXT_DARK,
        ha="center",
        va="center",
    )

    col1_cx = left_x + box_w / 2.0
    col2_cx = right_x + box_w / 2.0
    ax.text(
        col1_cx,
        subtitle_y,
        "Pathway predicted present (+)",
        fontsize=FS_COL_SUB,
        fontweight="bold",
        color=COL_TP,
        ha="center",
        va="center",
    )
    ax.text(
        col2_cx,
        subtitle_y,
        "Pathway predicted absent (−)",
        fontsize=FS_COL_SUB,
        fontweight="bold",
        color=COL_FP,
        ha="center",
        va="center",
    )

    # row_label_x sets a label-to-box gap that mirrors the vertical gap between
    # the top subtitle and the box top.
    row_label_x = 0.25
    outer_label_x = -0.10

    ax.text(
        row_label_x,
        top_y + box_h / 2.0,
        "Growth observed (+)",
        fontsize=FS_ROW_LABEL,
        fontweight="bold",
        color=COL_TP,
        ha="center",
        va="center",
        rotation=90,
    )
    ax.text(
        row_label_x,
        bottom_y + box_h / 2.0,
        "No growth (−)",
        fontsize=FS_ROW_LABEL,
        fontweight="bold",
        color=COL_FP,
        ha="center",
        va="center",
        rotation=90,
    )

    overall_cy = (bottom_y + top_y + box_h) / 2.0
    # One rotated text object with the parenthetical as math-italic.
    ax.text(
        outer_label_x,
        overall_cy,
        r"Experimental outcome $\mathit{(in\ vivo)}$",
        fontsize=FS_TOP_TITLE,
        fontweight="bold",
        color=TXT_DARK,
        ha="center",
        va="center",
        rotation=90,
    )


if __name__ == "__main__":
    # Matches the per-panel size the composite Figure 4 renders at.
    fig, ax = plt.subplots(figsize=(6.75, 6.0))
    create_quadrant_plot(ax)
    plt.tight_layout()

    output_file = Path("figures/figure4a.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()
