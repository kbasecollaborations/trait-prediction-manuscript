#!/usr/bin/env python3
"""Render Figure S9: taxonomic bias from GapMind concordance filtering.

Three panels:

A. Stacked-bar comparison of GTDB Class composition before vs after
   concordance filtering, side-by-side per phenotype.
B. Faith-PD scatter (concordant vs full) with a y = x reference. Phenotypes
   whose ``pd_concordant / pd_full`` ratio falls below 0.5 are annotated by
   name.
C. Grouped bar of mean train/test class overlap (full vs concordant
   training), per phenotype. The test set is held fixed; only the training
   subset changes between scenarios.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


DATA_FILE: Path = Path("data/outputs/figureS9/taxonomic_bias.tsv")
OUTPUT_FILE: Path = Path("figures/figure_s9.pdf")


def _phenotype_order(df: pd.DataFrame) -> list[str]:
    """Return phenotypes sorted alphabetically for stable plotting."""
    return sorted(df["phenotype"].unique())


def _class_palette(classes: list[str]) -> dict[str, tuple[float, float, float]]:
    """Build a stable colour mapping for class labels.

    "Unassigned" is forced to a neutral grey to visually separate it from
    real class assignments.
    """
    real = [c for c in classes if c != "Unassigned"]
    palette = sns.color_palette("Set2", n_colors=max(len(real), 1))
    mapping: dict[str, tuple[float, float, float]] = {
        c: palette[i] for i, c in enumerate(real)
    }
    if "Unassigned" in classes:
        mapping["Unassigned"] = (0.75, 0.75, 0.75)
    return mapping


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Stacked-bar of class composition (full vs concordant) per phenotype.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Long-form taxonomic-bias TSV.
    """
    comp = df[df["measure"] == "class_composition"].copy()
    phenotypes = _phenotype_order(df)
    classes = sorted(
        comp["class_label"].dropna().unique().tolist(),
        key=lambda c: (c == "Unassigned", c),
    )
    palette = _class_palette(classes)

    bar_width = 0.4
    x = np.arange(len(phenotypes))

    for ph_idx, phen in enumerate(phenotypes):
        for offset, scope in zip([-bar_width / 2, bar_width / 2], ["full", "concordant"]):
            sub = comp[(comp["phenotype"] == phen) & (comp["scope"] == scope)]
            bottom = 0.0
            for cls in classes:
                row = sub[sub["class_label"] == cls]
                val = float(row["value"].iloc[0]) if not row.empty else 0.0
                ax.bar(
                    x[ph_idx] + offset,
                    val,
                    width=bar_width,
                    bottom=bottom,
                    color=palette[cls],
                    edgecolor="white",
                    linewidth=0.3,
                    label=cls if (ph_idx == 0 and scope == "full") else None,
                )
                bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Genome fraction")
    ax.set_xlabel("Phenotype")

    # Tick lower-bar / upper-bar legend cue using a small subtitle line
    ax.set_title("(A) Class composition: full (left) vs concordant (right)", loc="left", fontsize=12)

    # Build legend from the unique class colours
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[c], label=c) for c in classes
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=9,
        title="GTDB Class",
    )


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter Faith PD concordant vs full, with y=x reference.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Long-form taxonomic-bias TSV.
    """
    pd_long = df[df["measure"] == "faith_pd"].pivot_table(
        index="phenotype", columns="scope", values="value"
    )
    ratio = df[df["measure"] == "faith_pd_ratio"].set_index("phenotype")["value"]

    ax.scatter(
        pd_long["full"],
        pd_long["concordant"],
        s=70,
        color="#377eb8",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Zoom to the data range so within-cluster variation is visible.
    lo = min(pd_long["full"].min(), pd_long["concordant"].min())
    hi = max(pd_long["full"].max(), pd_long["concordant"].max())
    span = hi - lo
    pad = max(span * 0.10, 1.0)
    lo_plot = max(lo - pad, 0.0)
    hi_plot = hi + pad
    ax.plot(
        [lo_plot, hi_plot],
        [lo_plot, hi_plot],
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=1,
        label="y = x",
    )
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlabel("Faith PD (full)")
    ax.set_ylabel("Faith PD (concordant)")
    ax.set_title(
        f"(B) Phylogenetic diversity retention (median ratio = {ratio.median():.2f})",
        loc="left",
        fontsize=12,
    )

    for phen, row in pd_long.iterrows():
        r = ratio.loc[phen] if phen in ratio.index else float("nan")
        if pd.notna(r) and r < 0.5:
            ax.annotate(
                phen,
                (row["full"], row["concordant"]),
                textcoords="offset points",
                xytext=(5, -3),
                fontsize=8,
                color="black",
            )


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Grouped bar of mean train/test class overlap (full vs concordant).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Long-form taxonomic-bias TSV.
    """
    overlap = df[df["measure"] == "train_test_class_overlap_mean"].pivot_table(
        index="phenotype", columns="scope", values="value"
    )
    phenotypes = _phenotype_order(df)
    overlap = overlap.reindex(phenotypes)

    x = np.arange(len(phenotypes))
    width = 0.4
    ax.bar(
        x - width / 2,
        overlap["full"],
        width=width,
        color="#4daf4a",
        edgecolor="white",
        label="Full training set",
    )
    ax.bar(
        x + width / 2,
        overlap["concordant"],
        width=width,
        color="#984ea3",
        edgecolor="white",
        label="Concordant training set",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean train/test class overlap")
    ax.set_xlabel("Phenotype")
    ax.set_title(
        "(C) Train/test class overlap (mean across 4 LOO splits; test set held fixed)",
        loc="left",
        fontsize=12,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


def create_figure(data_file: Path, output_file: Path) -> None:
    """Build and save Figure S13.

    Parameters
    ----------
    data_file : Path
        Path to the TSV produced by :mod:`scripts.figureS13.figureS13_data`.
    output_file : Path
        Where to write the PDF.
    """
    df = pd.read_csv(data_file, sep="\t")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    plot_panel_a(axes[0], df)
    plot_panel_b(axes[1], df)
    plot_panel_c(axes[2], df)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    create_figure(DATA_FILE, OUTPUT_FILE)
