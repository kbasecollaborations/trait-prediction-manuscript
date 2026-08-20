#!/usr/bin/env python3
"""Generate Supplementary Figure S2 (phylogenetic distance splits).

Per-phenotype boxplots of the minimum cophenetic distance from each test genome
to the training set, grouped by split type (random, in-clade, out-of-clade).

Reads ``data/outputs/figureS2/figureS2_data.tsv`` and writes
``figures/figure_s2.pdf``.

Run with::

    uv run python -m scripts.figureS2.figureS2_plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from matplotlib.patches import Patch

from scripts.create_data_splits import COMMON_PHENOTYPES
from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

DATA_FILE = Path("data/outputs/figureS2/figureS2_data.tsv")
OUTPUT_FILE = Path("figures/figure_s2.pdf")

SPLIT_ORDER = ["random", "in-clade", "out-of-clade"]
SPLIT_LABELS = {
    "random": "Random",
    "in-clade": "In-clade",
    "out-of-clade": "Out-of-clade",
}

# Split-type colours; in-clade and out-of-clade are drawn only in this figure.
SPLIT_COLORS = {
    "random": "#57BA64",  # green
    "in-clade": "#2B5164",  # dark slate
    "out-of-clade": "#785EF0",  # violet
}


def main() -> None:
    """Render Supplementary Figure S2."""
    df = pd.read_csv(DATA_FILE, sep="\t")
    print(f"Loaded {len(df)} rows from {DATA_FILE}")

    phenotypes = [p for p in COMMON_PHENOTYPES if p in set(df["phenotype"])]
    palette = [SPLIT_COLORS[s] for s in SPLIT_ORDER]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(0, len(phenotypes), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.06, zorder=0)

    sns.boxplot(
        data=df,
        x="phenotype",
        y="min_distance",
        hue="split_type",
        order=phenotypes,
        hue_order=SPLIT_ORDER,
        palette=palette,
        showfliers=False,
        # Keep the box fills at full saturation so they match the legend.
        saturation=1.0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="phenotype",
        y="min_distance",
        hue="split_type",
        order=phenotypes,
        hue_order=SPLIT_ORDER,
        dodge=True,
        alpha=0.35,
        size=2.5,
        # Raw points stay black; the dodged boxes carry the split-type colour.
        palette=["black"] * len(SPLIT_ORDER),
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Minimum cophenetic distance")
    ax.set_xticks(range(len(phenotypes)))
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.margins(x=0.01)

    # Handles are rebuilt because the stripplot adds a duplicate set.
    ax.legend(
        handles=[
            Patch(facecolor=SPLIT_COLORS[s], label=SPLIT_LABELS[s]) for s in SPLIT_ORDER
        ],
        title="Split type",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(SPLIT_ORDER),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
