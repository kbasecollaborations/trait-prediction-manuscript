#!/usr/bin/env python3
"""
Generate plots for Supplementary Figure S5: KOFAM features on concordant samples.

Produces a two-panel figure:
  (A) Paired comparison of cross-dataset balanced accuracy: full vs concordant
      training, both using KOFAM features. Each point is one phenotype-split
      combination; the diagonal shows parity.
  (B) Per-phenotype grouped box plots of cross-dataset balanced accuracy for
      full training (grey) vs concordant training (coloured).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

# Colours consistent with the main figures.
COLOR_FULL = "#7f7f7f"
COLOR_CONCORDANT = "#2E86AB"

COMMON_PHENOTYPES = [
    "Alanine",
    "Arginine",
    "Cellobiose",
    "Fructose",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Glycerol",
    "Histidine",
    "Maltose",
    "Mannitol",
    "Mannose",
    "Serine",
    "Sucrose",
    "m-Inositol",
]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Map internal labels to display labels."""
    out = df.copy()
    out["split_type"] = out["split_type"].map(
        {
            "random_split": "Random Split",
            "dataset_split": "Dataset Split",
            "phylo_ooc": "Out-of-Clade",
        }
    )
    out["training_type"] = out["training_type"].map(
        {"full": "Full", "concordant": "Concordant"}
    )
    out["test_subset"] = out["test_subset"].map(
        {
            "full": "Full Test",
            "concordant": "Concordant Test",
            "discordant": "Discordant Test",
        }
    )
    return out


def select_cross_dataset_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the cross-dataset results shown in Supplementary Figure S5.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results with display labels.

    Returns
    -------
    pd.DataFrame
        Cross-dataset results evaluated on concordant test samples for both
        full-training and concordant-training models.
    """
    return df[
        (df["split_type"] == "Dataset Split") & (df["test_subset"] == "Concordant Test")
    ].copy()


def plot_parity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel A: scatter of concordant vs full balanced accuracy per split-key.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Prepared results.
    """
    sub = select_cross_dataset_subset(df)
    pivot = sub.pivot_table(
        index=["key", "phenotype"],
        columns="training_type",
        values="balanced_accuracy",
        aggfunc="mean",
    ).dropna()

    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, zorder=0)
    ax.scatter(
        pivot["Full"],
        pivot["Concordant"],
        c=COLOR_CONCORDANT,
        edgecolors="black",
        linewidths=0.4,
        s=35,
        alpha=0.75,
        zorder=2,
    )
    ax.set_xlabel("Full Training (Balanced Accuracy)")
    ax.set_ylabel("Concordant Training (Balanced Accuracy)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.text(-0.08, 1.05, "(A)", transform=ax.transAxes, fontweight="bold", fontsize=14)

    n_improved = (pivot["Concordant"] > pivot["Full"]).sum()
    n_total = len(pivot)
    ax.text(
        0.95,
        0.05,
        f"Concordant better:\n{n_improved}/{n_total}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )


def plot_phenotype_boxes(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel B: grouped box plots per phenotype (cross-dataset, full test).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    df : pd.DataFrame
        Prepared results.
    """
    sub = select_cross_dataset_subset(df)

    phenotypes = [p for p in COMMON_PHENOTYPES if p in sub["phenotype"].unique()]
    x = np.arange(len(phenotypes))
    width = 0.35

    full_data = [
        sub[(sub["phenotype"] == p) & (sub["training_type"] == "Full")][
            "balanced_accuracy"
        ].values
        for p in phenotypes
    ]
    conc_data = [
        sub[(sub["phenotype"] == p) & (sub["training_type"] == "Concordant")][
            "balanced_accuracy"
        ].values
        for p in phenotypes
    ]

    for i in range(0, len(phenotypes), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.06, zorder=0)

    bp_full = ax.boxplot(
        full_data,
        positions=x - width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLOR_FULL, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    bp_conc = ax.boxplot(
        conc_data,
        positions=x + width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLOR_CONCORDANT, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlim(-0.5, len(phenotypes) - 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, ls=":", color="grey", lw=0.7)

    from matplotlib.patches import Patch

    ax.legend(
        [
            Patch(facecolor=COLOR_FULL, alpha=0.7),
            Patch(facecolor=COLOR_CONCORDANT, alpha=0.7),
        ],
        ["Full Training (KOFAM)", "Concordant Training (KOFAM)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    ax.text(-0.08, 1.05, "(B)", transform=ax.transAxes, fontweight="bold", fontsize=14)


def main() -> None:
    """Generate Supplementary Figure S5."""
    data_file = Path("data/outputs/figureS5/figureS5_kofam_concordant_results.csv")
    output_file = Path("figures/figure_s5.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows from {data_file}")

    # Minority-class-in-test filter (Methods). Figure S5 evaluates the
    # concordant subset of the held-out dataset, so concordant counts are used.
    from scripts.minority_filter import (
        concordant_minority_counts,
        filter_by_minority,
    )

    df = filter_by_minority(df, concordant_minority_counts())
    print(f"After minority-class filter: {len(df)} rows")
    plot_data = _prepare(df)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]}
    )

    plot_parity(axes[0], plot_data)
    plot_phenotype_boxes(axes[1], plot_data)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()

    ds = select_cross_dataset_subset(plot_data)
    print("\nCross-dataset mean balanced accuracy:")
    print(
        ds.groupby("training_type")["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
