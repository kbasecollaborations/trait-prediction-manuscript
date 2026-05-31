#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_gapmind_vs_ml(ax: plt.Axes, gapmind_data_dir: Path, ml_data_dir: Path) -> None:
    """Plot comparison of GapMind (strict) vs ML (intra-dataset) performance.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    gapmind_data_dir : Path
        Directory containing the GapMind metrics files.
    ml_data_dir : Path
        Directory containing the ML intra-dataset results.
    """
    gapmind_df = pd.read_csv(gapmind_data_dir / "gapmind_strict_metrics.tsv", sep="\t")

    cv_df = pd.read_csv(ml_data_dir / "intra_vs_inter" / "cv_results.csv")

    ml_stats = (
        cv_df[cv_df["representation"] == "full"]
        .groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ml_stats.columns = ["phenotype", "ml_balanced_accuracy", "ml_std"]

    combined_df = gapmind_df.merge(ml_stats, on="phenotype", how="inner")

    carbon_categories = {
        "Amino Acids": ["Alanine", "Arginine", "Histidine", "Serine"],
        "Sugars": [
            "Fructose",
            "Galactose",
            "Glucose",
            "Maltose",
            "Mannose",
            "Sucrose",
            "m-Inositol",
        ],
        "Others": ["Mannitol", "Glycerol", "Galacturonic-Acid", "Cellobiose"],
    }

    category_colors = {
        "Amino Acids": "#FFB6C1",  # Light pink
        "Sugars": "#98FB98",  # Light green
        "Others": "#87CEFA",  # Light blue
    }

    x_order = [c for category in carbon_categories.values() for c in category]

    combined_df = combined_df[combined_df["phenotype"].isin(x_order)]
    combined_df["phenotype"] = pd.Categorical(
        combined_df["phenotype"], categories=x_order, ordered=True
    )
    combined_df = combined_df.sort_values("phenotype")

    phenotype_colors = {}
    for category, phenotypes in carbon_categories.items():
        for phenotype in phenotypes:
            phenotype_colors[phenotype] = category_colors[category]

    x = np.arange(len(combined_df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        combined_df["balanced_accuracy"],
        width,
        label="GapMind (Strict)",
        color="#A23B72",  # Purple (from figure2)
        alpha=0.8,
    )

    bars2 = ax.bar(
        x + width / 2,
        combined_df["ml_balanced_accuracy"],
        width,
        yerr=combined_df["ml_std"],
        label="ML (intra-dataset)",
        color="#2E86AB",  # Blue (from figure2)
        alpha=0.8,
        capsize=3,
        error_kw={"linewidth": 1, "alpha": 0.7},
    )

    # Shade the background of each carbon source category.
    curr_pos = 0
    ax.margins(x=0.00)
    for category, compounds in carbon_categories.items():
        category_width = len(
            [p for p in compounds if p in combined_df["phenotype"].values]
        )
        if category_width > 0:
            ax.axvspan(
                curr_pos - 0.5,
                curr_pos + category_width - 0.5,
                color=category_colors[category],
                alpha=0.2,
                zorder=0,
            )
            curr_pos += category_width

    ax.set_xlabel("Carbon Source")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(combined_df["phenotype"], rotation=45, ha="right")

    for i, phenotype in enumerate(combined_df["phenotype"]):
        ax.get_xticklabels()[i].set_color(phenotype_colors[phenotype])

    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")
    ax.set_ylim(0, 1)

    # Random-performance reference line.
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.4, zorder=0)

    from matplotlib.patches import Patch

    method_handles = [bars1, bars2]
    method_labels = ["GapMind (Strict)", "ML (intra-dataset)"]

    first_legend = ax.legend(
        method_handles,
        method_labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.35, 1.15),
        ncol=2,
        frameon=False,
    )

    # Keep the method legend so the second legend does not replace it.
    ax.add_artist(first_legend)

    category_handles = [
        Patch(facecolor=color, alpha=0.3, label=category, edgecolor="none")
        for category, color in category_colors.items()
    ]

    ax.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.75, 1.15),
        title="Carbon source category",
        ncol=3,
        frameon=False,
    )


def create_figure(
    gapmind_data_dir: Path, ml_data_dir: Path, output_file: Path
) -> None:
    """Create figure comparing GapMind vs ML performance.

    Parameters
    ----------
    gapmind_data_dir : Path
        Directory containing GapMind metrics files.
    ml_data_dir : Path
        Directory containing ML results files.
    output_file : Path
        Path to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_gapmind_vs_ml(ax, gapmind_data_dir, ml_data_dir)

    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")

    png_file = output_file.with_suffix(".png")
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {png_file}")

    plt.close()


if __name__ == "__main__":
    gapmind_data_dir = Path("data/outputs/figure2")
    ml_data_dir = Path("data/outputs/figure3")
    output_file = Path("figures/misc/gapmind_vs_ml.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(gapmind_data_dir, ml_data_dir, output_file)
