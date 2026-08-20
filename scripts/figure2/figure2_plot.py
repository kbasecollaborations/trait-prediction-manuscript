#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
import seaborn as sns

from scripts.visualization import (
    configure_plot_style,
    hide_categorical_minor_ticks,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def plot_gapmind_comparison(ax: plt.Axes, data_dir: Path) -> None:
    """Plot comparison of strict vs loose GapMind confidence thresholds.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on.
    data_dir : Path
        Directory containing the GapMind metrics files.
    """
    strict_df = pd.read_csv(data_dir / "gapmind_strict_metrics.tsv", sep="\t")
    loose_df = pd.read_csv(data_dir / "gapmind_loose_metrics.tsv", sep="\t")

    strict_df["confidence"] = "Strict"
    loose_df["confidence"] = "Permissive"

    df = pd.concat([strict_df, loose_df], ignore_index=True)

    phenotypes = sorted(df["phenotype"].unique())

    x = np.arange(len(phenotypes))
    width = 0.35

    strict_data = strict_df.set_index("phenotype").reindex(phenotypes)
    loose_data = loose_df.set_index("phenotype").reindex(phenotypes)

    color_strict = "#6E2A4E"
    color_loose = "#A23B72"

    ax.bar(
        x - width / 2,
        strict_data["balanced_accuracy"],
        width,
        label="Strict",
        color=color_strict,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        loose_data["balanced_accuracy"],
        width,
        label="Permissive",
        color=color_loose,
        alpha=0.8,
    )

    mean_strict = strict_df["balanced_accuracy"].mean()
    mean_loose = loose_df["balanced_accuracy"].mean()

    ax.axhline(
        y=mean_strict,
        color=color_strict,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Strict Mean ({mean_strict:.2f})",
    )
    ax.axhline(
        y=mean_loose,
        color=color_loose,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Permissive Mean ({mean_loose:.2f})",
    )

    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", top=False, bottom=True)
    ax.tick_params(axis="y")
    ax.set_ylim(0, 1)

    ax.text(
        -0.05,
        1.05,
        "A",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
    )


def plot_baseline_comparison(axes: np.ndarray, data_dir: Path) -> None:
    """Plot comparison of nearest neighbor model against null models across split types.

    Parameters
    ----------
    axes : np.ndarray
        Array of matplotlib axes objects (2 subplots for 2 split types).
    data_dir : Path
        Directory containing the baseline metrics files.
    """
    random_df = pd.read_csv(data_dir / "random_split_baselines.tsv", sep="\t")
    phylo_df = pd.read_csv(data_dir / "out_of_clade_split_baselines.tsv", sep="\t")

    random_df["split"] = "Random split"
    phylo_df["split"] = "Out-of-Clade split"

    df = pd.concat([random_df, phylo_df], ignore_index=True)

    split_order = ["Random split", "Out-of-Clade split"]
    model_order = ["identity", "bernoulli", "nearest_neighbor"]
    model_labels = {
        "identity": "Identity",
        "bernoulli": "Bernoulli",
        "nearest_neighbor": "Nearest Neighbor",
    }

    df["model"] = df["model"].map(model_labels)

    palette = {
        "Identity": "#A0A0A0",
        "Bernoulli": "#707070",
        "Nearest Neighbor": "#000000",
    }

    markers = {
        "Identity": "o",
        "Bernoulli": "s",
        "Nearest Neighbor": "^",
    }

    phenotypes = sorted(df["phenotype"].unique())

    # Seeded so that re-running the script reproduces the same figure byte-wise.
    rng = np.random.default_rng(0)

    for idx, split_type in enumerate(split_order):
        ax = axes[idx]
        split_df = df[df["split"] == split_type]

        for model_name in model_labels.values():
            model_df = split_df[split_df["model"] == model_name]

            x_positions = []
            y_values = []
            for _, row in model_df.iterrows():
                x_pos = phenotypes.index(row["phenotype"])
                # Jitter so overlapping per-phenotype points are visible
                x_jitter = x_pos + rng.uniform(-0.15, 0.15)
                x_positions.append(x_jitter)
                y_values.append(row["balanced_accuracy"])

            ax.scatter(
                x_positions,
                y_values,
                marker=markers[model_name],
                color=palette[model_name],
                alpha=0.6,
                s=50,
                label=model_name,
            )

        for phenotype in phenotypes:
            phenotype_data = split_df[split_df["phenotype"] == phenotype]
            x_pos = phenotypes.index(phenotype)

            for model_name in model_labels.values():
                model_data = phenotype_data[phenotype_data["model"] == model_name][
                    "balanced_accuracy"
                ]
                if len(model_data) > 0:
                    mean_val = model_data.mean()
                    ax.plot(
                        [x_pos - 0.15, x_pos + 0.15],
                        [mean_val, mean_val],
                        color=palette[model_name],
                        linewidth=2,
                        alpha=0.9,
                    )

        ax.set_title(split_type, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

        ax.set_xticks(range(len(phenotypes)))
        ax.set_xticklabels(phenotypes)
        ax.set_xlim(-0.5, len(phenotypes) - 0.5)

        if idx == len(split_order) - 1:
            ax.set_xlabel("Phenotype")
            ax.tick_params(
                axis="x", which="both", top=False, bottom=True, labelbottom=True
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.tick_params(
                axis="x", which="both", top=False, bottom=True, labelbottom=False
            )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[model],
            color="w",
            markerfacecolor=palette[model],
            markersize=8,
            alpha=0.8,
            label=model,
            linestyle="None",
        )
        for model in model_labels.values()
    ]

    axes[0].legend(
        handles=legend_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
    )

    axes[0].text(
        -0.08,
        1.05,
        "B",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )


def create_figure(data_dir: Path, output_file: Path) -> None:
    """Create Figure 2 with multiple subplots.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data files.
    output_file : Path
        Path to save the output figure.
    """
    fig = plt.figure(figsize=(12, 14))

    # Panel A above the two-row panel B section.
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.6)
    gs_b = gs_main[1].subgridspec(2, 1, hspace=0.25)

    ax_a = fig.add_subplot(gs_main[0])
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)

    plot_gapmind_comparison(ax_a, data_dir)
    plot_baseline_comparison(np.array([ax_b1, ax_b2]), data_dir)

    plt.tight_layout()
    hide_categorical_minor_ticks(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data/outputs/figure2")
    output_file = Path("figures/figure2.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure(data_dir, output_file)
