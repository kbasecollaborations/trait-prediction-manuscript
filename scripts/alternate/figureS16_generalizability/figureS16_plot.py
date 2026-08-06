#!/usr/bin/env python3
"""Plot Supplementary Figure S16: top candidate non-canonical predictors of growth
recovered by concordant-trained models, for the six phenotypes with the most
recovered FN-discordant cases. Hypothesis generating, not validated."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (matplotlib style registration)
import seaborn as sns
from matplotlib.axes import Axes

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

DATA_FILE: Path = Path("data/outputs/figureS16/figureS16_recovered_features.tsv")
COUNTS_FILE: Path = Path("data/outputs/figureS16/figureS16_recovered_counts.tsv")
OUTPUT_FILE: Path = Path("figures/figure_s16.pdf")

N_PHENOTYPES: int = 6
N_FEATURES_PER_PANEL: int = 6
BAR_COLOR: str = "#2E86AB"
NAME_MAX_CHARS: int = 42


def _feature_label(feature: str, ko_name: str) -> str:
    """Build a compact y-axis label combining KO id and truncated function name.

    Parameters
    ----------
    feature : str
        KO identifier (e.g. ``"K01712"``).
    ko_name : str
        Functional annotation; may be empty.

    Returns
    -------
    str
        Label of the form ``"K01712: urocanate hydratase"``.
    """
    name = "" if pd.isna(ko_name) else str(ko_name)
    name = name.split(" [EC")[0].strip()
    if len(name) > NAME_MAX_CHARS:
        name = name[: NAME_MAX_CHARS - 1].rstrip() + "…"
    return f"{feature}: {name}" if name else feature


def plot_phenotype_panel(
    ax: Axes,
    phenotype: str,
    features: pd.DataFrame,
    n_recovered: int,
) -> None:
    """Draw the top candidate features for one phenotype as a horizontal bar chart.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    phenotype : str
        Phenotype name (panel title).
    features : pd.DataFrame
        Rows of the recovered-features table for this phenotype.
    n_recovered : int
        Number of recovered FN-discordant samples for this phenotype.
    """
    top = features.nlargest(N_FEATURES_PER_PANEL, "mean_shap_toward_growth")
    top = top.iloc[::-1]  # largest at the top of the bar chart

    labels = [
        _feature_label(row.feature, row.ko_name) for row in top.itertuples(index=False)
    ]
    positions = range(len(top))
    ax.barh(
        list(positions),
        top["mean_shap_toward_growth"].to_numpy(),
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.5,
        height=0.7,
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean SHAP toward growth")
    ax.set_title(f"{phenotype} (n = {n_recovered})", fontsize=11)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.margins(x=0.18)


def create_figure(output_file: Path) -> None:
    """Build and persist Supplementary Figure S16.

    Parameters
    ----------
    output_file : Path
        Destination PDF path.
    """
    features = pd.read_csv(DATA_FILE, sep="\t")
    counts = pd.read_csv(COUNTS_FILE, sep="\t")

    phenotypes = counts.nlargest(N_PHENOTYPES, "n_recovered_fn_discordant")[
        "phenotype"
    ].tolist()
    recovered_lookup = dict(
        zip(
            counts["phenotype"],
            counts["n_recovered_fn_discordant"],
            strict=True,
        )
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for ax, phenotype in zip(axes.flat, phenotypes, strict=False):
        plot_phenotype_panel(
            ax,
            phenotype,
            features[features["phenotype"] == phenotype],
            int(recovered_lookup[phenotype]),
        )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


def main() -> None:
    """Build Supplementary Figure S16."""
    create_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
