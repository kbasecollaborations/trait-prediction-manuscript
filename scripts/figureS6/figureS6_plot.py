#!/usr/bin/env python3
"""
Generate plots for Supplementary Figure S6: Learning curves for all 15 phenotypes.

Produces two figures:
  (A) Heatmap of balanced accuracy at each sample size × phenotype for
      concordant training on cross-dataset splits (full test).
  (B) Faceted learning-curve grid: phenotypes (rows) × split types (cols),
      showing concordant training only, on full test samples.
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


FEATURE_TYPE = "gapmind"
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


def normalize_sample_size_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize sample-size labels to a consistent string representation.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``sample_size`` stored as strings such as
        ``"50"``, ``"100"``, and ``"full"``.
    """
    out = df.copy()
    out["sample_size"] = out["sample_size"].astype(str).str.strip().str.lower()
    return out


def get_sample_size_order(df: pd.DataFrame) -> list[str]:
    """
    Return the canonical display order for sample sizes present in the data.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results with normalized sample-size labels.

    Returns
    -------
    list[str]
        Ordered sample-size labels.
    """
    return [size for size in ["50", "100", "200", "500", "full"] if size in set(df["sample_size"])]


def estimate_saturation_sizes(df: pd.DataFrame) -> dict[str, str]:
    """
    Estimate the smallest sample size that reaches 90% of full performance.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results with normalized sample-size labels.

    Returns
    -------
    dict[str, str]
        Mapping from phenotype to the earliest sample-size label that reaches
        at least 90% of the full cross-dataset performance. Phenotypes that do
        not saturate by 500 samples are assigned ``"500"``.
    """
    sub = df[
        (df["training_type"] == "Concordant")
        & (df["split_type"] == "Dataset Split")
        & (df["test_subset"] == "Full Test")
    ].copy()

    phenotypes = [p for p in COMMON_PHENOTYPES if p in sub["phenotype"].unique()]
    saturation_sizes: dict[str, str] = {}

    for pheno in phenotypes:
        ph = sub[sub["phenotype"] == pheno]
        full_perf = ph.loc[ph["sample_size"] == "full", "balanced_accuracy"].mean()
        threshold = 0.90 * full_perf

        for sample_size in ["50", "100", "200", "500"]:
            sample_perf = ph.loc[
                ph["sample_size"] == sample_size, "balanced_accuracy"
            ].mean()
            if pd.notna(sample_perf) and sample_perf >= threshold:
                saturation_sizes[pheno] = sample_size
                break
        else:
            saturation_sizes[pheno] = "500"

    return saturation_sizes


def plot_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    """
    Panel A: heatmap of mean balanced accuracy (concordant, dataset split, full test).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    sub = df[
        (df["training_type"] == "Concordant")
        & (df["split_type"] == "Dataset Split")
        & (df["test_subset"] == "Full Test")
    ].copy()

    sub = normalize_sample_size_labels(sub)

    # Build pivot: phenotype × sample_size
    pivot = (
        sub.groupby(["phenotype", "sample_size"])["balanced_accuracy"]
        .mean()
        .unstack("sample_size")
    )
    # Reorder columns by sample size
    size_order = get_sample_size_order(sub)
    pivot = pivot[size_order]
    # Reorder rows alphabetically
    pivot = pivot.reindex([p for p in COMMON_PHENOTYPES if p in pivot.index])

    # Rename 'full' column for display
    col_labels = [size if size != "full" else "Full" for size in size_order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Training Sample Size")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.55 else "black"
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Balanced Accuracy")
    ax.set_title(
        "Cross-dataset performance by sample size\n(concordant training, GapMind features)",
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def plot_learning_curves(df: pd.DataFrame, output_file: Path) -> None:
    """
    Panel B: faceted learning curves – phenotypes (rows) × split types (cols).

    Shows concordant training only on the full test set.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    sub = df[
        (df["training_type"] == "Concordant") & (df["test_subset"] == "Full Test")
    ].copy()

    phenotypes = [p for p in COMMON_PHENOTYPES if p in sub["phenotype"].unique()]
    split_types = ["Random Split", "Dataset Split", "Out-of-Clade"]

    n_rows = len(phenotypes)
    n_cols = len(split_types)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(12, 2.2 * n_rows),
        sharex=True,
        sharey=True,
    )

    color = "#ff7f0e"  # concordant orange, matching figure 7

    for ri, phenotype in enumerate(phenotypes):
        for ci, split_type in enumerate(split_types):
            ax = axes[ri, ci]
            subset = sub[
                (sub["phenotype"] == phenotype) & (sub["split_type"] == split_type)
            ]

            if subset.empty:
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes,
                )
            else:
                # Draw thin connecting lines per key-repeat
                subset = subset.copy()
                subset["key_repeat"] = subset["key"] + "_" + subset["repeat"].astype(str)
                for kr in subset["key_repeat"].unique():
                    ld = subset[subset["key_repeat"] == kr].sort_values("n_train_samples")
                    ax.plot(
                        ld["n_train_samples"],
                        ld["balanced_accuracy"],
                        color=color,
                        lw=0.8,
                        alpha=0.35,
                    )
                ax.scatter(
                    subset["n_train_samples"],
                    subset["balanced_accuracy"],
                    color=color,
                    marker="s",
                    s=20,
                    alpha=0.65,
                    edgecolors="black",
                    linewidths=0.3,
                )

            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, ls=":", color="grey", lw=0.5)

            if ri == 0:
                ax.set_title(split_type, fontweight="bold")
            if ri == n_rows - 1:
                ax.set_xlabel("Training Samples")
            if ci == 0:
                ax.set_ylabel("Bal. Acc.", fontsize=9)
                ax.text(
                    -0.35, 0.5, phenotype, transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontweight="bold",
                    fontsize=9,
                )

    plt.tight_layout(rect=[0.04, 0, 1, 0.98])
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved learning curves to {output_file}")
    plt.close()


def plot_saturation_summary(df: pd.DataFrame, output_file: Path) -> None:
    """
    Summary bar chart: estimated saturation sample size per phenotype.

    Saturation is defined as the smallest sample size at which cross-dataset
    mean balanced accuracy reaches >= 90 % of the full-dataset performance
    for that phenotype.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared results.
    output_file : Path
        Where to save the figure.
    """
    sub = df[
        (df["training_type"] == "Concordant")
        & (df["split_type"] == "Dataset Split")
        & (df["test_subset"] == "Full Test")
    ].copy()

    sub = normalize_sample_size_labels(sub)
    saturation_sizes = estimate_saturation_sizes(sub)

    # Sort by saturation size, then alphabetically
    order_map = {"50": 50, "100": 100, "200": 200, "500": 500}
    sorted_phenos = sorted(
        saturation_sizes, key=lambda p: (order_map[saturation_sizes[p]], p)
    )
    size_labels = [saturation_sizes[p] for p in sorted_phenos]
    sizes = [order_map[label] for label in size_labels]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    colors = ["#06A77D" if s <= 100 else "#2E86AB" if s <= 200 else "#DE8F05" for s in sizes]
    bars = ax.barh(range(len(sorted_phenos)), sizes, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(sorted_phenos)))
    ax.set_yticklabels(sorted_phenos)
    ax.set_xlabel("Estimated Saturation Sample Size")
    ax.set_title(
        "Most phenotypes saturate by 50--100 samples\n"
        "Cross-dataset 90% threshold"
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(sizes) + 20)

    # Annotate bars
    for bar, size_label in zip(bars, size_labels):
        ax.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            size_label, va="center", fontsize=9,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_specs = []
    if any(size <= 100 for size in sizes):
        legend_specs.append(
            (Patch(facecolor="#06A77D", edgecolor="black", linewidth=0.4), r"$\leq$ 100 samples")
        )
    if any(100 < size <= 200 for size in sizes):
        legend_specs.append(
            (Patch(facecolor="#2E86AB", edgecolor="black", linewidth=0.4), r"101--200 samples")
        )
    if any(size > 200 for size in sizes):
        legend_specs.append(
            (Patch(facecolor="#DE8F05", edgecolor="black", linewidth=0.4), r"$>$ 200 samples")
        )
    if legend_specs:
        handles, labels = zip(*legend_specs)
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            borderaxespad=0.0,
        )

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved saturation summary to {output_file}")
    plt.close()


def main() -> None:
    """Generate all Supplementary Figure S6 plots."""
    data_file = Path(
        f"data/outputs/figureS6/figureS6_learning_curves_{FEATURE_TYPE}.csv"
    )
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows from {data_file}")
    plot_data = normalize_sample_size_labels(_prepare(df))

    print("\nGenerating heatmap...")
    plot_heatmap(plot_data, output_dir / "figure_s6_heatmap.pdf")

    print("Generating learning curves...")
    plot_learning_curves(plot_data, output_dir / "figure_s6_learning_curves.pdf")

    print("Generating saturation summary...")
    plot_saturation_summary(plot_data, output_dir / "figure_s6_saturation.pdf")

    # Print summary statistics
    ds = plot_data[
        (plot_data["training_type"] == "Concordant")
        & (plot_data["split_type"] == "Dataset Split")
        & (plot_data["test_subset"] == "Full Test")
    ]
    if len(ds) > 0:
        print("\nCross-dataset concordant: mean balanced accuracy by phenotype × sample size:")
        summary = (
            ds.groupby(["phenotype", "sample_size"])["balanced_accuracy"]
            .mean()
            .unstack("sample_size")
            .round(3)
        )
        print(summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
