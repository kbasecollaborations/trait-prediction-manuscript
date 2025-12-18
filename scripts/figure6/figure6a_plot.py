#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_colors,
)

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()


def load_gapmind_data(gapmind_data_dir: Path) -> pd.DataFrame:
    """Load GapMind feature data for all phenotypes.

    Parameters
    ----------
    gapmind_data_dir : Path
        Directory containing GapMind feature files.

    Returns
    -------
    pd.DataFrame
        Combined GapMind feature data.
    """
    phenotype_dict = {
        "alanine": "Alanine",
        "arginine": "Arginine",
        "histidine": "Histidine",
        "serine": "Serine",
        "fructose": "Fructose",
        "galactose": "Galactose",
        "glucose": "Glucose",
        "maltose": "Maltose",
        "mannose": "Mannose",
        "sucrose": "Sucrose",
        "myoinositol": "m-Inositol",
        "mannitol": "Mannitol",
        "glycerol": "Glycerol",
        "galacturonate": "Galacturonic-Acid",
        "cellobiose": "Cellobiose",
    }
    phenotype_names = list(phenotype_dict.values())

    gapmind_data_dict = {}
    for phenotype_name in phenotype_names:
        gapmind_data_file = gapmind_data_dir / f"{phenotype_name}.tsv"
        if not gapmind_data_file.exists():
            continue
        gapmind_data = pd.read_csv(
            gapmind_data_file, sep="\t", index_col=0, dtype={"genomeID": str}
        )
        gapmind_data.columns = [
            f"{phenotype_name}-{col}" for col in gapmind_data.columns
        ]
        gapmind_data_dict[phenotype_name] = gapmind_data

    return pd.concat(gapmind_data_dict.values(), axis=1)


def load_gapmind_predictions(phenotype_dict: dict[str, str]) -> pd.DataFrame:
    """Load GapMind predictions.

    Parameters
    ----------
    phenotype_dict : dict[str, str]
        Mapping of phenotype keys to display names.

    Returns
    -------
    pd.DataFrame
        GapMind predictions (0/1 encoded).
    """
    import json

    from scripts.io import index_format_func

    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    if marine_ids_file.exists():
        with open(marine_ids_file, "r") as f:
            marine_ids_map = {v.rsplit("_", 2)[0]: k for k, v in json.load(f).items()}
    else:
        marine_ids_map = {}

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict.keys()]
    datasets = ["s__at-leaf-lit-pmi", "s__marine-seqs"]
    gapmind_data_list = [
        pd.read_csv(f"data/processed/gapmind/heatmap_csvs/{dataset}_categories.csv")
        for dataset in datasets
    ]
    gapmind_data = pd.concat(gapmind_data_list, axis=0)
    gapmind_data["genomeId"] = (
        gapmind_data["genome_id"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    gapmind_data.index = gapmind_data["genomeId"]
    gapmind_data.index = [marine_ids_map.get(ind, ind) for ind in gapmind_data.index]
    gapmind_data = gapmind_data.loc[:, gapmind_phenotype_subset]
    gapmind_data.columns = gapmind_data.columns.str.replace("Carbon__", "")
    gapmind_data.columns = gapmind_data.columns.map(phenotype_dict)

    replace_dict = {
        "complete": 1,
        "likely_complete": 1,
        "steps_missing_medium": 0,
        "steps_missing_low": 0,
        "steps_missing_high": 0,
        "incomplete": 0,
        "not_present": 0,
    }
    gapmind_data_replaced = gapmind_data.replace(replace_dict)
    return gapmind_data_replaced.astype(np.uint8)


def load_experimental_phenotypes(phenotype_dict: dict[str, str]) -> pd.DataFrame:
    """Load experimental phenotype data.

    Parameters
    ----------
    phenotype_dict : dict[str, str]
        Mapping of phenotype keys to display names.

    Returns
    -------
    pd.DataFrame
        Combined experimental phenotype data.
    """
    from scripts.io import read_phenotypes

    phenotype_files_all = Path("data/processed/phenotypes").glob("**/*.tsv")
    phenotype_files = []
    dataset_phenotype_name_map = defaultdict(set)
    for phenotype_file in phenotype_files_all:
        if phenotype_file.parent.stem in ["bacdive", "combined_phenotypes"]:
            continue
        dataset = phenotype_file.parent.stem
        phenotype_files.append(phenotype_file)
        phenotype_data = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})
        dataset_phenotype_name_map[dataset].add(phenotype_file.stem)

    common_phenotypes = sorted(set.intersection(*dataset_phenotype_name_map.values()))
    phenotype_files = [p for p in phenotype_files if p.stem in common_phenotypes]
    phenotype_set = read_phenotypes(phenotype_files)

    phenotypes_combined_dict = dict()
    for phenotype in phenotype_set:
        phenotype_name = phenotype.pindex.name
        if phenotype_name not in phenotypes_combined_dict:
            phenotypes_combined_dict[phenotype_name] = phenotype.phenotype_data.dropna()
        else:
            phenotypes_combined_dict[phenotype_name] = pd.concat(
                [phenotypes_combined_dict[phenotype_name], phenotype.phenotype_data],
                axis=0,
            )

    for phenotype_name, phenotype_data in phenotypes_combined_dict.items():
        phenotypes_combined_dict[phenotype_name] = phenotype_data[
            ~phenotype_data.index.duplicated(keep="first")
        ]

    return pd.concat(phenotypes_combined_dict.values(), axis=1)


def get_genomeid_dataset_map() -> dict[str, str]:
    """Get mapping of genome IDs to datasets.

    Returns
    -------
    dict[str, str]
        Mapping of genome ID to dataset name.
    """
    from scripts.io import read_phenotypes

    phenotype_files_all = Path("data/processed/phenotypes").glob("**/*.tsv")
    dataset_genome_name_map = defaultdict(set)

    for phenotype_file in phenotype_files_all:
        if phenotype_file.parent.stem in ["bacdive", "combined_phenotypes"]:
            continue
        dataset = phenotype_file.parent.stem
        phenotype_data = pd.read_csv(phenotype_file, sep="\t", dtype={"genomeID": str})
        dataset_genome_name_map[dataset].update(phenotype_data["genomeID"].unique())

    genomeid_dataset_map = dict()
    for dataset, genome_ids in dataset_genome_name_map.items():
        for gid in genome_ids:
            genomeid_dataset_map[gid] = dataset

    return genomeid_dataset_map


def identify_microbe_categories(
    phenotypes_combined: pd.DataFrame, gapmind_data_pheno: pd.DataFrame
) -> tuple[list[str], list[str], list[str]]:
    """Identify microbes in three categories.

    Parameters
    ----------
    phenotypes_combined : pd.DataFrame
        Experimental phenotype data.
    gapmind_data_pheno : pd.DataFrame
        GapMind predictions.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists of genome IDs for: (1) no exp growth but GapMind predicts growth,
        (2) all exp growth but GapMind incomplete, (3) top 20 most misclassified.
    """
    # Category 1: No experimental growth but GapMind predicts growth
    microbes_no_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 0).all(), axis=1)
    ].to_list()

    microbes_gapmind_predicts_growth = []
    for microbe in microbes_no_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if (gapmind_data_pheno.loc[microbe] == 1).any():
                microbes_gapmind_predicts_growth.append(microbe)

    # Category 2: All experimental growth but GapMind incomplete
    microbes_all_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 1).all(), axis=1)
    ].to_list()

    microbes_gapmind_missing_predictions = []
    for microbe in microbes_all_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if not (gapmind_data_pheno.loc[microbe] == 1).all():
                microbes_gapmind_missing_predictions.append(microbe)

    # Category 3: Top 20 most frequently misclassified
    misclassifications = dict()
    phenotype_names = phenotypes_combined.columns

    for phenotype_name in phenotype_names:
        exp_data = phenotypes_combined.loc[:, phenotype_name].dropna().astype(np.uint8)
        gapmind_data_pheno_subset = (
            gapmind_data_pheno.loc[:, phenotype_name].dropna().astype(np.uint8)
        )
        common_inds = exp_data.index.intersection(gapmind_data_pheno_subset.index)
        exp_data = exp_data.loc[common_inds]
        gapmind_data_pheno_subset = gapmind_data_pheno_subset.loc[common_inds]
        misclassified = exp_data[exp_data != gapmind_data_pheno_subset]
        misclassifications[phenotype_name] = misclassified

    missclassified_genomes = []
    for phenotype_name, misclassified in misclassifications.items():
        missclassified_genomes.extend(misclassified.index.unique().tolist())

    missclassified_counts = Counter(missclassified_genomes)
    most_common_misclassified = missclassified_counts.most_common(20)
    top_20_genomes = [genome_id for genome_id, _ in most_common_misclassified]

    return (
        microbes_gapmind_predicts_growth,
        microbes_gapmind_missing_predictions,
        top_20_genomes,
    )


def create_misclassification_plots(
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    gapmind_data_dir: Path | None = None,
) -> None:
    """Create misclassification plots on provided axes.

    Parameters
    ----------
    ax1 : plt.Axes
        Axes for "No Growth but GapMind predicts growth" plot.
    ax2 : plt.Axes
        Axes for "All Growth but GapMind incomplete" plot.
    ax3 : plt.Axes
        Axes for "Top 20 misclassified genomes" plot.
    gapmind_data_dir : Path | None
        Directory containing GapMind feature files. If None, uses default path.
    """
    if gapmind_data_dir is None:
        gapmind_data_dir = Path("data/results/new_outline/gapmind_features/all")

    # Load data
    phenotype_dict = {
        "alanine": "Alanine",
        "arginine": "Arginine",
        "histidine": "Histidine",
        "serine": "Serine",
        "fructose": "Fructose",
        "galactose": "Galactose",
        "glucose": "Glucose",
        "maltose": "Maltose",
        "mannose": "Mannose",
        "sucrose": "Sucrose",
        "myoinositol": "m-Inositol",
        "mannitol": "Mannitol",
        "glycerol": "Glycerol",
        "galacturonate": "Galacturonic-Acid",
        "cellobiose": "Cellobiose",
    }

    print("Loading GapMind predictions...")
    gapmind_data_pheno = load_gapmind_predictions(phenotype_dict)

    print("Loading experimental phenotypes...")
    phenotypes_combined = load_experimental_phenotypes(phenotype_dict)

    print("Getting dataset mapping...")
    genomeid_dataset_map = get_genomeid_dataset_map()

    print("Identifying microbe categories...")
    cat1_microbes, cat2_microbes, cat3_microbes = identify_microbe_categories(
        phenotypes_combined, gapmind_data_pheno
    )

    # Count microbes by dataset for each category
    def count_by_dataset(microbe_list: list[str]) -> dict[str, int]:
        counts = defaultdict(int)
        for microbe in microbe_list:
            dataset = genomeid_dataset_map.get(microbe, "unknown")
            counts[dataset] += 1
        return dict(counts)

    cat1_counts = count_by_dataset(cat1_microbes)
    cat2_counts = count_by_dataset(cat2_microbes)

    # Get misclassification counts for top 20 genomes
    misclassifications = dict()
    phenotype_names = phenotypes_combined.columns

    for phenotype_name in phenotype_names:
        exp_data = phenotypes_combined.loc[:, phenotype_name].dropna().astype(np.uint8)
        gapmind_data_pheno_subset = (
            gapmind_data_pheno.loc[:, phenotype_name].dropna().astype(np.uint8)
        )
        common_inds = exp_data.index.intersection(gapmind_data_pheno_subset.index)
        exp_data = exp_data.loc[common_inds]
        gapmind_data_pheno_subset = gapmind_data_pheno_subset.loc[common_inds]
        misclassified = exp_data[exp_data != gapmind_data_pheno_subset]
        misclassifications[phenotype_name] = misclassified

    missclassified_genomes = []
    for phenotype_name, misclassified in misclassifications.items():
        missclassified_genomes.extend(misclassified.index.unique().tolist())

    missclassified_counts = Counter(missclassified_genomes)

    datasets = ["atleaf", "lit", "pmi", "marine"]
    dataset_colors = get_dataset_colors()

    # Subplot 1: No growth but GapMind predicts growth
    cat1_data = pd.DataFrame({"count": [cat1_counts.get(d, 0) for d in datasets]})
    bars1 = ax1.bar(
        range(len(datasets)),
        cat1_data["count"],
        width=0.4,
        color=[dataset_colors[d] for d in datasets],
        alpha=0.7,
        zorder=2,
    )
    ax1.set_ylabel("Number of Microbes", fontsize=12)
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_title(
        "No Experimental Growth but GapMind Predicts Growth",
        fontsize=12,
        pad=10,
    )
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(format_dataset_names(datasets), rotation=45, ha="right")
    ax1.set_ylim(
        0, max(cat1_data["count"]) * 1.1 if cat1_data["count"].max() > 0 else 1
    )

    # Subplot 2: All growth but GapMind incomplete
    cat2_data = pd.DataFrame({"count": [cat2_counts.get(d, 0) for d in datasets]})
    bars2 = ax2.bar(
        range(len(datasets)),
        cat2_data["count"],
        width=0.4,
        color=[dataset_colors[d] for d in datasets],
        alpha=0.7,
        zorder=2,
    )
    ax2.set_ylabel("Number of Microbes", fontsize=12)
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_title(
        "Growth on All C Sources but GapMind Incomplete",
        fontsize=12,
        pad=10,
    )
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(format_dataset_names(datasets), rotation=45, ha="right")
    ax2.set_ylim(
        0, max(cat2_data["count"]) * 1.1 if cat2_data["count"].max() > 0 else 1
    )

    # Subplot 3: Top 20 most frequently misclassified genomes
    # Get category membership for each genome
    cat1_set = set(cat1_microbes)
    cat2_set = set(cat2_microbes)

    # Prepare data for plotting
    top_20_data = []
    for genome_id, count in missclassified_counts.most_common(20):
        # Determine category (excluding "both")
        in_cat1 = genome_id in cat1_set
        in_cat2 = genome_id in cat2_set

        if in_cat1:
            category = "No Growth"
            color = "#e377c2"  # Pink
        elif in_cat2:
            category = "All Growth"
            color = "#17becf"  # Cyan
        else:
            category = "Neither"
            color = "#7f7f7f"  # Gray

        top_20_data.append(
            {
                "genome_id": genome_id,
                "count": count,
                "category": category,
                "color": color,
            }
        )

    top_20_df = pd.DataFrame(top_20_data)

    # Create horizontal bar plot (vertical layout)
    bars3 = ax3.barh(
        range(len(top_20_df)),
        top_20_df["count"],
        height=0.6,
        color=top_20_df["color"],
        alpha=0.7,
        zorder=2,
    )
    ax3.set_xlabel("Number of Misclassifications", fontsize=12)
    ax3.set_ylabel("Microbe ID", fontsize=12)
    # ax3.set_title(
    #     "Top 20 Most Frequently Misclassified Microbes",
    #     fontsize=12,
    #     pad=20,
    # )
    ax3.set_yticks(range(len(top_20_df)))
    # Shorten genome IDs for readability - use only first part
    short_labels = [str(gid).split("_")[0] for gid in top_20_df["genome_id"]]
    ax3.set_yticklabels(short_labels, fontsize=8)
    ax3.invert_yaxis()  # Highest misclassified at top

    # Adjust tick parameters to prevent label overlap
    ax3.tick_params(axis='y', which='major', pad=2)

    # Set x-axis limit to make bars shorter
    max_count = top_20_df["count"].max()
    ax3.set_xlim(0, max_count * 1.1)

    # Add legend for categories at the top in one line
    from matplotlib.patches import Patch

    category_handles = [
        Patch(facecolor="#e377c2", alpha=0.7, label="No Growth"),
        Patch(facecolor="#17becf", alpha=0.7, label="All Growth"),
        Patch(facecolor="#7f7f7f", alpha=0.7, label="Neither"),
    ]
    ax3.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    # Calculate overlaps for summary
    cat1_set = set(cat1_microbes)
    cat2_set = set(cat2_microbes)
    cat3_set = set(cat3_microbes)

    overlap_12 = cat1_set.intersection(cat2_set)
    overlap_13 = cat1_set.intersection(cat3_set)
    overlap_23 = cat2_set.intersection(cat3_set)
    overlap_123 = cat1_set.intersection(cat2_set).intersection(cat3_set)

    # Print summary
    print(f"\n=== Figure 6A Summary ===")
    print(f"Category 1 (No growth, GM predicts growth): {len(cat1_microbes)} genomes")
    print(f"Category 2 (All growth, GM incomplete): {len(cat2_microbes)} genomes")
    print(f"Category 3 (Top 20 misclassified): {len(cat3_microbes)} genomes")
    print(f"\nOverlap 1-2: {len(overlap_12)} genomes")
    print(f"Overlap 1-3: {len(overlap_13)} genomes")
    print(f"Overlap 2-3: {len(overlap_23)} genomes")
    print(f"Overlap 1-2-3: {len(overlap_123)} genomes")


def create_figure6a(
    output_file: Path,
    gapmind_data_dir: Path | None = None,
) -> None:
    """Create standalone Figure 6A showing misclassification patterns.

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    gapmind_data_dir : Path | None
        Directory containing GapMind feature files. If None, uses default path.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right

    # Create a nested gridspec for the bottom row to center and narrow the third plot
    bottom_gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[0.2, 1, 0.2],
        wspace=0,
        left=gs[1, :].get_position(fig).x0,
        right=gs[1, :].get_position(fig).x1,
        bottom=gs[1, :].get_position(fig).y0,
        top=gs[1, :].get_position(fig).y1,
    )
    ax3 = fig.add_subplot(bottom_gs[0, 1])  # Center column only

    create_misclassification_plots(ax1, ax2, ax3, gapmind_data_dir)

    gs.tight_layout(fig)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure6a.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure6a(output_file)
