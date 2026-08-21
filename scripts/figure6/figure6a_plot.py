#!/usr/bin/env python3
"""Panel drawing functions for Figure 6A (problematic genomes for GapMind).

Compares the experimental phenotypes against the GapMind carbon-source calls to
count, per dataset, the genomes that never grow yet are predicted to grow and
the genomes that always grow yet lack a complete pathway, and to rank the twenty
genomes GapMind misclassifies most often.
"""

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers matplotlib styles)
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

    gapmind_phenotype_subset = [f"Carbon__{p}" for p in phenotype_dict]
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
    gapmind_data_replaced = gapmind_data.apply(lambda column: column.map(replace_dict))
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
    microbes_no_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 0).all(), axis=1)
    ].to_list()

    microbes_gapmind_predicts_growth = []
    for microbe in microbes_no_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if (gapmind_data_pheno.loc[microbe] == 1).any():
                microbes_gapmind_predicts_growth.append(microbe)

    microbes_all_exp_growth = phenotypes_combined.index[
        phenotypes_combined.apply(lambda x: (x.dropna() == 1).all(), axis=1)
    ].to_list()

    microbes_gapmind_missing_predictions = []
    for microbe in microbes_all_exp_growth:
        if microbe in gapmind_data_pheno.index:
            if not (gapmind_data_pheno.loc[microbe] == 1).all():
                microbes_gapmind_missing_predictions.append(microbe)

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


def plot_microbe_misclassification_ranking(
    ax: plt.Axes,
    gapmind_data_dir: Path | None = None,
) -> None:
    """Plot the top-20 most frequently misclassified genomes as a ranked bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the ranked horizontal bar chart.
    gapmind_data_dir : Path | None
        Directory containing GapMind feature files. If ``None``, uses the
        default path.
    """
    if gapmind_data_dir is None:
        gapmind_data_dir = Path("data/processed/gapmind_features/all")

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

    gapmind_data_pheno = load_gapmind_predictions(phenotype_dict)
    phenotypes_combined = load_experimental_phenotypes(phenotype_dict)
    cat1_microbes, cat2_microbes, _ = identify_microbe_categories(
        phenotypes_combined, gapmind_data_pheno
    )

    misclassifications: dict[str, pd.Series] = dict()
    for phenotype_name in phenotypes_combined.columns:
        exp_data = phenotypes_combined.loc[:, phenotype_name].dropna().astype(np.uint8)
        gapmind_data_pheno_subset = (
            gapmind_data_pheno.loc[:, phenotype_name].dropna().astype(np.uint8)
        )
        common_inds = exp_data.index.intersection(gapmind_data_pheno_subset.index)
        exp_data = exp_data.loc[common_inds]
        gapmind_data_pheno_subset = gapmind_data_pheno_subset.loc[common_inds]
        misclassified = exp_data[exp_data != gapmind_data_pheno_subset]
        misclassifications[phenotype_name] = misclassified

    missclassified_genomes: list[str] = []
    for misclassified in misclassifications.values():
        missclassified_genomes.extend(misclassified.index.unique().tolist())
    missclassified_counts = Counter(missclassified_genomes)

    cat1_set = set(cat1_microbes)
    cat2_set = set(cat2_microbes)

    top_20_data: list[dict[str, object]] = []
    for genome_id, count in missclassified_counts.most_common(20):
        if genome_id in cat1_set:
            color = "#8C2155"
        elif genome_id in cat2_set:
            color = "#17becf"
        else:
            color = "#7f7f7f"
        top_20_data.append({"genome_id": genome_id, "count": count, "color": color})

    top_20_df = pd.DataFrame(top_20_data)

    y_positions = np.arange(len(top_20_df))
    ax.barh(
        y_positions,
        top_20_df["count"],
        height=0.5,
        color=top_20_df["color"],
        alpha=0.75,
        align="center",
        zorder=2,
    )
    ax.set_xlabel("Number of misclassifications")
    ax.set_ylabel("Microbe ID", labelpad=8)
    ax.set_yticks(y_positions)
    # Assembly accessions need the numeric field to stay distinguishable; other
    # identifiers are unique before the first underscore.
    short_labels = [
        "_".join(str(gid).split("_")[:2])
        if str(gid).startswith(("GCF_", "GCA_"))
        else str(gid).split("_")[0]
        for gid in top_20_df["genome_id"]
    ]
    ax.set_yticklabels(short_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_ylim(float(len(top_20_df) - 0.5), -0.5)
    ax.tick_params(axis="y", which="major", pad=3)
    ax.set_xlim(0, float(top_20_df["count"].max()) * 1.1)
    ax.grid(axis="x", alpha=0.15, linewidth=0.6)

    from matplotlib.patches import Patch

    category_handles = [
        Patch(facecolor="#8C2155", alpha=0.75, label="No growth"),
        Patch(facecolor="#17becf", alpha=0.75, label="Universal growth"),
        Patch(facecolor="#7f7f7f", alpha=0.75, label="Neither"),
    ]
    ax.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        fontsize=8,
    )


def get_problematic_sample_summary(
    gapmind_data_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Summarize the two problematic-sample categories shown in Figure 6A.

    Parameters
    ----------
    gapmind_data_dir : Path | None
        Directory containing GapMind feature files. If None, uses the default
        path.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Per-dataset counts and totals for the two categories, and the category
        totals and overlap as metadata.
    """
    if gapmind_data_dir is None:
        gapmind_data_dir = Path("data/processed/gapmind_features/all")

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
    datasets = ["atleaf", "lit", "pmi", "marine"]

    gapmind_data_pheno = load_gapmind_predictions(phenotype_dict)
    phenotypes_combined = load_experimental_phenotypes(phenotype_dict)
    genomeid_dataset_map = get_genomeid_dataset_map()
    cat1_microbes, cat2_microbes, _ = identify_microbe_categories(
        phenotypes_combined, gapmind_data_pheno
    )

    def count_by_dataset(microbe_list: list[str]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for microbe in microbe_list:
            dataset = genomeid_dataset_map.get(microbe, "unknown")
            counts[dataset] += 1
        return counts

    category_counts = {
        "No growth,\nGM predicts": count_by_dataset(cat1_microbes),
        "Universal growth,\nGM incomplete": count_by_dataset(cat2_microbes),
    }
    # A category that has no genomes from a given dataset yields NaN, which would
    # otherwise poison the stacked-bar offsets and drop later segments.
    summary_df = (
        pd.DataFrame(category_counts)
        .T.reindex(columns=datasets, fill_value=0)
        .fillna(0)
    )
    summary_df["total"] = summary_df.sum(axis=1)

    metadata = {
        "category_1_total": len(cat1_microbes),
        "category_2_total": len(cat2_microbes),
        "category_overlap": len(set(cat1_microbes).intersection(cat2_microbes)),
    }
    return summary_df, metadata


def plot_problematic_sample_summary(
    ax: plt.Axes,
    gapmind_data_dir: Path | None = None,
) -> None:
    """Plot the problematic-sample summary as a per-dataset stacked bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the summary.
    gapmind_data_dir : Path | None
        Directory containing GapMind feature files. If None, uses the default
        path.
    """
    summary_df, _ = get_problematic_sample_summary(gapmind_data_dir)
    datasets = ["atleaf", "lit", "pmi", "marine"]
    dataset_colors = get_dataset_colors()

    y_positions = np.arange(len(summary_df))
    left = np.zeros(len(summary_df))
    for dataset in datasets:
        values = summary_df[dataset].to_numpy(dtype=float)
        ax.barh(
            y_positions,
            values,
            left=left,
            height=0.38,
            color=dataset_colors[dataset],
            edgecolor="white",
            linewidth=0.8,
            label=format_dataset_names([dataset])[0],
        )
        left += values

    for y_position, total in zip(y_positions, summary_df["total"], strict=True):
        ax.text(
            float(total) + 0.8,
            y_position,
            f"{int(total)}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of genomes")
    ax.set_xlim(0, float(summary_df["total"].max()) * 1.18)
    ax.grid(axis="x", alpha=0.15, linewidth=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        frameon=False,
        ncol=4,
        fontsize=8,
    )
