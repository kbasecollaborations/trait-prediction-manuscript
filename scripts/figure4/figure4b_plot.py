#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from scripts.visualization import (
    configure_plot_style,
    format_dataset_names,
    get_dataset_colors,
)
from scripts.figure4.style import (
    AXIS_LABEL_SIZE,
    LEGEND_FONT_SIZE,
    TICK_LABEL_SIZE,
)

plt.style.use(["science", "nature"])
configure_plot_style()

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


def calculate_confusion_matrix_by_dataset(
    phenotypes_combined: pd.DataFrame,
    gapmind_data_pheno: pd.DataFrame,
    genomeid_dataset_map: dict[str, str],
    phenotype_dict: dict[str, str],
) -> pd.DataFrame:
    """Calculate TP, TN, FP, FN for each phenotype and dataset.

    Parameters
    ----------
    phenotypes_combined : pd.DataFrame
        Experimental phenotype data.
    gapmind_data_pheno : pd.DataFrame
        GapMind predictions.
    genomeid_dataset_map : dict[str, str]
        Mapping of genome IDs to datasets.
    phenotype_dict : dict[str, str]
        Mapping of phenotype keys to display names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Dataset, Phenotype, TP, TN, FP, FN.
    """
    dataset_performance = {}

    for dataset in ["atleaf", "lit", "pmi", "marine"]:
        dataset_performance[dataset] = {}

        for phenotype_name in phenotype_dict.values():
            dataset_genome_ids = [
                gid for gid, d in genomeid_dataset_map.items() if d == dataset
            ]

            exp_data = (
                phenotypes_combined.loc[:, phenotype_name].dropna().astype(np.uint8)
            )
            gapmind_data_subset = (
                gapmind_data_pheno.loc[:, phenotype_name].dropna().astype(np.uint8)
            )

            common_inds = exp_data.index.intersection(gapmind_data_subset.index)
            common_inds = [gid for gid in common_inds if gid in dataset_genome_ids]

            if len(common_inds) == 0:
                dataset_performance[dataset][phenotype_name] = {
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                }
                continue

            exp_data_subset = exp_data.loc[common_inds]
            gapmind_data_subset = gapmind_data_subset.loc[common_inds]

            tp = ((exp_data_subset == 1) & (gapmind_data_subset == 1)).sum()
            tn = ((exp_data_subset == 0) & (gapmind_data_subset == 0)).sum()
            fp = ((exp_data_subset == 0) & (gapmind_data_subset == 1)).sum()
            fn = ((exp_data_subset == 1) & (gapmind_data_subset == 0)).sum()

            dataset_performance[dataset][phenotype_name] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }

    performance_data = []
    for dataset in dataset_performance:
        for phenotype in phenotype_dict.values():
            metrics = dataset_performance[dataset][phenotype]
            performance_data.append(
                {
                    "Dataset": dataset,
                    "Phenotype": phenotype,
                    "TP": metrics["TP"],
                    "TN": metrics["TN"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                }
            )

    return pd.DataFrame(performance_data)


def create_confusion_matrix_plots(
    ax1: plt.Axes,
    ax2: plt.Axes,
) -> None:
    """Create confusion matrix plots on provided axes.

    Parameters
    ----------
    ax1 : plt.Axes
        Axes for confusion matrix by phenotype plot (top).
    ax2 : plt.Axes
        Axes for confusion matrix by dataset plot (bottom).
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

    # Colorblind-friendly colors (Wong palette) - semantic mapping
    # Cool colors for correct predictions, warm colors for incorrect
    colors = {
        "TP": "#0072B2",  # Blue (correct)
        "TN": "#009E73",  # Teal (correct)
        "FP": "#E69F00",  # Orange (incorrect)
        "FN": "#D55E00",  # Vermillion (incorrect)
    }

    print("Loading GapMind predictions...")
    gapmind_data_pheno = load_gapmind_predictions(phenotype_dict)

    print("Loading experimental phenotypes...")
    phenotypes_combined = load_experimental_phenotypes(phenotype_dict)

    print("Getting dataset mapping...")
    genomeid_dataset_map = get_genomeid_dataset_map()

    print("Calculating confusion matrix metrics...")
    performance_df = calculate_confusion_matrix_by_dataset(
        phenotypes_combined, gapmind_data_pheno, genomeid_dataset_map, phenotype_dict
    )

    # Top plot: combined performance across all datasets for each phenotype.
    phenotype_combined = performance_df.groupby("Phenotype")[
        ["TP", "TN", "FP", "FN"]
    ].sum()

    phenotypes = phenotype_combined.index.tolist()
    x = np.arange(len(phenotypes))
    width = 0.45

    tp_vals = phenotype_combined["TP"].values
    tn_vals = phenotype_combined["TN"].values
    fp_vals = phenotype_combined["FP"].values
    fn_vals = phenotype_combined["FN"].values

    ax1.bar(x, tp_vals, width, label="TP", color=colors["TP"])
    ax1.bar(x, tn_vals, width, bottom=tp_vals, label="TN", color=colors["TN"])
    ax1.bar(x, fp_vals, width, bottom=tp_vals + tn_vals, label="FP", color=colors["FP"])
    ax1.bar(
        x,
        fn_vals,
        width,
        bottom=tp_vals + tn_vals + fp_vals,
        label="FN",
        color=colors["FN"],
    )

    ax1.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax1.set_xlabel("Phenotype", fontsize=AXIS_LABEL_SIZE, labelpad=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax1.tick_params(axis="x", pad=0.5)
    ax1.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax1.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.4,
        columnspacing=1.0,
    )
    ax1.grid(axis="y", alpha=0.3)

    # Bottom plot: combined performance across all phenotypes for each dataset.
    dataset_combined = performance_df.groupby("Dataset")[["TP", "TN", "FP", "FN"]].sum()

    datasets = dataset_combined.index.tolist()
    x = np.arange(len(datasets))
    width = 0.35  # Narrower bars for bottom subplot

    tp_vals = dataset_combined["TP"].values
    tn_vals = dataset_combined["TN"].values
    fp_vals = dataset_combined["FP"].values
    fn_vals = dataset_combined["FN"].values

    ax2.bar(x, tp_vals, width, label="TP", color=colors["TP"])
    ax2.bar(x, tn_vals, width, bottom=tp_vals, label="TN", color=colors["TN"])
    ax2.bar(x, fp_vals, width, bottom=tp_vals + tn_vals, label="FP", color=colors["FP"])
    ax2.bar(
        x,
        fn_vals,
        width,
        bottom=tp_vals + tn_vals + fp_vals,
        label="FN",
        color=colors["FN"],
    )

    ax2.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax2.set_xlabel("Dataset", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(format_dataset_names(datasets), fontsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="x", pad=1)
    ax2.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax2.grid(axis="y", alpha=0.3)

    print("\n=== Figure 4B Summary ===")
    print(f"Total TP: {phenotype_combined['TP'].sum()}")
    print(f"Total TN: {phenotype_combined['TN'].sum()}")
    print(f"Total FP: {phenotype_combined['FP'].sum()}")
    print(f"Total FN: {phenotype_combined['FN'].sum()}")


def create_figure4b(output_file: Path) -> None:
    """Create standalone Figure 4B showing confusion matrix plots.

    Parameters
    ----------
    output_file : Path
        Path to save the output figure.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 8.5))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1], hspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    create_confusion_matrix_plots(ax1, ax2)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    output_file = Path("figures/figure4b.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_figure4b(output_file)
