#!/usr/bin/env python3
"""Generate the supplementary stable-feature comparison table (histidine).

Writes ``sections/table_feature_comparison.tex``.

Run with::

    uv run python -m scripts.tables.histidine_feature_table
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.tables.kegg_module_coverage import pathway_coverage_line


def get_ko_description(ko_id: str, ko_dict: dict[str, Any]) -> str:
    """Look up the canonical description of a KO identifier.

    Parameters
    ----------
    ko_id : str
        KO identifier (e.g., ``"K01712"``).
    ko_dict : dict[str, Any]
        KO dictionary loaded from ``data/external/mapping/KO_dictionary.json``.

    Returns
    -------
    str
        ``"{ko_id}: {description}"`` if known, otherwise the raw KO identifier.
    """
    ko_key = f"KO:{ko_id}"
    if ko_key in ko_dict["term_hash"]:
        name = ko_dict["term_hash"][ko_key]["name"].split("; ", maxsplit=1)[-1]
        return f"{ko_id}: {name}"
    return ko_id


def _cluster_labels(
    kos: list[str],
    ko_to_cluster: dict[str, int] | None,
) -> dict[int, str]:
    """Assign compact labels to clusters with multiple displayed KOs.

    Parameters
    ----------
    kos : list[str]
        KOs pooled across all held-out-dataset rows in one table column.
    ko_to_cluster : dict[str, int] | None
        Mapping from KO to redundancy-cluster identifier.

    Returns
    -------
    dict[int, str]
        Mapping from qualifying cluster identifiers to display labels.
    """
    if ko_to_cluster is None:
        return {}
    cluster_members: dict[int, set[str]] = defaultdict(set)
    for ko in kos:
        if ko in ko_to_cluster:
            cluster_members[int(ko_to_cluster[ko])].add(ko)
    repeated_clusters = sorted(
        cluster_id
        for cluster_id, members in cluster_members.items()
        if len(members) > 1
    )
    return {
        cluster_id: f"Cluster {chr(ord('A') + index)}"
        for index, cluster_id in enumerate(repeated_clusters)
    }


def _format_clustered_kos(
    kos: list[str],
    ko_to_cluster: dict[str, int] | None,
    cluster_labels: dict[int, str],
    ko_dict: dict[str, Any],
) -> str:
    """Format a list of KOs grouped by their redundancy cluster.

    Parameters
    ----------
    kos : list[str]
        KO identifiers to format.
    ko_to_cluster : dict[str, int] | None
        Mapping from KO to integer cluster identifier for the phenotype.
        ``None`` indicates no clustering metadata available.
    cluster_labels : dict[int, str]
        Labels assigned from KOs pooled across the table column.
    ko_dict : dict[str, Any]
        KO dictionary used to fetch human-readable descriptions.

    Returns
    -------
    str
        LaTeX-ready string with one cluster per ``\\newline`` block. Singletons
        are listed last.
    """
    if not kos:
        return "None"

    if ko_to_cluster is None:
        descriptions = [get_ko_description(ko, ko_dict) for ko in kos]
        return " \\newline ".join(descriptions)

    lines: list[str] = []
    for ko in kos:
        cluster_id = ko_to_cluster.get(ko)
        if cluster_id in cluster_labels:
            lines.append(
                f"[{cluster_labels[int(cluster_id)]}] {get_ko_description(ko, ko_dict)}"
            )
        else:
            lines.append(get_ko_description(ko, ko_dict))

    return " \\newline ".join(lines)


def create_feature_table(
    phenotype: str,
    output_file: Path,
) -> None:
    """Write the LaTeX feature table for full-data and concordant training.

    Parameters
    ----------
    phenotype : str
        Phenotype shown as the worked example.
    output_file : Path
        Destination path for the LaTeX fragment.
    """
    comparison_files = {
        "Full data": Path("data/outputs/figure4/feature_comparison_summary.csv"),
        "Concordant": Path(
            "data/outputs/figure5/figure5b_feature_comparison_summary.csv"
        ),
    }
    ko_dict_file = Path("data/external/mapping/KO_dictionary.json")
    cluster_file = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")

    comparison_dfs = {
        training_set: pd.read_csv(path)
        for training_set, path in comparison_files.items()
    }
    with open(ko_dict_file) as handle:
        ko_dict = json.load(handle)
    cluster_mapping: dict[str, dict[str, int]] = {}
    if cluster_file.exists():
        with open(cluster_file) as handle:
            cluster_mapping = json.load(handle)
    ko_to_cluster = cluster_mapping.get(phenotype)

    displayed_kos: list[str] = []
    for comparison_df in comparison_dfs.values():
        phenotype_rows = comparison_df[comparison_df["phenotype"] == phenotype]
        for column in ("intersection", "unique_to_individual"):
            for value in phenotype_rows[column].dropna():
                displayed_kos.extend(value.split(";"))
    cluster_labels = _cluster_labels(displayed_kos, ko_to_cluster)

    datasets = ["atleaf", "lit", "marine"]
    dataset_display = {"atleaf": "ATLeaf", "lit": "Biolog", "marine": "Marine"}

    latex_lines: list[str] = []
    latex_lines.append("\\begin{landscape}")
    latex_lines.append("\\begin{table}[p]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\begin{tabularx}{\\linewidth}{llXX}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "\\textbf{Training set} & \\textbf{Held-out dataset} & "
        "\\textbf{Shared stable features} & \\textbf{Unique to held-out-alone model} \\\\"
    )
    latex_lines.append("\\hline")

    for training_set, comparison_df in comparison_dfs.items():
        rows: list[tuple[str, list[str], list[str]]] = []
        shared_pool: list[str] = []
        unique_pool: list[str] = []

        for dataset in datasets:
            matching = comparison_df[
                (comparison_df["phenotype"] == phenotype)
                & (comparison_df["test_dataset"] == dataset)
            ]
            if matching.empty:
                continue
            row = matching.iloc[0]

            intersection_raw = row["intersection"]
            intersection_kos = (
                intersection_raw.split(";")
                if pd.notna(intersection_raw) and intersection_raw
                else []
            )
            shared_pool.extend(intersection_kos)

            unique_raw = row["unique_to_individual"]
            unique_kos = (
                unique_raw.split(";") if pd.notna(unique_raw) and unique_raw else []
            )
            unique_pool.extend(unique_kos)
            rows.append((dataset_display[dataset], intersection_kos, unique_kos))

        for index, (display, intersection_kos, unique_kos) in enumerate(rows):
            common_cell = _format_clustered_kos(
                intersection_kos, ko_to_cluster, cluster_labels, ko_dict
            )
            unique_cell = _format_clustered_kos(
                unique_kos, ko_to_cluster, cluster_labels, ko_dict
            )
            if index == 0:
                latex_lines.append(
                    f"\\multirow{{{len(datasets)}}}{{*}}{{{training_set}}} & "
                    f"{display} & {common_cell} & {unique_cell} \\\\"
                )
            else:
                latex_lines.append(f" & {display} & {common_cell} & {unique_cell} \\\\")

        coverage = pathway_coverage_line(
            phenotype, shared_pool, unique_pool, ko_to_cluster
        )
        if coverage is not None:
            latex_lines.append(
                f"\\multicolumn{{4}}{{l}}{{\\footnotesize {coverage}}} \\\\"
            )

        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabularx}")
    latex_lines.append(
        "\\caption{\\textbf{Stable KOFAM features for histidine under full-data "
        "and concordant training.} Shared features occur in both the "
        "three-dataset and held-out-alone models; unique features occur only "
        "in the held-out-alone model. Cluster labels mark SHAP-supervised "
        "redundancy groups, and module coverage counts one representative per "
        "cluster across the three comparisons. Features appeared in at least "
        "70\\% of 20 training runs.}"
    )
    latex_lines.append("\\label{tab:feature_comparison}")
    latex_lines.append("\\end{table}")
    latex_lines.append("\\end{landscape}")

    with open(output_file, "w") as handle:
        handle.write("\n".join(latex_lines) + "\n")

    print(f"LaTeX table written to {output_file}")


if __name__ == "__main__":
    output_file = Path("sections/table_feature_comparison.tex")
    create_feature_table("Histidine", output_file)
