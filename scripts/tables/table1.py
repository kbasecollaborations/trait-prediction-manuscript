#!/usr/bin/env python3
"""Generate the Supplementary Table S2 of stable features for the full-data analysis.

The table is built from ``data/outputs/figure4/feature_comparison_summary.csv``,
which records, for each (phenotype, held-out dataset) pair, the stable feature
lists produced by the model trained on the other three datasets and by the
model trained on the held-out dataset alone. KO members are grouped by their
SHAP-supervised redundancy cluster (computed in ``scripts/feature_clustering``)
so readers can see which features represent the same biological signal.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


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
        name = ko_dict["term_hash"][ko_key]["name"]
        return f"{ko_id}: {name}"
    return ko_id


def _format_clustered_kos(
    kos: list[str],
    ko_to_cluster: dict[str, int] | None,
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
    ko_dict : dict[str, Any]
        KO dictionary used to fetch human-readable descriptions.

    Returns
    -------
    str
        LaTeX-ready string with one cluster per ``\\newline`` block. KOs with no
        cluster sibling (singletons) are listed last under "Singletons".
    """
    if not kos:
        return "None"

    if ko_to_cluster is None:
        descriptions = [get_ko_description(ko, ko_dict) for ko in kos]
        return " \\newline ".join(descriptions)

    cluster_to_members: dict[int, list[str]] = defaultdict(list)
    singletons: list[str] = []
    for ko in kos:
        if ko in ko_to_cluster:
            cluster_to_members[int(ko_to_cluster[ko])].append(ko)
        else:
            singletons.append(ko)

    lines: list[str] = []
    cluster_label_idx = 0
    for cluster_id, members in sorted(cluster_to_members.items()):
        if len(members) > 1:
            cluster_label_idx += 1
            label = f"Cluster {chr(ord('A') + cluster_label_idx - 1)}"
            for ko in members:
                lines.append(f"[{label}] {get_ko_description(ko, ko_dict)}")
        else:
            singletons.extend(members)

    for ko in singletons:
        lines.append(get_ko_description(ko, ko_dict))

    return " \\newline ".join(lines)


def create_feature_table(
    phenotypes: list[str],
    output_file: Path,
) -> None:
    """Write the LaTeX feature-comparison table for the full-data analysis.

    Parameters
    ----------
    phenotypes : list[str]
        Phenotype names whose rows are included in the table.
    output_file : Path
        Destination path for the LaTeX fragment.
    """
    comparison_file = Path("data/outputs/figure4/feature_comparison_summary.csv")
    ko_dict_file = Path("data/external/mapping/KO_dictionary.json")
    cluster_file = Path("data/outputs/clustering/ko_clusters_shap_hclust.json")

    comparison_df = pd.read_csv(comparison_file)
    with open(ko_dict_file) as handle:
        ko_dict = json.load(handle)
    cluster_mapping: dict[str, dict[str, int]] = {}
    if cluster_file.exists():
        with open(cluster_file) as handle:
            cluster_mapping = json.load(handle)

    datasets = ["atleaf", "lit", "marine"]
    dataset_display = {"atleaf": "ATLeaf", "lit": "Biolog", "marine": "Marine"}

    latex_lines: list[str] = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{|l|l|p{4.4cm}|p{4.4cm}|}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "\\textbf{Phenotype} & \\textbf{Held-out dataset} & "
        "\\textbf{Shared stable features} & \\textbf{Unique to held-out-alone model} \\\\"
    )
    latex_lines.append("\\hline")

    for phenotype in phenotypes:
        first_row = True
        ko_to_cluster = cluster_mapping.get(phenotype)

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
            common_cell = _format_clustered_kos(intersection_kos, ko_to_cluster, ko_dict)

            unique_raw = row["unique_to_individual"]
            unique_kos = (
                unique_raw.split(";")
                if pd.notna(unique_raw) and unique_raw
                else []
            )
            unique_cell = _format_clustered_kos(unique_kos, ko_to_cluster, ko_dict)

            display = dataset_display[dataset]
            if first_row:
                latex_lines.append(
                    f"\\multirow{{{len(datasets)}}}{{*}}{{{phenotype}}} & "
                    f"{display} & {common_cell} & {unique_cell} \\\\"
                )
                first_row = False
            else:
                latex_lines.append(
                    f" & {display} & {common_cell} & {unique_cell} \\\\"
                )

        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")
    latex_lines.append(
        "\\caption{Stable KOFAM features identified by machine learning models "
        "for selected phenotypes. \\textbf{Shared stable features} are KOs "
        "present in both the model trained on the three non-held-out datasets "
        "and the model trained on the held-out dataset alone. \\textbf{Unique "
        "to held-out-alone model} are KOs present only when training on the "
        "held-out dataset alone. KOs are bracketed with a cluster label "
        "(``[Cluster A]'', etc.) when they belong to the same SHAP-supervised "
        "redundancy cluster (Methods); features within a cluster represent the "
        "same biological signal even when their KO identifiers differ. Features "
        "shown are consistent across multiple random seeds (appearing in "
        "$\\geq$70\\% of 20 training runs).}"
    )
    latex_lines.append("\\label{tab:feature_comparison}")
    latex_lines.append("\\end{table}")

    with open(output_file, "w") as handle:
        handle.write("\n".join(latex_lines))

    print(f"LaTeX table written to {output_file}")


if __name__ == "__main__":
    phenotypes = ["Histidine"]
    output_file = Path("sections/table_feature_comparison.tex")
    create_feature_table(phenotypes, output_file)
