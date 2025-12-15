#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


def get_ko_description(ko_id: str, ko_dict: dict) -> str:
    """Get description for a KO ID.

    Parameters
    ----------
    ko_id : str
        KO identifier (e.g., 'K01712').
    ko_dict : dict
        Dictionary mapping KO IDs to their descriptions.

    Returns
    -------
    str
        KO description with ID, or just the ID if not found.
    """
    ko_key = f"KO:{ko_id}"
    if ko_key in ko_dict["term_hash"]:
        name = ko_dict["term_hash"][ko_key]["name"]
        return f"{ko_id}: {name}"
    return ko_id


def create_feature_table(
    phenotypes: list[str],
    output_file: Path,
) -> None:
    """Create LaTeX table showing common and unique features for each dataset (concordant samples).

    Parameters
    ----------
    phenotypes : list[str]
        List of phenotypes to include in the table.
    output_file : Path
        Output path for the LaTeX table file.
    """
    # Load data from Figure 5B outputs
    comparison_file = Path(
        "data/outputs/figure5/figure5b_feature_comparison_summary.csv"
    )
    ko_dict_file = Path("data/external/mapping/KO_dictionary.json")

    comparison_df = pd.read_csv(comparison_file)

    with open(ko_dict_file) as f:
        ko_dict = json.load(f)

    datasets = ["atleaf", "lit", "marine"]

    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{|l|l|p{4cm}|p{4cm}|}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "\\textbf{Phenotype} & \\textbf{Dataset} & "
        "\\textbf{Common Features} & \\textbf{Unique Features} \\\\"
    )
    latex_lines.append("\\hline")

    # Process each phenotype
    for phenotype in phenotypes:
        first_row = True

        for dataset in datasets:
            matching_rows = comparison_df[
                (comparison_df["phenotype"] == phenotype)
                & (comparison_df["test_dataset"] == dataset)
            ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                # Get intersection features (common between combined and individual)
                intersection_str = row["intersection"]
                if pd.notna(intersection_str) and intersection_str:
                    intersection_kos = intersection_str.split(";")
                    common_desc = [
                        get_ko_description(ko, ko_dict) for ko in intersection_kos
                    ]
                    common_str = " \\newline ".join(common_desc)
                else:
                    common_str = "None"

                # Get unique features (only in individual dataset)
                unique_str = row["unique_to_individual"]
                if pd.notna(unique_str) and unique_str:
                    unique_kos = unique_str.split(";")
                    unique_desc = [get_ko_description(ko, ko_dict) for ko in unique_kos]
                    unique_str = " \\newline ".join(unique_desc)
                else:
                    unique_str = "None"

                # Format dataset name
                dataset_display = {
                    "atleaf": "ATLeaf",
                    "lit": "Literature",
                    "marine": "Marine",
                }[dataset]

                # Add row to table
                if first_row:
                    latex_lines.append(
                        f"\\multirow{{3}}{{*}}{{{phenotype}}} & {dataset_display} & {common_str} & {unique_str} \\\\"
                    )
                    first_row = False
                else:
                    latex_lines.append(
                        f" & {dataset_display} & {common_str} & {unique_str} \\\\"
                    )

        latex_lines.append("\\hline")

    # Close table
    latex_lines.append("\\end{tabular}")
    latex_lines.append(
        "\\caption{Stable features identified by machine learning models for selected phenotypes "
        "when trained on GapMind-concordant samples only. \\textbf{Common Features} are those "
        "appearing in both the combined (all datasets) and individual dataset models. "
        "\\textbf{Unique Features} are those appearing only in the individual dataset model. "
        "Features shown are consistent across multiple random seeds (appearing in $\\geq$70\\% "
        "of training runs).}"
    )
    latex_lines.append("\\label{tab:feature_comparison_concordant}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table written to {output_file}")


if __name__ == "__main__":
    # phenotypes = ["Histidine", "Galactose"]
    phenotypes = ["Histidine"]
    output_file = Path("sections/table_feature_comparison_concordant.tex")
    create_feature_table(phenotypes, output_file)
