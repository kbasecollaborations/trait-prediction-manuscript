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
    """Create LaTeX table showing common and unique features across datasets.

    Parameters
    ----------
    phenotypes : list[str]
        List of phenotypes to include in the table.
    output_file : Path
        Output path for the LaTeX table file.
    """
    # Load data
    combined_file = Path("data/outputs/figure4/all_datasets_combined_shap_features.json")
    comparison_file = Path("data/outputs/figure4/feature_comparison_summary.csv")
    ko_dict_file = Path("data/external/mapping/KO_dictionary.json")

    with open(combined_file) as f:
        combined_features = json.load(f)

    comparison_df = pd.read_csv(comparison_file)

    with open(ko_dict_file) as f:
        ko_dict = json.load(f)

    datasets = ["atleaf", "lit", "marine"]

    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{|l|p{3.5cm}|p{3.5cm}|p{3.5cm}|p{3.5cm}|}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "\\textbf{Phenotype} & \\textbf{Common Features} & "
        "\\multicolumn{3}{c|}{\\textbf{Unique Features}} \\\\"
    )
    latex_lines.append("\\cline{3-5}")
    latex_lines.append(
        " & & \\textbf{ATLeaf} & \\textbf{Literature} & \\textbf{Marine} \\\\"
    )
    latex_lines.append("\\hline")

    # Process each phenotype
    for phenotype in phenotypes:
        # Get intersection features for each dataset
        intersection_features = {ds: [] for ds in datasets}
        unique_features = {ds: [] for ds in datasets}

        for dataset in datasets:
            matching_rows = comparison_df[
                (comparison_df["phenotype"] == phenotype)
                & (comparison_df["test_dataset"] == dataset)
            ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                # Get intersection features
                intersection_str = row["intersection"]
                if pd.notna(intersection_str) and intersection_str:
                    intersection_kos = intersection_str.split(";")
                    intersection_features[dataset] = intersection_kos

                # Get unique features
                unique_str = row["unique_to_individual"]
                if pd.notna(unique_str) and unique_str:
                    unique_kos = unique_str.split(";")
                    unique_features[dataset] = unique_kos

        # Common features = intersection of all three datasets' intersection sets
        if intersection_features["atleaf"] and intersection_features["lit"] and intersection_features["marine"]:
            common_kos = set(intersection_features["atleaf"]) & set(intersection_features["lit"]) & set(intersection_features["marine"])
        else:
            common_kos = set()

        common_desc = [get_ko_description(ko, ko_dict) for ko in sorted(common_kos)]
        common_str = " \\newline ".join(common_desc) if common_desc else "None"

        # Format unique features for each dataset (each on a new line)
        atleaf_desc = [get_ko_description(ko, ko_dict) for ko in unique_features["atleaf"]]
        atleaf_str = " \\newline ".join(atleaf_desc) if atleaf_desc else "None"

        lit_desc = [get_ko_description(ko, ko_dict) for ko in unique_features["lit"]]
        lit_str = " \\newline ".join(lit_desc) if lit_desc else "None"

        marine_desc = [get_ko_description(ko, ko_dict) for ko in unique_features["marine"]]
        marine_str = " \\newline ".join(marine_desc) if marine_desc else "None"

        # Add row to table
        latex_lines.append(
            f"{phenotype} & {common_str} & {atleaf_str} & {lit_str} & {marine_str} \\\\"
        )
        latex_lines.append("\\hline")

    # Close table
    latex_lines.append("\\end{tabular}")
    latex_lines.append(
        "\\caption{Common and unique stable features identified by machine learning "
        "models for selected phenotypes. Common features are those identified across "
        "all datasets combined, while unique features are specific to individual datasets.}"
    )
    latex_lines.append("\\label{tab:feature_comparison}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table written to {output_file}")


if __name__ == "__main__":
    phenotypes = ["Histidine", "Galactose"]
    output_file = Path("sections/table_feature_comparison.tex")
    create_feature_table(phenotypes, output_file)
