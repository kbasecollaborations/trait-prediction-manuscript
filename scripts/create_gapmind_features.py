"""Extract GapMind features from .steps files.

This script processes GapMind output files to extract pathway step scores for all
phenotypes, not just a subset. The extracted features are saved at different confidence
thresholds (all, high, medium, low).

GapMind scores:
- 2 = high confidence
- 1 = medium confidence
- 0 = low confidence
- -1 = no score (missing)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.io import index_format_func

warnings.filterwarnings("ignore")


def load_marine_ids_map() -> dict[str, str]:
    """Load marine strain ID mapping.

    Returns
    -------
    dict[str, str]
        Mapping from genome ID to strain ID
    """
    marine_ids_file = Path("data/interim/features/marine/strain_genomeid_map.json")
    with open(marine_ids_file, "r") as f:
        return {v: k for k, v in json.load(f).items()}


def load_gapmind_steps_data() -> pd.DataFrame:
    """Load GapMind steps data from carbon.sum.steps files.

    Returns
    -------
    pd.DataFrame
        Combined GapMind steps data from all datasets
    """
    gapmind_steps_data = []
    gapmind_steps_files = Path(
        "data/processed/gapmind/pmi_gapmind_output_files/pmi_gapmind_output_files/"
    ).glob("**/carbon.sum.steps")

    for gapmind_steps_file in gapmind_steps_files:
        dataset = gapmind_steps_file.parent.name
        df = pd.read_csv(gapmind_steps_file, sep="\t")
        df["dataset"] = dataset
        gapmind_steps_data.append(df)

    return pd.concat(gapmind_steps_data)


def load_organism_data(
    datasets: list[str], marine_ids_map: dict[str, str]
) -> pd.DataFrame:
    """Load organism ID to genome ID mapping.

    Parameters
    ----------
    datasets : list[str]
        List of dataset names
    marine_ids_map : dict[str, str]
        Mapping from genome ID to marine strain ID

    Returns
    -------
    pd.DataFrame
        Organism data with genomeID column
    """
    orgid_data_dict = {
        dataset: pd.read_csv(
            f"data/processed/gapmind/{dataset.removeprefix('s__')}.org",
            sep="\t",
            index_col=0,
        )
        for dataset in datasets
    }

    for dataset in datasets:
        orgid_data_dict[dataset]["dataset"] = dataset

    orgid_data = pd.concat(orgid_data_dict.values(), axis=0)
    orgid_data["genomeID"] = (
        orgid_data["genomeName"]
        .str.split(" ")
        .str[-1]
        .apply(index_format_func)
        .astype(str)
    )
    orgid_data["genomeID"] = [
        marine_ids_map.get(ind, ind) for ind in orgid_data["genomeID"]
    ]

    return orgid_data


def get_phenotype_name_map() -> dict[str, str]:
    """Get mapping from GapMind pathway IDs to standardized phenotype names.

    Returns
    -------
    dict[str, str]
        Mapping from pathway ID to phenotype name
    """
    return {
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


def create_feature_matrices(
    gapmind_steps_raw: pd.DataFrame, orgid_genomeid_map: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Create feature matrices for each phenotype.

    Parameters
    ----------
    gapmind_steps_raw : pd.DataFrame
        Raw GapMind steps data
    orgid_genomeid_map : dict[str, str]
        Mapping from organism ID to genome ID

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping phenotype names to feature matrices
    """
    phenotype_name_map = get_phenotype_name_map()
    feature_matrix_dict = {}

    # Get all unique pathways, excluding the header row
    all_pathways = gapmind_steps_raw["pathway"].unique()
    all_pathways = [p for p in all_pathways if p != "pathway"]

    for pathway_id in all_pathways:
        # Get the standardized phenotype name or use the pathway ID as-is
        phenotype_name = phenotype_name_map.get(pathway_id, pathway_id)

        # Filter data for this pathway
        gapmind_data = gapmind_steps_raw[gapmind_steps_raw["pathway"] == pathway_id]

        # Create pivot table: rows=organisms, columns=steps, values=scores
        feature_matrix = gapmind_data.dropna(subset=["score"]).pivot(
            index="orgId", columns="step", values="score"
        )

        # Fill missing scores with -1
        feature_matrix.fillna(-1, inplace=True)
        feature_matrix.columns.name = ""

        # Map organism IDs to genome IDs
        feature_matrix.index = feature_matrix.index.map(orgid_genomeid_map)
        feature_matrix.index.name = "genomeID"

        feature_matrix_dict[phenotype_name] = feature_matrix.astype(np.int16)

    return feature_matrix_dict


def save_feature_matrices(
    feature_matrix_dict: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Save feature matrices at different confidence thresholds.

    Parameters
    ----------
    feature_matrix_dict : dict[str, pd.DataFrame]
        Dictionary mapping phenotype names to feature matrices
    output_dir : Path
        Output directory for saving feature matrices
    """
    conf_threshold_dict = {
        "all": -1,
        "high_conf": 2,
        "medium_conf": 1,
        "low_conf": 0,
    }

    for phenotype_name, feature_matrix in feature_matrix_dict.items():
        print(f"{phenotype_name} has {feature_matrix.shape[1]} features")

        for conf_name, conf_threshold in conf_threshold_dict.items():
            results_dir = output_dir / conf_name
            results_dir.mkdir(parents=True, exist_ok=True)

            # Threshold the feature matrix
            feature_matrix_threshold = feature_matrix[feature_matrix >= conf_threshold]

            # For specific confidence levels, binarize the data
            if conf_name != "all":
                feature_matrix_threshold[feature_matrix_threshold >= 0] = 1
                feature_matrix_threshold = feature_matrix_threshold.fillna(0).astype(
                    np.uint8
                )

            # Save to file
            output_file = results_dir / f"{phenotype_name}.tsv"
            feature_matrix_threshold.to_csv(output_file, sep="\t")


def main() -> None:
    """Main function to extract GapMind features."""
    # Setup
    output_dir = Path("data/processed/gapmind_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["s__at-leaf-lit-pmi", "s__marine-seqs"]

    # Load data
    print("Loading marine IDs mapping...")
    marine_ids_map = load_marine_ids_map()

    print("Loading GapMind steps data...")
    gapmind_steps_raw = load_gapmind_steps_data()

    print("Loading organism data...")
    orgid_data = load_organism_data(datasets, marine_ids_map)
    orgid_genomeid_map = orgid_data["genomeID"].to_dict()

    print("Creating feature matrices...")
    feature_matrix_dict = create_feature_matrices(gapmind_steps_raw, orgid_genomeid_map)

    print(f"\nExtracting features for {len(feature_matrix_dict)} phenotypes:")
    print("Saving feature matrices at different confidence thresholds...")
    save_feature_matrices(feature_matrix_dict, output_dir)

    print(f"\nFeatures saved to {output_dir}")


if __name__ == "__main__":
    main()
