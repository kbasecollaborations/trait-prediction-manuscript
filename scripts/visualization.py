"""
Visualization utilities and color schemes for consistent plotting across figures.
"""

import seaborn as sns


def get_dataset_colors():
    """
    Get a colorblind-friendly color palette for datasets.

    Returns
    -------
    dict
        Dictionary mapping dataset names to colors.
    """
    # Use seaborn's colorblind palette (designed for accessibility)
    palette = sns.color_palette("colorblind", n_colors=4)

    # Map datasets to colors (alphabetically sorted for consistency)
    dataset_color_map = {
        "atleaf": palette[0],  # Blue
        "lit": palette[1],     # Orange
        "marine": palette[2],  # Green
        "pmi": palette[3],     # Red
    }

    return dataset_color_map


def get_dataset_color_list(datasets):
    """
    Get a list of colors for a given list of datasets.

    Parameters
    ----------
    datasets : list
        List of dataset names.

    Returns
    -------
    list
        List of colors corresponding to the datasets.
    """
    color_map = get_dataset_colors()
    return [color_map.get(dataset, "#000000") for dataset in datasets]


def get_dataset_display_names():
    """
    Get a mapping of dataset names to their display names for plots.

    Returns
    -------
    dict
        Dictionary mapping dataset names to display names.
    """
    return {
        "atleaf": "AtLeaf",
        "marine": "Marine",
        "lit": "Literature",
        "pmi": "Populus",
    }


def format_dataset_names(datasets):
    """
    Convert a list of dataset names to their display names.

    Parameters
    ----------
    datasets : list
        List of dataset names.

    Returns
    -------
    list
        List of display names corresponding to the datasets.
    """
    name_map = get_dataset_display_names()
    return [name_map.get(dataset, dataset) for dataset in datasets]
