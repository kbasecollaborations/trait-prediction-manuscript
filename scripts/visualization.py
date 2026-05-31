"""Color schemes and plot-style helpers shared across figures."""

import matplotlib.pyplot as plt
import seaborn as sns


def configure_plot_style() -> None:
    """Set matplotlib font sizes for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 12,
        }
    )


def get_dataset_colors():
    """
    Get a colorblind-friendly color palette for datasets.

    Returns
    -------
    dict
        Dictionary mapping dataset names to colors.
    """
    palette = sns.color_palette("colorblind", n_colors=4)

    dataset_color_map = {
        "atleaf": palette[0],
        "lit": palette[1],
        "marine": palette[2],
        "pmi": palette[3],
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
        "lit": "Biolog",
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
