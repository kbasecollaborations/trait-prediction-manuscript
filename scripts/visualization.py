"""Shared colour palette and matplotlib style helpers for the figure scripts.

Each semantic category has one colour, used wherever that category is drawn.

Dataset            ATLeaf #0173B2  Biolog #DE8F05  Marine #029E73  Populus #A83E00
Split type         random #57BA64  cross-dataset #2E86AB
                   in-clade #2B5164  out-of-clade #785EF0
GapMind            permissive #A23B72  strict #6E2A4E  feature set #D7263D
Concordance        concordant #6A4C93  discordant #E89149
                   full-data arm #9E9E9E  full + BacDive #F5B25C
Confusion outcome  TP #499DD4  TN #009E54  FP #E1B22F  FN #7D083B
Neutral            Figure 4C pooled bar #595959; reference lines and phenotype
                   banding use grey at low alpha

Colours local to a single figure are defined in that figure's own script:
Figure 2B models, 4A failure causes, 6A-right modes, 6B metrics, 7C/D
strategies and S4 QC status.
"""

import re

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator

# Under the "science" style, ScalarFormatter renders numbers as mathtext, e.g.
# "$\mathdefault{0.25}$", and uses a unicode minus.
_MATHTEXT_LABEL = re.compile(r"^\$\\mathdefault\{(?P<value>.*)\}\$$")


def _is_numeric_label(text: str) -> bool:
    """Return whether a rendered tick label is a number.

    Parameters
    ----------
    text : str
        Tick label text, possibly wrapped in mathtext by the plotting style.

    Returns
    -------
    bool
        ``True`` when the label denotes a number.
    """
    match = _MATHTEXT_LABEL.match(text)
    if match is not None:
        text = match["value"]
    text = text.strip("$").replace("\N{MINUS SIGN}", "-").replace(",", "")
    text = text.removesuffix(r"\%").removesuffix("%")
    try:
        float(text)
    except ValueError:
        return False
    return True


def _is_categorical_axis(axis: Axis) -> bool:
    """Return whether an axis carries fixed, non-numeric tick labels.

    A categorical axis is one whose ticks were placed explicitly and whose
    labels are not numbers. An axis with explicit but numeric tick labels is a
    continuous scale and is left alone.

    Parameters
    ----------
    axis : Axis
        The ``xaxis`` or ``yaxis`` of an axes.

    Returns
    -------
    bool
        ``True`` when the axis is categorical.
    """
    if not isinstance(axis.get_major_locator(), FixedLocator):
        return False
    labels = [tick.get_text().strip() for tick in axis.get_ticklabels()]
    for text in labels:
        if not text:
            continue
        if not _is_numeric_label(text):
            return True
    # Every label blank means the labels were deliberately hidden on a shared
    # categorical axis (e.g. the upper panel of a stacked pair).
    return not any(labels)


def hide_categorical_minor_ticks(fig: Figure) -> None:
    """Remove minor ticks from every categorical axis in a figure.

    The ``science`` style turns minor ticks on for all axes, but a minor tick
    between two categories (e.g. two phenotypes) has no meaning. Call this
    immediately before saving a figure.

    Parameters
    ----------
    fig : Figure
        Figure whose axes should be cleaned up.
    """
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if _is_categorical_axis(axis):
                axis.set_minor_locator(NullLocator())


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
    dataset_color_map = {
        "atleaf": "#0173B2",
        "lit": "#DE8F05",
        "marine": "#029E73",
        "pmi": "#A83E00",
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
        "atleaf": "ATLeaf",
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
