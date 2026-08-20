"""Color schemes and plot-style helpers shared across figures.

Canonical colour vocabulary
---------------------------
One semantic category gets one colour across every main and supplementary
figure, and no colour ever denotes two different categories. Values below are
the single source of truth; a figure script that hardcodes a hex must use one
of these.

Dataset identity (``get_dataset_colors``)
    ATLeaf #0173B2, Biolog #DE8F05, Marine #029E73, Populus #A83E00

Evaluation split type
    random / in-distribution holdout  #57BA64
    cross-dataset (leave-one-dataset-out)  #2E86AB
    in-clade  #CA9161
    out-of-clade  #785EF0

GapMind (rule-based)
    permissive / default threshold  #A23B72
    strict threshold  #6E2A4E
    GapMind feature set fed to an ML model  #D7263D  (a different referent from
        the rule-based tool, so deliberately its own hue)

Concordance
    concordant (filtered training set, concordant test subset)  #6A4C93
    discordant (out of applicability domain)  #E89149
    full / unfiltered training arm  #9E9E9E
    full + BacDive  #F5B25C

Confusion outcome (GapMind prediction vs experiment)
    TP #499DD4, TN #009E54, FP #E1B22F, FN #7D083B

Neutral
    aggregate over all datasets  #9E9E9E
    chance, parity, zero and banding chrome  grey at low alpha

Figure-local vocabularies carry NO hue. Their categories are already named on
an axis or in a legend, so they use a neutral ramp plus a redundant
non-colour channel. This keeps the hue budget for categories that recur:
    Fig 2B baseline models     luminance ramp + markers o / s / ^
    Fig 4A failure causes      local swatches keyed by inline bold labels
    Fig 6A-right problem modes #4D4D4D / #909090 / #C8C8C8
    Fig 6B metrics             one ink #3F3F3F + markers o / s / ^ + line style
    Fig 7C/D strategies        one fill #BFBFBF + hatch (// \\ none xx)
    Fig S4 QC status           #3A3A3A / #8C8C8C / #BFBFBF / #E0E0E0

Why some categories moved off the Okabe-Ito values they used to hold: under
deuteranopia the orange #DE8F05 and vermillion #D55E00 sit only dE 9 apart, so
Populus and the false-negative colour both left vermillion; and the old
confusion palette was byte-identical to the dataset palette (TP == ATLeaf,
TN == Marine) inside Figure 4. Only about 21 mutually distinguishable
publication-quality colours exist at all, which is why the figure-local
vocabularies above are deliberately neutral rather than hued.
"""

import matplotlib.pyplot as plt


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

    The first three are seaborn ``colorblind`` indices 0-2. ``pmi`` uses a
    darkened vermillion instead of index 3 (#D55E00) because index 3 collapses
    onto the ``lit`` orange under deuteranopia; see the module docstring.

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
