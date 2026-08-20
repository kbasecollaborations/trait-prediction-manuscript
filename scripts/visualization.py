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
    in-clade  #2B5164
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

Tier 2, figure-local vocabularies. Each is keyed by its own legend or by its
own axis labels, so a Tier-2 colour never carries a manuscript-wide meaning.
Tier-2 colours may therefore repeat ACROSS figures; what they must never do is
collide with a Tier-1 category drawn in the SAME figure.
    Fig 2B baseline models     #A0A0A0 / #707070 / #000000 + markers o / s / ^
    Fig 4A failure causes      local swatches keyed by inline bold labels
    Fig 6A-right problem modes #8C2155 / #17BECF / #7F7F7F
    Fig 4C pooled/aggregate bar #595959
    Fig 6B metrics             #3B4CC0 / #F2C230 / #D62728
                               + markers o / s / ^ + line style
    Fig 7C/D strategies        #0173B2 / #CC78BC / #7F7F7F / #146B3A
    Fig S4 QC status           success #2E7D5B; failure modes neutral
                               #8C8C8C / #BFBFBF / #E0E0E0

Figure 7's strategy blue is the ATLeaf hex. That reuse is deliberate and safe:
Figure 7 draws no dataset series at all, so inside that figure the colour
cannot be read as dataset identity.

Verify any change with an established colour-vision model, not a hand-rolled
one: naive dichromat simulation clips out-of-gamut results toward white and
invents collapses that are not real. ``uv run --with colorspacious`` gives
sRGB1+CVD at severity 100, which reproduces the known weak Okabe-Ito pair
(orange vs vermillion, dE 18) and the red/green collapse.

Why some categories moved off the Okabe-Ito values they used to hold: Biolog
orange and the old Populus vermillion sat ~10 dE apart under dichromacy, and
the old confusion palette was byte-identical to the dataset palette
(TP == ATLeaf, TN == Marine) inside Figure 4, which also draws datasets.
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
