#!/usr/bin/env python3
"""Figure S1: circular phylogeny of the 629 GTDB-placed genomes.

Replaces the iTOL-rendered ``figure_s1.png``. The tree is the pruned GTDB
reference tree (``data/processed/phylogeny/gtdb-pruned.nwk``) drawn with
pyCirclize, matching the circular-tree convention used in the traitweaver
repository (``bin/plot_circular_tree.py``). pyCirclize is matplotlib-based, so
the dataset colours are the same ``scripts.visualization.get_dataset_colors``
palette as the main text.

Layout follows the usual conventions for a several-hundred-tip tree on one page:
tip labels are dropped, dataset membership is an inner colour ring, and taxonomy
is carried by an outer clade bar with radial labels rather than per-tip text. The
tree is a cladogram, so the tips align at the ring and no scale bar is implied;
the fan opens at the bottom to keep every clade label clear of the legend band.

Dataset membership is read from ``data/processed/phenotypes/<dataset>/`` (the
authoritative source; ``genome_to_dataset.tsv`` is missing 24 of the tips).
GTDB lineages come from the pangenome ANI assignments, bridged to marine strain
codes via ``marine_strain_genomeid_map.json``; the 152 tips without a lineage
inherit the majority lineage of their smallest ancestral clade that has one.

Run with:
    uv run python -m scripts.figureS1.figureS1_plot
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ete3 import Tree, TreeNode
from matplotlib.patches import Patch
from pycirclize import Circos
from pycirclize.utils.plot import get_label_params_by_rad

from scripts.visualization import get_dataset_colors, get_dataset_display_names

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "nature"])
except Exception:
    pass
plt.rcParams.update({"text.usetex": False, "svg.fonttype": "none"})

TREE_FILE = Path("data/processed/phylogeny/gtdb-pruned.nwk")
ASSIGNMENTS_FILE = Path("data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv")
STRAIN_MAP_FILE = Path("data/interim/marine_strain_genomeid_map.json")
PHENOTYPE_DIR = Path("data/processed/phenotypes")
PDF_OUT = Path("figures/figure_s1.pdf")

DATASETS: tuple[str, ...] = ("lit", "atleaf", "marine", "pmi")

# Radii in pyCirclize units: 0 at the centre, 100 at the outer edge.
R_TREE = (13.0, 76.0)
R_DATASET = (78.0, 83.0)  # inner ring: dataset membership
R_CLADE = (85.0, 90.0)  # outer bar: GTDB order
R_LABEL = 92.0  # radial clade labels start here

# The fan opens at the bottom, wide enough that no clade label points into the
# legend band below the tree.
FAN_GAP_DEG = 38.0

MIN_CLADE_TIPS = 8  # orders smaller than this get a bar but no label
FIG_SIDE = 5.25  # inches; see the note in plot()
LEGEND_BAND = 0.72  # inches reserved below the axes for the two keys

FS_CLADE = 5.5
FS_LEGEND = 6.0
FS_TITLE = 6.8

# Distinct hues for the six GTDB classes present; deliberately unrelated to the
# dataset palette so the two rings never read as the same encoding.
CLASS_COLORS: dict[str, str] = {
    "Gammaproteobacteria": "#7B6FA8",
    "Alphaproteobacteria": "#C7799B",
    "Actinomycetia": "#5B8C6E",
    "Bacteroidia": "#C58A46",
    "Bacilli": "#7EA6C4",
    "Campylobacteria": "#A8574E",
}
UNKNOWN_COLOR = "#BBBBBB"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset_membership(genomes: list[str]) -> dict[str, str]:
    """Map each genome to the dataset whose phenotype tables contain it.

    Parameters
    ----------
    genomes : list[str]
        Genome identifiers to resolve (the tree tips).

    Returns
    -------
    dict[str, str]
        Mapping from genome id to dataset key. Genomes found in no dataset are
        omitted.

    Raises
    ------
    FileNotFoundError
        If a per-dataset phenotype directory is missing.
    """
    membership: dict[str, str] = {}
    for dataset in DATASETS:
        directory = PHENOTYPE_DIR / dataset
        if not directory.is_dir():
            raise FileNotFoundError(f"missing phenotype directory {directory}")
        ids: set[str] = set()
        for path in directory.glob("*.tsv"):
            ids |= set(pd.read_csv(path, sep="\t", dtype=str)["genomeID"])
        for genome in genomes:
            if genome in ids:
                membership[genome] = dataset
    return membership


def _parse_lineage(lineage: str) -> dict[str, str]:
    """Parse a GTDB lineage string into a ``{rank_prefix: name}`` mapping."""
    return {
        part.split("__", 1)[0]: part.split("__", 1)[1]
        for part in lineage.split(";")
        if "__" in part
    }


def load_lineages(tree: Tree) -> dict[str, dict[str, str]]:
    """Resolve a GTDB lineage for every tip, propagating down the tree if needed.

    Tips are matched against the pangenome assignments table directly, by
    accession prefix, or through the marine strain-code map. Tips that remain
    unmatched inherit the majority lineage of the smallest ancestral clade that
    contains at least one matched tip.

    Parameters
    ----------
    tree : Tree
        The pruned GTDB tree.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from tip name to a ``{rank_prefix: name}`` lineage dict.
    """
    assignments = pd.read_csv(ASSIGNMENTS_FILE, sep="\t", dtype=str)
    by_name = {
        name: lineage
        for name, lineage in zip(
            assignments["Genome name"], assignments["gtdb_taxonomy_id"]
        )
        if isinstance(lineage, str)
    }
    by_accession = {
        name.split("_ASM")[0]: lineage
        for name, lineage in by_name.items()
        if name.startswith(("GCF_", "GCA_"))
    }
    strain_map: dict[str, str] = json.loads(STRAIN_MAP_FILE.read_text())
    by_strain = {
        code: by_accession[assembly.split("_ASM")[0]]
        for code, assembly in strain_map.items()
        if assembly.split("_ASM")[0] in by_accession
    }

    known: dict[str, dict[str, str]] = {}
    for leaf in tree:
        lineage = (
            by_name.get(leaf.name)
            or by_accession.get(leaf.name)
            or by_strain.get(leaf.name)
        )
        if lineage:
            known[leaf.name] = _parse_lineage(lineage)

    lineages = dict(known)
    for leaf in tree:
        if leaf.name in known:
            continue
        node: TreeNode = leaf
        while node.up is not None:
            node = node.up
            relatives = [known[tip.name] for tip in node if tip.name in known]
            if relatives:
                lineages[leaf.name] = {
                    rank: Counter(
                        d[rank] for d in relatives if rank in d
                    ).most_common(1)[0][0]
                    for rank in "dpcofg"
                    if any(rank in d for d in relatives)
                }
                break
    return lineages


def contiguous_runs(labels: list[str]) -> list[tuple[str, int, int]]:
    """Collapse a label sequence into ``(label, start_index, end_index)`` runs."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], start, i - 1))
            start = i
    return runs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(membership: dict[str, str], lineages: dict[str, dict[str, str]]) -> plt.Figure:
    """Render the full Figure S1 panel.

    Parameters
    ----------
    membership : dict[str, str]
        Tip name to dataset key.
    lineages : dict[str, dict[str, str]]
        Tip name to ``{rank_prefix: name}`` lineage mapping.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure, sized for inclusion at ``\\textwidth``.
    """
    half_span = (360.0 - FAN_GAP_DEG) / 2.0
    circos, tv = Circos.initialize_from_tree(
        TREE_FILE,
        start=-half_span,
        end=half_span,
        r_lim=R_TREE,
        ladderize=True,
        leaf_label_size=0,  # 629 tips: per-tip labels would be unreadable
        # Drawn as a cladogram. The figure's claim is about which clades each
        # dataset occupies, not about divergence, and Figure S2 reports the
        # distances numerically. Flush tips also let the dataset ring sit
        # directly against them, and correctly signal to the reader that no
        # branch length here is meant to be measured, so no scale bar belongs.
        ignore_branch_length=True,
        line_kws=dict(lw=0.3, color="#333333"),
    )

    # pyCirclize decides the leaf order, so key every annotation by tip name and
    # read the order back off the TreeViz rather than assuming it matches ete3's.
    order = list(tv.leaf_labels)
    sector = tv.track.parent_sector

    dataset_colors = get_dataset_colors()
    dataset_track = sector.add_track(R_DATASET)
    for dataset, first, last in contiguous_runs([membership[n] for n in order]):
        dataset_track.rect(
            first, last + 1, color=dataset_colors.get(dataset, UNKNOWN_COLOR), lw=0.0
        )

    orders = [lineages[n]["o"] for n in order]
    classes = [lineages[n].get("c", "") for n in order]
    order_to_class = dict(zip(orders, classes))
    clade_track = sector.add_track(R_CLADE)
    for name, first, last in contiguous_runs(orders):
        clade_track.rect(
            first,
            last + 1,
            color=CLASS_COLORS.get(order_to_class.get(name, ""), UNKNOWN_COLOR),
            lw=0.0,
        )
        n_tips = last - first + 1
        if n_tips < MIN_CLADE_TIPS:
            continue
        center = (first + last + 1) / 2.0
        # Track.text() only takes the rotation from get_label_params_by_rad and
        # then centres the anchor, which straddles the label across the bar.
        # Taking the full parameter set instead anchors each label at its inner
        # end so it radiates outward, flipped on the left half to stay readable.
        label_kws = get_label_params_by_rad(
            clade_track.x_to_rad(center), "vertical", outer=True
        )
        clade_track.text(
            f"{name} ({n_tips})",
            x=center,
            r=R_LABEL,
            adjust_rotation=False,
            size=FS_CLADE,
            color="#222222",
            **label_kws,
        )

    # FIG_SIDE is tuned so the tight bounding box, which grows to include the
    # radial clade labels, lands at roughly \textwidth. Any larger and LaTeX
    # scales the figure down, shrinking every label point size with it.
    height = FIG_SIDE + LEGEND_BAND
    fig = plt.figure(figsize=(FIG_SIDE, height))
    ax = fig.add_axes(
        [0.0, LEGEND_BAND / height, 1.0, FIG_SIDE / height], projection="polar"
    )
    circos.plotfig(ax=ax)
    _add_legends(fig, membership, classes)
    return fig


def _add_legends(
    fig: plt.Figure, membership: dict[str, str], classes: list[str]
) -> None:
    """Draw both keys in the band reserved below the tree.

    The fan gap is wide enough that no clade label points into that band, so the
    two never collide.
    """
    display = get_dataset_display_names()
    colors = get_dataset_colors()
    counts = Counter(membership.values())
    fig.legend(
        handles=[
            Patch(facecolor=colors[d], label=f"{display[d]} ({counts[d]})")
            for d in DATASETS
            if counts[d]
        ],
        title="Dataset",
        loc="lower left",
        bbox_to_anchor=(0.015, 0.0),
        ncol=2,
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.35,
        columnspacing=1.0,
        fontsize=FS_LEGEND,
        title_fontsize=FS_TITLE,
    )

    class_counts = Counter(c for c in classes if c)
    fig.legend(
        handles=[
            Patch(facecolor=CLASS_COLORS.get(c, UNKNOWN_COLOR), label=c)
            for c, _ in class_counts.most_common()
        ],
        title="GTDB class",
        loc="lower right",
        bbox_to_anchor=(0.985, 0.0),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.35,
        columnspacing=1.0,
        fontsize=FS_LEGEND,
        title_fontsize=FS_TITLE,
    )


def main() -> None:
    """Build Figure S1 and write the vector PDF used by the manuscript."""
    tree = Tree(str(TREE_FILE), format=1)
    tips = [leaf.name for leaf in tree]
    fig = plot(load_dataset_membership(tips), load_lineages(tree))
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
