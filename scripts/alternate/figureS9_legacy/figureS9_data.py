#!/usr/bin/env python3
"""Quantify taxonomic bias from GapMind concordance filtering.

For each shared phenotype, compares the full experimental set against the
concordant subset on three measures: GTDB Class composition, Faith PD, and
train/test class overlap across the four LOO splits (test set held fixed). Writes
``data/outputs/taxonomic_bias/taxonomic_bias.tsv`` in long form.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from ete3 import Tree

from scripts.create_data_splits import COMMON_PHENOTYPES, DATASET_SUBSET
from scripts.figure5.figure5a_data import (
    get_concordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)

ASSIGNMENTS_FILE: Path = Path(
    "data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv"
)
STRAIN_MAP_FILE: Path = Path("data/interim/marine_strain_genomeid_map.json")
TREE_FILE: Path = Path("data/processed/phylogeny/gtdb-pruned.nwk")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
SPLITS_DIR: Path = Path("data/processed/train_test_splits/dataset_split")
OUTPUT_FILE: Path = Path("data/outputs/taxonomic_bias/taxonomic_bias.tsv")

UNASSIGNED_LABEL: str = "Unassigned"


def _parse_lineage(lineage: str) -> dict[str, str]:
    """Parse a GTDB lineage string into a {rank: name} dict.

    Parameters
    ----------
    lineage : str
        Semicolon-separated GTDB lineage of the form
        ``d__Bacteria;p__...;c__Class;...``.

    Returns
    -------
    dict[str, str]
        Mapping from one-letter rank prefix (``d``, ``p``, ``c``, ``o``, ``f``,
        ``g``, ``s``) to the rank name (without the prefix).
    """
    out: dict[str, str] = {}
    for part in lineage.split(";"):
        if "__" in part:
            rank, name = part.split("__", 1)
            out[rank] = name
    return out


def build_genome_to_class(
    assignments_file: Path = ASSIGNMENTS_FILE,
    strain_map_file: Path = STRAIN_MAP_FILE,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build a ``genome_id -> GTDB Class`` lookup for all genomes in the project.

    Direct matches use the ``Genome name`` column of the pangenome assignments
    table. Marine strain short codes (e.g. ``A2R07``) are bridged via the
    project's strain-id JSON map. For genomes that lack a GTDB taxonomy entry
    (typically IMG-only genomes), the class is inferred from the genus parsed
    out of the ``Genome name`` column using the most common class observed for
    that genus among genomes that have a GTDB taxonomy. Genomes whose genus
    cannot be resolved are labelled ``"Unassigned"``.

    Parameters
    ----------
    assignments_file : Path, optional
        Path to ``assignments.ani.merged_mmseqs90.tsv``.
    strain_map_file : Path, optional
        Path to ``marine_strain_genomeid_map.json`` used to bridge short
        marine strain codes to GenBank accessions.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Tuple of (lookup, genus_to_class). The first maps known
        ``genome_id -> class_name`` (genomes resolvable from the assignments
        table or the strain map). The second is a ``genus -> class`` table
        derived from the GTDB-classified rows; callers should use it as a
        fallback when an unknown genome's name begins with a recognised genus
        token.
    """
    df = pd.read_csv(assignments_file, sep="\t", dtype=str)
    df["Genome name"] = df["Genome name"].fillna("")

    name_to_class: dict[str, str] = {}
    genus_to_class_counts: dict[str, dict[str, int]] = {}
    for gname, lineage in zip(df["Genome name"], df["gtdb_taxonomy_id"]):
        if not lineage or pd.isna(lineage):
            continue
        ranks = _parse_lineage(lineage)
        cls = ranks.get("c") or UNASSIGNED_LABEL
        if gname:
            name_to_class[gname] = cls
        genus = ranks.get("g")
        if genus:
            genus_to_class_counts.setdefault(genus, {}).setdefault(cls, 0)
            genus_to_class_counts[genus][cls] += 1

    genus_to_class: dict[str, str] = {
        g: max(counts.items(), key=lambda kv: kv[1])[0]
        for g, counts in genus_to_class_counts.items()
    }

    strain_map: dict[str, str] = json.loads(strain_map_file.read_text())
    accession_to_class: dict[str, str] = {}
    for gname, cls in name_to_class.items():
        # Genome name is e.g. "GCF_000152765.1_ASM15276v1_genomic.fna"
        if gname.startswith(("GCF_", "GCA_")):
            accession = gname.split("_ASM")[0]
            accession_to_class.setdefault(accession, cls)

    code_to_class: dict[str, str] = {}
    for code, mapped in strain_map.items():
        accession = mapped.split("_ASM")[0]
        cls = accession_to_class.get(accession)
        if cls is not None:
            code_to_class[code] = cls

    lookup: dict[str, str] = {}
    lookup.update(name_to_class)
    lookup.update(code_to_class)

    for gname in df["Genome name"]:
        if not gname or gname in lookup:
            continue
        # Genus often appears as the first underscore-separated token.
        genus = gname.split("_", 1)[0]
        cls = genus_to_class.get(genus, UNASSIGNED_LABEL)
        lookup[gname] = cls

    return lookup, genus_to_class


def _genus_from_id(genome_id: str) -> str | None:
    """Heuristically extract a genus token from a genome identifier.

    Returns ``None`` for purely numeric / accession identifiers (e.g.
    ``"1663.224"``, ``"GCF_001421325.1"``) where the leading token does not
    look like a Latinised genus name.
    """
    head = genome_id.split("_", 1)[0]
    if not head:
        return None
    if head[0].isdigit() or head.startswith(("GCF", "GCA", "RS")):
        return None
    return head


def assign_classes(
    genome_ids: Iterable[str],
    lookup: dict[str, str],
    genus_to_class: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign a class label to each genome id.

    Parameters
    ----------
    genome_ids : Iterable[str]
        Iterable of genome identifiers (matching either the ``Genome name``
        column of the assignments table or a marine strain short code).
    lookup : dict[str, str]
        The ``genome_id -> class`` map returned by :func:`build_genome_to_class`.
    genus_to_class : dict[str, str] | None, optional
        Genus to class fallback. When supplied, genomes not in ``lookup`` whose
        leading token matches a known genus inherit that genus's class.

    Returns
    -------
    dict[str, str]
        Mapping of input ``genome_id`` to class label (``"Unassigned"`` when
        unknown).
    """
    out: dict[str, str] = {}
    for gid in genome_ids:
        cls = lookup.get(gid)
        if cls is None and genus_to_class is not None:
            genus = _genus_from_id(gid)
            if genus is not None:
                cls = genus_to_class.get(genus)
        out[gid] = cls if cls is not None else UNASSIGNED_LABEL
    return out


def class_composition(class_assignments: dict[str, str]) -> dict[str, float]:
    """Compute the fraction of genomes belonging to each class.

    Parameters
    ----------
    class_assignments : dict[str, str]
        Mapping of genome id to class label.

    Returns
    -------
    dict[str, float]
        Mapping of class name to fraction of genomes assigned to that class.
        Returns an empty dict when ``class_assignments`` is empty.
    """
    if not class_assignments:
        return {}
    counts: dict[str, int] = {}
    for cls in class_assignments.values():
        counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    return {cls: n / total for cls, n in counts.items()}


def faith_pd(tree: Tree, tips: Iterable[str]) -> float:
    """Compute Faith's PD as the sum of branch lengths spanning the given tips.

    The tree is copied and pruned to the intersection of ``tips`` and the
    tree's leaves; the sum of the remaining edge lengths is returned. The root
    edge is not subtracted separately.

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree from ETE 3.
    tips : Iterable[str]
        Iterable of leaf names to retain. Tips not present in the tree are
        ignored.

    Returns
    -------
    float
        Sum of branch lengths in the pruned subtree. Returns ``0.0`` when fewer
        than two tips are present in the tree.
    """
    leaf_set = set(tree.get_leaf_names())
    keep = [t for t in tips if t in leaf_set]
    if len(keep) < 2:
        return 0.0
    pruned = tree.copy(method="newick")
    pruned.prune(keep, preserve_branch_length=True)
    total = 0.0
    for node in pruned.traverse():
        if node.up is not None:
            total += float(node.dist)
    return total


def load_split_indices(
    phenotype: str, dataset: str
) -> tuple[set[str], set[str]] | None:
    """Load train and test genome ids for a single LOO split.

    Parameters
    ----------
    phenotype : str
        Phenotype name (e.g. ``"Histidine"``).
    dataset : str
        Held-out dataset name (the test dataset).

    Returns
    -------
    tuple[set[str], set[str]] | None
        Tuple of (train_genomes, test_genomes). ``None`` if the split files do
        not exist on disk.
    """
    train_dsets = [d for d in DATASET_SUBSET if d != dataset]
    folder = SPLITS_DIR / phenotype / f"train({'+'.join(train_dsets)}),test({dataset})"
    train_file = folder / "y_train.tsv"
    val_file = folder / "y_val.tsv"
    test_file = folder / "y_test.tsv"
    if not (train_file.exists() and test_file.exists()):
        return None
    y_train = pd.read_csv(train_file, sep="\t", index_col=0, dtype={"genomeID": str})
    y_test = pd.read_csv(test_file, sep="\t", index_col=0, dtype={"genomeID": str})
    train_ids = set(y_train.index.astype(str))
    if val_file.exists():
        y_val = pd.read_csv(val_file, sep="\t", index_col=0, dtype={"genomeID": str})
        train_ids |= set(y_val.index.astype(str))
    test_ids = set(y_test.index.astype(str))
    return train_ids, test_ids


def class_overlap(
    train_ids: set[str],
    test_ids: set[str],
    class_lookup: dict[str, str],
    genus_to_class: dict[str, str],
) -> float:
    """Fraction of test-set classes that are also represented in the training set.

    Parameters
    ----------
    train_ids : set[str]
        Genome ids in the training (and validation) set.
    test_ids : set[str]
        Genome ids in the test set.
    class_lookup : dict[str, str]
        Genome id to class label lookup.
    genus_to_class : dict[str, str]
        Genus to class fallback for genomes outside the lookup.

    Returns
    -------
    float
        Fraction in [0, 1]. ``float('nan')`` if the test set has no classes.
    """
    test_classes = set(assign_classes(test_ids, class_lookup, genus_to_class).values())
    train_classes = set(
        assign_classes(train_ids, class_lookup, genus_to_class).values()
    )
    if not test_classes:
        return float("nan")
    overlap = test_classes & train_classes
    return len(overlap) / len(test_classes)


def compute_phenotype_rows(
    phenotype: str,
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    class_lookup: dict[str, str],
    genus_to_class: dict[str, str],
    tree: Tree,
) -> list[dict[str, object]]:
    """Compute every output row for a single phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype name.
    gapmind_predictions : pd.DataFrame
        GapMind loose-threshold predictions (rows = genomes).
    experimental_phenotypes : pd.DataFrame
        Combined experimental phenotype matrix.
    class_lookup : dict[str, str]
        Genome id to class label lookup.
    genus_to_class : dict[str, str]
        Genus to class fallback for genomes outside the lookup.
    tree : Tree
        Phylogenetic tree used for Faith PD.

    Returns
    -------
    list[dict[str, object]]
        Long-form rows ready to append to a pandas DataFrame. Each row has at
        least ``phenotype``, ``measure``, ``scope`` and ``value``; rows for the
        composition measure also carry ``class_label``; rows for the overlap
        measure also carry ``test_dataset``.
    """
    rows: list[dict[str, object]] = []

    # "full" scope: genomes with a non-missing experimental measurement.
    if phenotype not in experimental_phenotypes.columns:
        return rows
    exp = experimental_phenotypes[phenotype].dropna()
    full_genomes = set(exp.index.astype(str))
    concordant_genomes = get_concordant_samples(
        gapmind_predictions, experimental_phenotypes, phenotype
    )
    concordant_genomes = {str(g) for g in concordant_genomes}

    full_assignments = assign_classes(full_genomes, class_lookup, genus_to_class)
    conc_assignments = assign_classes(concordant_genomes, class_lookup, genus_to_class)
    full_comp = class_composition(full_assignments)
    conc_comp = class_composition(conc_assignments)
    all_classes = sorted(set(full_comp) | set(conc_comp))
    for cls in all_classes:
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "class_composition",
                "scope": "full",
                "class_label": cls,
                "value": full_comp.get(cls, 0.0),
                "n_genomes": len(full_genomes),
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "class_composition",
                "scope": "concordant",
                "class_label": cls,
                "value": conc_comp.get(cls, 0.0),
                "n_genomes": len(concordant_genomes),
            }
        )

    pd_full = faith_pd(tree, full_genomes)
    pd_conc = faith_pd(tree, concordant_genomes)
    pd_ratio = pd_conc / pd_full if pd_full > 0 else float("nan")
    rows.append(
        {
            "phenotype": phenotype,
            "measure": "faith_pd",
            "scope": "full",
            "value": pd_full,
            "n_genomes": len(full_genomes & set(tree.get_leaf_names())),
        }
    )
    rows.append(
        {
            "phenotype": phenotype,
            "measure": "faith_pd",
            "scope": "concordant",
            "value": pd_conc,
            "n_genomes": len(concordant_genomes & set(tree.get_leaf_names())),
        }
    )
    rows.append(
        {
            "phenotype": phenotype,
            "measure": "faith_pd_ratio",
            "scope": "concordant_over_full",
            "value": pd_ratio,
        }
    )

    # Train/test class overlap per LOO split; the test set is held fixed in
    # both scopes and only the training set changes.
    overlaps_full: list[float] = []
    overlaps_conc: list[float] = []
    for ds in DATASET_SUBSET:
        split = load_split_indices(phenotype, ds)
        if split is None:
            continue
        train_ids, test_ids = split
        train_ids_conc = train_ids & concordant_genomes
        ov_full = class_overlap(train_ids, test_ids, class_lookup, genus_to_class)
        ov_conc = class_overlap(train_ids_conc, test_ids, class_lookup, genus_to_class)
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "train_test_class_overlap",
                "scope": "full",
                "test_dataset": ds,
                "value": ov_full,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "train_test_class_overlap",
                "scope": "concordant",
                "test_dataset": ds,
                "value": ov_conc,
                "n_train": len(train_ids_conc),
                "n_test": len(test_ids),
            }
        )
        overlaps_full.append(ov_full)
        overlaps_conc.append(ov_conc)

    if overlaps_full:
        mean_full = sum(overlaps_full) / len(overlaps_full)
        mean_conc = sum(overlaps_conc) / len(overlaps_conc)
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "train_test_class_overlap_mean",
                "scope": "full",
                "test_dataset": "aggregate",
                "value": mean_full,
            }
        )
        rows.append(
            {
                "phenotype": phenotype,
                "measure": "train_test_class_overlap_mean",
                "scope": "concordant",
                "test_dataset": "aggregate",
                "value": mean_conc,
            }
        )

    return rows


def main(phenotypes: list[str] | None = None) -> Path:
    """Generate the taxonomic-bias TSV.

    Parameters
    ----------
    phenotypes : list[str] | None, optional
        Subset of phenotypes to process. Defaults to all 15 shared phenotypes.

    Returns
    -------
    Path
        Path to the written TSV.
    """
    if phenotypes is None:
        phenotypes = COMMON_PHENOTYPES

    print("Loading taxonomy lookup ...")
    class_lookup, genus_to_class = build_genome_to_class()
    n_unassigned = sum(1 for v in class_lookup.values() if v == UNASSIGNED_LABEL)
    print(
        f"  built lookup for {len(class_lookup)} genomes "
        f"({n_unassigned} unassigned); {len(genus_to_class)} genera in fallback"
    )

    print("Loading tree ...")
    tree = Tree(str(TREE_FILE), format=1)
    print(f"  tree has {len(tree.get_leaves())} leaves")

    print("Loading GapMind + experimental phenotypes ...")
    gapmind = load_gapmind_predictions(GAPMIND_FILE)
    experimental = load_experimental_phenotypes(PHENOTYPE_DIR)

    all_rows: list[dict[str, object]] = []
    for phen in phenotypes:
        print(f"Processing {phen} ...")
        rows = compute_phenotype_rows(
            phen, gapmind, experimental, class_lookup, genus_to_class, tree
        )
        all_rows.extend(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotype",
        type=str,
        default=None,
        help="Process a single phenotype (defaults to all 15).",
    )
    args = parser.parse_args()
    phens = [args.phenotype] if args.phenotype else None
    main(phens)
