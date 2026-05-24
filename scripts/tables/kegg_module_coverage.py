#!/usr/bin/env python3
"""Compute KEGG-module coverage for stable feature clusters per phenotype.

Used by the Table S2 / Table S3 generators to append a one-line summary
showing what fraction of the shared (and unique) stable cluster
representatives map to the canonical KEGG catabolism module for the
target phenotype.

This is a deliberately small module: the manuscript only needs the
mapping for the 15 shared phenotypes, and only catabolism modules are
relevant (growth on a carbon source implies degradation, not
biosynthesis).
"""

import re
from pathlib import Path

# Phenotype -> canonical KEGG module identifier(s). Multiple modules are
# allowed when the phenotype's pathway is split across module entries.
# Phenotypes with no dedicated catabolism module in KEGG are mapped to
# None and reported as "no canonical KEGG module" in the table.
PHENOTYPE_TO_MODULES: dict[str, tuple[str, ...] | None] = {
    "Histidine": ("M00045",),
    "Galactose": ("M00632",),
    "Galacturonic-Acid": ("M00631",),
    "Glucose": ("M00001",),
    "Arginine": None,
    "Alanine": None,
    "Serine": None,
    "Cellobiose": None,
    "Fructose": None,
    "Glycerol": None,
    "Maltose": None,
    "Mannitol": None,
    "Mannose": None,
    "Sucrose": None,
    "m-Inositol": None,
}

MODULE_NAMES: dict[str, str] = {
    "M00045": "Histidine degradation",
    "M00632": "Galactose degradation, Leloir pathway",
    "M00631": "D-Galacturonate degradation (bacteria)",
    "M00001": "Glycolysis (Embden-Meyerhof pathway)",
}


_MODULE_KO_CACHE: dict[str, frozenset[str]] | None = None


def _load_module_kos(
    module_def_path: Path = Path("data/external/mapping/module-definitions.tsv"),
) -> dict[str, frozenset[str]]:
    """Parse the KEGG module-definitions table into module-id -> KO members.

    The module definitions are encoded with parentheses, commas (alternative
    enzymes), plus signs (complex subunits), and spaces (sequential steps).
    For the purpose of "is this KO part of this module" we ignore the
    grammar and simply extract every ``K\\d{5}`` token from the definition
    column.

    Parameters
    ----------
    module_def_path : Path
        Path to the KEGG module-definitions TSV file.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping from module identifier (``"M00045"`` etc.) to the set of
        member KOs across all alternative-step rows for that module.
    """
    global _MODULE_KO_CACHE
    if _MODULE_KO_CACHE is not None:
        return _MODULE_KO_CACHE

    ko_pattern = re.compile(r"K\d{5}")
    membership: dict[str, set[str]] = {}
    with open(module_def_path) as handle:
        header = handle.readline()
        if not header.startswith("Module ID"):
            raise ValueError(
                f"Unexpected header in module definitions file: {header!r}"
            )
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            module_id, _alt, _name, definition = parts[0], parts[1], parts[2], parts[3]
            members = membership.setdefault(module_id, set())
            members.update(ko_pattern.findall(definition))

    _MODULE_KO_CACHE = {mid: frozenset(kos) for mid, kos in membership.items()}
    return _MODULE_KO_CACHE


def pathway_coverage_line(
    phenotype: str,
    shared_kos: list[str],
    unique_kos: list[str],
    ko_to_cluster: dict[str, int] | None,
) -> str | None:
    """Build a single LaTeX-ready coverage line for a phenotype's table block.

    For each phenotype with a canonical KEGG catabolism module assigned in
    :data:`PHENOTYPE_TO_MODULES`, we collapse the shared and unique KOs
    into cluster representatives (one KO per cluster) and report the
    fraction of representatives that hit the canonical module.

    Parameters
    ----------
    phenotype : str
        Phenotype name (must be a key of ``PHENOTYPE_TO_MODULES``).
    shared_kos : list[str]
        All KOs reported as "shared" for this phenotype, pooled across the
        three held-out-dataset comparisons.
    unique_kos : list[str]
        All KOs reported as "unique to held-out-alone" for this phenotype,
        pooled across the three held-out-dataset comparisons.
    ko_to_cluster : dict[str, int] | None
        SHAP-supervised redundancy-cluster mapping for this phenotype, as
        produced by ``scripts/feature_clustering.py``. ``None`` falls back
        to treating each KO as its own cluster.

    Returns
    -------
    str | None
        LaTeX fragment to be inserted as a coverage row, or ``None`` when
        the phenotype has no canonical KEGG catabolism module assigned (in
        which case the table generator skips the row).
    """
    modules = PHENOTYPE_TO_MODULES.get(phenotype)
    if modules is None:
        return None

    module_kos: set[str] = set()
    for module_id in modules:
        module_kos.update(_load_module_kos().get(module_id, frozenset()))

    def _cluster_representatives(kos: list[str]) -> list[str]:
        """Pick one KO per cluster (smallest KO id by convention).

        Falls back to the raw KO list when no clustering metadata is
        available for the phenotype.
        """
        if ko_to_cluster is None:
            return sorted(set(kos))
        seen_clusters: dict[int, str] = {}
        singletons: list[str] = []
        for ko in sorted(set(kos)):
            cluster = ko_to_cluster.get(ko)
            if cluster is None:
                singletons.append(ko)
                continue
            if cluster not in seen_clusters:
                seen_clusters[cluster] = ko
        return list(seen_clusters.values()) + singletons

    shared_reps = _cluster_representatives(shared_kos)
    unique_reps = _cluster_representatives(unique_kos)

    def _fraction_in_module(reps: list[str]) -> str:
        if not reps:
            return "0/0"
        hits = sum(1 for ko in reps if ko in module_kos)
        return f"{hits}/{len(reps)} ({hits / len(reps) * 100:.0f}\\%)"

    module_label = ", ".join(
        f"{mid} \\textit{{{MODULE_NAMES.get(mid, mid)}}}" for mid in modules
    )
    return (
        f"Pathway coverage ({module_label}): "
        f"shared {_fraction_in_module(shared_reps)}; "
        f"unique {_fraction_in_module(unique_reps)}"
    )
