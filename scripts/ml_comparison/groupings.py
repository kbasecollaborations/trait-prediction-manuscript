"""KO -> KEGG-pathway group assignments for group lasso.

Groups come from KEGG reference pathway membership
(``data/external/mapping/pathway-ko-membership.tsv``) rather than KEGG modules.
Each KO is assigned to its first pathway membership (alphabetical by pathway ID);
KOs not in any pathway become singleton groups.

The group assignment is global (does not vary by phenotype) and computed
on first call against the feature matrix on disk. Cached in
``data/processed/kegg_pathway_groupings.json`` between runs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

FEATURE_FILE = Path("data/processed/features_reduced/combined_datasets/kofam.tsv")
PATHWAY_FILE = Path("data/external/mapping/pathway-ko-membership.tsv")
CACHE_FILE = Path("data/processed/kegg_pathway_groupings.json")


def _build_groupings() -> dict[str, int]:
    """Return {KO_id: group_id} for every KO in the kofam feature matrix."""
    feat_cols = pd.read_csv(
        FEATURE_FILE, sep="\t", index_col=0, nrows=0
    ).columns.tolist()
    pathways = pd.read_csv(PATHWAY_FILE, sep="\t")
    ko_to_pathway: dict[str, str] = {}
    for _, row in pathways.sort_values("Pathway ID").iterrows():
        kos = str(row["KO IDs"]).split(",")
        for ko in kos:
            ko_to_pathway.setdefault(ko.strip(), str(row["Pathway ID"]))
    group_label_to_id: dict[str, int] = {}
    next_id = 1
    out: dict[str, int] = {}
    for ko in feat_cols:
        pid = ko_to_pathway.get(ko)
        if pid is None:
            # singleton group per ungrouped KO
            out[ko] = -hash(ko) % 100_000_000
            continue
        if pid not in group_label_to_id:
            group_label_to_id[pid] = next_id
            next_id += 1
        out[ko] = group_label_to_id[pid]
    return out


def load_or_build_kegg_groupings() -> dict[str, int]:
    """Return {KO_id: group_id}, reading ``CACHE_FILE`` when it is readable and
    rebuilding it otherwise."""
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open() as f:
                return json.load(f)
        except Exception:
            pass
    out = _build_groupings()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("w") as f:
        json.dump(out, f)
    return out


@lru_cache(maxsize=1)
def _full_groupings() -> dict[str, int]:
    return load_or_build_kegg_groupings()


def kegg_module_groupings_for_phenotype(phenotype: str) -> list[int]:
    """Return a list[int] aligned with the columns of the KOFAM feature matrix.

    The phenotype argument is accepted but ignored; group assignment is global
    across phenotypes.
    """
    feat_cols = pd.read_csv(
        FEATURE_FILE, sep="\t", index_col=0, nrows=0
    ).columns.tolist()
    g = _full_groupings()
    return [int(g.get(ko, -hash(ko) % 100_000_000)) for ko in feat_cols]
