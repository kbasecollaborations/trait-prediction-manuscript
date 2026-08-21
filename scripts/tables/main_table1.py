#!/usr/bin/env python3
"""Regenerate the main-text Table 1 (concordance-vs-full feature comparison).

Run with::

    uv run python -m scripts.tables.main_table1
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.tables.kegg_module_coverage import (
    PATHWAY_NAMES,
    PHENOTYPE_TO_PATHWAY_MAPS,
    pathway_kos_for_phenotype,
)

HELD_OUT_DATASETS: tuple[str, ...] = ("atleaf", "lit", "marine")

# Ordered by canonical pathway map (carbohydrates, then amino acids), with
# Histidine first as the worked example.
PHENOTYPE_ORDER: tuple[str, ...] = (
    "Histidine",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Cellobiose",
    "Maltose",
    "Sucrose",
    "Mannose",
    "Fructose",
    "Mannitol",
    "m-Inositol",
    "Glycerol",
    "Alanine",
    "Serine",
    "Arginine",
)

# Two-line LaTeX renders for pathway names too long for the "KEGG pathway map"
# column; names not listed render as a single italic line from PATHWAY_NAMES.
PATHWAY_NAME_WRAPS: dict[str, str] = {
    "map00040": "\\textit{Pentose and glucuronate} \\\\ \\textit{interconversions}",
    "map00250": "\\textit{Alanine, aspartate and} \\\\ \\textit{glutamate metabolism}",
    "map00260": "\\textit{Glycine, serine and} \\\\ \\textit{threonine metabolism}",
}

# Short display names for the example column, replacing verbose KEGG dictionary
# names. KOs not listed fall back to the KEGG dictionary's descriptive name.
KO_DISPLAY_NAMES: dict[str, str] = {
    "K10228": "mannitol permease",
    "K02770": "PTS fructose IIC",
    "K00009": "mannitol-1-P 5-dehydrogenase",
    "K00010": "myo-inositol 2-dehydrogenase",
    "K02444": "glycerol-3-P regulon repressor",
}


def _display_ko_name(ko_id: str, ko_dict: dict[str, Any]) -> str:
    """Return a readable, LaTeX-safe enzyme name for a KO identifier.

    KOs absent from ``KO_DISPLAY_NAMES`` fall back to the dictionary's
    descriptive segment (the part after the gene symbols, with the trailing
    ``[EC:...]`` bracket removed).

    Parameters
    ----------
    ko_id : str
        KO identifier such as ``"K01712"``.
    ko_dict : dict[str, Any]
        KO dictionary loaded from ``KO_dictionary.json``.

    Returns
    -------
    str
        LaTeX-safe enzyme name.
    """
    if ko_id in KO_DISPLAY_NAMES:
        return KO_DISPLAY_NAMES[ko_id]
    term = ko_dict.get("term_hash", {}).get(f"KO:{ko_id}")
    if not term:
        return ""
    name = term.get("name", "")
    segments = name.split(";")
    desc = segments[1] if len(segments) > 1 else segments[0]
    desc = re.sub(r"\s*\[EC:[^]]+\]\s*$", "", desc).strip()
    return desc.replace("_", r"\_")


def _invert_cluster_mapping(
    ko_to_cluster: dict[str, int],
) -> dict[int, set[str]]:
    """Invert a KO -> cluster mapping to cluster -> set of KO members."""
    out: dict[int, set[str]] = {}
    for ko, cid in ko_to_cluster.items():
        out.setdefault(cid, set()).add(ko)
    return out


def _kos_to_cluster_ids(
    kos: list[str], ko_to_cluster: dict[str, int]
) -> set[str | int]:
    """Map a list of KOs to the union of their cluster IDs.

    Unmapped KOs become per-KO synthetic singleton IDs (``"sing:K01234"``)
    so they are still represented as distinct clusters when intersecting or
    differencing two sets.
    """
    ids: set[str | int] = set()
    for ko in kos:
        ids.add(ko_to_cluster[ko] if ko in ko_to_cluster else f"sing:{ko}")
    return ids


def _cluster_in_pathway(
    cluster_id: str | int,
    cluster_to_kos: dict[int, set[str]],
    pathway_kos: set[str],
) -> bool:
    """Return True iff the cluster contains at least one KO on the pathway.

    Singleton clusters (synthetic ``"sing:<KO>"`` IDs) reduce to a direct
    KO-membership check.
    """
    if isinstance(cluster_id, str) and cluster_id.startswith("sing:"):
        return cluster_id.split(":", 1)[1] in pathway_kos
    return any(k in pathway_kos for k in cluster_to_kos.get(cluster_id, ()))


def _shared_unique_counts(
    phenotype: str,
    combined_results: dict[str, list[str]],
    individual_results: dict[str, dict[str, list[str]]],
    ko_to_cluster: dict[str, int],
    pathway_kos: set[str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute summed cluster-level shared and unique counts for one phenotype.

    Parameters
    ----------
    phenotype : str
        Phenotype to score.
    combined_results : dict[str, list[str]]
        Per-split combined-of-three stable KO lists, keyed by the
        ``"{phenotype}_train(...),test({dataset})"`` convention.
    individual_results : dict[str, dict[str, list[str]]]
        Per-(dataset, phenotype) held-out-alone stable KO lists.
    ko_to_cluster : dict[str, int]
        Cluster mapping for the phenotype.
    pathway_kos : set[str]
        KOs on the row's assigned KEGG reference pathway map.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ``((shared_total, shared_in_pathway), (unique_total, unique_in_pathway))``.
    """
    cluster_to_kos = _invert_cluster_mapping(ko_to_cluster)
    shared_total = shared_hit = unique_total = unique_hit = 0
    for ds in HELD_OUT_DATASETS:
        combined_key = next(
            (
                k
                for k in combined_results
                if k.startswith(f"{phenotype}_train(") and k.endswith(f",test({ds})")
            ),
            None,
        )
        if (
            combined_key is None
            or ds not in individual_results
            or phenotype not in individual_results[ds]
        ):
            continue
        combined_clusters = _kos_to_cluster_ids(
            combined_results[combined_key], ko_to_cluster
        )
        individual_clusters = _kos_to_cluster_ids(
            individual_results[ds][phenotype], ko_to_cluster
        )
        shared = combined_clusters & individual_clusters
        unique = individual_clusters - combined_clusters
        shared_total += len(shared)
        unique_total += len(unique)
        if pathway_kos:
            shared_hit += sum(
                1 for c in shared if _cluster_in_pathway(c, cluster_to_kos, pathway_kos)
            )
            unique_hit += sum(
                1 for c in unique if _cluster_in_pathway(c, cluster_to_kos, pathway_kos)
            )
    return (shared_total, shared_hit), (unique_total, unique_hit)


def _concordant_shared_kos(
    phenotype: str,
    conc_combined: dict[str, list[str]],
    conc_individual: dict[str, dict[str, list[str]]],
) -> set[str]:
    """Return KOs shared between the three-dataset and held-out-alone concordant models.

    A KO counts as shared for a held-out dataset when it appears in both the
    combined-of-three concordant stable list and the held-out-alone concordant
    stable list; the union is taken across the three held-out datasets.

    Parameters
    ----------
    phenotype : str
        Phenotype to score.
    conc_combined : dict[str, list[str]]
        Per-split combined-of-three concordant stable KO lists.
    conc_individual : dict[str, dict[str, list[str]]]
        Per-(dataset, phenotype) held-out-alone concordant stable KO lists.

    Returns
    -------
    set[str]
        KOs shared across at least one held-out comparison.
    """
    shared: set[str] = set()
    for ds in HELD_OUT_DATASETS:
        combined_key = next(
            (
                k
                for k in conc_combined
                if k.startswith(f"{phenotype}_train(") and k.endswith(f",test({ds})")
            ),
            None,
        )
        if (
            combined_key is None
            or ds not in conc_individual
            or phenotype not in conc_individual[ds]
        ):
            continue
        shared |= set(conc_combined[combined_key]) & set(conc_individual[ds][phenotype])
    return shared


def _recurrence(
    phenotype: str,
    ko: str,
    conc_combined: dict[str, list[str]],
    conc_individual: dict[str, dict[str, list[str]]],
) -> int:
    """Count held-out datasets in which a KO is concordant shared-stable.

    Parameters
    ----------
    phenotype : str
        Phenotype to score.
    ko : str
        KO identifier to test.
    conc_combined : dict[str, list[str]]
        Concordant combined-of-three stable KO lists.
    conc_individual : dict[str, dict[str, list[str]]]
        Concordant held-out-alone stable KO lists.

    Returns
    -------
    int
        Number of held-out datasets (0--3) in which ``ko`` appears in both the
        combined and held-out-alone concordant stable lists.
    """
    count = 0
    for ds in HELD_OUT_DATASETS:
        combined_key = next(
            (
                k
                for k in conc_combined
                if k.startswith(f"{phenotype}_train(") and k.endswith(f",test({ds})")
            ),
            None,
        )
        if (
            combined_key is None
            or ds not in conc_individual
            or phenotype not in conc_individual[ds]
        ):
            continue
        if ko in set(conc_combined[combined_key]) & set(conc_individual[ds][phenotype]):
            count += 1
    return count


def _format_pct(hits: int, denom: int) -> str:
    """Format a hits/denom fraction as ``"NN\\%"`` (or ``"--"`` when empty)."""
    if denom == 0:
        return "--"
    return f"{hits / denom * 100:.0f}\\%"


def _shared_cluster_representatives(
    phenotype: str,
    conc_combined: dict[str, list[str]],
    conc_individual: dict[str, dict[str, list[str]]],
    ko_to_cluster: dict[str, int],
    cluster_to_kos: dict[int, set[str]],
    pathway_kos: set[str],
) -> list[str]:
    """Return one representative KO per shared concordant-stable cluster.

    A cluster counts as shared for a held-out comparison when it is present in
    both the combined-of-three and held-out-alone concordant models. Each shared
    cluster contributes its highest-recurrence member KO (ties broken by KO id);
    clusters are ordered pathway-resident first, then by the number of held-out
    comparisons in which they recur, then by representative KO id.

    Parameters
    ----------
    phenotype : str
        Phenotype to score.
    conc_combined, conc_individual : dict
        Concordant combined-of-three and held-out-alone stable KO lists.
    ko_to_cluster : dict[str, int]
        SHAP-supervised redundancy-cluster mapping for the phenotype.
    cluster_to_kos : dict[int, set[str]]
        Inverse of ``ko_to_cluster``.
    pathway_kos : set[str]
        KOs on the phenotype's assigned KEGG reference pathway map(s).

    Returns
    -------
    list[str]
        Representative KOs, ordered for display (empty when no cluster is shared).
    """
    cluster_recurrence: dict[str | int, int] = {}
    cluster_members: dict[str | int, set[str]] = {}
    for ds in HELD_OUT_DATASETS:
        combined_key = next(
            (
                k
                for k in conc_combined
                if k.startswith(f"{phenotype}_train(") and k.endswith(f",test({ds})")
            ),
            None,
        )
        if (
            combined_key is None
            or ds not in conc_individual
            or phenotype not in conc_individual[ds]
        ):
            continue
        combined_kos = conc_combined[combined_key]
        individual_kos = conc_individual[ds][phenotype]
        combined_clusters = _kos_to_cluster_ids(combined_kos, ko_to_cluster)
        individual_clusters = _kos_to_cluster_ids(individual_kos, ko_to_cluster)
        for cid in combined_clusters & individual_clusters:
            cluster_recurrence[cid] = cluster_recurrence.get(cid, 0) + 1
            members = cluster_members.setdefault(cid, set())
            for ko in (*combined_kos, *individual_kos):
                if ko_to_cluster.get(ko, f"sing:{ko}") == cid:
                    members.add(ko)
    if not cluster_recurrence:
        return []

    def _representative(cid: str | int) -> str:
        return sorted(
            cluster_members[cid],
            key=lambda ko: (
                -_recurrence(phenotype, ko, conc_combined, conc_individual),
                ko,
            ),
        )[0]

    reps = {cid: _representative(cid) for cid in cluster_recurrence}

    def _resident(cid: str | int) -> bool:
        return bool(pathway_kos) and _cluster_in_pathway(
            cid, cluster_to_kos, pathway_kos
        )

    ordered = sorted(
        cluster_recurrence,
        key=lambda cid: (not _resident(cid), -cluster_recurrence[cid], reps[cid]),
    )
    return [reps[cid] for cid in ordered]


def _example_cell(
    phenotype: str,
    conc_combined: dict[str, list[str]],
    conc_individual: dict[str, dict[str, list[str]]],
    ko_to_cluster: dict[str, int],
    cluster_to_kos: dict[int, set[str]],
    pathway_kos: set[str],
    ko_dict: dict[str, Any],
    max_examples: int = 2,
) -> str:
    """Build the example column from the concordant shared-stable feature KOs.

    Features whose redundancy cluster touches the phenotype's KEGG reference
    pathway map are preferred; off-pathway features are used only when no
    on-pathway feature is stable. Within the chosen pool, features are ranked by
    recurrence across held-out datasets (ties broken by KO id) and the top
    ``max_examples`` are rendered. When the raw-KO intersection is empty but
    clusters are shared, the column falls back to a representative KO per shared
    cluster via :func:`_shared_cluster_representatives`.

    Parameters
    ----------
    phenotype : str
        Phenotype to score.
    conc_combined, conc_individual : dict
        Concordant combined-of-three and held-out-alone stable KO lists.
    ko_to_cluster : dict[str, int]
        SHAP-supervised redundancy-cluster mapping for the phenotype.
    cluster_to_kos : dict[int, set[str]]
        Inverse of ``ko_to_cluster``.
    pathway_kos : set[str]
        KOs on the phenotype's assigned KEGG reference pathway map(s).
    ko_dict : dict[str, Any]
        KO dictionary loaded from ``KO_dictionary.json``.
    max_examples : int, optional
        Maximum number of example KOs to render. Defaults to ``2``.

    Returns
    -------
    str
        Rendered ``"K..... name; K..... name"`` cell, or ``"---"`` when the
        phenotype has no shared cluster at all.
    """
    features = _concordant_shared_kos(phenotype, conc_combined, conc_individual)
    if features:

        def _cluster_touches_pathway(ko: str) -> bool:
            cid = ko_to_cluster.get(ko)
            if cid is None:
                return ko in pathway_kos
            return _cluster_in_pathway(cid, cluster_to_kos, pathway_kos)

        on_pathway = [
            ko for ko in features if pathway_kos and _cluster_touches_pathway(ko)
        ]
        pool = on_pathway or list(features)
        chosen = sorted(
            pool,
            key=lambda ko: (
                -_recurrence(phenotype, ko, conc_combined, conc_individual),
                ko,
            ),
        )[:max_examples]
    else:
        chosen = _shared_cluster_representatives(
            phenotype,
            conc_combined,
            conc_individual,
            ko_to_cluster,
            cluster_to_kos,
            pathway_kos,
        )[:max_examples]

    if not chosen:
        return "---"
    return "; ".join(f"{ko} {_display_ko_name(ko, ko_dict)}" for ko in chosen)


def _pathway_cell(phenotype: str) -> str:
    """Render the "KEGG pathway map" cell for a phenotype."""
    map_ids = PHENOTYPE_TO_PATHWAY_MAPS.get(phenotype, ())
    if not map_ids:
        return "---"
    blocks: list[str] = []
    for map_id in map_ids:
        name_render = PATHWAY_NAME_WRAPS.get(
            map_id,
            f"\\textit{{{PATHWAY_NAMES.get(map_id, map_id)}}}",
        )
        blocks.append(f"{map_id} \\\\ {name_render}")
    return "\\makecell[l]{" + " \\\\[2pt] ".join(blocks) + "}"


def build_table(
    full_combined_json: Path = Path(
        "data/outputs/figure4/combined_splits_shap_features.json"
    ),
    full_individual_json: Path = Path(
        "data/outputs/figure4/individual_datasets_shap_features.json"
    ),
    conc_combined_json: Path = Path(
        "data/outputs/figure5/figure5b_combined_splits_shap_features.json"
    ),
    conc_individual_json: Path = Path(
        "data/outputs/figure5/figure5b_individual_datasets_shap_features.json"
    ),
    cluster_json: Path = Path("data/outputs/clustering/ko_clusters_shap_hclust.json"),
    ko_dictionary_json: Path = Path("data/external/mapping/KO_dictionary.json"),
    output_path: Path = Path("sections/table_main_feature_comparison.tex"),
    highlight_phenotype: str = "Histidine",
) -> None:
    """Regenerate ``sections/table_main_feature_comparison.tex`` from source.

    Parameters
    ----------
    full_combined_json, full_individual_json : Path
        Raw SHAP-stable KO lists for the combined-of-three and held-out-alone
        models under full-data training.
    conc_combined_json, conc_individual_json : Path
        Same under concordant-only training.
    cluster_json : Path
        SHAP-supervised cluster JSON (per-phenotype KO -> cluster id).
    ko_dictionary_json : Path
        KO dictionary JSON used to render enzyme names in the example column.
    output_path : Path
        Destination ``.tex`` file (overwritten).
    highlight_phenotype : str, optional
        Phenotype whose row receives bold emphasis on the headline numeric
        cells. Defaults to ``"Histidine"``.
    """
    with open(full_combined_json) as handle:
        comb_full: dict[str, list[str]] = json.load(handle)
    with open(full_individual_json) as handle:
        ind_full: dict[str, dict[str, list[str]]] = json.load(handle)
    with open(conc_combined_json) as handle:
        comb_conc: dict[str, list[str]] = json.load(handle)
    with open(conc_individual_json) as handle:
        ind_conc: dict[str, dict[str, list[str]]] = json.load(handle)
    with open(cluster_json) as handle:
        cluster_mapping: dict[str, dict[str, int]] = json.load(handle)
    with open(ko_dictionary_json) as handle:
        ko_dict: dict[str, Any] = json.load(handle)

    def _row_cells(phenotype: str) -> tuple[str, str, str, str, str]:
        ko_to_cluster = cluster_mapping.get(phenotype, {})
        pathway_kos = pathway_kos_for_phenotype(phenotype)
        cluster_to_kos = _invert_cluster_mapping(ko_to_cluster)

        (sf_t, sf_h), (uf_t, uf_h) = _shared_unique_counts(
            phenotype, comb_full, ind_full, ko_to_cluster, pathway_kos
        )
        (sc_t, sc_h), (uc_t, uc_h) = _shared_unique_counts(
            phenotype, comb_conc, ind_conc, ko_to_cluster, pathway_kos
        )

        def _fmt(total: int, hits: int) -> str:
            if total == 0:
                return "0"
            return f"{total} ({_format_pct(hits, total)})"

        shared_cell = f"{_fmt(sf_t, sf_h)} $\\rightarrow$ {_fmt(sc_t, sc_h)}"
        unique_cell = f"{_fmt(uf_t, uf_h)} $\\rightarrow$ {_fmt(uc_t, uc_h)}"

        example = _example_cell(
            phenotype,
            comb_conc,
            ind_conc,
            ko_to_cluster,
            cluster_to_kos,
            pathway_kos,
            ko_dict,
        )

        return (
            phenotype,
            _pathway_cell(phenotype),
            shared_cell,
            unique_cell,
            example,
        )

    lines: list[str] = []
    lines.append("\\FloatBarrier")
    lines.append("\\begin{table}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.25}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{|l|l|c|c|p{4.6cm}|}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Phenotype} & "
        "\\textbf{KEGG pathway map(s)} & "
        "\\makecell{\\textbf{Shared stable clusters} \\\\ \\textbf{(\\% in pathway)} \\\\ \\textbf{Full $\\rightarrow$ Conc.}} & "
        "\\makecell{\\textbf{Unique stable clusters} \\\\ \\textbf{(\\% in pathway)} \\\\ \\textbf{Full $\\rightarrow$ Conc.}} & "
        "\\makecell[l]{\\textbf{Example shared concordant-} \\\\ \\textbf{stable feature}} \\\\"
    )
    lines.append("\\hline")

    def _emit_row(phenotype: str) -> None:
        ph, pathway_cell, shared_cell, pct_cell, example = _row_cells(phenotype)
        if phenotype == highlight_phenotype:
            shared_render = f"\\textbf{{{shared_cell}}}"
            pct_render = f"\\textbf{{{pct_cell}}}"
        else:
            shared_render = shared_cell
            pct_render = pct_cell
        lines.append(
            f"{ph} & {pathway_cell} & {shared_render} & {pct_render} & {example} \\\\"
        )
        lines.append("\\hline")

    for phenotype in PHENOTYPE_ORDER:
        _emit_row(phenotype)

    lines.append("\\end{tabular}%")
    lines.append("}")
    # Asterisked caption: prints "Table N" without re-incrementing the counter
    # (already advanced in the Legends section of tables_figures.tex).
    lines.append("\\caption*{\\textbf{Table \\arabic{table}}}")
    lines.append("\\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"LaTeX table body written to {output_path}")


if __name__ == "__main__":
    build_table()
