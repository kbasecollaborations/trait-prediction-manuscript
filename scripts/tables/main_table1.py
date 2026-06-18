#!/usr/bin/env python3
"""Regenerate the main-text Table 1 (concordance-vs-full feature comparison).

Run with ``uv run python -m scripts.tables.main_table1``.
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

# Phenotype display order: by canonical pathway map for natural grouping
# (carbohydrates first, then amino acids), Histidine first as worked example.
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

# Manual two-line italic splits for KEGG pathway names that are too long to
# fit on a single line of the wrapped "KEGG pathway map" column. Keys are
# KEGG pathway IDs; values are the full LaTeX render with an explicit \\
# break between two italic spans. Names not listed render as a single
# italic line via the canonical PATHWAY_NAMES entry.
PATHWAY_NAME_WRAPS: dict[str, str] = {
    "map00040": "\\textit{Pentose and glucuronate} \\\\ \\textit{interconversions}",
    "map00250": "\\textit{Alanine, aspartate and} \\\\ \\textit{glutamate metabolism}",
    "map00260": "\\textit{Glycine, serine and} \\\\ \\textit{threonine metabolism}",
}

# Hand-curated example concordant-stable KOs for the rightmost column.
# These are biologically representative concordant-stable signals selected
# from the per-phenotype SHAP-stable feature lists; the automatic
# "must recur across held-out splits" rule is too strict and leaves most
# rows empty, so we keep these as authoritative override.
EXAMPLE_KO_OVERRIDES: dict[str, str] = {
    "Histidine": "K01468 imidazolonepropionase; K01712 urocanate hydratase",
    "Galacturonic-Acid": "K18981 uronate dehydrogenase",
    "Galactose": "K01190 beta-galactosidase",
    "Glucose": "K01785 aldose 1-epimerase",
    "Cellobiose": "K05349 beta-glucosidase; K01443 NAG-6-P deacetylase",
    "Maltose": "K01187 alpha-glucosidase",
    "Sucrose": "K01187 alpha-glucosidase; K01193 beta-fructofuranosidase",
    "Fructose": "K00882 1-phosphofructokinase; K02770 PTS fructose IIC",
    "Mannitol": (
        "K02027 sugar transport SBP; K10228 mannitol permease; "
        "K00009 mannitol-1-P 5-dehydrogenase"
    ),
    "m-Inositol": "K00010 myo-inositol 2-dehydrogenase; K03335 inosose dehydratase",
    "Glycerol": "K02440 glycerol uptake facilitator; K02444 DeoR regulator",
}


def _short_ko_name(ko_id: str, ko_dict: dict[str, Any]) -> str:
    """Return a short, human-readable name for a KO identifier.

    Parameters
    ----------
    ko_id : str
        KO identifier such as ``"K01712"``.
    ko_dict : dict[str, Any]
        KO dictionary loaded from ``KO_dictionary.json``.

    Returns
    -------
    str
        Short, LaTeX-safe enzyme name (no EC bracket, no trailing semicolons).
    """
    ko_key = f"KO:{ko_id}"
    term = ko_dict.get("term_hash", {}).get(ko_key)
    if not term:
        return ""
    name = term.get("name", "")
    name = name.split(";")[0].strip()
    name = re.sub(r"\s*\[EC:[^]]+\]\s*$", "", name).strip()
    name = name.replace("_", r"\_")
    return name


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
        shared |= set(conc_combined[combined_key]) & set(
            conc_individual[ds][phenotype]
        )
    return shared


def _assert_overrides_are_shared_stable(
    conc_combined: dict[str, list[str]],
    conc_individual: dict[str, dict[str, list[str]]],
) -> None:
    """Verify each ``EXAMPLE_KO_OVERRIDES`` KO is a genuine concordant shared-stable feature.

    The example column is hand-curated (see ``EXAMPLE_KO_OVERRIDES``). This guard
    enforces that every listed KO actually recurs as a shared concordant-stable
    feature for its phenotype, so the curated examples cannot silently drift away
    from the data the figure and table report.

    Parameters
    ----------
    conc_combined : dict[str, list[str]]
        Concordant combined-of-three stable KO lists.
    conc_individual : dict[str, dict[str, list[str]]]
        Concordant held-out-alone stable KO lists.

    Raises
    ------
    ValueError
        If any override KO is absent from its phenotype's concordant
        shared-stable set.
    """
    problems: list[str] = []
    for phenotype, text in EXAMPLE_KO_OVERRIDES.items():
        shared = _concordant_shared_kos(phenotype, conc_combined, conc_individual)
        for ko in re.findall(r"K\d{5}", text):
            if ko not in shared:
                problems.append(f"{phenotype}:{ko}")
    if problems:
        raise ValueError(
            "EXAMPLE_KO_OVERRIDES contains KOs that are not concordant "
            "shared-stable features: " + ", ".join(problems)
        )


def _format_pct(hits: int, denom: int) -> str:
    """Format a hits/denom fraction as ``"NN\\%"`` (or ``"--"`` when empty)."""
    if denom == 0:
        return "--"
    return f"{hits / denom * 100:.0f}\\%"


def _filter_example_to_pathway_clusters(
    example_text: str,
    ko_to_cluster: dict[str, int] | None,
    cluster_to_kos: dict[int, set[str]],
    pathway_kos: set[str],
) -> str:
    """Drop semicolon-separated KO segments whose cluster is not pathway-resident.

    Parameters
    ----------
    example_text : str
        Display-ready example cell value.
    ko_to_cluster : dict[str, int] | None
        Cluster mapping for the phenotype.
    cluster_to_kos : dict[int, set[str]]
        Inverse of ``ko_to_cluster``.
    pathway_kos : set[str]
        KOs on the assigned KEGG reference pathway map.

    Returns
    -------
    str
        Filtered cell value, or ``"---"`` if every segment was dropped.
    """
    if not pathway_kos:
        return example_text
    segments = [s.strip() for s in example_text.split(";") if s.strip()]
    keep: list[str] = []
    for seg in segments:
        match = re.match(r"^(K\d{5})\b", seg)
        if not match:
            continue
        ko = match.group(1)
        cid: str | int | None = (
            ko_to_cluster.get(ko) if ko_to_cluster else None
        )
        if cid is None:
            in_path = ko in pathway_kos
        else:
            in_path = _cluster_in_pathway(cid, cluster_to_kos, pathway_kos)
        if in_path:
            keep.append(seg)
    if not keep:
        return "---"
    return "; ".join(keep)


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
    cluster_json: Path = Path(
        "data/outputs/clustering/ko_clusters_shap_hclust.json"
    ),
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

    # Guard: every curated example KO must be a genuine concordant shared-stable
    # feature (the pathway-residency filter alone does not check this).
    _assert_overrides_are_shared_stable(comb_conc, ind_conc)

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

        example_override = EXAMPLE_KO_OVERRIDES.get(phenotype, "---")
        example = _filter_example_to_pathway_clusters(
            example_override, ko_to_cluster, cluster_to_kos, pathway_kos
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
        "\\makecell[l]{\\textbf{Example shared concordant-} \\\\ \\textbf{stable KO in pathway}} \\\\"
    )
    lines.append("\\hline")

    def _emit_row(phenotype: str) -> None:
        ph, pathway_cell, shared_cell, pct_cell, example = _row_cells(phenotype)
        # Use bold-text emphasis on the headline numeric cells only.
        if phenotype == highlight_phenotype:
            shared_render = f"\\textbf{{{shared_cell}}}"
            pct_render = f"\\textbf{{{pct_cell}}}"
        else:
            shared_render = shared_cell
            pct_render = pct_cell
        lines.append(
            f"{ph} & {pathway_cell} & {shared_render} & {pct_render} & "
            f"{example} \\\\"
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
