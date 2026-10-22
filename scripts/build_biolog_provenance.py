#!/usr/bin/env python3
"""Build a per-genome Biolog assay-provenance table for the deposited genomes.

The curator-maintained master spreadsheets record provenance one row per strain,
with the ``PATRIC Genome ID`` cell holding one or more comma-separated genome
identifiers. This script explodes those cells to one row per genome, restricts
the result to the genomes deposited in ``biolog_genomes.tsv``, and merges the two
spreadsheet generations so that neither is silently dropped.

Genomes absent from both spreadsheets are the Cedar Creek (CCESR) isolates held in
public KBase narrative 89639; ``kbase_ccesr_links.tsv`` lists them with their KBase
phenotype sets, so every deposited genome gets a row.

Run with::

    uv run python -m scripts.build_biolog_provenance
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

GENOME_TABLE = Path("data/zenodo/biolog_genomes.tsv")
SHEETS: tuple[tuple[str, Path], ...] = (
    ("master_428", Path("data/Biolog_metadata/Biolog_Class_Master_428.xlsx")),
    ("master_178", Path("data/Biolog_metadata/S1_Biolog_Class_Master_178.xlsx")),
)
"""Spreadsheet generations in precedence order; earlier entries win on conflict."""

OUTPUT_FILE = Path("data/zenodo/biolog_assay_provenance.tsv")

CCESR_LINKS = Path("data/Biolog_metadata/kbase_ccesr_links.tsv")
"""Public KBase copies of the CCESR genomes (narrative 89639) and their phenotype sets."""

CCESR_SOURCE = "Cedar Creek (CCESR) isolate collection"
CONTRIBUTED_SOURCE = "unpublished contribution"
NOT_RECORDED = "source not recorded"

MISATTRIBUTED: frozenset[str] = frozenset({"615.104"})
"""Rows whose PMIDs describe other organisms (here a different Serratia strain and a plant virus)."""

PROXY_IDS: frozenset[str] = frozenset({"2654975.3", "470.9249", "1663.246", "83333.108"})
"""Reviewed pairings with another strain's record that no text cue flags.

2654975.3 is a Paenibacillus genome carrying the Xanthomonas vesicatoria row; 470.9249
(Acinetobacter baumannii AB043) carries the Acinetobacter sp. ATCC 17979 row; 1663.246
(Arthrobacter PCH119) carries the calls of Arthrobacter PCH103, not those of its PCH119 set;
83333.108 (Escherichia coli K-12 ER3446, a recombinant built for a 2015 study) carries the
2010 Model SEED E. coli K-12 row.
"""

DONOR_CULTURES: dict[str, str] = {"1663.246": "PCH103"}
"""Proxy genomes carrying another PCH culture's calls, mapped to that culture's code."""

_PROXY_CUE = re.compile(r"close genome|closest", re.IGNORECASE)
_PMID_LIST = re.compile(r"\d{6,9}(?:\s*[,;/]\s*\d{6,9})*")
_PCH_CODE = re.compile(r"(PCH\d+)")

_ID_SEPARATORS = re.compile(r"[,;/\s]+")

_FIELDS: dict[str, str] = {
    "Assay type": "assay_type",
    "Data Source (PMID)": "data_source",
    "Genome Original Source": "genome_original_source",
    "Organisms name": "sheet_organism_name",
}


def split_genome_ids(cell: object) -> list[str]:
    """Split a ``PATRIC Genome ID`` cell into individual genome identifiers.

    Parameters
    ----------
    cell : object
        Raw cell value; may be missing, a single identifier, or several
        identifiers separated by commas, semicolons, slashes, or whitespace.

    Returns
    -------
    list[str]
        Stripped identifiers, empty if the cell holds no usable value.
    """
    if pd.isna(cell):
        return []
    return [token.strip() for token in _ID_SEPARATORS.split(str(cell)) if token.strip()]


def load_sheet(path: Path, label: str) -> pd.DataFrame:
    """Read one master spreadsheet as one row per genome identifier.

    Parameters
    ----------
    path : Path
        Spreadsheet to read; its first worksheet is used.
    label : str
        Value written to the ``provenance_source`` column for these rows.

    Returns
    -------
    pd.DataFrame
        One row per genome identifier, indexed by ``genome_id``, with a
        ``record_key`` naming the spreadsheet row it came from. Trailing blank
        and legend rows carrying no genome identifier are dropped.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    sheet = pd.read_excel(path, sheet_name=0, dtype=str)
    records: list[dict[str, object]] = []
    for index, row in sheet.iterrows():
        genome_ids = split_genome_ids(row.get("PATRIC Genome ID"))
        if not genome_ids:
            continue
        fields = {out: row.get(src) for src, out in _FIELDS.items() if src in sheet.columns}
        for genome_id in genome_ids:
            records.append(
                {
                    "genome_id": genome_id,
                    "provenance_source": label,
                    "record_key": f"{label}:{index}",
                    **fields,
                }
            )
    frame = pd.DataFrame.from_records(records)
    return frame.drop_duplicates(subset="genome_id", keep="first").set_index("genome_id")


def build() -> pd.DataFrame:
    """Join provenance onto the deposited Biolog genome table.

    Returns
    -------
    pd.DataFrame
        One row per deposited genome, in the genome table's original order.
        ``data_source`` holds PMIDs or a standard label for a source without one,
        ``genome_match`` whether the phenotype record names this genome
        (``exact``), is shared with other deposited genomes (``shared_record``),
        or comes from another strain (``proxy``), and ``kbase_phenotypeset``
        the public KBase copy of a CCESR genome's growth calls.

    Raises
    ------
    ValueError
        If a genome has neither a spreadsheet record nor a CCESR link.
    """
    genomes = pd.read_csv(GENOME_TABLE, sep="\t", dtype=str)
    genomes["genome_id"] = genomes["genome_id"].str.strip()

    merged = pd.DataFrame(index=pd.Index(genomes["genome_id"], name="genome_id"))
    for label, path in SHEETS:
        sheet = load_sheet(path, label)
        incoming = sheet.reindex(merged.index)
        for column in incoming.columns:
            if column not in merged.columns:
                merged[column] = pd.NA
            merged[column] = merged[column].fillna(incoming[column])

    links = pd.read_csv(CCESR_LINKS, sep="\t", dtype=str, keep_default_na=False)
    links = links.set_index("genome_id")

    out = genomes[["genome_id", "organism"]].merge(
        merged.reset_index(), on="genome_id", how="left"
    )
    unrecorded = out["provenance_source"].isna()
    in_ccesr = out["genome_id"].isin(links.index)
    if (unrecorded & ~in_ccesr).any():
        missing = out.loc[unrecorded & ~in_ccesr, "genome_id"].tolist()
        raise ValueError(f"no provenance for {missing}")
    out.loc[unrecorded, ["provenance_source", "assay_type"]] = ["kbase_89639", "Biolog"]

    label = out["data_source"].fillna("").str.strip()
    contributed = ~unrecorded & label.ne("") & ~label.str.fullmatch(_PMID_LIST)
    misattributed = out["genome_id"].isin(MISATTRIBUTED)
    out.loc[in_ccesr & (contributed | unrecorded), "data_source"] = CCESR_SOURCE
    out.loc[~in_ccesr & contributed, "data_source"] = CONTRIBUTED_SOURCE
    out.loc[misattributed, "data_source"] = NOT_RECORDED

    # Genomes with the same PCH code share that culture's phenotype record;
    # DONOR_CULTURES moves a genome that carries another culture's calls.
    culture = out["organism"].str.extract(_PCH_CODE)[0]
    culture = out["genome_id"].map(DONOR_CULTURES).fillna(culture)
    record = ("ccesr:" + culture).fillna(out["record_key"])
    proxy = out["genome_original_source"].fillna("").str.contains(_PROXY_CUE)
    out["genome_match"] = "exact"
    out.loc[record.map(record.value_counts()).gt(1), "genome_match"] = "shared_record"
    out.loc[proxy | out["genome_id"].isin(PROXY_IDS), "genome_match"] = "proxy"
    out["kbase_phenotypeset"] = out["genome_id"].map(links["kbase_phenotypeset"])

    out = out.drop(columns="record_key")
    ordered = [
        "genome_id",
        "organism",
        "provenance_source",
        "assay_type",
        "data_source",
        "genome_match",
        "kbase_phenotypeset",
    ]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def main() -> None:
    table = build()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print(f"Wrote {OUTPUT_FILE} ({len(table)} genomes)")
    for column in ("assay_type", "provenance_source", "genome_match"):
        print(f"\n  by {column}:")
        print(table[column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
