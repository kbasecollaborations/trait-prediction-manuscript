#!/usr/bin/env python3
"""Build a per-genome Biolog assay-provenance table for the deposited genomes.

The curator-maintained master spreadsheets record provenance one row per strain,
with the ``PATRIC Genome ID`` cell holding one or more comma-separated genome
identifiers. This script explodes those cells to one row per genome, restricts
the result to the genomes deposited in ``biolog_genomes.tsv``, and merges the two
spreadsheet generations so that neither is silently dropped.

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
        One row per genome identifier, indexed by ``genome_id``. Trailing blank
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
    for _, row in sheet.iterrows():
        genome_ids = split_genome_ids(row.get("PATRIC Genome ID"))
        if not genome_ids:
            continue
        fields = {out: row.get(src) for src, out in _FIELDS.items() if src in sheet.columns}
        for genome_id in genome_ids:
            records.append({"genome_id": genome_id, "provenance_source": label, **fields})
    frame = pd.DataFrame.from_records(records)
    return frame.drop_duplicates(subset="genome_id", keep="first").set_index("genome_id")


def build() -> pd.DataFrame:
    """Join spreadsheet provenance onto the deposited Biolog genome table.

    Returns
    -------
    pd.DataFrame
        One row per deposited genome carrying a spreadsheet record, in the
        genome table's original order.
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

    out = genomes[["genome_id", "organism"]].merge(
        merged.reset_index(), on="genome_id", how="inner"
    )
    out = out[out["provenance_source"].notna()]
    ordered = ["genome_id", "organism", "provenance_source", "assay_type", "data_source"]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def main() -> None:
    table = build()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print(f"Wrote {OUTPUT_FILE} ({len(table)} genomes)")
    print("\n  by assay type:")
    print(table["assay_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
