"""Build the authoritative substrate identity table for all four phenotype datasets.

Every published phenotype column is mapped to exactly one raw source column, with the
molecule and, where known, the Biolog plate well recorded alongside it. ``CORRECTIONS``
lists every binding that differs from the original pipeline, which stripped leading
stereochemical descriptors and resolved the resulting name collisions by source column
order rather than by chemistry.

Outputs
-------
``data/processed/substrate_identity.csv``
    Full table: one row per (dataset, published phenotype).
``data/zenodo/substrate_identity_all.csv``
    The same full table, as published in the Zenodo deposit.
``data/zenodo/substrate_identity_common15.csv``
    The 15 phenotypes shared by all four datasets, for the Zenodo deposit and the
    supplementary data listing.

Run with::

    uv run python -m scripts.build_substrate_identity
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DIR: Final[Path] = REPO_ROOT / "data/raw/phenotypes"
NAME_MAP: Final[Path] = RAW_DIR / "phenotype_common_names.csv"
FULL_OUT: Final[Path] = REPO_ROOT / "data/processed/substrate_identity.csv"
ALL_OUT: Final[Path] = REPO_ROOT / "data/zenodo/substrate_identity_all.csv"
COMMON15_OUT: Final[Path] = REPO_ROOT / "data/zenodo/substrate_identity_common15.csv"

#: Datasets keyed by the raw handoff file stem and the name-map column.
DATASETS: Final[dict[str, str]] = {
    "lit": "Lit",
    "atleaf": "AT Leaf",
    "pmi": "PMI",
}

COMMON_PHENOTYPES: Final[list[str]] = [
    "Alanine",
    "Arginine",
    "Cellobiose",
    "Fructose",
    "Galactose",
    "Galacturonic-Acid",
    "Glucose",
    "Glycerol",
    "Histidine",
    "Maltose",
    "Mannitol",
    "Mannose",
    "Serine",
    "Sucrose",
    "m-Inositol",
]

#: Reviewed bindings, keyed by ``(dataset, published phenotype)``.
#:
#: ``source`` is the exact raw column header to bind to. ``rename_to`` renames the
#: published file where the generic name cannot denote a single molecule. ``note`` is
#: published as ``binding_note``, so it states the evidence for the binding only.
CORRECTIONS: Final[dict[tuple[str, str], dict[str, str]]] = {
    # Populus: columns bound to the wrong molecule by last-wins.
    ("pmi", "Glucose"): {
        "source": "a-D-Glucose",
        "note": "No Populus profile grows on L-glucose (PM2A C2, 0/58); alpha-D-glucose (PM1 "
        "C9, 58/58) is D-glucose, the substrate every other dataset assays.",
    },
    ("pmi", "Arabinose"): {
        "source": "L-Arabinose",
        "note": "L-arabinose (PM1 A2, 51/58) is the plant pentose that ATLeaf and Marine assay; "
        "D-arabinose (PM2A B5, 13/58) is the outlier.",
    },
    ("pmi", "Aspartic-Acid"): {
        "source": "L-Aspartic Acid",
        "note": "L-aspartate (PM1 A7, 44/58) is the proteinogenic, anaplerotic form and "
        "the one bound in the Biolog compilation; D-aspartate (PM1 D2, 9/58) needs a racemase.",
    },
    ("pmi", "Fucose"): {
        "source": "L-Fucose",
        "note": "L-fucose (PM1 B4, 31/58) is the abundant deoxyhexose modelled by GapMind; "
        "D-fucose (PM2A B11, 5/58) is rare and often only a gratuitous inducer.",
    },
    ("pmi", "Arabitol"): {
        "source": "D-Arabitol",
        "note": "D-arabitol (PM2A B6, 52/58) is the widespread form and the one the "
        "Biolog compilation assayed; L-arabitol (PM2A B7, 19/58) is the outlier.",
    },
    ("pmi", "Galactonic-Acid-g-Lactone"): {
        "source": "D-Galactonic Acid-g-Lactone",
        "note": "D-galactonolactone (PM1 C2, 40/58), which yields the dgo-pathway substrate "
        "D-galactonate, is the only galactonolactone in the Biolog compilation; "
        "L-galactonolactone (PM1 H9, 13/58) is the outlier.",
    },
    ("pmi", "Methyl-D-Glucoside"): {
        "source": "b-Methyl-D-Glucoside",
        "note": "beta-methyl-D-glucoside (PM1 E8, 23/58) is the form the Biolog compilation "
        "assayed; the anomers do not interconvert (locked methyl acetal) and use "
        "different uptake systems, so the alpha- well (PM2A C6, 4/58) is a different "
        "trait and is listed in discarded.",
    },
    # Populus: constitutional isomers that no generic name can denote.
    ("pmi", "Hydroxy-Butyric-Acid"): {
        "source": "g-Hydroxy Butyric Acid",
        "rename_to": "g-Hydroxy-Butyric-Acid",
        "note": "alpha/beta/gamma-hydroxybutyrate are 2-, 3- and 4-hydroxybutyrate: "
        "constitutional isomers on unrelated pathways. Bound to the gamma isomer (PM2A "
        "E9), as in the Biolog compilation.",
    },
    ("pmi", "Cyclodextrin"): {
        "source": "g-Cyclodextrin",
        "rename_to": "g-Cyclodextrin",
        "note": "alpha/beta/gamma-cyclodextrin are cyclic glucans of 6, 7 and 8 glucose units "
        "with different formulae. Bound to gamma-cyclodextrin (PM2A A5); "
        "beta-cyclodextrin is published as a separate column.",
    },
    # Populus: the workbook names PM1 D8 'b-Methyl-D-Galactoside', but Biolog Cat. #12111,
    # the opm R package and DuctApe (CAS 3396-99-4 = PubChem CID 76935) all give that well
    # as the alpha anomer; the beta anomer is PM2A C7 (CAS 1824-94-8, CID 94214). PM1 and
    # PM2A share no carbon source, so the two columns are not a duplicate measurement.
    ("pmi", "b-Methyl-D-Galactoside"): {
        "source": "b-Methyl-D-Galactoside",
        "rename_to": "a-Methyl-D-Galactoside",
        "note": "Workbook row 47 lies between a-Keto-Butyric Acid (PM1 D7) and a-D-Lactose "
        "(PM1 D9), so it is PM1 well D8, alpha-methyl-D-galactoside (a MelB/MelA "
        "substrate); the workbook labels it 'b-Methyl-D-Galactoside', the name of PM2A C7.",
    },
    ("pmi", "b-Methyl-D-Galactoside.1"): {
        "source": "b-Methyl-D-Galactoside.1",
        "rename_to": "b-Methyl-D-Galactoside",
        "note": "Workbook row 131 is PM2A well C7, beta-methyl-D-galactoside (a LacY/LacZ "
        "substrate). The raw Populus matrix, data/raw/phenotypes/pmi_phenotypes.tsv, "
        "repeats the header 'b-Methyl-D-Galactoside', and source_column names this "
        "second copy 'b-Methyl-D-Galactoside.1'.",
    },
    # Literature: columns bound to the wrong molecule.
    ("lit", "Malic-Acid"): {
        "source": "Carbon-L-Malic-Acid",
        "note": "L-malate (226/270) is the TCA intermediate and the isomer bound in Populus; "
        "D-malate (151/305) is a separate trait that depends on the D-malate "
        "dehydrogenase DmlA.",
    },
    ("lit", "Fucose"): {
        "source": "Carbon-L-Fucose",
        "note": "L-fucose is the canonical deoxyhexose modelled by GapMind and the isomer "
        "bound in Populus.",
    },
    ("lit", "Hydroxy-Butyric-Acid"): {
        "source": "Carbon-g-Hydroxy-Butyric-Acid",
        "rename_to": "g-Hydroxy-Butyric-Acid",
        "note": "alpha- and gamma-hydroxybutyric acid (2- and 4-hydroxybutyrate) are "
        "constitutional isomers; bound to the gamma isomer, as in Populus.",
    },
    # Literature: N-Acetyl- was stripped, changing the compound.
    ("lit", "Glucosamine"): {
        "source": "Carbon-N-Acetyl-D-Glucosamine",
        "rename_to": "N-Acetyl-D-Glucosamine",
        "note": "N-acetyl-D-glucosamine (GlcNAc), the compound of the Populus "
        "N-Acetyl-D-Glucosamine column.",
    },
    ("lit", "Galactosamine"): {
        "source": "Carbon-N-Acetyl-D-Galactosamine",
        "rename_to": "N-Acetyl-D-Galactosamine",
        "note": "N-acetyl-D-galactosamine (GalNAc), the compound of the Populus "
        "N-Acetyl-D-Galactosamine column.",
    },
    ("lit", "Neuraminic-Acid"): {
        "source": "Carbon-N-Acetyl-Neuraminic-Acid",
        "rename_to": "N-Acetyl-Neuraminic-Acid",
        "note": "N-acetylneuraminic acid (sialic acid).",
    },
    # Marine: the original notebook hardcoded 'Galacturonate-lmw' -> 'Galacturonic-Acid',
    # binding the shared name to the pectin polysaccharide. ``marine_published_name`` no
    # longer applies that rename, so the monomer takes the shared name instead.
    ("marine", "Galacturonate"): {
        "source": "galacturonate",
        "rename_to": "Galacturonic-Acid",
        "note": "'D-Galacturonic acid sodium salt' (Gralka 2023 SI Table 2 row 80, 0.5 M, "
        "6 C atoms) is the monomer ATLeaf, Biolog and Populus assay. 'Galacturonate "
        "polysaccharides LM from apple' (row 137, dosed in %, no C-atom count) is a "
        "pectin polysaccharide, published as the separate column Galacturonate-lmw.",
    },
}

#: Populus molecule names where the workbook column name is not the compound.
PMI_MOLECULES: Final[dict[str, str]] = {
    "a-Methyl-D-Galactoside": "a-Methyl-D-Galactoside",
    "b-Methyl-D-Galactoside": "b-Methyl-D-Galactoside",
}

#: Biolog plate wells for the substrates that matter to the shared analysis or that were
#: corrected. Verified against the Biolog 00A-042 Rev D plate-map booklet and the DuctApe
#: reference table, which agree on 191 of 192 wells.
BIOLOG_WELLS: Final[dict[str, str]] = {
    "a-D-Glucose": "PM1 C9",
    "L-Glucose": "PM2A C2",
    "L-Arabinose": "PM1 A2",
    "D-Arabinose": "PM2A B5",
    "L-Aspartic Acid": "PM1 A7",
    "D-Aspartic Acid": "PM1 D2",
    "L-Fucose": "PM1 B4",
    "D-Fucose": "PM2A B11",
    "D-Arabitol": "PM2A B6",
    "L-Arabitol": "PM2A B7",
    "D-Galactonic Acid-g-Lactone": "PM1 C2",
    "L-Galactonic Acid-g-Lactone": "PM1 H9",
    "b-Methyl-D-Glucoside": "PM1 E8",
    "a-Methyl-D-Glucoside": "PM2A C6",
    "a-Hydroxy Butyric Acid": "PM1 E7",
    "b-Hydroxy Butyric Acid": "PM2A E8",
    "g-Hydroxy Butyric Acid": "PM2A E9",
    "a-Cyclodextrin": "PM2A A3",
    "b-Cyclodextrin": "PM2A A4",
    "g-Cyclodextrin": "PM2A A5",
    "L-Malic Acid": "PM1 G12",
    "D-Malic Acid": "PM1 G11",
    "D,L-Malic Acid": "PM1 C3",
    "L-Serine": "PM1 G3",
    "D-Serine": "PM1 B1",
    "L-Alanine": "PM1 G5",
    "D-Alanine": "PM1 A9",
    "D-Galactose": "PM1 A6",
    "D-Mannose": "PM1 A11",
    "D-Fructose": "PM1 C7",
    "D-Mannitol": "PM1 B11",
    "D-Cellobiose": "PM1 F11",
    "Maltose": "PM1 C10",
    "Sucrose": "PM1 D11",
    "Glycerol": "PM1 B3",
    "m-Inositol": "PM1 F3",
    "L-Arginine": "PM2A G4",
    "L-Histidine": "PM2A G6",
    "D-Galacturonic Acid": "PM1 H10",
    "L-Threonine": "PM1 G4",
    "D-Threonine": "PM1 F4",
    "L-Tartaric Acid": "PM2A F12",
    "D-Tartaric Acid": "PM2A F11",
    # The Biolog literature set drops the anomeric prefix that the plate map carries.
    "D-Glucose": "PM1 C9",
    "D-Galacturonic-Acid": "PM1 H10",
}

#: Molecule names for substrates whose raw name is not already unambiguous.
#:
#: ATLeaf entries are quoted from Table S1 of Schaefer, Pacheco et al. 2023
#: (doi:10.1126/science.adf5121, ETH Research Collection); the handoff matrix is that
#: paper's Table S2 verbatim (8,765 non-null cells compared, 0 mismatches). Three column
#: names are corruptions of the authors' own spellings (Gluconate -> Glucote, Succinate
#: -> Succite, Bensoate -> Beoate); the filenames are left as they are and the true
#: compound is recorded here.
#:
#: Marine entries are the 'present as' column of Gralka et al. 2023 SI Table 2.
ATLEAF_MOLECULES: Final[dict[str, str]] = {
    "Glucose": "D-glucose monohydrate, 5 mM (Schaefer Table S1)",
    "Serine": "L-Serine, 10 mM (Schaefer Table S1)",
    "Alanine": "L-Alanine, 10 mM (Schaefer Table S1)",
    "Arginine": "L-Arginine, 5 mM (Schaefer Table S1)",
    "Histidine": "L-Histidine, 5 mM (Schaefer Table S1)",
    "Galacturonic-Acid": "D-Galacturonic acid, 5 mM (Schaefer Table S1)",
    "Galactose": "D-galactose, 5 mM (Schaefer Table S1)",
    "Mannose": "D-mannose, 5 mM (Schaefer Table S1)",
    "Mannitol": "D-mannitol, 5 mM (Schaefer Table S1)",
    "Fructose": "D-fructose, 5 mM (Schaefer Table S1)",
    "Cellobiose": "Cellobiose, 2.5 mM (Schaefer Table S1; D- via the authors' BiGG id cellb)",
    "Maltose": "D-Maltose, 2.5 mM (Schaefer Table S1)",
    "Sucrose": "Sucrose, 2.5 mM (Schaefer Table S1)",
    "Glycerol": "Glycerol, 10 mM, achiral (Schaefer Table S1)",
    "m-Inositol": "Myo-inositol, 5 mM (Schaefer Table S1)",
    "Arabinose": "L-arabinose, 6 mM (Schaefer Table S1)",
    "Xylose": "D-xylose, 6 mM (Schaefer Table S1)",
    "Glucote": "Gluconic acid sodium salt, 5 mM (Schaefer Table S1)",
    "Succite": "Succinic acid disodium salt, 7.5 mM (Schaefer Table S1)",
    "Beoate": "Potassium benzoate, 4.3 mM (Schaefer Table S1)",
    "Methanol_La3": "Methanol with 20 uM LaCl3, 30 mM (Schaefer Table S1)",
    "Coniferyl_alcohol": "Coniferyl alcohol, 3 mM (Schaefer Table S1; no isomer given)",
    "Aspartate": "L-Asparic acid [sic] sodium salt, 7.5 mM (Schaefer Table S1)",
    "Threonine": "L-Threonine, 7.5 mM (Schaefer Table S1)",
    "Glutamate": "L-Glutamic acid sodium salt, 6 mM (Schaefer Table S1)",
}

MARINE_MOLECULES: Final[dict[str, str]] = {
    "Glucose": "D-Glucose",
    "Serine": "L-Serine",
    "Alanine": "DL-Alanine (racemate; the other three datasets assay L only)",
    "Arginine": "L-Arginine",
    "Histidine": "L-histidine HCl",
    "Galacturonic-Acid": "D-Galacturonic acid sodium salt",
    "Galactose": "D-Galactose",
    "Mannose": "D-mannose",
    "Mannitol": "D-Mannitol",
    "Fructose": "D-fructose",
    "Cellobiose": "D-cellobiose",
    "Maltose": "D-maltose monohydrate",
    "Sucrose": "Sucrose",
    "Glycerol": "Glycerol",
    "m-Inositol": "m-Inositol",
    "Arabinose": "L-Arabinose",
    "Glucosamine": "D-Glucosamine HCl",
    "Galactosamine": "D-(+)-Galactosamine hydrochloride",
    "Galacturonate-lmw": "Galacturonate polysaccharides LM from apple (sodium salt)",
}


def read_header(path: Path) -> list[str]:
    """Return the phenotype column names of a raw handoff TSV.

    Read via pandas rather than :mod:`csv` so duplicate headers are de-duplicated
    identically to :func:`scripts.harmonize_phenotypes.read_raw`; the Populus matrix
    contains ``b-Methyl-D-Galactoside`` twice, once per plate.

    Parameters
    ----------
    path
        Path to a tab-separated file whose first column is ``genomeID``.

    Returns
    -------
    list[str]
        Column names excluding ``genomeID`` and any unnamed trailing column, with
        duplicates suffixed ``.1`` as pandas does.
    """
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    return [c for c in header[1:] if c and not c.startswith("Unnamed")]


def legacy_published_name(dataset: str, index: int, legacy: pd.DataFrame) -> str:
    """Return the published column name the original pipeline assigned.

    Parameters
    ----------
    dataset
        One of ``lit``, ``atleaf``, ``pmi``.
    index
        Positional index of the source column.
    legacy
        The 191x3 name map, read with ``dtype=str``.

    Returns
    -------
    str
        Published phenotype name with spaces replaced by hyphens.
    """
    names = legacy[DATASETS[dataset]].dropna().tolist()
    return names[index].replace(" ", "-")


def biolog_well(source: str) -> str:
    """Return the Biolog plate well for a source column, if known.

    The Biolog literature set prefixes its columns with ``Carbon-`` and hyphenates spaces,
    so both spellings are tried against :data:`BIOLOG_WELLS`.

    Parameters
    ----------
    source
        Raw source column name from either Biolog-derived dataset.

    Returns
    -------
    str
        Plate and well, for example ``PM1 C9``, or an empty string if not catalogued.
    """
    bare = source.removeprefix("Carbon-")
    return (
        BIOLOG_WELLS.get(source)
        or BIOLOG_WELLS.get(bare)
        or BIOLOG_WELLS.get(bare.replace("-", " "), "")
    )


def marine_published_name(source: str) -> str:
    """Return the published column name for a Marine source column.

    Applies ``name.replace(" ", "-").capitalize()``, mapping ``Inositol`` to
    ``m-Inositol``. The original ``Galacturonate-lmw`` rename is not applied.

    Parameters
    ----------
    source
        Raw Marine column name.

    Returns
    -------
    str
        Published phenotype name.
    """
    name = source.replace(" ", "-").capitalize()
    return "m-Inositol" if name == "Inositol" else name


def build() -> pd.DataFrame:
    """Assemble the substrate identity table for all four datasets.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, published phenotype), including the source column it binds
        to, the molecule, the Biolog well where applicable, and the evidence for reviewed
        bindings.

    Raises
    ------
    AssertionError
        If any dataset would emit two published files under the same name, or if a
        correction names a source column that does not exist.
    """
    legacy = pd.read_csv(NAME_MAP, dtype=str)
    rows: list[dict[str, object]] = []

    for dataset in ("lit", "atleaf", "pmi", "marine"):
        source_columns = read_header(RAW_DIR / f"{dataset}_phenotypes.tsv")
        # Group source columns by the published name they originally collapsed into,
        # so the discarded members of each collision can be recorded.
        by_published: dict[str, list[str]] = defaultdict(list)
        for index, source in enumerate(source_columns):
            published = (
                marine_published_name(source)
                if dataset == "marine"
                else legacy_published_name(dataset, index, legacy)
            )
            by_published[published].append(source)

        for published, members in by_published.items():
            correction = CORRECTIONS.get((dataset, published), {})
            # Original rule: the last source column wins.
            original_source = members[-1]
            source = correction.get("source", original_source)
            assert source in source_columns, (
                f"{dataset}: unknown source column {source!r}"
            )
            final_name = correction.get("rename_to", published)
            molecule = source.removeprefix("Carbon-") if dataset == "lit" else source
            if dataset == "atleaf":
                molecule = ATLEAF_MOLECULES.get(published, source)
            elif dataset == "marine":
                molecule = MARINE_MOLECULES.get(final_name, source)
            elif dataset == "pmi":
                molecule = PMI_MOLECULES.get(final_name, source)
            rows.append(
                {
                    "dataset": dataset,
                    "phenotype": final_name,
                    "source_column": source,
                    "molecule": molecule,
                    "biolog_well": biolog_well(source)
                    if dataset in ("lit", "pmi")
                    else "",
                    "collapsed_group": "; ".join(members) if len(members) > 1 else "",
                    "discarded": "; ".join(m for m in members if m != source),
                    "binding_note": correction.get("note", ""),
                    "in_common15": final_name in COMMON_PHENOTYPES,
                }
            )

    table = pd.DataFrame(rows).sort_values(["dataset", "phenotype"], ignore_index=True)
    for dataset, group in table.groupby("dataset"):
        duplicates = group.loc[group.phenotype.duplicated(), "phenotype"].tolist()
        assert not duplicates, f"{dataset}: duplicate published names {duplicates}"
    return table


def main() -> None:
    """Write the full and common-15 substrate identity tables."""
    table = build()
    FULL_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMMON15_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FULL_OUT, index=False)
    table.to_csv(ALL_OUT, index=False)

    common = table[table.in_common15].copy()
    common = common[
        [
            "phenotype",
            "dataset",
            "molecule",
            "source_column",
            "biolog_well",
            "collapsed_group",
            "discarded",
        ]
    ].sort_values(["phenotype", "dataset"], ignore_index=True)
    common.to_csv(COMMON15_OUT, index=False)

    reviewed = table.binding_note.ne("")
    print(f"{FULL_OUT}: {len(table)} rows, {int(reviewed.sum())} reviewed bindings")
    print(
        f"{COMMON15_OUT}: {len(common)} rows ({common.phenotype.nunique()} phenotypes x 4 datasets)"
    )
    for dataset, group in table.groupby("dataset"):
        print(
            f"  {dataset}: {len(group)} published phenotypes, "
            f"{int(group.binding_note.ne('').sum())} reviewed"
        )


if __name__ == "__main__":
    main()
