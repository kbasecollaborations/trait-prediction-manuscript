"""Assemble the Zenodo deposit under ``data/zenodo/``.

The manifest below is curated rather than a wholesale copy of ``data/``: it lists the
inputs the analysis scripts consume but cannot regenerate, the figure source data needed
to redraw every panel without retraining, and the trained model checkpoints. Anything a
depositor can rebuild from those (raw tool output, intermediate genome files) is excluded
and named in :data:`EXCLUDED` with the reason.

Deliberately omitted from the archive, with reasons:

``data/processed/gapmind/pmi_gapmind_output_files/`` (2.9 GB)
    Raw GapMind output. Read only by :mod:`scripts.create_gapmind_features`, whose
    product ``data/processed/gapmind_features/`` is included.
``data/raw/all_seqs/``, ``data/raw/genomes_w_phenotype/``
    Genome and proteome sequences. Proteomes ship as a separate archive; assemblies are
    retrievable from the accessions in ``genome_accessions.tsv``.
``*_backup_pre_enantiomer_fix/``, ``_stale_*``
    Working copies kept for rollback during the 2026-08 substrate identity correction.
``*.log``, ``*.pkl``
    Run logs and cached intermediates; both are rebuilt from the archived inputs.

Run with ``uv run python -m scripts.build_zenodo_deposit`` (add ``--dry-run`` to preview).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEPOSIT_DIR: Final[Path] = REPO_ROOT / "data/zenodo"
ARCHIVE_NAME: Final[str] = "trait-prediction-data.zip"

#: Inputs the pipeline consumes but does not generate.
INPUTS: Final[tuple[str, ...]] = (
    "data/raw/phenotypes",
    "data/raw/bacdive_dataset/metabolic_phenotypes.tsv",
    "data/external/mapping/KO_dictionary.json",
    "data/external/mapping/pathway-ko-membership.tsv",
    "data/external/mapping/module-definitions.tsv",
    "data/interim/features",
    "data/interim/gapmind",
    "data/processed/phylogeny",
    "data/processed/pangenome",
    "data/processed/gapmind/heatmap_csvs",
    "data/processed/gapmind_features",
    "data/processed/features_reduced",
    "data/processed/phenotypes",
    "data/processed/substrate_identity.csv",
    "data/processed/train_test_splits",
    "data/processed/bacdive",
)

#: Figure and table source data, so every panel can be redrawn without retraining.
FIGURE_DATA: Final[tuple[str, ...]] = tuple(
    f"data/outputs/{name}"
    for name in (
        "figure1", "figure2", "figure3", "figure4", "figure5", "figure6", "figure7",
        "figureS2", "figureS3", "figureS5", "figureS6",
        "bacdive", "clustering", "agreement_analysis",
        "stats", "pangenome_completeness", "measurement_reliability",
    )
) + ("data/outputs/leakage_feature_dup_per_split.csv",)

#: Trained CatBoost checkpoints (105 models).
MODELS: Final[tuple[str, ...]] = (
    "data/outputs/full_data_models",
    "data/outputs/concordant_full_models",
    "data/outputs/concordant_models",
)

#: Paths never archived, with the reason recorded for the README.
EXCLUDED: Final[dict[str, str]] = {
    "data/processed/gapmind/pmi_gapmind_output_files": "raw GapMind output; derived features are included",
    "data/raw/all_seqs": "proteomes ship as a separate archive",
    "data/raw/genomes_w_phenotype": "assemblies retrievable from the deposited accessions",
    "data/outputs_backup_pre_enantiomer_fix": "rollback copy from the substrate identity correction",
    "data/processed/phenotypes_backup_pre_enantiomer_fix": "rollback copy",
    "data/processed/train_test_splits_backup_pre_enantiomer_fix": "rollback copy",
    "data/outputs/figureS7": "retired learning-curve figure; the current Figure S7 is drawn from data/outputs/bacdive",
    "data/outputs/concordance_meta": "exploratory meta-classifier; no figure, table, or reported value uses it",
}

#: Working artefacts skipped inside archived directories.
SKIP_SUFFIXES: Final[tuple[str, ...]] = (".log", ".pkl")

#: Per-dataset phenotype matrices published loose alongside the archive.
PHENOTYPE_TABLES: Final[dict[str, str]] = {
    "atleaf": "atleaf_phenotypes.tsv",
    "lit": "biolog_phenotypes.tsv",
    "marine": "marine_phenotypes.tsv",
    "pmi": "populus_phenotypes.tsv",
}


def iter_files(relative: str) -> list[Path]:
    """Return every file under a manifest entry.

    Parameters
    ----------
    relative
        Repository-relative path to a file or directory.

    Returns
    -------
    list[Path]
        Absolute paths, empty when the entry does not exist.
    """
    target = REPO_ROOT / relative
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(f for f in target.rglob("*") if f.is_file())
    return []


def human(size: int) -> str:
    """Format a byte count for display.

    Parameters
    ----------
    size
        Number of bytes.

    Returns
    -------
    str
        Human-readable size, for example ``85MB``.
    """
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def write_phenotype_tables(dry_run: bool) -> None:
    """Rebuild the loose per-dataset phenotype matrices from the corrected labels.

    Parameters
    ----------
    dry_run
        When ``True``, report intended output without writing.
    """
    source = REPO_ROOT / "data/processed/phenotypes"
    for dataset, filename in PHENOTYPE_TABLES.items():
        files = sorted((source / dataset).glob("*.tsv"))
        if not files:
            print(f"  WARNING: no phenotype files for {dataset}")
            continue
        frame = pd.concat(
            [pd.read_csv(f, sep="\t", dtype=str).set_index("genomeID") for f in files],
            axis=1,
        )
        print(f"  {filename}: {frame.shape[0]} genomes x {frame.shape[1]} substrates")
        if not dry_run:
            frame.to_csv(DEPOSIT_DIR / filename, sep="\t")


def build_archive(dry_run: bool) -> int:
    """Write the deposit archive and return its uncompressed byte total.

    Parameters
    ----------
    dry_run
        When ``True``, size the manifest without writing the archive.

    Returns
    -------
    int
        Total uncompressed size of the archived files.
    """
    total = 0
    members: list[Path] = []
    for label, group in (("inputs", INPUTS), ("figure data", FIGURE_DATA), ("models", MODELS)):
        group_total = 0
        for entry in group:
            files = [f for f in iter_files(entry) if f.suffix not in SKIP_SUFFIXES]
            if not files:
                print(f"  MISSING  {entry}")
                continue
            group_total += sum(f.stat().st_size for f in files)
            members.extend(files)
        print(f"  {human(group_total):>8s}  {label}")
        total += group_total

    if not dry_run:
        archive = DEPOSIT_DIR / ARCHIVE_NAME
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in members:
                zf.write(f, f.relative_to(REPO_ROOT))
        print(f"  wrote {archive.name} ({human(archive.stat().st_size)} compressed)")
    return total


def write_checksums(dry_run: bool) -> None:
    """Regenerate ``MD5SUMS.txt`` over every loose file in the deposit.

    Parameters
    ----------
    dry_run
        When ``True``, report without writing.
    """
    lines = []
    for f in sorted(DEPOSIT_DIR.iterdir()):
        if not f.is_file() or f.name == "MD5SUMS.txt":
            continue
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        lines.append(f"{digest}  {f.name}")
    print(f"  MD5SUMS.txt: {len(lines)} entries")
    if not dry_run:
        (DEPOSIT_DIR / "MD5SUMS.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Rebuild the deposit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preview without writing")
    args = parser.parse_args()

    DEPOSIT_DIR.mkdir(parents=True, exist_ok=True)
    print("Phenotype matrices:")
    write_phenotype_tables(args.dry_run)
    print("\nArchive manifest:")
    total = build_archive(args.dry_run)
    print(f"  {human(total):>8s}  total uncompressed")
    print("\nExcluded:")
    for path, reason in EXCLUDED.items():
        size = sum(f.stat().st_size for f in iter_files(path))
        print(f"  {human(size):>8s}  {path}  ({reason})")
    print("\nChecksums:")
    write_checksums(args.dry_run)
    if args.dry_run:
        print("\n(dry run: nothing written)")


if __name__ == "__main__":
    main()
