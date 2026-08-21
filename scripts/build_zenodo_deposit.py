"""Assemble the Zenodo deposit under ``data/zenodo/``.

The manifest is curated rather than a wholesale copy of ``data/``: the inputs the
analysis scripts consume but cannot regenerate, the figure source data needed to redraw
every panel without retraining, and the trained model checkpoints. Anything rebuildable
from those is excluded and named in :data:`EXCLUDED` with the reason; genome assemblies
are retrievable from the accessions in ``genome_accessions.tsv``. Run logs, pickle
caches, and resume checkpoints are skipped inside archived directories by :func:`ships`.

Run with::

    uv run python -m scripts.build_zenodo_deposit
    uv run python -m scripts.build_zenodo_deposit --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEPOSIT_DIR: Final[Path] = REPO_ROOT / "data/zenodo"
ARCHIVE_NAME: Final[str] = "trait-prediction-data.zip"

#: Inputs the pipeline consumes but does not generate. Entries cover what the published
#: figures, tables, and reported numbers need, plus the complete annotation matrices
#: (KOFAM, RAST, GapMind), which are deposited for reuse even where no published panel
#: reads them.
INPUTS: Final[tuple[str, ...]] = (
    # Source phenotype tables, including the legacy name map the harmonisation reads.
    "data/raw/phenotypes",
    # BacDive phenotypes for the data-volume comparison (Figure S7).
    "data/raw/bacdive_dataset/metabolic_phenotypes.tsv",
    # KEGG reference mappings behind the feature-annotation tables (Figure 4, Table 1, Table S3).
    "data/external/mapping/KO_dictionary.json",
    "data/external/mapping/pathway-ko-membership.tsv",
    "data/external/mapping/module-definitions.tsv",
    # Unfiltered per-dataset annotation matrices, the marine identifier map, and the
    # BacDive matrices.
    "data/interim/features",
    "data/interim/gapmind",
    # Pruned GTDB tree and distance matrix (Figures 2B, S1, S2, phylogeny splits).
    "data/processed/phylogeny",
    # Species assignments for the pangenome completeness audit (Figure S4).
    "data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv",
    "data/processed/pangenome/gtdb_species_clade_id.value_counts.with_pangenome_mmseqs90.tsv",
    # GapMind category calls behind Figure 6A, and the per-phenotype feature files.
    "data/processed/gapmind/heatmap_csvs",
    "data/processed/gapmind_features",
    # Filtered feature matrices the models train on.
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
        "figure1",
        "figure2",
        "figure3",
        "figure4",
        "figure5",
        "figure6",
        "figure7",
        "figureS2",
        "figureS3",
        "figureS5",
        "learning_curves",
        "figure5_fp_only",
        "figure5_fn_discovery",
        "bacdive",
        "clustering",
        "agreement_analysis",
        "ml_comparison",
        "stats",
        "pangenome_completeness",
        "measurement_reliability",
    )
) + ("data/outputs/leakage_feature_dup_per_split.csv",)

#: Trained CatBoost checkpoints: 30 deployment models plus the 75 concordant folds.
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
    "data/outputs/learning_curves_histidine": "retired histidine-only learning-curve run; the current Figure S6 is drawn from data/outputs/learning_curves",
    "data/processed/phenotypes_backup_pre_enantiomer_fix": "rollback copy",
    "data/processed/train_test_splits_backup_pre_enantiomer_fix": "rollback copy",
    "data/outputs/concordance_meta": "exploratory meta-classifier; no figure, table, or reported value uses it",
    "data/processed/unique_core_faas": "pangenome core-gene FASTAs; regenerable from the proteomes",
    "data/processed/pangenome/old": "superseded pre-mmseqs90 assignments",
}

#: Working artefacts skipped inside archived directories.
SKIP_SUFFIXES: Final[tuple[str, ...]] = (".log", ".pkl")


def ships(path: Path) -> bool:
    """Return ``True`` when a file belongs in the deposit.

    Parameters
    ----------
    path
        Candidate file inside an archived directory.

    Returns
    -------
    bool
        ``False`` for run logs, pickle caches, and resume checkpoints.
    """
    return path.suffix not in SKIP_SUFFIXES and not path.stem.endswith("_checkpoint")


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
        # Underscore-prefixed subdirectories are scratch by repository
        # convention (``_chunks_*`` checkpoints, ``_stale_*`` quarantines).
        # They are intermediate or superseded, and the consolidated CSV
        # alongside them carries the same results.
        return sorted(
            f
            for f in target.rglob("*")
            if f.is_file()
            and not any(part.startswith("_") for part in f.relative_to(target).parts)
        )
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
    for label, group in (
        ("inputs", INPUTS),
        ("figure data", FIGURE_DATA),
        ("models", MODELS),
    ):
        group_total = 0
        for entry in group:
            files = [f for f in iter_files(entry) if ships(f)]
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
    parser.add_argument(
        "--dry-run", action="store_true", help="preview without writing"
    )
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
