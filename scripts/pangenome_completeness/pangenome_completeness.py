#!/usr/bin/env python3
"""Calculate genome completeness as the fraction of species core genes recovered
by MMseqs2 search above identity and coverage thresholds.

Reads genome protein .faa files, the per-species core-gene .faa files, and the
genome-to-species mapping TSV; writes one row per genome with the core genes
expected, the core genes present, and the resulting completeness.

Run with: bash scripts/pangenome_completeness/run_local.sh
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MAPPING_PATH = Path(
    "data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv"
)
DEFAULT_OUTPUT_PATH = Path("completeness_results.tsv")
# TODO: consider 0.80, as used by PPanGGOLiN and PATO; higher thresholds may miss
# divergent homologs and underestimate completeness.
DEFAULT_MIN_IDENTITY = 0.90
DEFAULT_MIN_COVERAGE = 0.80
DEFAULT_EVALUE = 1e-3
DEFAULT_JOBS = 1


@dataclass
class CompletenessResult:
    """Result of completeness calculation for a single genome.

    Attributes
    ----------
    genome_id : str
        Identifier for the genome.
    species : str
        GTDB species clade ID assigned to this genome.
    core_genes_found : int
        Number of core genes found above thresholds.
    core_genes_total : int
        Total number of core genes in the species pangenome.
    completeness : float
        Fraction of core genes found (0.0-1.0).
    status : str
        Processing status: 'success', 'no_species', 'no_core_genes', or 'error'.
    error_message : str
        Error message if status is 'error', empty otherwise.
    """

    genome_id: str
    species: str = ""
    core_genes_found: int = 0
    core_genes_total: int = 0
    completeness: float = 0.0
    status: str = "success"
    error_message: str = ""


def load_genome_species_mapping(mapping_file: Path) -> dict[str, str]:
    """Load genome-to-species mapping from TSV file.

    Parameters
    ----------
    mapping_file : Path
        Path to the mapping TSV file. Expected columns include 'Genome name'
        and 'gtdb_species_clade_id'.

    Returns
    -------
    dict[str, str]
        Mapping from genome name to GTDB species clade ID.
        Only genomes with non-empty species assignments are included.

    Raises
    ------
    FileNotFoundError
        If the mapping file does not exist.
    ValueError
        If required columns are missing from the file.
    """
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    df = pd.read_csv(mapping_file, sep="\t")

    required_cols = ["Genome name", "gtdb_species_clade_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in mapping file: {missing}")

    df = df[df["gtdb_species_clade_id"].notna() & (df["gtdb_species_clade_id"] != "")]

    return dict(zip(df["Genome name"], df["gtdb_species_clade_id"], strict=False))


def discover_genome_files(all_seqs_dir: Path) -> list[Path]:
    """Discover all .faa files in the genome sequences directory.

    Parameters
    ----------
    all_seqs_dir : Path
        Directory containing genome protein sequence files.

    Returns
    -------
    list[Path]
        List of paths to .faa files found in the directory.
    """
    return sorted(all_seqs_dir.glob("*.faa"))


def extract_genome_name(faa_path: Path) -> str:
    """Extract genome name from a .faa filename.

    Handles various naming conventions:
    - 'Species_name_ID.fna.RAST.faa' -> 'Species_name_ID'
    - 'Species_name_ID.RAST.faa' -> 'Species_name_ID'
    - '104336.16.RAST.faa' -> '104336.16'
    - '104336.16.faa' -> '104336.16'

    Parameters
    ----------
    faa_path : Path
        Path to the .faa file.

    Returns
    -------
    str
        Genome name with extensions removed.
    """
    name = faa_path.stem
    for suffix in [".fna.RAST", ".RAST", ".fna"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def get_core_genes_path(core_genes_dir: Path, species_id: str) -> Path | None:
    """Get path to core genes file for a species.

    Tries '{species_id}_unique_core_reps.faa' then '{species_id}.faa'.

    Parameters
    ----------
    core_genes_dir : Path
        Directory containing core gene .faa files.
    species_id : str
        GTDB species clade ID (e.g., 's__Pedobacter_agri--RS_GCF_000258495.1').

    Returns
    -------
    Path | None
        Path to the core genes file if it exists, None otherwise.
    """
    for suffix in ["_unique_core_reps.faa", ".faa"]:
        core_path = core_genes_dir / f"{species_id}{suffix}"
        if core_path.exists():
            return core_path
    return None


def count_fasta_sequences(fasta_path: Path) -> int:
    """Count the number of sequences in a FASTA file.

    Parameters
    ----------
    fasta_path : Path
        Path to the FASTA file.

    Returns
    -------
    int
        Number of sequences (header lines starting with '>').
    """
    count = 0
    with fasta_path.open() as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def create_mmseqs2_db(fasta_path: Path, db_path: Path) -> None:
    """Create an MMseqs2 database from a FASTA file.

    Parameters
    ----------
    fasta_path : Path
        Path to input FASTA file.
    db_path : Path
        Path for output MMseqs2 database.

    Raises
    ------
    subprocess.CalledProcessError
        If MMseqs2 createdb fails.
    """
    subprocess.run(
        ["mmseqs", "createdb", str(fasta_path), str(db_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def run_mmseqs2_search(
    query_db: Path,
    target_db: Path,
    result_db: Path,
    tmp_dir: Path,
    min_identity: float,
    min_coverage: float,
    evalue: float = DEFAULT_EVALUE,
) -> None:
    """Run MMseqs2 search between query and target databases.

    Parameters
    ----------
    query_db : Path
        Path to query MMseqs2 database (genome proteins).
    target_db : Path
        Path to target MMseqs2 database (core genes).
    result_db : Path
        Path for output result database.
    tmp_dir : Path
        Temporary directory for MMseqs2 intermediate files.
    min_identity : float
        Minimum sequence identity threshold (0.0-1.0).
    min_coverage : float
        Minimum coverage threshold (0.0-1.0).
    evalue : float
        E-value threshold for hits.

    Raises
    ------
    subprocess.CalledProcessError
        If MMseqs2 search fails.
    """
    subprocess.run(
        [
            "mmseqs",
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp_dir),
            "--remove-tmp-files",
            "--min-seq-id",
            str(min_identity),
            "-c",
            str(min_coverage),
            "--cov-mode",
            "0",  # coverage of query and target
            "-e",
            str(evalue),
            "--alignment-mode",
            "3",  # global alignment
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def convert_mmseqs2_results(
    query_db: Path,
    target_db: Path,
    result_db: Path,
    output_tsv: Path,
) -> None:
    """Convert MMseqs2 results to tabular format.

    Parameters
    ----------
    query_db : Path
        Path to query MMseqs2 database.
    target_db : Path
        Path to target MMseqs2 database.
    result_db : Path
        Path to result database from search.
    output_tsv : Path
        Path for output TSV file.

    Raises
    ------
    subprocess.CalledProcessError
        If MMseqs2 convertalis fails.
    """
    subprocess.run(
        [
            "mmseqs",
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(output_tsv),
            "--format-output",
            "query,target,pident,alnlen,qcov,tcov,evalue",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def count_unique_core_genes_found(results_tsv: Path) -> int:
    """Count unique core genes (targets) found in MMseqs2 results.

    Parameters
    ----------
    results_tsv : Path
        Path to MMseqs2 results TSV file.

    Returns
    -------
    int
        Number of unique target sequences hit.
    """
    if not results_tsv.exists() or results_tsv.stat().st_size == 0:
        return 0

    df = pd.read_csv(
        results_tsv,
        sep="\t",
        header=None,
        names=["query", "target", "pident", "alnlen", "qcov", "tcov", "evalue"],
    )
    return df["target"].nunique()


def calculate_single_genome_completeness(
    genome_faa: Path,
    core_genes_faa: Path,
    min_identity: float,
    min_coverage: float,
    evalue: float = DEFAULT_EVALUE,
) -> tuple[int, int]:
    """Calculate completeness for a single genome against its core genes.

    Parameters
    ----------
    genome_faa : Path
        Path to genome protein FASTA file.
    core_genes_faa : Path
        Path to core genes FASTA file.
    min_identity : float
        Minimum sequence identity threshold (0.0-1.0).
    min_coverage : float
        Minimum coverage threshold (0.0-1.0).
    evalue : float
        E-value threshold for hits.

    Returns
    -------
    tuple[int, int]
        (core_genes_found, core_genes_total)

    Raises
    ------
    subprocess.CalledProcessError
        If any MMseqs2 command fails.
    """
    core_genes_total = count_fasta_sequences(core_genes_faa)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        query_db = tmp_path / "query_db"
        target_db = tmp_path / "target_db"
        result_db = tmp_path / "result_db"
        mmseqs_tmp = tmp_path / "mmseqs_tmp"
        results_tsv = tmp_path / "results.tsv"

        mmseqs_tmp.mkdir()

        create_mmseqs2_db(genome_faa, query_db)
        create_mmseqs2_db(core_genes_faa, target_db)

        run_mmseqs2_search(
            query_db,
            target_db,
            result_db,
            mmseqs_tmp,
            min_identity,
            min_coverage,
            evalue,
        )

        convert_mmseqs2_results(query_db, target_db, result_db, results_tsv)
        core_genes_found = count_unique_core_genes_found(results_tsv)

    return core_genes_found, core_genes_total


def process_genome(
    genome_faa: Path,
    genome_species_mapping: dict[str, str],
    core_genes_dir: Path,
    min_identity: float,
    min_coverage: float,
    evalue: float = DEFAULT_EVALUE,
) -> CompletenessResult:
    """Process a single genome and calculate its completeness.

    Parameters
    ----------
    genome_faa : Path
        Path to genome protein FASTA file.
    genome_species_mapping : dict[str, str]
        Mapping from genome names to species IDs.
    core_genes_dir : Path
        Directory containing core gene files.
    min_identity : float
        Minimum sequence identity threshold.
    min_coverage : float
        Minimum coverage threshold.
    evalue : float
        E-value threshold for hits.

    Returns
    -------
    CompletenessResult
        Result containing completeness metrics and status.
    """
    genome_id = extract_genome_name(genome_faa)

    species = genome_species_mapping.get(genome_id, "")
    if not species:
        return CompletenessResult(
            genome_id=genome_id,
            status="no_species",
            error_message="No species mapping found",
        )

    core_genes_path = get_core_genes_path(core_genes_dir, species)
    if core_genes_path is None:
        return CompletenessResult(
            genome_id=genome_id,
            species=species,
            status="no_core_genes",
            error_message=f"Core genes file not found: {species}.faa",
        )

    try:
        found, total = calculate_single_genome_completeness(
            genome_faa,
            core_genes_path,
            min_identity,
            min_coverage,
            evalue,
        )
        completeness = found / total if total > 0 else 0.0

        return CompletenessResult(
            genome_id=genome_id,
            species=species,
            core_genes_found=found,
            core_genes_total=total,
            completeness=completeness,
            status="success",
        )
    except subprocess.CalledProcessError as e:
        return CompletenessResult(
            genome_id=genome_id,
            species=species,
            status="error",
            error_message=f"MMseqs2 error: {e.stderr}",
        )
    except Exception as e:
        return CompletenessResult(
            genome_id=genome_id,
            species=species,
            status="error",
            error_message=str(e),
        )


def _process_genome_wrapper(args: tuple) -> CompletenessResult:
    """Wrapper for process_genome to work with ProcessPoolExecutor."""
    return process_genome(*args)


def process_all_genomes(
    all_seqs_dir: Path,
    core_genes_dir: Path,
    mapping_file: Path,
    min_identity: float,
    min_coverage: float,
    evalue: float,
    jobs: int,
    quiet: bool,
) -> list[CompletenessResult]:
    """Process all genomes and calculate completeness.

    Parameters
    ----------
    all_seqs_dir : Path
        Directory containing genome protein .faa files.
    core_genes_dir : Path
        Directory containing core gene .faa files.
    mapping_file : Path
        Path to genome-species mapping TSV file.
    min_identity : float
        Minimum sequence identity threshold.
    min_coverage : float
        Minimum coverage threshold.
    evalue : float
        E-value threshold for hits.
    jobs : int
        Number of parallel workers.
    quiet : bool
        If True, suppress progress output.

    Returns
    -------
    list[CompletenessResult]
        Results for all processed genomes.
    """
    genome_species_mapping = load_genome_species_mapping(mapping_file)
    if not quiet:
        print(f"Loaded {len(genome_species_mapping)} genome-species mappings")

    genome_files = discover_genome_files(all_seqs_dir)
    if not quiet:
        print(f"Found {len(genome_files)} genome .faa files")

    if not genome_files:
        return []

    results: list[CompletenessResult] = []

    args_list = [
        (gf, genome_species_mapping, core_genes_dir, min_identity, min_coverage, evalue)
        for gf in genome_files
    ]

    if jobs == 1:
        for i, args in enumerate(args_list, 1):
            result = process_genome(*args)
            results.append(result)
            if not quiet:
                print(
                    f"[{i}/{len(genome_files)}] {result.genome_id}: "
                    f"{result.status} ({result.completeness:.2%})"
                )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_process_genome_wrapper, args): args[0]
                for args in args_list
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)
                if not quiet:
                    print(
                        f"[{completed}/{len(genome_files)}] {result.genome_id}: "
                        f"{result.status} ({result.completeness:.2%})"
                    )

    return results


def write_results(results: list[CompletenessResult], output_path: Path) -> None:
    """Write completeness results to TSV file.

    Parameters
    ----------
    results : list[CompletenessResult]
        List of completeness results.
    output_path : Path
        Path for output TSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "genome_id": r.genome_id,
            "species": r.species,
            "core_genes_expected": r.core_genes_total,
            "core_genes_present": r.core_genes_found,
            "completeness_pct": round(r.completeness * 100, 2),
            "status": r.status,
            "error_message": r.error_message,
        }
        for r in results
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)


def print_summary(results: list[CompletenessResult]) -> None:
    """Print summary statistics of completeness results.

    Parameters
    ----------
    results : list[CompletenessResult]
        List of completeness results.
    """
    total = len(results)
    if total == 0:
        print("\nNo genomes processed.")
        return

    print("\n" + "=" * 60)
    print("Pangenome Completeness Summary")
    print("=" * 60)

    print(f"\nTotal genomes: {total}")

    print("\nStatus Breakdown:")
    print("-" * 30)
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    for status in ["success", "no_species", "no_core_genes", "error"]:
        count = status_counts.get(status, 0)
        pct = 100 * count / total
        print(f"  {status:15s}: {count:6d} ({pct:5.1f}%)")

    successful = [r for r in results if r.status == "success"]
    if successful:
        completeness_values = [r.completeness for r in successful]
        print("\nCompleteness Statistics (successful genomes):")
        print("-" * 30)
        print(f"  Count:  {len(successful):6d}")
        print(f"  Mean:   {sum(completeness_values) / len(completeness_values):6.2%}")
        print(f"  Min:    {min(completeness_values):6.2%}")
        print(f"  Max:    {max(completeness_values):6.2%}")

        print("\nCompleteness Distribution:")
        print("-" * 30)
        brackets = [
            (0.9, 1.0, "90-100%"),
            (0.8, 0.9, "80-90%"),
            (0.7, 0.8, "70-80%"),
            (0.5, 0.7, "50-70%"),
            (0.0, 0.5, "<50%"),
        ]
        for low, high, label in brackets:
            count = sum(
                1
                for v in completeness_values
                if low <= v < high or (high == 1.0 and v == 1.0)
            )
            pct = 100 * count / len(successful)
            print(f"  {label:10s}: {count:6d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)


def check_mmseqs2_installed() -> bool:
    """Check if MMseqs2 is installed and accessible.

    Returns
    -------
    bool
        True if MMseqs2 is found, False otherwise.
    """
    return shutil.which("mmseqs") is not None


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the pangenome completeness script.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calculate genome completeness based on core gene presence "
            "from species pangenomes using MMseqs2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all-seqs",
        type=Path,
        required=True,
        help="Directory containing genome protein .faa files.",
    )
    parser.add_argument(
        "--core-genes",
        type=Path,
        required=True,
        help="Directory containing core gene .faa files (named by gtdb_species_clade_id).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to genome-to-species mapping TSV file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output TSV path for completeness results.",
    )
    parser.add_argument(
        "--min-identity",
        type=float,
        default=DEFAULT_MIN_IDENTITY,
        help="Minimum sequence identity threshold for MMseqs2 (0.0-1.0).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="Minimum coverage threshold for MMseqs2 (0.0-1.0).",
    )
    parser.add_argument(
        "--evalue",
        "-e",
        type=float,
        default=DEFAULT_EVALUE,
        help="E-value threshold for MMseqs2.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=DEFAULT_JOBS,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args(argv)

    if not check_mmseqs2_installed():
        print("Error: MMseqs2 is not installed or not in PATH.", file=sys.stderr)
        print(
            "Please install MMseqs2: https://github.com/soedinglab/MMseqs2",
            file=sys.stderr,
        )
        return 1

    if not args.all_seqs.is_dir():
        print(
            f"Error: --all-seqs directory does not exist: {args.all_seqs}",
            file=sys.stderr,
        )
        return 1

    if not args.core_genes.is_dir():
        print(
            f"Error: --core-genes directory does not exist: {args.core_genes}",
            file=sys.stderr,
        )
        return 1

    if not args.mapping.exists():
        print(f"Error: --mapping file does not exist: {args.mapping}", file=sys.stderr)
        return 1

    if not 0.0 <= args.min_identity <= 1.0:
        print("Error: --min-identity must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    if not 0.0 <= args.min_coverage <= 1.0:
        print("Error: --min-coverage must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    if args.jobs < 1:
        print("Error: --jobs must be at least 1", file=sys.stderr)
        return 1

    if args.evalue <= 0:
        print("Error: --evalue must be positive", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Processing genomes from: {args.all_seqs}")
        print(f"Using core genes from: {args.core_genes}")
        print(f"Identity threshold: {args.min_identity:.0%}")
        print(f"Coverage threshold: {args.min_coverage:.0%}")
        print(f"E-value threshold: {args.evalue:.0e}")
        print(f"Parallel workers: {args.jobs}")
        print()

    try:
        results = process_all_genomes(
            all_seqs_dir=args.all_seqs,
            core_genes_dir=args.core_genes,
            mapping_file=args.mapping,
            min_identity=args.min_identity,
            min_coverage=args.min_coverage,
            evalue=args.evalue,
            jobs=args.jobs,
            quiet=args.quiet,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    write_results(results, args.output)
    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    if not args.quiet:
        print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
