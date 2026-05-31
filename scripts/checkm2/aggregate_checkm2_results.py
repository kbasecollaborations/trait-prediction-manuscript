#!/usr/bin/env python3
"""Aggregate CheckM2 quality reports and split genomes into pass/fail QC lists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


# MIMAG quality thresholds
# https://www.nature.com/articles/nbt.3893
DEFAULT_MIN_COMPLETENESS = 50.0  # Minimum completeness (%)
DEFAULT_MAX_CONTAMINATION = 10.0  # Maximum contamination (%)

# High-quality thresholds (MIMAG)
HIGH_QUALITY_COMPLETENESS = 90.0
HIGH_QUALITY_CONTAMINATION = 5.0


def load_checkm2_results(
    input_paths: Sequence[Path],
) -> pd.DataFrame:
    """Load and concatenate CheckM2 quality report files.

    Parameters
    ----------
    input_paths : Sequence[Path]
        Paths to quality_report.tsv files or directories containing them.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all CheckM2 results.

    Raises
    ------
    FileNotFoundError
        If no quality_report.tsv files are found.
    """
    dfs: list[pd.DataFrame] = []

    for path in input_paths:
        if path.is_file() and path.name == "quality_report.tsv":
            dfs.append(pd.read_csv(path, sep="\t"))
        elif path.is_dir():
            report_file = path / "quality_report.tsv"
            if report_file.exists():
                dfs.append(pd.read_csv(report_file, sep="\t"))
            else:
                for subfile in path.rglob("quality_report.tsv"):
                    dfs.append(pd.read_csv(subfile, sep="\t"))

    if not dfs:
        raise FileNotFoundError(
            f"No quality_report.tsv files found in: {input_paths}"
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Drop genomes appearing in multiple runs, keeping the first.
    if combined["Name"].duplicated().any():
        print(
            f"Warning: Found {combined['Name'].duplicated().sum()} duplicate "
            "genome entries. Keeping first occurrence.",
            file=sys.stderr,
        )
        combined = combined.drop_duplicates(subset=["Name"], keep="first")

    return combined


def classify_genome_quality(
    df: pd.DataFrame,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
    max_contamination: float = DEFAULT_MAX_CONTAMINATION,
) -> pd.DataFrame:
    """Classify genomes by quality tier based on MIMAG standards.

    Parameters
    ----------
    df : pd.DataFrame
        CheckM2 results with 'Completeness' and 'Contamination' columns.
    min_completeness : float
        Minimum completeness threshold for medium quality.
    max_contamination : float
        Maximum contamination threshold for medium quality.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional quality classification columns.
    """
    df = df.copy()

    df["Quality_Tier"] = "Low"

    medium_mask = (df["Completeness"] >= min_completeness) & (
        df["Contamination"] <= max_contamination
    )
    df.loc[medium_mask, "Quality_Tier"] = "Medium"

    high_mask = (df["Completeness"] >= HIGH_QUALITY_COMPLETENESS) & (
        df["Contamination"] <= HIGH_QUALITY_CONTAMINATION
    )
    df.loc[high_mask, "Quality_Tier"] = "High"

    df["Low_Completeness"] = df["Completeness"] < min_completeness
    df["High_Contamination"] = df["Contamination"] > max_contamination
    df["Fails_QC"] = df["Low_Completeness"] | df["High_Contamination"]

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of genome quality.

    Parameters
    ----------
    df : pd.DataFrame
        Classified CheckM2 results.
    """
    total = len(df)
    print("\n" + "=" * 60)
    print("CheckM2 Results Summary")
    print("=" * 60)

    print(f"\nTotal genomes: {total}")

    print("\nQuality Tier Distribution:")
    print("-" * 30)
    for tier in ["High", "Medium", "Low"]:
        count = (df["Quality_Tier"] == tier).sum()
        pct = 100 * count / total if total > 0 else 0
        print(f"  {tier:8s}: {count:6d} ({pct:5.1f}%)")

    print("\nQC Failure Breakdown:")
    print("-" * 30)
    low_comp = df["Low_Completeness"].sum()
    high_cont = df["High_Contamination"].sum()
    both = (df["Low_Completeness"] & df["High_Contamination"]).sum()
    print(f"  Low completeness only:    {low_comp - both:6d}")
    print(f"  High contamination only:  {high_cont - both:6d}")
    print(f"  Both issues:              {both:6d}")
    print(f"  Total failing QC:         {df['Fails_QC'].sum():6d}")

    print("\nCompleteness Statistics:")
    print("-" * 30)
    print(f"  Mean:   {df['Completeness'].mean():6.2f}%")
    print(f"  Median: {df['Completeness'].median():6.2f}%")
    print(f"  Min:    {df['Completeness'].min():6.2f}%")
    print(f"  Max:    {df['Completeness'].max():6.2f}%")

    print("\nContamination Statistics:")
    print("-" * 30)
    print(f"  Mean:   {df['Contamination'].mean():6.2f}%")
    print(f"  Median: {df['Contamination'].median():6.2f}%")
    print(f"  Min:    {df['Contamination'].min():6.2f}%")
    print(f"  Max:    {df['Contamination'].max():6.2f}%")

    print("\n" + "=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the script.

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
        description="Aggregate CheckM2 results and identify low-quality genomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help=(
            "Path(s) to CheckM2 output directory or quality_report.tsv file. "
            "Multiple paths can be provided to aggregate results."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("checkm2_aggregated.tsv"),
        help="Output path for aggregated results with quality classifications.",
    )
    parser.add_argument(
        "--failed-list",
        type=Path,
        default=Path("failed_genomes.txt"),
        help="Output path for list of genome names that fail QC.",
    )
    parser.add_argument(
        "--passed-list",
        type=Path,
        default=Path("passed_genomes.txt"),
        help="Output path for list of genome names that pass QC.",
    )
    parser.add_argument(
        "--min-completeness",
        type=float,
        default=DEFAULT_MIN_COMPLETENESS,
        help="Minimum completeness threshold (%%).",
    )
    parser.add_argument(
        "--max-contamination",
        type=float,
        default=DEFAULT_MAX_CONTAMINATION,
        help="Maximum contamination threshold (%%).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress printing summary statistics.",
    )

    args = parser.parse_args(argv)

    print(f"Loading CheckM2 results from {len(args.input)} path(s)...")
    try:
        df = load_checkm2_results(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} genome records.")

    df = classify_genome_quality(
        df,
        min_completeness=args.min_completeness,
        max_contamination=args.max_contamination,
    )

    if not args.no_summary:
        print_summary(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nAggregated results saved to: {args.output}")

    failed_genomes = df.loc[df["Fails_QC"], "Name"]
    args.failed_list.parent.mkdir(parents=True, exist_ok=True)
    failed_genomes.to_csv(args.failed_list, index=False, header=False)
    print(f"Failed genome list saved to: {args.failed_list} ({len(failed_genomes)} genomes)")

    passed_genomes = df.loc[~df["Fails_QC"], "Name"]
    args.passed_list.parent.mkdir(parents=True, exist_ok=True)
    passed_genomes.to_csv(args.passed_list, index=False, header=False)
    print(f"Passed genome list saved to: {args.passed_list} ({len(passed_genomes)} genomes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
