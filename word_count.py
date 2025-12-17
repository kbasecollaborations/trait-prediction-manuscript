#!/usr/bin/env python3
"""Word count helper for this LaTeX manuscript.

This script parses ``main.tex`` to find ``\\input{sections/...}`` lines and
uses the external ``texcount`` tool to compute LaTeX-aware word counts for
each section file. It then prints a simple table of per-section counts and
an overall total.

Run from the repository root as::

    python word_count.py

You can optionally point it at a different main file::

    python word_count.py --main other_main.tex

Notes
-----
- This script **requires** the external ``texcount`` program, which is
  distributed with most TeX installations (e.g. TeX Live, MiKTeX).
- LaTeX commands and markup are handled by ``texcount``, so counts should
  reflect the actual manuscript text rather than TeX keywords.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class SectionInfo:
    """Information about a manuscript section.

    Parameters
    ----------
    label : str
        Human-readable label for the section (e.g. ``"ABSTRACT"``).
    path : pathlib.Path
        Path to the corresponding ``.tex`` file.
    """

    label: str
    path: pathlib.Path


def parse_main_for_sections(main_path: pathlib.Path) -> List[SectionInfo]:
    """Parse a LaTeX main file to locate section inputs.

    The function scans ``main_path`` line by line, tracking the most recent
    comment line (starting with ``%``). When it encounters an
    ``\\input{sections/...}`` command on a non-comment line, it records a
    ``SectionInfo`` where the label is taken from the preceding comment, if
    present, or otherwise from the basename of the input file.

    Parameters
    ----------
    main_path : pathlib.Path
        Path to the main LaTeX file (e.g. ``main.tex``).

    Returns
    -------
    List[SectionInfo]
        Ordered list of section descriptors in the order they appear in the
        main file.

    Raises
    ------
    FileNotFoundError
        If ``main_path`` does not exist.
    """

    if not main_path.is_file():
        raise FileNotFoundError(f"Main LaTeX file not found: {main_path}")

    sections: List[SectionInfo] = []
    current_label: str | None = None
    input_pattern = re.compile(r"\\input\{([^}]*)\}")

    text = main_path.read_text(encoding="utf8")
    for line in text.splitlines():
        stripped = line.strip()

        # Track comment lines as potential labels, but skip completely
        # commented-out \input lines.
        if stripped.startswith("%"):
            comment_text = stripped.lstrip("%").strip()
            if comment_text:
                current_label = comment_text
            continue

        match = input_pattern.search(line)
        if not match:
            continue

        tex_target = match.group(1)
        # Only care about inputs from sections/; ignore others such as
        # bibliography or external macros if present.
        if not tex_target.startswith("sections/"):
            continue

        if not tex_target.endswith(".tex"):
            tex_target = f"{tex_target}.tex"

        section_path = main_path.parent / tex_target
        label = current_label if current_label is not None else section_path.stem
        sections.append(SectionInfo(label=label, path=section_path))
        current_label = None

    return sections


def texcount_words(tex_file: pathlib.Path) -> int:
    """Count words in a LaTeX file using ``texcount``.

    Parameters
    ----------
    tex_file : pathlib.Path
        Path to the LaTeX file whose words should be counted.

    Returns
    -------
    int
        The word count reported by ``texcount``.

    Raises
    ------
    RuntimeError
        If ``texcount`` is not available or returns an unexpected result.
    """

    try:
        # ``-1`` makes texcount print just the total word count for the file.
        result = subprocess.run(
            ["texcount", "-1", str(tex_file)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # ``texcount`` not installed
        msg = (
            "texcount is required but was not found on your PATH. "
            "Install a TeX distribution that includes texcount (e.g. TeX Live) "
            "or install texcount separately."
        )
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"texcount failed for {tex_file}: {exc.stderr or exc.stdout}"
        raise RuntimeError(msg) from exc

    output = result.stdout.strip()
    first_token = output.split()[0] if output else "0"

    try:
        # texcount often returns values like "262+1+0" where the first
        # number is the body text word count and subsequent numbers
        # correspond to headers and captions. We take the first part as
        # the primary word count.
        primary = first_token.split("+")[0]
        return int(primary)
    except ValueError as exc:
        msg = f"Unexpected texcount output for {tex_file}: {output!r}"
        raise RuntimeError(msg) from exc


def format_section_label(label: str) -> str:
    """Normalize a section label for display.

    Parameters
    ----------
    label : str
        Raw label string, typically from a LaTeX comment.

    Returns
    -------
    str
        Cleaned label suitable for table output.
    """

    cleaned = " ".join(label.split())
    return cleaned


def main() -> None:
    """Entry point for the word count script.

    Parses command-line arguments, locates section files referenced from the
    main LaTeX file, runs ``texcount`` on each, and prints a summary table
    with per-section and total word counts.
    """

    parser = argparse.ArgumentParser(description="Word counts per LaTeX section")
    parser.add_argument(
        "--main",
        type=pathlib.Path,
        default=pathlib.Path("main.tex"),
        help="Path to the main LaTeX file (default: main.tex)",
    )
    args = parser.parse_args()

    main_path = args.main
    try:
        sections = parse_main_for_sections(main_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return

    if not sections:
        print(f"No sections found in {main_path} (no \\input{{sections/...}} lines).")
        return

    print(f"Section word counts using texcount (main: {main_path}):\n")

    # Compute counts and track maximum label width for pretty printing.
    counts: list[tuple[str, pathlib.Path, int]] = []
    max_label_len = 0
    total_words = 0

    for section in sections:
        label = format_section_label(section.label)
        max_label_len = max(max_label_len, len(label))

        if not section.path.is_file():
            print(f"WARNING: Skipping missing file: {section.path}")
            continue

        try:
            count = texcount_words(section.path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            continue

        counts.append((label, section.path, count))
        total_words += count

    label_col_width = max_label_len + 2

    for label, path, count in counts:
        print(f"{label:<{label_col_width}} {count:7d}  ({path})")

    print("\n" + "-" * (label_col_width + 12))
    print(f"{'TOTAL':<{label_col_width}} {total_words:7d}")


if __name__ == "__main__":  # pragma: no cover
    main()
