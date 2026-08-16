#!/usr/bin/env python3
"""Word count helper for this LaTeX manuscript.

This script parses ``main.tex`` to find manuscript ``\\input{sections/...}``
lines and uses the external ``texcount`` tool to compute LaTeX-aware word
counts for each manuscript section. It then reports counts against the ISME
Original Article limits documented in ``docs/isme_guidelines.md``.

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


DEFAULT_GUIDELINES_PATH = pathlib.Path("docs/isme_guidelines.md")
MAIN_BODY_SECTION_STEMS = ("introduction", "methods", "results", "discussion")
REPORT_SECTION_STEMS = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "acknowledgements",
)


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


@dataclass(frozen=True)
class IsmeGuidelines:
    """Word-count limits for an ISME Original Article.

    Parameters
    ----------
    main_body_limit : int
        Maximum word count for the main body text.
    abstract_limit : int
        Maximum word count for the unstructured abstract.
    source_path : pathlib.Path
        Documentation file used to extract the limits.
    """

    main_body_limit: int
    abstract_limit: int
    source_path: pathlib.Path


@dataclass(frozen=True)
class SectionCount:
    """Word count for one manuscript section.

    Parameters
    ----------
    section : SectionInfo
        Section metadata.
    words : int
        Primary text word count reported by ``texcount``.
    """

    section: SectionInfo
    words: int


def parse_main_for_sections(main_path: pathlib.Path) -> list[SectionInfo]:
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
    list[SectionInfo]
        Ordered list of section descriptors in the order they appear in the
        main file.

    Raises
    ------
    FileNotFoundError
        If ``main_path`` does not exist.
    """

    if not main_path.is_file():
        raise FileNotFoundError(f"Main LaTeX file not found: {main_path}")

    sections: list[SectionInfo] = []
    current_label: str | None = None
    input_pattern = re.compile(r"\\input\{([^}]*)\}")

    text = main_path.read_text(encoding="utf8")
    for line in text.splitlines():
        stripped = line.strip()

        # Track comment lines as potential labels, but skip completely
        # commented-out \input lines.
        if stripped.startswith("%"):
            comment_text = stripped.lstrip("%").strip()
            if comment_text and not _is_decorative_comment(comment_text):
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


def _is_decorative_comment(comment_text: str) -> bool:
    """Return whether a LaTeX comment is a visual separator.

    Parameters
    ----------
    comment_text : str
        Comment text after removing the leading percent sign.

    Returns
    -------
    bool
        ``True`` when the comment is only punctuation used as a separator.
    """

    return bool(re.fullmatch(r"[-=~_* ]+", comment_text))


def load_isme_guidelines(path: pathlib.Path) -> IsmeGuidelines:
    """Load ISME Original Article word-count limits from local docs.

    Parameters
    ----------
    path : pathlib.Path
        Local guidelines file.

    Returns
    -------
    IsmeGuidelines
        Extracted Original Article and abstract word-count limits.

    Raises
    ------
    FileNotFoundError
        If the guidelines file does not exist.
    RuntimeError
        If the expected limits cannot be found.
    """

    if not path.is_file():
        raise FileNotFoundError(f"ISME guidelines file not found: {path}")

    text = path.read_text(encoding="utf8")
    original_article_match = re.search(
        r"Original Article[^\n]*maximum word count:\s*([\d,]+)",
        text,
        flags=re.IGNORECASE,
    )
    abstract_match = re.search(
        r"(?:Unstructured\s+)?Abstract[^\n]*maximum word count:\s*([\d,]+)",
        text,
        flags=re.IGNORECASE,
    )

    if original_article_match is None or abstract_match is None:
        msg = f"Could not find ISME Original Article limits in {path}"
        raise RuntimeError(msg)

    return IsmeGuidelines(
        main_body_limit=_parse_int(original_article_match.group(1)),
        abstract_limit=_parse_int(abstract_match.group(1)),
        source_path=path,
    )


def _parse_int(text: str) -> int:
    """Parse an integer that may contain thousands separators.

    Parameters
    ----------
    text : str
        Integer text.

    Returns
    -------
    int
        Parsed integer value.
    """

    return int(text.replace(",", ""))


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

    tex_source = "%TC:macro \\keywords [ignore]\n" + tex_file.read_text()

    try:
        # ``-1`` makes texcount print just the total word count for the file.
        result = subprocess.run(
            ["texcount", "-1", "-"],
            check=True,
            capture_output=True,
            input=tex_source,
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


def section_stem(section: SectionInfo) -> str:
    """Return the normalized stem for a section file.

    Parameters
    ----------
    section : SectionInfo
        Section descriptor.

    Returns
    -------
    str
        Lower-case file stem.
    """

    return section.path.stem.lower()


def count_report_sections(sections: list[SectionInfo]) -> list[SectionCount]:
    """Count sections that belong in the manuscript word-count report.

    Parameters
    ----------
    sections : list[SectionInfo]
        Sections parsed from ``main.tex``.

    Returns
    -------
    list[SectionCount]
        Counts for reportable sections.
    """

    report_stems = set(REPORT_SECTION_STEMS)
    counts: list[SectionCount] = []

    for section in sections:
        if section_stem(section) not in report_stems:
            continue
        counts.append(SectionCount(section=section, words=texcount_words(section.path)))

    return counts


def format_overage(words: int, limit: int) -> str:
    """Format how far a count is over or under a word-count limit.

    Parameters
    ----------
    words : int
        Observed word count.
    limit : int
        Maximum word-count limit.

    Returns
    -------
    str
        Human-readable overage status.
    """

    overage = words - limit
    if overage > 0:
        return f"{overage} over"
    return f"0 over ({abs(overage)} under)"


def print_isme_report(
    main_path: pathlib.Path,
    guidelines: IsmeGuidelines,
    counts: list[SectionCount],
) -> None:
    """Print the ISME word-count report.

    Parameters
    ----------
    main_path : pathlib.Path
        Manuscript main file used for the report.
    guidelines : IsmeGuidelines
        ISME Original Article word-count limits.
    counts : list[SectionCount]
        Per-section word counts.
    """

    by_stem = {section_stem(count.section): count for count in counts}
    abstract_count = by_stem.get("abstract")
    abstract_words = abstract_count.words if abstract_count is not None else 0
    main_body_words = sum(
        by_stem[stem].words for stem in MAIN_BODY_SECTION_STEMS if stem in by_stem
    )

    print(f"Word count report for {main_path}\n")
    print(f"ISME Original Article limits (from {guidelines.source_path}):")
    print(
        f"  Abstract:  {abstract_words:5d} / {guidelines.abstract_limit:<5d} "
        f"({format_overage(abstract_words, guidelines.abstract_limit)})"
    )
    print(
        f"  Main body: {main_body_words:5d} / {guidelines.main_body_limit:<5d} "
        f"({format_overage(main_body_words, guidelines.main_body_limit)})"
    )

    print("\nSection word counts:")
    label_width = max(
        (len(format_section_label(count.section.label)) for count in counts),
        default=len("Section"),
    )
    for count in counts:
        label = format_section_label(count.section.label)
        print(f"  {label:<{label_width}} {count.words:6d}")


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
    parser.add_argument(
        "--guidelines",
        type=pathlib.Path,
        default=DEFAULT_GUIDELINES_PATH,
        help="Path to the ISME guidelines file (default: docs/isme_guidelines.md)",
    )
    args = parser.parse_args()

    main_path = args.main
    try:
        sections = parse_main_for_sections(main_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return

    try:
        guidelines = load_isme_guidelines(args.guidelines)
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
        return

    if not sections:
        print(f"No sections found in {main_path} (no \\input{{sections/...}} lines).")
        return

    try:
        counts = count_report_sections(sections)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    print_isme_report(main_path=main_path, guidelines=guidelines, counts=counts)


if __name__ == "__main__":  # pragma: no cover
    main()
