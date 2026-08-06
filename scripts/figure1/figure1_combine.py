#!/usr/bin/env python3
"""Combine Figure 1 panels into one single-page vector PDF.

Run with:
    uv run python -m scripts.figure1.figure1_combine
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = tuple(ROOT / "figures" / f"figure1{panel}.pdf" for panel in "abc")
OUTPUT = ROOT / "figures" / "figure1.pdf"


def combine_panels(panel_files: tuple[Path, ...], output_file: Path) -> None:
    """Stack PDF panels vertically in a single vector PDF.

    Parameters
    ----------
    panel_files : tuple[Path, ...]
        Ordered single-page PDF panels to combine.
    output_file : Path
        Destination for the combined PDF.

    Raises
    ------
    FileNotFoundError
        If an input panel does not exist.
    subprocess.CalledProcessError
        If LaTeX compilation or PDF validation fails.
    RuntimeError
        If the combined PDF does not contain exactly one page.
    """
    missing = [panel for panel in panel_files if not panel.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Figure 1 panels: {missing}")

    panel_paths = [panel.as_posix() for panel in panel_files]
    source = rf"""\documentclass[border=0pt]{{standalone}}
\usepackage{{graphicx}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{minipage}}{{12in}}
\centering
\includegraphics[width=\linewidth]{{{panel_paths[0]}}}\par\vspace{{0.5cm}}
\includegraphics[width=\linewidth]{{{panel_paths[1]}}}\par\vspace{{0.15cm}}
\includegraphics[width=\linewidth]{{{panel_paths[2]}}}
\end{{minipage}}
\end{{document}}
"""

    with tempfile.TemporaryDirectory(prefix="figure1-") as tmp:
        build_dir = Path(tmp)
        tex_file = build_dir / "figure1.tex"
        tex_file.write_text(source, encoding="utf-8")
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_file.name,
            ],
            cwd=build_dir,
            check=True,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_dir / "figure1.pdf", output_file)

    info = subprocess.run(
        ["pdfinfo", output_file],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages = next(
        (
            line.partition(":")[2].strip()
            for line in info.splitlines()
            if line.startswith("Pages:")
        ),
        None,
    )
    if pages != "1":
        raise RuntimeError(f"Expected a single-page PDF: {output_file}")


if __name__ == "__main__":
    combine_panels(PANELS, OUTPUT)
    print(f"Saved combined Figure 1 to {OUTPUT}")
