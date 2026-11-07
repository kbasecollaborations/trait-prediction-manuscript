"""Build the graphical abstract from Figure 1 panels A and C.

Panel A (BioRender workflow schematic) sits above panel C (performance
summary). The panel letters are cropped away because the graphical abstract
stands alone. The BioRender licence citation sits directly under panel A, so
it is not read as covering panel C.

Run with::

    uv run python -m scripts.figure1.graphical_abstract
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL_A = ROOT / "figures" / "figure1a.pdf"
PANEL_C = ROOT / "figures" / "figure1c.pdf"
OUTPUT = ROOT / "figures" / "graphical_abstract.pdf"

# Licence KA2A1S5SUD requires this exact citation alongside any use of panel A.
BIORENDER_CITATION = (
    "Created in BioRender. Kishore, D. (2026) https://BioRender.com/tz7t8n7"
)

# Crops in pt (left, bottom, right, top). Panel A's letter ends at x = 12.6 pt
# and its first drawing starts near x = 30 pt; panel C's letter ends near
# x = 11 pt, its table starts at x = 43 pt and its rules end near x = 551 pt
# on a 592.56 pt page.
TRIM_A = "16pt 0 0 0"
TRIM_C = "20pt 0 36pt 0"


def build_graphical_abstract(
    panel_a: Path, panel_c: Path, citation: str, output_file: Path
) -> None:
    """Stack panels A and C into a single-page vector PDF.

    Parameters
    ----------
    panel_a : Path
        Figure 1A PDF (BioRender export).
    panel_c : Path
        Figure 1C PDF.
    citation : str
        BioRender citation printed under panel A.
    output_file : Path
        Destination for the graphical abstract PDF.

    Raises
    ------
    FileNotFoundError
        If an input panel does not exist.
    subprocess.CalledProcessError
        If LaTeX compilation or PDF validation fails.
    RuntimeError
        If the output PDF does not contain exactly one page.
    """
    missing = [panel for panel in (panel_a, panel_c) if not panel.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Figure 1 panels: {missing}")

    source = rf"""\documentclass[border=8pt]{{standalone}}
\usepackage{{graphicx}}
\usepackage[scaled]{{helvet}}
\renewcommand{{\familydefault}}{{\sfdefault}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{minipage}}{{10in}}
\centering
\includegraphics[width=\linewidth,trim={TRIM_A},clip]{{{panel_a.as_posix()}}}\par\vspace{{0.15cm}}
{{\raggedleft\scriptsize {citation}\par}}\vspace{{0.5cm}}
\includegraphics[width=\linewidth,trim={TRIM_C},clip]{{{panel_c.as_posix()}}}
\end{{minipage}}
\end{{document}}
"""

    with tempfile.TemporaryDirectory(prefix="graphical-abstract-") as tmp:
        build_dir = Path(tmp)
        tex_file = build_dir / "graphical_abstract.tex"
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
        shutil.copyfile(build_dir / "graphical_abstract.pdf", output_file)

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
    build_graphical_abstract(PANEL_A, PANEL_C, BIORENDER_CITATION, OUTPUT)
    print(f"Saved graphical abstract to {OUTPUT}")
