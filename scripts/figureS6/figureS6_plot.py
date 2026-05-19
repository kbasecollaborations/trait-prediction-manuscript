#!/usr/bin/env python3
"""
Deprecated compatibility entrypoint for the old Supplementary Figure S6 script.
"""

from scripts.alternate.figureS10.figureS10_plot import main as figure_s10_main
from scripts.alternate.figureS9.figureS9_plot import main as figure_s9_main
from scripts.figureS8.figureS8_plot import main as figure_s8_main


def main() -> None:
    """Generate manuscript-numbered supplementary figures S8--S10."""
    print(
        "scripts.figureS6.figureS6_plot is deprecated; "
        "generating figure_s8.pdf, figure_s9.pdf, and figure_s10.pdf instead."
    )
    figure_s8_main()
    figure_s9_main()
    figure_s10_main()
    print("\nDone!")


if __name__ == "__main__":
    main()
