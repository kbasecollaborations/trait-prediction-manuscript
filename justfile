# Default build: ONE pdf with the supplementary appended (build/main.pdf).
# This is what Overleaf and co-authors compile - no xr, no cross-document tricks.
main:
    latexmk -pdf -f main.tex

# Submission build: TWO pdfs, supplementary split out as ISME requires.
# main.tex and supplement.tex cross-reference each other via xr-hyper, so each
# must be compiled after the other's .aux exists - hence the repeated rounds.
# Note this overwrites build/main.pdf with the split (supplement-free) version.
submit:
    latexmk -pdf -f -g -usepretex -pretex='\def\splitsupp{}' main.tex
    latexmk -pdf -f -g supplement.tex
    latexmk -pdf -f -g -usepretex -pretex='\def\splitsupp{}' main.tex
    latexmk -pdf -f -g supplement.tex

# ISME word count. Counts the RENDERED pdf (Introduction..Discussion), which is
# what an editor sees: includes headings, inline math and \ref numbers that
# texcount/word_count.py silently skip. Strips line numbers and [1-4] citations,
# since ISME excludes "tables, figures, and references". Needs a current
# build/main.pdf - run `just main` first. Works in either build mode.
words:
    #!/usr/bin/env bash
    set -euo pipefail
    test -f build/main.pdf || { echo "build/main.pdf missing - run 'just main' first"; exit 1; }
    pdf=$(pdftotext -nopgbrk build/main.pdf - \
      | awk '/^Introduction$/{f=1} /^Author Contributions$/{f=0} f' \
      | grep -vxE '[[:space:]]*[0-9]+[[:space:]]*' \
      | sed -E 's/\[[0-9]+([,–-][0-9]+)*\]//g' | wc -w | tr -d ' ')
    echo "rendered PDF (authoritative): $pdf / 5000  [$(( pdf - 5000 )) over]"
    echo
    uv run python word_count.py
    echo
    echo "note: word_count.py undercounts - it omits headings, inline math, and bare \\ref numbers."

# Watch and auto-compile, continuing past errors
watch:
    latexmk -pdf -pvc -f

# Clean build artifacts first, then watch
watch-clean:
    latexmk -C
    latexmk -pdf -pvc -f

# Force clean including build directory
clean:
    latexmk -C
    rm -rf build/
