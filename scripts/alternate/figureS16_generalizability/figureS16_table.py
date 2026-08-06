#!/usr/bin/env python3
"""Build the Supplementary Table of candidate non-canonical predictors from the
Figure S16 data, writing ``sections/table_recovered_features.tex``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES_FILE: Path = Path("data/outputs/figureS16/figureS16_recovered_features.tsv")
COUNTS_FILE: Path = Path("data/outputs/figureS16/figureS16_recovered_counts.tsv")
OUTPUT_FILE: Path = Path("sections/table_recovered_features.tex")

MIN_RECOVERED: int = 20
TOP_FEATURES: int = 5
NAME_MAX_CHARS: int = 48


def latex_escape(text: str) -> str:
    """Escape the LaTeX special characters that occur in KO function names.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        LaTeX-safe text.
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "[": r"{[}",
        "]": r"{]}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_feature(feature: str, ko_name: object, shap: float) -> str:
    """Format one candidate feature as a LaTeX cell fragment.

    Parameters
    ----------
    feature : str
        KO identifier.
    ko_name : object
        Functional annotation (may be NaN).
    shap : float
        Mean signed SHAP toward growth.

    Returns
    -------
    str
        Fragment of the form ``"K01712: urocanate hydratase ($+0.85$)"``.
    """
    name = "" if pd.isna(ko_name) else str(ko_name)
    name = name.split(" [EC")[0].strip()
    if len(name) > NAME_MAX_CHARS:
        name = name[: NAME_MAX_CHARS - 1].rstrip() + "..."
    label = f"{feature}: {name}" if name else feature
    return f"{latex_escape(label)} (${shap:+.2f}$)"


def build_table(features: pd.DataFrame, counts: pd.DataFrame) -> str:
    """Assemble the LaTeX table string.

    Parameters
    ----------
    features : pd.DataFrame
        Contents of ``figureS16_recovered_features.tsv``.
    counts : pd.DataFrame
        Contents of ``figureS16_recovered_counts.tsv``.

    Returns
    -------
    str
        Complete LaTeX ``table`` environment.
    """
    kept = counts[counts["n_recovered_fn_discordant"] >= MIN_RECOVERED]
    kept = kept.sort_values("n_recovered_fn_discordant", ascending=False)

    lines: list[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|c|p{10.5cm}|}",
        r"\hline",
        r"\textbf{Phenotype} & \textbf{N} & "
        r"\textbf{Top candidate features (KO: function; mean SHAP toward growth)} \\",
        r"\hline",
    ]

    for row in kept.itertuples(index=False):
        phenotype = row.phenotype
        n_recovered = int(row.n_recovered_fn_discordant)
        sub = features[features["phenotype"] == phenotype].nlargest(
            TOP_FEATURES, "mean_shap_toward_growth"
        )
        cells = [
            format_feature(r.feature, r.ko_name, r.mean_shap_toward_growth)
            for r in sub.itertuples(index=False)
        ]
        feature_cell = r" \newline ".join(cells)
        lines.append(f"{latex_escape(phenotype)} & {n_recovered} & {feature_cell} \\\\")
        lines.append(r"\hline")

    lines.extend(
        [
            r"\end{tabular}",
            r"\caption{Candidate non-canonical predictors of growth recovered by "
            r"concordant-trained models. For each phenotype, FN-discordant held-out "
            r"genomes (GapMind predicts no growth, growth observed experimentally) "
            r"that the concordant-trained model nonetheless classifies correctly as "
            r"growth were pooled across held-out datasets. \textbf{N} is the number "
            r"of such recovered genomes. Features are ranked by mean signed CatBoost "
            r"SHAP contribution toward the growth class; positive values indicate "
            r"features that push the prediction toward growth. These features are "
            r"hypothesis-generating candidates, not validated mechanisms. Only "
            rf"phenotypes with at least {MIN_RECOVERED} recovered genomes are shown.}}",
            r"\label{tab:recovered_features}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build and write the recovered-features supplementary table."""
    features = pd.read_csv(FEATURES_FILE, sep="\t")
    counts = pd.read_csv(COUNTS_FILE, sep="\t")
    table = build_table(features, counts)
    OUTPUT_FILE.write_text(table)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
