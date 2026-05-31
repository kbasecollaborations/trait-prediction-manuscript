#!/usr/bin/env python3
"""Generate Supplementary Figure S16 data: per-phenotype GapMind concordance rate
versus cross-dataset prediction accuracy.

Concordance rate comes from the per-(phenotype, dataset) concordance counts;
cross-dataset balanced accuracy comes from the full-data Figure 3 ``dataset_split``
results, which never use concordance information.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

CONCORDANCE_FILE: Path = Path("data/outputs/figureS8/concordance_counts.tsv")
ML_RESULTS_FILE: Path = Path("data/outputs/figure3/ml_results.csv")
OUTPUT_DIR: Path = Path("data/outputs/figureS16")


def concordance_rate_per_phenotype(path: Path) -> pd.DataFrame:
    """Compute the per-phenotype concordance rate from the concordance counts.

    Parameters
    ----------
    path : Path
        Path to ``concordance_counts.tsv`` (per phenotype and dataset).

    Returns
    -------
    pd.DataFrame
        Columns ``phenotype``, ``concordance_rate``, ``n_labelled`` (the number
        of genomes with both a GapMind call and an experimental phenotype).
    """
    counts = pd.read_csv(path, sep="\t")
    grouped = counts.groupby("phenotype")[
        ["n_concordant", "n_discordant_FP", "n_discordant_FN"]
    ].sum()
    labelled = (
        grouped["n_concordant"]
        + grouped["n_discordant_FP"]
        + grouped["n_discordant_FN"]
    )
    return pd.DataFrame(
        {
            "phenotype": grouped.index,
            "concordance_rate": grouped["n_concordant"] / labelled,
            "n_labelled": labelled,
        }
    ).reset_index(drop=True)


def cross_dataset_accuracy_per_phenotype(path: Path) -> pd.DataFrame:
    """Compute the per-phenotype cross-dataset balanced accuracy of the full-data model.

    Parameters
    ----------
    path : Path
        Path to the Figure 3 ``ml_results.csv``.

    Returns
    -------
    pd.DataFrame
        Columns ``phenotype`` and ``cross_dataset_ba`` (mean balanced accuracy
        across the leave-one-dataset-out evaluations).
    """
    results = pd.read_csv(path)
    cross = results[results["split_type"] == "dataset_split"]
    grouped = cross.groupby("phenotype")["balanced_accuracy"].mean()
    return pd.DataFrame(
        {"phenotype": grouped.index, "cross_dataset_ba": grouped.values}
    )


def main() -> None:
    """Build and persist the Supplementary Figure S16 data table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    concordance = concordance_rate_per_phenotype(CONCORDANCE_FILE)
    accuracy = cross_dataset_accuracy_per_phenotype(ML_RESULTS_FILE)
    merged = concordance.merge(accuracy, on="phenotype").sort_values(
        "concordance_rate"
    )

    output_file = OUTPUT_DIR / "figureS16_phenotype_generalizability.tsv"
    merged.to_csv(output_file, sep="\t", index=False)

    rho, p_value = spearmanr(merged["concordance_rate"], merged["cross_dataset_ba"])
    print(merged.to_string(index=False))
    print(f"\nSaved {output_file}")
    print(
        f"Spearman rho = {rho:.3f}, p = {p_value:.4f}, "
        f"n = {len(merged)} phenotypes"
    )


if __name__ == "__main__":
    main()
