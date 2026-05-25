#!/usr/bin/env python3
"""Backfill an ``n_minority_test`` column on existing per-cell ML result CSVs.

Each per-split row in the affected CSVs gets a column recording the
minority-class count of its held-out test set, computed using the
appropriate test-set definition (full, concordant, or discordant). This makes
downstream filtering (``filter_by_minority``) a column lookup instead of a
re-derivation from the phenotype files.

Random-split rows (no ``test(<ds>)`` token in the key) and rows whose
``phenotype`` is not present in the GapMind / experimental data are written
with ``n_minority_test`` left empty.

Run once after data generation; the data-generation scripts are also updated
to populate this column on subsequent runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from scripts.minority_filter import (
    annotate_minority_test as _add_minority_column,
    concordant_minority_counts,
    discordant_minority_counts,
    full_test_minority_counts,
)


def backfill_csv(
    path: Path,
    counts_provider: Callable[[], dict[tuple[str, str], int]],
    *,
    sep: str = ",",
    phenotype_column: str = "phenotype",
    test_dataset_column: str | None = None,
    key_column: str = "key",
    multi_counts_column: str | None = None,
) -> None:
    """Read ``path``, append ``n_minority_test``, write back in place.

    ``multi_counts_column`` is used when a single CSV mixes test-subset
    definitions (e.g., ``test_subset`` in ``full``/``concordant``/``discordant``);
    the column value selects the appropriate counts source per row.
    """
    if not path.exists():
        print(f"  [skip] {path} (does not exist)")
        return
    df = pd.read_csv(path, sep=sep)
    if "n_minority_test" in df.columns:
        df = df.drop(columns="n_minority_test")

    if multi_counts_column is None:
        counts = counts_provider()
        out = _add_minority_column(
            df,
            counts,
            phenotype_column=phenotype_column,
            test_dataset_column=test_dataset_column,
            key_column=key_column,
        )
    else:
        # multi-subset: counts_provider must return a dict[str, dict] keyed by subset
        provider_dict = counts_provider()  # type: ignore[assignment]
        parts: list[pd.DataFrame] = []
        for subset, sub_df in df.groupby(multi_counts_column):
            counts = provider_dict.get(
                subset, provider_dict.get("full", full_test_minority_counts())
            )
            parts.append(
                _add_minority_column(
                    sub_df,
                    counts,
                    phenotype_column=phenotype_column,
                    test_dataset_column=test_dataset_column,
                    key_column=key_column,
                )
            )
        out = pd.concat(parts, ignore_index=True)

    out.to_csv(path, sep=sep, index=False)
    n_with = out["n_minority_test"].notna().sum()
    print(
        f"  [ok]  {path}: {len(out)} rows, {n_with} with n_minority_test populated"
    )


def main() -> None:
    """Backfill all known per-cell ML result CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    # Memoise — the minority-counts helpers each scan the entire phenotype tree.
    _full = {"_cache": None}
    _conc = {"_cache": None}
    _disc = {"_cache": None}

    def full() -> dict[tuple[str, str], int]:
        if _full["_cache"] is None:
            _full["_cache"] = full_test_minority_counts()
        return _full["_cache"]

    def conc() -> dict[tuple[str, str], int]:
        if _conc["_cache"] is None:
            _conc["_cache"] = concordant_minority_counts()
        return _conc["_cache"]

    def disc() -> dict[tuple[str, str], int]:
        if _disc["_cache"] is None:
            _disc["_cache"] = discordant_minority_counts()
        return _disc["_cache"]

    def full_concordant_discordant() -> dict[str, dict[tuple[str, str], int]]:
        return {"full": full(), "concordant": conc(), "discordant": disc()}

    # Figures whose held-out test set is the full labelled subset.
    # Tuples: (path, sep, key_column_override)
    full_test_csvs: list[tuple[Path, str, str]] = [
        (Path("data/outputs/figure3/ml_results.csv"), ",", "key"),
        (Path("data/outputs/figure3/gapmind_random_split_metrics.tsv"), "\t", "key"),
        (Path("data/outputs/figure3/gapmind_dataset_split_metrics.tsv"), "\t", "key"),
        (Path("data/outputs/figure3/figure3c_results.csv"), ",", "train_test_config"),
        (Path("data/outputs/figure7/figure7b_confident_ml_results.csv"), ",", "key"),
        (Path("data/outputs/figure7/figure7c_dataset_split_results.csv"), ",", "split"),
        (Path("data/outputs/figure7/figure7d_all_results.csv"), ",", "key"),
    ]
    print("== Backfilling figures evaluated on the full held-out test ==")
    for path, sep, key_col in full_test_csvs:
        backfill_csv(path, full, sep=sep, key_column=key_col)

    # Fig 5A KOFAM and its GapMind reference: test set is concordant.
    print("\n== Backfilling Fig 5A (concordant test) ==")
    for path in [
        Path("data/outputs/figure5/figure5a_concordant_ml_results.csv"),
        Path("data/outputs/figure5/figure5a_concordant_ml_results_gapmind.csv"),
        Path("data/outputs/figure5/figure5a_concordant_ml_results_gapmind_raw.csv"),
    ]:
        backfill_csv(path, conc)

    # Fig 5C: concordant-trained, discordant test.
    print("\n== Backfilling Fig 5C (discordant test) ==")
    backfill_csv(
        Path("data/outputs/figure5/figure5c_concordant_train_different_test.csv"),
        disc,
    )

    # Fig 5D / S7: full test (TSV with a dedicated held_out_dataset column).
    print("\n== Backfilling Fig 5D / S7 (full test, TSV) ==")
    backfill_csv(
        Path("data/outputs/figure5/figure5d_full_test.tsv"),
        full,
        sep="\t",
        test_dataset_column="held_out_dataset",
    )

    # Figures with per-row test_subset (full / concordant / discordant).
    print("\n== Backfilling per-test_subset CSVs (S5, S6, S7) ==")
    for path in [
        Path("data/outputs/figureS5/figureS5_kofam_concordant_results.csv"),
        Path("data/outputs/figureS7/figureS7_learning_curves_kofam.csv"),
        Path("data/outputs/figureS6/figure_s6_data_requirements_kofam.csv"),
    ]:
        backfill_csv(
            path,
            full_concordant_discordant,
            multi_counts_column="test_subset",
        )


if __name__ == "__main__":
    main()
