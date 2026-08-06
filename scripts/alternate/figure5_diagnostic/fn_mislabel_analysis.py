#!/usr/bin/env python3
"""Rate of candidate experimental mislabels among unexplained false negatives.

Links the transport-gap split from ``fn_transport_gap.py`` (each GapMind
false-negative grower flagged as ``explained_transportgap`` when its enzyme-step
machinery is complete, else an unexplained residual) to two signals that a
"grows" label may be wrong:

1. Confident genomic-consensus disagreement: the model predicts no-growth at
   confidence >= 0.9 and GapMind also calls 0.
2. Cross-dataset twin conflict (Supplementary Text S13 data): a near-identical
   strain (patristic distance <= 0.01) in another dataset carries the opposite
   label.

Exploratory; prints a report and writes ``data/outputs/figure5_fn_discovery/
fn_mislabel_analysis.csv``. Not wired into any manuscript pipeline.

Run with::

    uv run python -m scripts.alternate.figure5_diagnostic.fn_mislabel_analysis
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

FN_SCORES: Path = Path(
    "data/outputs/figure5_fn_discovery/fn_negatives_genome_scores.csv"
)
PER_SAMPLE: Path = Path("data/outputs/figure7/figure7_per_sample.tsv")
TWIN_PAIRS: Path = Path("data/outputs/measurement_reliability/part1_pairs_thr0.01.csv")
OUT: Path = Path("data/outputs/figure5_fn_discovery/fn_mislabel_analysis.csv")

CONF_THRESHOLD: float = 0.9


def load_fn_scores() -> pd.DataFrame:
    """Load the per-genome false-negative transport-gap scores.

    Returns
    -------
    pd.DataFrame
        False-negative rows only (grow == 1) with ``explained_transportgap``.
    """
    df = pd.read_csv(FN_SCORES, dtype={"genome": str})
    return df[df["grow"] == 1].copy()


def load_per_sample() -> pd.DataFrame:
    """Load the per-sample model outputs restricted to false-negative test genomes.

    Returns
    -------
    pd.DataFrame
        Rows where ``y_true == 1`` and ``gapmind_pred == 0`` with derived
        ``rescued`` and ``consensus_mislabel`` flags.
    """
    df = pd.read_csv(PER_SAMPLE, sep="\t", dtype={"genome": str})
    fn = df[(df["y_true"] == 1) & (df["gapmind_pred"] == 0)].copy()
    fn["rescued"] = (fn["y_pred"] == 1).astype(int)
    fn["consensus_mislabel"] = (
        (fn["y_pred"] == 0)
        & (fn["gapmind_pred"] == 0)
        & (fn["confidence"] >= CONF_THRESHOLD)
    ).astype(int)
    # One row per (phenotype, genome)
    return fn[
        ["phenotype", "genome", "confidence", "rescued", "consensus_mislabel"]
    ].drop_duplicates(["phenotype", "genome"])


def twin_conflict_set() -> set[tuple[str, str]]:
    """Build the set of (phenotype, genome) with a conflicting cross-dataset twin.

    Returns
    -------
    set[tuple[str, str]]
        (phenotype, genome) pairs where the genome has a near-identical
        (distance <= 0.01) strain in another dataset with the opposite label.
    """
    pairs = pd.read_csv(TWIN_PAIRS, dtype={"g1": str, "g2": str})
    conf = pairs[(pairs["cross"]) & (pairs["conflict"] == 1)]
    out: set[tuple[str, str]] = set()
    for _, r in conf.iterrows():
        out.add((r["pheno"], r["g1"]))
        out.add((r["pheno"], r["g2"]))
    return out


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cross-tabulate a binary signal by the transport-gap explanation flag.

    Parameters
    ----------
    df : pd.DataFrame
        Joined false-negative table.
    col : str
        Binary column name (e.g. ``consensus_mislabel``).

    Returns
    -------
    pd.DataFrame
        Counts and rate of ``col`` for explained vs residual false negatives.
    """
    rows = []
    for explained, label in [
        (True, "explained_transportgap"),
        (False, "residual_unexplained"),
    ]:
        sub = df[df["explained_transportgap"] == explained]
        rows.append(
            {
                "group": label,
                "n": len(sub),
                f"{col}_n": int(sub[col].sum()),
                f"{col}_rate": round(sub[col].mean(), 3) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the mislabel-concentration analysis and print the report."""
    fn_scores = load_fn_scores()
    per_sample = load_per_sample()
    twins = twin_conflict_set()

    merged = fn_scores.merge(per_sample, on=["phenotype", "genome"], how="inner")
    merged["twin_conflict"] = [
        int((p, g) in twins) for p, g in zip(merged["phenotype"], merged["genome"])
    ]
    merged.to_csv(OUT, index=False)

    n_fn_total = len(fn_scores)
    n_matched = len(merged)
    print("=== FN mislabel-concentration analysis ===")
    print(
        f"FN (GapMind-negative growers) in transport-gap table: {n_fn_total}; "
        f"matched to per-sample model outputs: {n_matched}"
    )
    print(
        f"  explained (transport-gap): {(merged['explained_transportgap']).sum()}; "
        f"residual (unexplained): {(~merged['explained_transportgap']).sum()}"
    )

    print("\n-- Model rescue rate (y_pred == 1) --")
    print(rate_table(merged, "rescued").to_string(index=False))

    print(
        "\n-- Confident-consensus candidate mislabel (conf>=0.9, model & GapMind both no-grow) --"
    )
    cm = rate_table(merged, "consensus_mislabel")
    print(cm.to_string(index=False))

    exp = merged[merged["explained_transportgap"]]
    res = merged[~merged["explained_transportgap"]]
    table = [
        [
            int(res["consensus_mislabel"].sum()),
            int((res["consensus_mislabel"] == 0).sum()),
        ],
        [
            int(exp["consensus_mislabel"].sum()),
            int((exp["consensus_mislabel"] == 0).sum()),
        ],
    ]
    odds, p = fisher_exact(table)
    print(
        f"  Fisher (residual vs explained, candidate-mislabel enrichment): "
        f"OR={odds:.2f}, p={p:.2e}"
    )

    print("\n-- Cross-dataset twin conflict (S13, distance<=0.01) corroboration --")
    tw = rate_table(merged, "twin_conflict")
    print(tw.to_string(index=False))
    print(
        f"  Note: cross-dataset near-identical twins are sparse (S13 reports ~18 "
        f"cross pairs), so twin coverage here is {int(merged['twin_conflict'].sum())} "
        f"of {n_matched} matched FN; treat as thin corroboration only."
    )

    total_res = len(res)
    mislabel_res = int(res["consensus_mislabel"].sum())
    print(
        f"\nHeadline: of {total_res} unexplained-residual FN, "
        f"{mislabel_res} ({mislabel_res / max(total_res, 1):.1%}) are confident-consensus "
        f"candidate mislabels; explained-transport-gap FN mislabel rate is "
        f"{exp['consensus_mislabel'].mean():.1%}."
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
