"""Cross-dataset label conflicts among near-identical strains (Supplementary Text S12).

Strain pairs ``(g1, g2, d1, d2, cross, dist, pheno)`` are read from
``data/outputs/measurement_reliability/part1_pairs_thr0.01.csv``; only ``y1``, ``y2``, and
``conflict`` are recomputed, from ``data/processed/phenotypes``. The Part 1 tables are
written back to ``data/outputs/measurement_reliability/``.

Three logistic models quantify the cross-dataset excess:

``unadjusted``
    ``conflict ~ 1 + is_cross``
``adjusted``
    ``conflict ~ 1 + is_cross + expected_conflict``, the rate under independence
    ``p1 (1 - p2) + p2 (1 - p1)`` from the two datasets' positive-call rates.
``conditional``
    ``conflict ~ 1 + is_cross + max(p1, p2) + min(p1, p2)``.

Run with::

    uv run python -m scripts.measurement_reliability.label_conflict
``--verify-pre-fix`` recomputes the statistics from the pre-enantiomer-fix label
snapshot and asserts that they match the values stored in ``PRE_FIX_EXPECTED``. Part 2
of the analysis (``part2_*.csv``) is not regenerated: it derives from model predictions
rather than labels, and its flagging rules were not recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

from scripts.create_data_splits import COMMON_PHENOTYPES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
OUT_DIR: Final[Path] = REPO_ROOT / "data/outputs/measurement_reliability"
PAIRS_FILE: Final[Path] = OUT_DIR / "part1_pairs_thr0.01.csv"
PHENOTYPE_DIR: Final[Path] = REPO_ROOT / "data/processed/phenotypes"
PRE_FIX_PHENOTYPE_DIR: Final[Path] = (
    REPO_ROOT / "data/processed/phenotypes_backup_pre_enantiomer_fix"
)

#: Datasets contributing labels, in the order used by the output tables.
DATASETS: Final[tuple[str, ...]] = ("atleaf", "lit", "marine", "pmi")

#: Distance cut-offs reported in ``part1_conflict_by_threshold.csv``.
THRESHOLDS: Final[tuple[float, ...]] = (0.0, 0.001, 0.002, 0.005, 0.01)

#: Stored pre-fix values checked by ``--verify-pre-fix``.
PRE_FIX_EXPECTED: Final[dict[str, float]] = {
    "cross_conflicts": 81,
    "within_conflicts": 782,
    "fisher_or": 3.984,
    "unadjusted_or": 3.984,
    "adjusted_or": 2.846,
    "conditional_or": 1.746,
}


def load_labels(phenotype_dir: Path) -> dict[tuple[str, str], pd.Series]:
    """Read every per-phenotype label table.

    Parameters
    ----------
    phenotype_dir
        Directory holding one ``<dataset>/<phenotype>.tsv`` file per phenotype.

    Returns
    -------
    dict[tuple[str, str], pd.Series]
        Labels keyed by ``(dataset, phenotype)``, indexed by genome identifier, with
        untested genomes dropped.
    """
    labels: dict[tuple[str, str], pd.Series] = {}
    for dataset in DATASETS:
        for path in sorted((phenotype_dir / dataset).glob("*.tsv")):
            frame = pd.read_csv(path, sep="\t", dtype={"genomeID": str}).set_index(
                "genomeID"
            )
            labels[(dataset, path.stem)] = frame[path.stem].dropna()
    return labels


def score_pairs(
    pairs: pd.DataFrame, labels: dict[tuple[str, str], pd.Series]
) -> pd.DataFrame:
    """Attach labels and conflict flags to the strain-pair enumeration.

    Parameters
    ----------
    pairs
        Rows of ``part1_pairs_thr0.01.csv``; only the identifier, distance, and phenotype
        columns are read, so an existing ``conflict`` column is ignored.
    labels
        Labels as returned by :func:`load_labels`.

    Returns
    -------
    pd.DataFrame
        One row per scored comparison, with ``y1``, ``y2``, ``conflict``, the two
        positive-call rates, and the regression covariates.
    """
    rates = {key: float(series.mean()) for key, series in labels.items()}
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        first = labels.get((pair.d1, pair.pheno))
        second = labels.get((pair.d2, pair.pheno))
        if first is None or second is None:
            continue
        if pair.g1 not in first.index or pair.g2 not in second.index:
            continue
        y1, y2 = int(first.loc[pair.g1]), int(second.loc[pair.g2])
        p1, p2 = rates[(pair.d1, pair.pheno)], rates[(pair.d2, pair.pheno)]
        rows.append(
            {
                "g1": pair.g1,
                "g2": pair.g2,
                "d1": pair.d1,
                "d2": pair.d2,
                "cross": bool(pair.cross),
                "dist": float(pair.dist),
                "pheno": pair.pheno,
                "y1": y1,
                "y2": y2,
                "conflict": int(y1 != y2),
                "thr": float(pair.thr),
                "p1": p1,
                "p2": p2,
                "expected_conflict": p1 * (1 - p2) + p2 * (1 - p1),
                "pmax": max(p1, p2),
                "pmin": min(p1, p2),
            }
        )
    return pd.DataFrame(rows)


def fit_logit(scored: pd.DataFrame, covariates: list[str]) -> dict[str, float]:
    """Fit ``conflict ~ 1 + is_cross + covariates`` and summarise the cross-dataset term.

    Parameters
    ----------
    scored
        Scored comparisons from :func:`score_pairs`.
    covariates
        Additional column names to adjust for; empty for the unadjusted model.

    Returns
    -------
    dict[str, float]
        Odds ratio, 95% confidence interval, and p-value for ``is_cross``.
    """
    design = scored[["is_cross", *covariates]].astype(float)
    model = sm.Logit(scored["conflict"], sm.add_constant(design)).fit(disp=0)
    coefficient = model.params["is_cross"]
    error = model.bse["is_cross"]
    return {
        "odds_ratio": float(np.exp(coefficient)),
        "ci_low": float(np.exp(coefficient - 1.96 * error)),
        "ci_high": float(np.exp(coefficient + 1.96 * error)),
        "p_value": float(model.pvalues["is_cross"]),
    }


def build_tables(
    scored: pd.DataFrame, labels: dict[tuple[str, str], pd.Series]
) -> dict[str, pd.DataFrame]:
    """Derive every regenerated table from the scored comparisons.

    Parameters
    ----------
    scored
        Scored comparisons from :func:`score_pairs`.
    labels
        Labels as returned by :func:`load_labels`, used for the per-dataset base rates.

    Returns
    -------
    dict[str, pd.DataFrame]
        Output file stem mapped to its table.
    """
    scored = scored.assign(is_cross=scored["cross"].astype(int))
    cross = scored[scored["cross"]]
    within = scored[~scored["cross"]]

    models = pd.DataFrame(
        [
            {"model": name, **fit_logit(scored, covariates)}
            for name, covariates in (
                ("unadjusted", []),
                ("adjusted_expected_conflict", ["expected_conflict"]),
                ("conditional_both_rates", ["pmax", "pmin"]),
            )
        ]
    )
    fisher_or, fisher_p = fisher_exact(
        [
            [int(cross["conflict"].sum()), int((1 - cross["conflict"]).sum())],
            [int(within["conflict"].sum()), int((1 - within["conflict"]).sum())],
        ]
    )
    models = pd.concat(
        [
            models,
            pd.DataFrame(
                [
                    {
                        "model": "fisher_exact",
                        "odds_ratio": float(fisher_or),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": float(fisher_p),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    by_phenotype = (
        cross.groupby("pheno")["conflict"]
        .agg(n="size", conflicts="sum")
        .assign(rate=lambda f: (f["conflicts"] / f["n"]).round(3))
        .reset_index()
        .sort_values("rate", ascending=False, ignore_index=True)
    )

    by_threshold = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "cross_pairs": cross_at.groupby(["g1", "g2"]).ngroups,
                "cross_pheno_comparisons": len(cross_at),
                "cross_conflicts": int(cross_at["conflict"].sum()),
                "cross_conflict_rate": round(float(cross_at["conflict"].mean()), 4),
                "within_pairs": within_at.groupby(["g1", "g2"]).ngroups,
                "within_pheno_comparisons": len(within_at),
                "within_conflicts": int(within_at["conflict"].sum()),
                "within_conflict_rate": round(float(within_at["conflict"].mean()), 4),
            }
            for threshold in THRESHOLDS
            for cross_at, within_at in [
                (cross[cross["dist"] <= threshold], within[within["dist"] <= threshold])
            ]
        ]
    )

    by_dataset_pair = (
        cross.assign(
            pairkey=cross.apply(lambda r: "-".join(sorted((r["d1"], r["d2"]))), axis=1)
        )
        .groupby("pairkey")
        .apply(
            lambda g: pd.Series(
                {
                    "n_strain_pairs": g.groupby(["g1", "g2"]).ngroups,
                    "n_comparisons": len(g),
                    "n_conflicts": int(g["conflict"].sum()),
                    "conflict_rate": round(float(g["conflict"].mean()), 4),
                }
            ),
            include_groups=False,
        )
        .astype({"n_strain_pairs": int, "n_comparisons": int, "n_conflicts": int})
        .reset_index()
        .sort_values("conflict_rate", ascending=False, ignore_index=True)
    )

    cross_pairs_list = (
        cross.groupby(["g1", "g2", "d1", "d2", "dist"])["conflict"]
        .agg(n_shared_pheno="size", n_conflict="sum")
        .assign(
            conflict_rate=lambda f: (f["n_conflict"] / f["n_shared_pheno"]).round(3)
        )
        .reset_index()
        .sort_values("conflict_rate", ascending=False, ignore_index=True)
    )

    shared = {
        key: series for key, series in labels.items() if key[1] in COMMON_PHENOTYPES
    }
    base_rates = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_labels": int(
                    sum(len(s) for (d, _), s in shared.items() if d == dataset)
                ),
                "n_growth": int(
                    sum(int(s.sum()) for (d, _), s in shared.items() if d == dataset)
                ),
            }
            for dataset in DATASETS
        ]
    ).assign(growth_rate=lambda f: (f["n_growth"] / f["n_labels"]).round(3))

    baserate_control = pd.DataFrame(
        [
            {
                "group": group,
                "n": len(subset),
                "observed_conflict": round(float(subset["conflict"].mean()), 4),
                "expected_conflict_baserate": round(
                    float(subset["expected_conflict"].mean()), 3
                ),
                "obs_minus_exp": round(
                    float(
                        subset["conflict"].mean() - subset["expected_conflict"].mean()
                    ),
                    4,
                ),
                "obs_over_exp": round(
                    float(
                        subset["conflict"].mean() / subset["expected_conflict"].mean()
                    ),
                    3,
                ),
            }
            for group, subset in (("cross", cross), ("within", within))
        ]
    )

    return {
        "part1_pairs_thr0.01": scored[
            [
                "g1",
                "g2",
                "d1",
                "d2",
                "cross",
                "dist",
                "pheno",
                "y1",
                "y2",
                "conflict",
                "thr",
            ]
        ],
        "part1_conflict_by_phenotype": by_phenotype,
        "part1_conflict_by_threshold": by_threshold,
        "part1_conflict_by_dataset_pair": by_dataset_pair,
        "part1_cross_pairs_list": cross_pairs_list,
        "dataset_base_rates": base_rates,
        "c2_baserate_control": baserate_control,
        "label_conflict_models": models,
    }


def write_summary(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write the headline numbers to ``summary_key_numbers.csv``.

    Parameters
    ----------
    tables
        Regenerated tables from :func:`build_tables`.
    out_dir
        Destination directory.
    """
    cross_row = tables["part1_conflict_by_threshold"].iloc[-1]
    fisher = tables["label_conflict_models"].set_index("model").loc["fisher_exact"]
    cross_rate = cross_row["cross_conflicts"] / cross_row["cross_pheno_comparisons"]
    within_rate = cross_row["within_conflicts"] / cross_row["within_pheno_comparisons"]
    summary = pd.DataFrame(
        [
            ("cross-dataset close pairs (d<=0.01)", str(int(cross_row["cross_pairs"]))),
            ("cross conflict rate (d<=0.01)", f"{cross_rate:.3f}"),
            ("within conflict rate (d<=0.01)", f"{within_rate:.3f}"),
            (
                "Fisher OR / p",
                f"{fisher['odds_ratio']:.2f} / {fisher['p_value']:.1e}",
            ),
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(out_dir / "summary_key_numbers.csv", index=False)


def verify_pre_fix(pairs: pd.DataFrame) -> None:
    """Assert the stored pre-fix statistics are reproduced from the pre-fix labels.

    Parameters
    ----------
    pairs
        The strain-pair enumeration.

    Raises
    ------
    AssertionError
        If any statistic deviates from the stored pre-fix value.
    """
    scored = score_pairs(pairs, load_labels(PRE_FIX_PHENOTYPE_DIR))
    tables = build_tables(scored, load_labels(PRE_FIX_PHENOTYPE_DIR))
    models = tables["label_conflict_models"].set_index("model")["odds_ratio"]
    counts = tables["part1_conflict_by_threshold"].iloc[-1]
    checks = {
        "cross_conflicts": counts["cross_conflicts"],
        "within_conflicts": counts["within_conflicts"],
        "fisher_or": models["fisher_exact"],
        "unadjusted_or": models["unadjusted"],
        "adjusted_or": models["adjusted_expected_conflict"],
        "conditional_or": models["conditional_both_rates"],
    }
    for key, expected in PRE_FIX_EXPECTED.items():
        got = float(checks[key])
        assert abs(got - expected) < 0.005, f"{key}: expected {expected}, got {got:.3f}"
        print(f"  ok  {key:18s} {got:.3f}")


def main() -> None:
    """Regenerate the label-conflict tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypes",
        type=Path,
        default=PHENOTYPE_DIR,
        help="phenotype label directory",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    parser.add_argument(
        "--verify-pre-fix",
        action="store_true",
        help="reproduce the retired pre-enantiomer-fix values instead of writing",
    )
    args = parser.parse_args()

    pairs = pd.read_csv(PAIRS_FILE, dtype={"g1": str, "g2": str})
    if args.verify_pre_fix:
        print("Reproducing the 2026-06-25 tables from the pre-fix labels:")
        verify_pre_fix(pairs)
        return

    labels = load_labels(args.phenotypes)
    scored = score_pairs(pairs, labels)
    tables = build_tables(scored, labels)

    args.out.mkdir(parents=True, exist_ok=True)
    for stem, table in tables.items():
        table.to_csv(args.out / f"{stem}.csv", index=False)
    write_summary(tables, args.out)

    counts = tables["part1_conflict_by_threshold"].iloc[-1]
    print(f"labels from {args.phenotypes}")
    cross_n, within_n = (
        int(counts["cross_pheno_comparisons"]),
        int(counts["within_pheno_comparisons"]),
    )
    cross_k, within_k = int(counts["cross_conflicts"]), int(counts["within_conflicts"])
    print(
        f"  cross {cross_k}/{cross_n} = {cross_k / cross_n * 100:.1f}%,"
        f" within {within_k}/{within_n} = {within_k / within_n * 100:.1f}%"
    )
    print(tables["label_conflict_models"].to_string(index=False))


if __name__ == "__main__":
    main()
