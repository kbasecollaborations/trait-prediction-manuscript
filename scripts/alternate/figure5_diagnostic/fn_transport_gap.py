#!/usr/bin/env python3
"""Characterise GapMind false negatives as a diagnostic of the rule's blind spots.

Reframes the retained-false-negative signal away from novel-mechanism discovery
(which the feature analyses found no robust support for) toward what the false
negatives robustly DO reveal: where GapMind's canonical-pathway completeness rule
fails, and whether that failure is recoverable.

For each phenotype every GapMind step (columns of the GapMind step-score matrix,
values -1/0/1/2, present := score >= 1) is classified as a TRANSPORTER or an
ENZYME/other step by a name lexicon. Per genome we then compute an enzyme-step
completeness and a transporter-step completeness. Three analyses follow, all
restricted to GapMind-NEGATIVE genomes (the pathway GapMind scored incomplete),
split into false negatives (FN, experiment = grow) and true negatives (TN, no
growth):

1. Transport-gap map: do FN growers have the enzymatic machinery present while
   the transporter step is what is missing (relative to TN and to concordant
   positives)?
2. Rescue heuristic: among GapMind negatives, how well does enzyme-step
   completeness (and, independently, KOFAM enzyme-gene presence) recover the
   false negatives (AUROC + an operating point recall/precision)?
3. Mysterious-FN split: partition FN into "explained" (enzyme machinery present,
   a transport/annotation gap) versus an "unexplained residual" (enzymes also
   absent) that is the candidate pool for genuine novelty or experimental
   mislabels.

Compute-only diagnostic; writes to ``data/outputs/figure5_fn_discovery/``.

Run with::

    uv run python -m scripts.alternate.figure5_diagnostic.fn_transport_gap
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)
from scripts.alternate.figure5_diagnostic.fn_mechanism_shap import (
    build_symbol_to_ko,
    load_ko_descriptions,
)

GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
GAPMIND_STEP_FILE: Path = Path("data/interim/features/combined_datasets/gapmind.tsv")
KO_DICT_FILE: Path = Path("data/external/mapping/KO_dictionary.json")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
OUT_DIR: Path = Path("data/outputs/figure5_fn_discovery")
FEATURE_DIRS: tuple[str, ...] = ("lit", "marine", "atleaf", "pmi")

MIN_CLASS: int = 15

# Transporter step lexicon for GapMind carbon/amino-acid step names.
TRANSPORT_RE: re.Pattern[str] = re.compile(
    r"(PTS|ABC|permease|transport|MFS|SSS|TRAP|symport|uptake|porter|facilitat|"
    r"SWEET|PLT\d|HMIT|GLUT|Slc|STP\d|MST\d|"
    r"cycA|bra[CDEFG]|art[JMQPI]|his[JMPQ]|ggu[AB]|mgl[ABC]|man[XYZP]|fru[ABI]|fruII|fruP|"
    r"mtl[AEFG]|cmt[AB]|glpF|glpT|glpS|exuT|kdgT|uhpT|ptsG|crr|mal[EFGK]|lamB|thu[EFG]|"
    r"msm[EFGK]|ugp|ara[EFGH]|xyl[EFGH]|rbs[ABC]|glcP|gluP|glcU|glcT|glcV|snatA|sstT|dctA|"
    r"uxuT|gntP|gnt[UT]|iat[PA]|iolT|smoK|frc[ABC]|treP|scrT|susC|SGLT|"
    r"HSERO|TM17\d|TT_C\d|PS417_0420|BT1758|SemiSWEET|manMFS)",
    re.IGNORECASE,
)


def is_transport(step: str) -> bool:
    """Return whether a GapMind step name denotes a transporter.

    Parameters
    ----------
    step : str
        GapMind step name (without the ``<Phenotype>-`` prefix).

    Returns
    -------
    bool
        ``True`` if the step matches the transporter lexicon.
    """
    return bool(TRANSPORT_RE.search(step))


def load_kofam_presence() -> pd.DataFrame:
    """Load the union KOFAM presence matrix across the four datasets.

    Returns
    -------
    pd.DataFrame
        Binary (genomes x KOs) presence matrix.
    """
    parts: list[pd.DataFrame] = []
    for dataset in FEATURE_DIRS:
        frame = pd.read_csv(
            Path(f"data/interim/features/{dataset}/kofam.tsv"), sep="\t", index_col=0
        )
        frame.index = frame.index.astype(str)
        parts.append(frame)
    combined = pd.concat(parts).groupby(level=0).max().fillna(0)
    return (combined > 0).astype(int)


def enzyme_step_kos(
    enzyme_steps: list[str], symbol_to_ko: dict[str, set[str]]
) -> set[str]:
    """Map enzyme step symbols to KOFAM KO ids (best effort).

    Parameters
    ----------
    enzyme_steps : list[str]
        Enzyme (non-transport) GapMind step names.
    symbol_to_ko : dict[str, set[str]]
        Gene-symbol -> KO index.

    Returns
    -------
    set[str]
        KO ids backing the enzyme steps.
    """
    kos: set[str] = set()
    for step in enzyme_steps:
        kos |= symbol_to_ko.get(step.lower(), set())
    return kos


def analyse(
    gapmind_predictions: pd.DataFrame,
    experimental_phenotypes: pd.DataFrame,
    step_matrix: pd.DataFrame,
    kofam: pd.DataFrame,
    symbol_to_ko: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the transport-gap, rescue and split analyses for every phenotype.

    Parameters
    ----------
    gapmind_predictions : pd.DataFrame
        GapMind loose per-phenotype calls.
    experimental_phenotypes : pd.DataFrame
        Experimental phenotypes.
    step_matrix : pd.DataFrame
        GapMind step-score matrix (genomes x ``<Phenotype>-<step>``).
    kofam : pd.DataFrame
        KOFAM presence matrix.
    symbol_to_ko : dict[str, set[str]]
        Gene-symbol -> KO index.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(per_phenotype_summary, pooled_genome_table)``.
    """
    summary_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []

    for phenotype in gapmind_predictions.columns:
        if phenotype not in experimental_phenotypes.columns:
            continue
        cols = [c for c in step_matrix.columns if c.startswith(f"{phenotype}-")]
        if not cols:
            continue
        steps = [c.split("-", 1)[1] for c in cols]
        t_cols = [c for c, s in zip(cols, steps) if is_transport(s)]
        e_cols = [c for c, s in zip(cols, steps) if not is_transport(s)]
        if not e_cols or not t_cols:
            continue
        e_steps = [c.split("-", 1)[1] for c in e_cols]

        common = (
            gapmind_predictions.index.intersection(experimental_phenotypes.index)
            .intersection(step_matrix.index)
        )
        gm = gapmind_predictions.loc[common, phenotype]
        exp = experimental_phenotypes.loc[common, phenotype]
        valid = gm.notna() & exp.notna()
        gm, exp = gm[valid], exp[valid]
        genomes = gm.index

        present = (step_matrix.loc[genomes, cols] >= 1).astype(int)
        enzyme_score = present[e_cols].mean(axis=1)
        transport_score = present[t_cols].mean(axis=1)

        # Independent KOFAM enzyme-gene completeness.
        e_kos = sorted(enzyme_step_kos(e_steps, symbol_to_ko) & set(kofam.columns))
        kofam_enzyme = (
            kofam.loc[kofam.index.intersection(genomes), e_kos].mean(axis=1)
            if e_kos
            else pd.Series(dtype=float)
        )

        neg = gm == 0
        fn = genomes[neg & (exp == 1)]
        tn = genomes[neg & (exp == 0)]
        cpos = genomes[(gm == 1) & (exp == 1)]
        if len(fn) < MIN_CLASS or len(tn) < MIN_CLASS:
            continue

        # Rescue threshold: enzyme completeness comparable to real growers.
        tau = float(np.percentile(enzyme_score.loc[cpos], 25)) if len(cpos) else 0.5
        neg_idx = genomes[neg]
        y = (exp.loc[neg_idx] == 1).astype(int).values
        e_neg = enzyme_score.loc[neg_idx].values
        t_neg = transport_score.loc[neg_idx].values

        auroc_e = roc_auc_score(y, e_neg) if len(set(y)) == 2 else np.nan
        auroc_t = roc_auc_score(y, t_neg) if len(set(y)) == 2 else np.nan
        auroc_kofam = np.nan
        if len(e_kos) and len(kofam_enzyme):
            k_neg = kofam_enzyme.reindex(neg_idx).fillna(0).values
            if len(set(y)) == 2:
                auroc_kofam = roc_auc_score(y, k_neg)

        rescued = enzyme_score.loc[neg_idx] >= tau
        tp = int(((exp.loc[neg_idx] == 1) & rescued).sum())
        fp = int(((exp.loc[neg_idx] == 0) & rescued).sum())
        recall_fn = tp / max(len(fn), 1)
        precision = tp / max(tp + fp, 1)

        explained = int((enzyme_score.loc[fn] >= tau).sum())
        residual = len(fn) - explained

        summary_rows.append(
            {
                "phenotype": phenotype,
                "n_fn": len(fn),
                "n_tn": len(tn),
                "n_enzyme_steps": len(e_cols),
                "n_transport_steps": len(t_cols),
                "fn_enzyme_score": round(float(enzyme_score.loc[fn].mean()), 3),
                "tn_enzyme_score": round(float(enzyme_score.loc[tn].mean()), 3),
                "cpos_enzyme_score": round(float(enzyme_score.loc[cpos].mean()), 3)
                if len(cpos)
                else np.nan,
                "fn_transport_score": round(float(transport_score.loc[fn].mean()), 3),
                "tn_transport_score": round(float(transport_score.loc[tn].mean()), 3),
                "cpos_transport_score": round(float(transport_score.loc[cpos].mean()), 3)
                if len(cpos)
                else np.nan,
                "auroc_enzyme_rescue": round(float(auroc_e), 3),
                "auroc_transport": round(float(auroc_t), 3),
                "auroc_kofam_enzyme": round(float(auroc_kofam), 3)
                if not np.isnan(auroc_kofam)
                else np.nan,
                "rescue_recall_fn": round(recall_fn, 3),
                "rescue_precision": round(precision, 3),
                "fn_explained_transportgap": explained,
                "fn_residual_unexplained": residual,
            }
        )

        for g in neg_idx:
            pooled_rows.append(
                {
                    "phenotype": phenotype,
                    "genome": g,
                    "grow": int(exp.loc[g]),
                    "enzyme_score": float(enzyme_score.loc[g]),
                    "transport_score": float(transport_score.loc[g]),
                    "explained_transportgap": bool(enzyme_score.loc[g] >= tau),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(pooled_rows)


def main() -> None:
    """Run the analyses, write CSVs, and print the headline summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ko_descriptions = load_ko_descriptions(KO_DICT_FILE)
    symbol_to_ko = build_symbol_to_ko(ko_descriptions)

    gm = load_gapmind_predictions(GAPMIND_FILE)
    exp = load_experimental_phenotypes(PHENOTYPE_DIR)
    step_matrix = pd.read_csv(GAPMIND_STEP_FILE, sep="\t", index_col=0)
    step_matrix.index = step_matrix.index.astype(str)
    kofam = load_kofam_presence()

    summary, pooled = analyse(gm, exp, step_matrix, kofam, symbol_to_ko)
    summary = summary.sort_values("auroc_enzyme_rescue", ascending=False)
    summary.to_csv(OUT_DIR / "fn_transport_gap_summary.csv", index=False)
    pooled.to_csv(OUT_DIR / "fn_negatives_genome_scores.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("=== Per-phenotype transport-gap / rescue / split ===")
    show = [
        "phenotype", "n_fn", "n_tn",
        "fn_enzyme_score", "tn_enzyme_score", "fn_transport_score", "tn_transport_score",
        "auroc_enzyme_rescue", "auroc_kofam_enzyme",
        "rescue_recall_fn", "rescue_precision",
        "fn_explained_transportgap", "fn_residual_unexplained",
    ]
    print(summary[show].to_string(index=False))

    # Pooled rescue metrics.
    y = (pooled["grow"] == 1).astype(int).values
    auroc_pool = roc_auc_score(y, pooled["enzyme_score"].values)
    total_fn = int((pooled["grow"] == 1).sum())
    total_tn = int((pooled["grow"] == 0).sum())
    exp_fn = int(((pooled["grow"] == 1) & pooled["explained_transportgap"]).sum())
    exp_tn = int(((pooled["grow"] == 0) & pooled["explained_transportgap"]).sum())
    print("\n=== Pooled over all GapMind-negative genomes ===")
    print(f"  FN (grow) = {total_fn}, TN (no grow) = {total_tn}")
    print(f"  AUROC enzyme-completeness -> growth among negatives: {auroc_pool:.3f}")
    print(
        f"  Transport-gap 'explained' FN: {exp_fn}/{total_fn} "
        f"({exp_fn / max(total_fn,1):.1%}); unexplained residual: {total_fn - exp_fn}"
    )
    print(
        f"  Rescue heuristic (enzyme complete -> predict grow): "
        f"recall_FN={exp_fn / max(total_fn,1):.3f}, "
        f"precision={exp_fn / max(exp_fn + exp_tn,1):.3f} "
        f"(false alarms on TN={exp_tn}/{total_tn})"
    )
    print(f"\nWrote {OUT_DIR}/fn_transport_gap_summary.csv and fn_negatives_genome_scores.csv")


if __name__ == "__main__":
    main()
