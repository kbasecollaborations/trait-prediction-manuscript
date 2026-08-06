#!/usr/bin/env python3
"""Reclassify false-negative-driven features by mechanistic relevance.

KOs are classed as mechanistically relevant (substrate transporters,
transcriptional regulators, catabolic enzymes), irrelevant co-occurrence
(housekeeping genes), or unknown. For every false-negative-introduced feature
(stable in the concordant+false-negative model but not the concordant-only
model) and for the top false-negative-vs-true-negative discriminators on the
unfiltered feature set, reports the mechanistic class and the prevalence in
false negatives versus true negatives.

Writes ``fn_introduced_reclassified.csv`` and ``fn_contrastive_reclassified.csv``
under ``data/outputs/figure5_fn_discovery/``.

Run with::

    uv run python -m scripts.alternate.figure5_diagnostic.fn_mechanism_reclassify
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from scripts.alternate.figure5_diagnostic.fn_mechanism_shap import (
    build_symbol_to_ko,
    canonical_ko_set,
    load_ko_descriptions,
)
from scripts.figure5.figure5cd_data import (
    load_experimental_phenotypes,
    load_gapmind_predictions,
)

KO_DICT_FILE: Path = Path("data/external/mapping/KO_dictionary.json")
GAPMIND_FILE: Path = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
GAPMIND_STEP_FILE: Path = Path("data/interim/features/combined_datasets/gapmind.tsv")
PHENOTYPE_DIR: Path = Path("data/processed/phenotypes")
FN_INTRODUCED: Path = Path(
    "data/outputs/figure5_fn_discovery/fn_introduced_features.csv"
)
CONCORDANT_STABLE: Path = Path(
    "data/outputs/figure5_fn_discovery/concordant_stable_features.json"
)
OUT_DIR: Path = Path("data/outputs/figure5_fn_discovery")
FEATURE_DIRS: tuple[str, ...] = ("lit", "marine", "atleaf", "pmi")
MIN_CLASS: int = 15
TOP_K_CONTRASTIVE: int = 40

# Mechanistic classification lexicons, matched against the KO description.
TRANSPORTER_RE = re.compile(
    r"(transport|permease|\bPTS\b|\bABC\b|symport|antiport|\bporter\b|\bMFS\b|"
    r"\bSSS\b|\bTRAP\b|uptake|importer|exporter|efflux|channel|\bTonB\b|"
    r"substrate-binding|ATP-binding.*transport|SemiSWEET|\bSWEET\b|permease)",
    re.IGNORECASE,
)
REGULATOR_RE = re.compile(
    r"(transcriptional regulator|transcription factor|repressor|activator|"
    r"two-component|response regulator|sensor (histidine )?kinase|sigma factor|"
    r"\bHTH\b|DNA-binding|anti-sigma|regulatory protein)",
    re.IGNORECASE,
)
# Housekeeping and other co-occurrence markers.
HOUSEKEEPING_RE = re.compile(
    r"(ribosomal|\btRNA\b|\brRNA\b|DNA (repair|polymerase|replication|gyrase|"
    r"helicase|primase|ligase|topoisomerase)|recombinase|restriction enzyme|"
    r"photolyase|cell division|elongation factor|translation|chaperone|GroEL|"
    r"DnaK|cold.shock|heat.shock|HSP\d|peptidoglycan|cell wall|flagell|pilus|"
    r"pili|twitching|chemotaxis|autotransporter adhesin|CRISPR|transposase|"
    r"integrase|prophage|toxin-antitoxin|Ku\b|non-homologous end)",
    re.IGNORECASE,
)
UNKNOWN_RE = re.compile(
    r"(uncharacteri[sz]ed|putative|hypothetical|unknown function|\bDUF\d|\bUPF\d|"
    r"domain-containing protein|membrane protein$)",
    re.IGNORECASE,
)
EC_RE = re.compile(r"\[EC:")


def classify(description: str) -> str:
    """Classify a KO by mechanistic relevance from its description.

    Housekeeping is matched first so that a DNA-repair ATPase is not classed as a
    transporter, then transporter, regulator, enzyme, unknown, other.

    Parameters
    ----------
    description : str
        KO ``symbol; name [EC:...]`` string.

    Returns
    -------
    str
        One of ``transporter``, ``regulator``, ``enzyme``, ``unknown``,
        ``housekeeping``, ``other``.
    """
    if not description:
        return "unknown"
    if HOUSEKEEPING_RE.search(description):
        return "housekeeping"
    if TRANSPORTER_RE.search(description):
        return "transporter"
    if REGULATOR_RE.search(description):
        return "regulator"
    if UNKNOWN_RE.search(description):
        return "unknown"
    if EC_RE.search(description):
        return "enzyme"
    return "other"


RELEVANT: frozenset[str] = frozenset({"transporter", "regulator", "enzyme"})


def load_kofam_presence() -> pd.DataFrame:
    """Load the union KOFAM presence matrix across the four datasets.

    Returns
    -------
    pd.DataFrame
        Binary genomes x KOs presence matrix.
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


def fn_tn_sets(
    gm: pd.DataFrame, exp: pd.DataFrame, phenotype: str, index: pd.Index
) -> tuple[list[str], list[str]]:
    """Return (false_negative, true_negative) genome ids for a phenotype.

    Parameters
    ----------
    gm : pd.DataFrame
        GapMind loose predictions.
    exp : pd.DataFrame
        Experimental phenotypes.
    phenotype : str
        Phenotype name.
    index : pd.Index
        Genome ids present in the feature matrix.

    Returns
    -------
    tuple[list[str], list[str]]
        False-negative and true-negative genome id lists.
    """
    common = gm.index.intersection(exp.index).intersection(index)
    a = gm.loc[common, phenotype]
    b = exp.loc[common, phenotype]
    m = a.notna() & b.notna()
    fn = list(common[m & (a == 0) & (b == 1)])
    tn = list(common[m & (a == 0) & (b == 0)])
    return fn, tn


def annotate_fn_introduced(
    ko_desc: dict[str, str],
    gm: pd.DataFrame,
    exp: pd.DataFrame,
    kofam: pd.DataFrame,
) -> pd.DataFrame:
    """Add mechanistic class and FN/TN prevalence to the FN-introduced features.

    Parameters
    ----------
    ko_desc : dict[str, str]
        KO -> description.
    gm, exp : pd.DataFrame
        GapMind and experimental phenotype tables.
    kofam : pd.DataFrame
        KOFAM presence matrix.

    Returns
    -------
    pd.DataFrame
        The FN-introduced table with ``mech_class``, ``prev_fn``,
        ``prev_tn`` and ``grower_enriched`` columns.
    """
    df = pd.read_csv(FN_INTRODUCED)
    df["mech_class"] = [
        classify(ko_desc.get(k, d)) for k, d in zip(df["ko"], df["description"])
    ]
    prev_fn, prev_tn = [], []
    cache: dict[str, tuple[list[str], list[str]]] = {}
    for _, row in df.iterrows():
        ph, ko = row["phenotype"], row["ko"]
        if ph not in cache:
            cache[ph] = fn_tn_sets(gm, exp, ph, kofam.index)
        fn, tn = cache[ph]
        prev_fn.append(
            kofam.loc[fn, ko].mean() if ko in kofam.columns and fn else np.nan
        )
        prev_tn.append(
            kofam.loc[tn, ko].mean() if ko in kofam.columns and tn else np.nan
        )
    df["prev_fn"] = np.round(prev_fn, 3)
    df["prev_tn"] = np.round(prev_tn, 3)
    df["grower_enriched"] = df["prev_fn"] > df["prev_tn"]
    return df


def contrastive_candidates(
    ko_desc: dict[str, str],
    symbol_to_ko: dict[str, set[str]],
    step_columns: list[str],
    gm: pd.DataFrame,
    exp: pd.DataFrame,
    kofam: pd.DataFrame,
    concordant_stable: dict[str, list[str]],
) -> pd.DataFrame:
    """Top FN-vs-TN discriminators per phenotype on the unfiltered feature set.

    For each phenotype, trains a balanced CatBoost to separate false negatives
    from true negatives on all KOFAM features, ranks by mean SHAP toward growth,
    and keeps the mechanistically relevant, grower-enriched, non-canonical KOs
    that are not already stable features of the concordant model.

    Parameters
    ----------
    ko_desc : dict[str, str]
        KO -> description.
    symbol_to_ko : dict[str, set[str]]
        Gene-symbol -> KO index (for canonical set construction).
    step_columns : list[str]
        GapMind step-matrix column names.
    gm, exp : pd.DataFrame
        GapMind and experimental phenotype tables.
    kofam : pd.DataFrame
        KOFAM presence matrix.
    concordant_stable : dict[str, list[str]]
        Concordant-model stable features per split key.

    Returns
    -------
    pd.DataFrame
        Candidate rows across phenotypes.
    """
    # Concordant stable KOs pooled per phenotype (strip the split-key suffix).
    conc_by_pheno: dict[str, set[str]] = {}
    for key, feats in concordant_stable.items():
        ph = key.split("_", 1)[0]
        conc_by_pheno.setdefault(ph, set()).update(feats)

    rows: list[dict[str, object]] = []
    phenotypes = [c for c in gm.columns if c in exp.columns]
    for ph in phenotypes:
        fn, tn = fn_tn_sets(gm, exp, ph, kofam.index)
        if len(fn) < MIN_CLASS or len(tn) < MIN_CLASS:
            continue
        x = kofam.loc[fn + tn]
        y = pd.Series([1] * len(fn) + [0] * len(tn), index=x.index)
        keep = x.columns[(x.sum(0) >= 3) & (x.sum(0) <= len(x) - 3)]
        x = x[keep]
        model = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            random_state=42,
            thread_count=4,
            verbose=False,
            auto_class_weights="Balanced",
        )
        model.fit(x, y)
        shap = model.get_feature_importance(
            Pool(x, y), type="ShapValues", thread_count=4
        )[:, :-1]
        signed = pd.Series(shap[y.values == 1].mean(0), index=x.columns)
        canonical, _, _ = canonical_ko_set(ph, step_columns, symbol_to_ko)
        conc = conc_by_pheno.get(ph, set())
        top = signed.sort_values(ascending=False).head(TOP_K_CONTRASTIVE)
        for ko in top.index:
            desc = ko_desc.get(ko, "")
            mech = classify(desc)
            pf = kofam.loc[fn, ko].mean()
            pt = kofam.loc[tn, ko].mean()
            rows.append(
                {
                    "phenotype": ph,
                    "ko": ko,
                    "description": desc[:70],
                    "mech_class": mech,
                    "shap_toward_growth": round(float(top[ko]), 4),
                    "prev_fn": round(float(pf), 3),
                    "prev_tn": round(float(pt), 3),
                    "grower_enriched": pf > pt,
                    "is_canonical": ko in canonical,
                    "in_concordant_stable": ko in conc,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both reclassifications and print the mechanistically-relevant candidates."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 62)
    pd.set_option("display.max_rows", 200)

    ko_desc = load_ko_descriptions(KO_DICT_FILE)
    symbol_to_ko = build_symbol_to_ko(ko_desc)
    step_columns = pd.read_csv(GAPMIND_STEP_FILE, sep="\t", nrows=0).columns.tolist()
    gm = load_gapmind_predictions(GAPMIND_FILE)
    exp = load_experimental_phenotypes(PHENOTYPE_DIR)
    kofam = load_kofam_presence()
    concordant_stable = json.loads(CONCORDANT_STABLE.read_text())

    fni = annotate_fn_introduced(ko_desc, gm, exp, kofam)
    fni.to_csv(OUT_DIR / "fn_introduced_reclassified.csv", index=False)
    print("=" * 90)
    print("(A) FN-INTRODUCED STABLE FEATURES (concordant+FN, not concordant), by class")
    print("=" * 90)
    print(fni["mech_class"].value_counts().to_string())
    rel = fni[fni["mech_class"].isin(RELEVANT) & fni["grower_enriched"]]
    print(f"\nMechanistically relevant AND grower-enriched: {len(rel)} of {len(fni)}")
    print(
        rel.sort_values(["mech_class", "prev_fn"], ascending=[True, False])[
            [
                "phenotype",
                "ko",
                "mech_class",
                "prev_fn",
                "prev_tn",
                "in_gapmind_steps",
                "description",
            ]
        ].to_string(index=False)
    )

    cand = contrastive_candidates(
        ko_desc, symbol_to_ko, step_columns, gm, exp, kofam, concordant_stable
    )
    cand.to_csv(OUT_DIR / "fn_contrastive_reclassified.csv", index=False)
    print("\n" + "=" * 90)
    print(
        "(B) UNFILTERED FN-vs-TN CONTRASTIVE: relevant, grower-enriched, non-canonical,"
    )
    print(
        "    NOT already a concordant stable feature (candidate alternate mechanisms)"
    )
    print("=" * 90)
    keep = cand[
        cand["mech_class"].isin(RELEVANT)
        & cand["grower_enriched"]
        & (~cand["is_canonical"])
        & (~cand["in_concordant_stable"])
        & (cand["prev_fn"] - cand["prev_tn"] >= 0.10)
    ].copy()
    keep["prev_gap"] = (keep["prev_fn"] - keep["prev_tn"]).round(3)
    for cls in ("transporter", "regulator", "enzyme"):
        sub = keep[keep["mech_class"] == cls].sort_values("prev_gap", ascending=False)
        print(f"\n--- {cls.upper()} ({len(sub)}) ---")
        if len(sub):
            print(
                sub[
                    [
                        "phenotype",
                        "ko",
                        "prev_fn",
                        "prev_tn",
                        "prev_gap",
                        "shap_toward_growth",
                        "description",
                    ]
                ].to_string(index=False)
            )
    print("\nWrote fn_introduced_reclassified.csv and fn_contrastive_reclassified.csv")


if __name__ == "__main__":
    main()
