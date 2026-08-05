"""Per-dataset concordant-subset diagnostics for poor Fig 5A performers:
class composition, genome overlap, phylum composition with Jensen-Shannon
divergence, and per-fold dataset_split and phylo_ooc breakdowns. Prints
Markdown tables to stdout.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS: tuple[str, ...] = ("atleaf", "lit", "marine", "pmi")
GAPMIND_PRED: Path = REPO_ROOT / "data/outputs/figure2/gapmind_phenotypes_loose.tsv"
PHENOTYPE_DIR: Path = REPO_ROOT / "data/processed/phenotypes"
FIG5A_CSV: Path = REPO_ROOT / "data/outputs/figure5/figure5a_concordant_ml_results.csv"
TAXONOMY_TSV: Path = (
    REPO_ROOT / "data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv"
)

POOR: tuple[str, ...] = ("Glucose", "Serine", "Galacturonic-Acid")
GOOD: tuple[str, ...] = ("Histidine", "Mannitol", "m-Inositol")
PHENS: tuple[str, ...] = POOR + GOOD
PHYLO_OOC_PHENS: tuple[str, ...] = ("Glucose", "Serine", "Alanine")


def load_gapmind() -> pd.DataFrame:
    """Load GapMind loose predictions indexed by genomeID."""
    return pd.read_csv(GAPMIND_PRED, sep="\t", index_col=0, dtype={"genomeID": str})


def load_dataset_labels(dataset: str, phenotype: str) -> pd.Series | None:
    """Load experimental labels for one (dataset, phenotype) pair."""
    path = PHENOTYPE_DIR / dataset / f"{phenotype}.tsv"
    if not path.exists():
        return None
    return (
        pd.read_csv(path, sep="\t", dtype={"genomeID": str})
        .set_index("genomeID")[phenotype]
        .dropna()
    )


def concordant_summary(gapmind: pd.DataFrame) -> pd.DataFrame:
    """Per (phenotype, dataset) concordant counts and positive fraction."""
    rows = []
    for phen in PHENS:
        for ds in DATASETS:
            labels = load_dataset_labels(ds, phen)
            if labels is None or phen not in gapmind.columns:
                rows.append(
                    dict(phenotype=phen, dataset=ds, n_total=0, n_conc=0,
                         n_conc_pos=0, n_conc_neg=0, pos_frac=np.nan,
                         minority=0)
                )
                continue
            common = labels.index.intersection(gapmind.index)
            labels = labels.loc[common]
            preds = gapmind.loc[common, phen]
            mask = labels == preds
            conc = labels[mask]
            n_pos = int((conc == 1).sum())
            n_neg = int((conc == 0).sum())
            n_conc = n_pos + n_neg
            rows.append(
                dict(
                    phenotype=phen,
                    dataset=ds,
                    n_total=len(labels),
                    n_conc=n_conc,
                    n_conc_pos=n_pos,
                    n_conc_neg=n_neg,
                    pos_frac=(n_pos / n_conc if n_conc else np.nan),
                    minority=min(n_pos, n_neg),
                )
            )
    return pd.DataFrame(rows)


def genome_overlap() -> pd.DataFrame:
    """Pairwise genomeID overlap counts between the four source datasets.

    Uses Glucose as a representative phenotype to enumerate genomeIDs assayed
    per dataset; the genome set per dataset is essentially the same across
    phenotypes because phenotype TSVs share the dataset's genome list.
    """
    sets: dict[str, set[str]] = {}
    for ds in DATASETS:
        ids: set[str] = set()
        for phen_file in (PHENOTYPE_DIR / ds).glob("*.tsv"):
            df = pd.read_csv(phen_file, sep="\t", dtype={"genomeID": str})
            ids.update(df["genomeID"].astype(str).tolist())
        sets[ds] = ids
    mat = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=int)
    for a in DATASETS:
        for b in DATASETS:
            mat.loc[a, b] = len(sets[a] & sets[b])
    return mat, sets


def load_taxonomy() -> dict[str, str]:
    """Return a ``genomeID -> GTDB class`` lookup that spans all four datasets.

    Reuses the project's :func:`scripts.alternate.figureS9_legacy.figureS9_data.build_genome_to_class`
    and :func:`assign_classes`, which already bridge marine short codes and
    IMG-named PMI genomes via a strain map plus genus-based fallback.
    """
    from scripts.alternate.figureS9_legacy.figureS9_data import (
        assign_classes,
        build_genome_to_class,
    )

    lookup, genus_to_class = build_genome_to_class()
    # Collect every genome that appears in any dataset's phenotype TSV
    all_genomes: set[str] = set()
    for ds in DATASETS:
        for path in (PHENOTYPE_DIR / ds).glob("*.tsv"):
            df = pd.read_csv(path, sep="\t", dtype={"genomeID": str})
            all_genomes.update(df["genomeID"].astype(str).tolist())

    assigned = assign_classes(all_genomes, lookup, genus_to_class)
    return assigned


def js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence between two label-keyed proportions."""
    idx = p.index.union(q.index)
    pp = p.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    qq = q.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    pp = pp / pp.sum() if pp.sum() else pp
    qq = qq / qq.sum() if qq.sum() else qq
    m = 0.5 * (pp + qq)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pp, m) + 0.5 * kl(qq, m)


def phylum_composition(
    gapmind: pd.DataFrame, taxonomy: dict[str, str]
) -> pd.DataFrame:
    """For each held-out dataset, top-5 GTDB classes (concordant subset) and JSD vs train pool."""
    rows = []
    for phen in POOR:
        ds_genomes: dict[str, set[str]] = {}
        for ds in DATASETS:
            labels = load_dataset_labels(ds, phen)
            if labels is None or phen not in gapmind.columns:
                ds_genomes[ds] = set()
                continue
            common = labels.index.intersection(gapmind.index)
            labels = labels.loc[common]
            preds = gapmind.loc[common, phen]
            ds_genomes[ds] = set(labels.index[labels == preds])

        ds_dist: dict[str, pd.Series] = {}
        for ds in DATASETS:
            classes = [taxonomy.get(g, "Unassigned") for g in ds_genomes[ds]]
            ds_dist[ds] = pd.Series(Counter(classes))

        # For each held-out dataset, compute JSD against the training pool.
        for held in DATASETS:
            train_ds = [d for d in DATASETS if d != held]
            train_counts = pd.Series(dtype=float)
            for d in train_ds:
                train_counts = train_counts.add(ds_dist[d], fill_value=0)
            test_counts = ds_dist[held]
            tr_total = train_counts.sum()
            te_total = test_counts.sum()
            jsd = js_divergence(train_counts, test_counts) if te_total and tr_total else np.nan
            top_train = train_counts.sort_values(ascending=False).head(4)
            top_test = test_counts.sort_values(ascending=False).head(4)
            top_train_str = ", ".join(
                f"{p}={c/tr_total:.0%}" for p, c in top_train.items()
            ) if tr_total else "(none)"
            top_test_str = ", ".join(
                f"{p}={c/te_total:.0%}" for p, c in top_test.items()
            ) if te_total else "(none)"
            rows.append(
                dict(
                    phenotype=phen,
                    held_out=held,
                    n_train_conc=int(tr_total),
                    n_test_conc=int(te_total),
                    jsd_class=round(jsd, 3) if not np.isnan(jsd) else np.nan,
                    top_train_classes=top_train_str,
                    top_test_classes=top_test_str,
                )
            )
    return pd.DataFrame(rows)


def per_fold_breakdown(gapmind: pd.DataFrame) -> pd.DataFrame:
    """Per-fold breakdown for dataset_split, with train/test majority class."""
    results = pd.read_csv(FIG5A_CSV)
    import re

    def held_out(key: str) -> str | None:
        m = re.search(r"test\(([^)]+)\)", str(key))
        return m.group(1) if m else None

    def train_ds(key: str) -> list[str]:
        m = re.search(r"train\(([^)]+)\)", str(key))
        return m.group(1).split("+") if m else []

    # Cache concordant pos/neg counts per (phenotype, dataset).
    cache: dict[tuple[str, str], tuple[int, int]] = {}

    def conc_counts(phen: str, ds: str) -> tuple[int, int]:
        if (phen, ds) in cache:
            return cache[(phen, ds)]
        labels = load_dataset_labels(ds, phen)
        if labels is None or phen not in gapmind.columns:
            cache[(phen, ds)] = (0, 0)
            return 0, 0
        common = labels.index.intersection(gapmind.index)
        labels = labels.loc[common]
        preds = gapmind.loc[common, phen]
        conc = labels[labels == preds]
        out = (int((conc == 1).sum()), int((conc == 0).sum()))
        cache[(phen, ds)] = out
        return out

    rows = []
    for _, r in results[
        (results.split_type == "dataset_split")
        & (results.phenotype.isin(POOR))
    ].iterrows():
        held = held_out(r["key"])
        trains = train_ds(r["key"])
        test_pos, test_neg = conc_counts(r["phenotype"], held)
        train_pos = sum(conc_counts(r["phenotype"], d)[0] for d in trains)
        train_neg = sum(conc_counts(r["phenotype"], d)[1] for d in trains)
        rows.append(
            dict(
                phenotype=r["phenotype"],
                held_out=held,
                n_train=int(r["n_train"]),
                n_test=int(r["n_test"]),
                n_min_test=r["n_minority_test"],
                BA=round(r["balanced_accuracy"], 3),
                sens=round(r["sensitivity"], 3),
                spec=round(r["specificity"], 3),
                train_pos_frac=(
                    train_pos / (train_pos + train_neg)
                    if (train_pos + train_neg)
                    else np.nan
                ),
                test_pos_frac=(
                    test_pos / (test_pos + test_neg)
                    if (test_pos + test_neg)
                    else np.nan
                ),
                train_majority=("pos" if train_pos >= train_neg else "neg"),
                test_majority=("pos" if test_pos >= test_neg else "neg"),
                passes_filter=(
                    "yes"
                    if (not pd.isna(r["n_minority_test"])) and r["n_minority_test"] >= 10
                    else "no"
                ),
            )
        )
    return pd.DataFrame(rows)


def phylo_ooc_breakdown() -> pd.DataFrame:
    """Per-fold phylo_ooc summary for Glucose, Serine, Alanine."""
    results = pd.read_csv(FIG5A_CSV)
    sub = results[
        (results.split_type == "phylo_ooc")
        & (results.phenotype.isin(PHYLO_OOC_PHENS))
    ].copy()
    out = (
        sub.groupby("phenotype")["balanced_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )
    detail = sub[
        ["phenotype", "key", "n_train", "n_test", "balanced_accuracy",
         "sensitivity", "specificity"]
    ].copy()
    detail["balanced_accuracy"] = detail["balanced_accuracy"].round(3)
    detail["sensitivity"] = detail["sensitivity"].round(3)
    detail["specificity"] = detail["specificity"].round(3)
    return out, detail


def _print_df(df: pd.DataFrame, index: bool = True) -> None:
    """Print a DataFrame as text (avoids optional tabulate dependency)."""
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 60,
    ):
        print(df.to_string(index=index))


def main() -> None:
    """Run all diagnostics and print tables."""
    gapmind = load_gapmind()

    print("# Per-dataset concordant class composition\n")
    summary = concordant_summary(gapmind)
    pivot_pos = summary.pivot(index="phenotype", columns="dataset", values="pos_frac")
    pivot_n = summary.pivot(index="phenotype", columns="dataset", values="n_conc")
    pivot_min = summary.pivot(index="phenotype", columns="dataset", values="minority")
    print("## Positive fraction\n")
    _print_df(pivot_pos.round(2))
    print("\n## n_concordant (minority in parens)\n")
    combined = pivot_n.astype(str) + " (" + pivot_min.astype(str) + ")"
    _print_df(combined)

    print("\n# Genome overlap across source datasets\n")
    mat, sets = genome_overlap()
    _print_df(mat)
    sizes = {d: len(sets[d]) for d in DATASETS}
    print(f"\nDataset sizes (unique genomeIDs): {sizes}")

    print("\n# Phylum composition shift (concordant subset)\n")
    taxonomy = load_taxonomy()
    phylo = phylum_composition(gapmind, taxonomy)
    _print_df(phylo, index=False)

    print("\n# Per-fold breakdown — dataset_split, poor performers\n")
    pf = per_fold_breakdown(gapmind)
    pf2 = pf.copy()
    pf2["train_pos_frac"] = pf2["train_pos_frac"].round(2)
    pf2["test_pos_frac"] = pf2["test_pos_frac"].round(2)
    _print_df(pf2, index=False)

    print("\n# Phylo_ooc breakdown — Glucose, Serine, Alanine\n")
    summ, detail = phylo_ooc_breakdown()
    print("## Per-phenotype summary\n")
    _print_df(summ)
    print("\n## Per-fold detail\n")
    _print_df(detail, index=False)


if __name__ == "__main__":
    main()
