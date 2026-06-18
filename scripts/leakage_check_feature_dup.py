"""Quantify feature-level data leakage in random-holdout splits.

Genome IDs are disjoint between train and test, but two genomes with different
IDs can have identical (or near-identical) KOFAM feature vectors and the same
label, with one in train and one in test of the same split. To the model this is
an effective memorized duplicate. This script measures exact and near feature
duplicates across all random splits and bounds the phenomenon globally.

Load path: scripts.ml_splits.load_split_data (random_split), which uses the
published feature matrix data/processed/features_reduced/combined_datasets/kofam.tsv.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from scripts.ml_splits import load_split_data


def _row_hashes(X: pd.DataFrame) -> list[str]:
    """Hash each (binary) row of a feature matrix to a hex digest.

    Parameters
    ----------
    X : pd.DataFrame
        Binary feature matrix (genomeID index, KO columns).

    Returns
    -------
    list[str]
        One hex digest per row, in row order.
    """
    arr = X.to_numpy(dtype=np.uint8)
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in arr]


def global_bound(feature_file: str) -> dict[str, int]:
    """Count duplicate feature vectors across the whole combined matrix.

    Parameters
    ----------
    feature_file : str
        Path to the combined kofam.tsv feature matrix.

    Returns
    -------
    dict[str, int]
        n_genomes, n_distinct_vectors, n_genomes_in_dup_groups, n_dup_groups.
    """
    df = pd.read_csv(feature_file, sep="\t", index_col=0)
    # ensure binary
    arr = (df.to_numpy() > 0).astype(np.uint8)
    hashes = [hashlib.sha1(row.tobytes()).hexdigest() for row in arr]
    counts = Counter(hashes)
    n_genomes = len(hashes)
    n_distinct = len(counts)
    dup_groups = {h: c for h, c in counts.items() if c > 1}
    n_in_dup_groups = sum(dup_groups.values())
    return {
        "n_genomes": n_genomes,
        "n_distinct_vectors": n_distinct,
        "n_genomes_in_dup_groups": n_in_dup_groups,
        "n_dup_groups": len(dup_groups),
    }


def analyze_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Analyze exact and near feature duplicates for one random split.

    Training pool = train + val. Test = held-out set.

    Returns
    -------
    dict
        Per-split metrics plus example exact-duplicate pairs.
    """
    # Align columns to a common order (they should already match).
    cols = X_train.columns
    Xtr = pd.concat([X_train, X_val], axis=0)
    ytr = pd.concat([y_train, y_val], axis=0)
    Xtr = Xtr[cols]
    Xte = X_test[cols]

    tr_arr = (Xtr.to_numpy() > 0).astype(np.uint8)
    te_arr = (Xte.to_numpy() > 0).astype(np.uint8)
    ytr_arr = np.asarray(ytr).astype(int)
    yte_arr = np.asarray(y_test).astype(int)
    tr_ids = Xtr.index.to_numpy()
    te_ids = Xte.index.to_numpy()

    n_test = len(te_ids)

    # --- Exact duplicates via hashing ---
    tr_hashes = [hashlib.sha1(r.tobytes()).hexdigest() for r in tr_arr]
    te_hashes = [hashlib.sha1(r.tobytes()).hexdigest() for r in te_arr]
    # map train hash -> set of labels and list of (id, label)
    tr_hash_map: dict[str, list[tuple[str, int]]] = {}
    for h, gid, lab in zip(tr_hashes, tr_ids, ytr_arr):
        tr_hash_map.setdefault(h, []).append((str(gid), int(lab)))

    n_exact = 0
    n_exact_same = 0
    n_exact_diff = 0
    exact_examples: list[dict] = []
    for i, h in enumerate(te_hashes):
        matches = tr_hash_map.get(h)
        if not matches:
            continue
        n_exact += 1
        test_lab = int(yte_arr[i])
        match_labels = {lab for _, lab in matches}
        # same-label leak if any matching train genome shares the test label
        if test_lab in match_labels:
            n_exact_same += 1
        else:
            n_exact_diff += 1
        if len(match_labels) > 1:
            mixed = True
        else:
            mixed = False
        if len(exact_examples) < 50:
            # pick a representative matching train genome (prefer same-label)
            same_lab_match = next((g for g, lab in matches if lab == test_lab), None)
            rep = same_lab_match if same_lab_match is not None else matches[0][0]
            rep_lab = next((lab for g, lab in matches if g == rep), matches[0][1])
            exact_examples.append(
                {
                    "test_id": str(te_ids[i]),
                    "train_id": rep,
                    "test_label": test_lab,
                    "train_label": int(rep_lab),
                    "n_train_matches": len(matches),
                    "mixed_train_labels": mixed,
                }
            )

    # --- Near duplicates via numpy ---
    # shared present KOs = te_arr @ tr_arr.T
    shared = te_arr.astype(np.int32) @ tr_arr.astype(np.int32).T  # (n_test, n_train)
    tr_card = tr_arr.sum(axis=1).astype(np.int32)  # (n_train,)
    te_card = te_arr.sum(axis=1).astype(np.int32)  # (n_test,)
    # union = |A| + |B| - intersection
    union = te_card[:, None] + tr_card[None, :] - shared
    union_safe = np.where(union == 0, 1, union)  # avoid div by zero (both all-zero)
    jaccard = shared / union_safe
    # both all-zero vectors -> define Jaccard 1.0 (identical empty sets)
    both_zero = (te_card[:, None] == 0) & (tr_card[None, :] == 0)
    jaccard = np.where(both_zero, 1.0, jaccard)
    # Hamming distance = symmetric difference = |A| + |B| - 2*intersection
    hamming = te_card[:, None] + tr_card[None, :] - 2 * shared

    max_jac = jaccard.max(axis=1)  # best similarity per test genome
    argmax_jac = jaccard.argmax(axis=1)
    min_ham = hamming.min(axis=1)

    def _count_jac(thr: float) -> int:
        return int((max_jac >= thr).sum())

    def _count_ham(thr: int) -> int:
        return int((min_ham <= thr).sum())

    jac_099 = _count_jac(0.99)
    jac_098 = _count_jac(0.98)
    jac_095 = _count_jac(0.95)
    ham_0 = _count_ham(0)
    ham_2 = _count_ham(2)
    ham_5 = _count_ham(5)
    ham_10 = _count_ham(10)

    # near-dup (Jaccard>=0.98) label agreement with nearest (max-jaccard) train genome
    n_near = 0
    n_near_same = 0
    near_examples: list[dict] = []
    for i in range(n_test):
        if max_jac[i] >= 0.98:
            n_near += 1
            j = int(argmax_jac[i])
            test_lab = int(yte_arr[i])
            train_lab = int(ytr_arr[j])
            if test_lab == train_lab:
                n_near_same += 1
            if len(near_examples) < 50:
                near_examples.append(
                    {
                        "test_id": str(te_ids[i]),
                        "train_id": str(tr_ids[j]),
                        "jaccard": round(float(max_jac[i]), 4),
                        "hamming": int(hamming[i, j]),
                        "test_label": test_lab,
                        "train_label": train_lab,
                    }
                )

    return {
        "n_test": n_test,
        "n_exact": n_exact,
        "n_exact_same": n_exact_same,
        "n_exact_diff": n_exact_diff,
        "jac_099": jac_099,
        "jac_098": jac_098,
        "jac_095": jac_095,
        "ham_0": ham_0,
        "ham_2": ham_2,
        "ham_5": ham_5,
        "ham_10": ham_10,
        "n_near098": n_near,
        "n_near098_same": n_near_same,
        "exact_examples": exact_examples,
        "near_examples": near_examples,
    }


def main() -> None:
    feature_file = "data/processed/features_reduced/combined_datasets/kofam.tsv"

    print("=" * 70)
    print("GLOBAL BOUND on the full combined kofam matrix")
    print("=" * 70)
    gb = global_bound(feature_file)
    for k, v in gb.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("Loading random splits ...")
    print("=" * 70)
    sd = load_split_data(split_types=["random_split"])
    splits = sd["random_split"]
    print(f"  loaded {len(splits)} random splits")
    print()

    rows: list[dict] = []
    all_exact_examples: list[dict] = []
    all_near_examples: list[dict] = []

    for key, d in splits.items():
        # phenotype = key without trailing _<seed>
        phenotype = key.rsplit("_", 1)[0]
        res = analyze_split(
            d["X_train"], d["y_train"],
            d["X_val"], d["y_val"],
            d["X_test"], d["y_test"],
        )
        row = {"split": key, "phenotype": phenotype}
        for k, v in res.items():
            if k in ("exact_examples", "near_examples"):
                continue
            row[k] = v
        rows.append(row)
        for ex in res["exact_examples"]:
            ex2 = dict(ex)
            ex2["split"] = key
            ex2["phenotype"] = phenotype
            all_exact_examples.append(ex2)
        for ex in res["near_examples"]:
            ex2 = dict(ex)
            ex2["split"] = key
            ex2["phenotype"] = phenotype
            all_near_examples.append(ex2)

    df = pd.DataFrame(rows)

    num_cols = [
        "n_test", "n_exact", "n_exact_same", "n_exact_diff",
        "jac_099", "jac_098", "jac_095",
        "ham_0", "ham_2", "ham_5", "ham_10",
        "n_near098", "n_near098_same",
    ]

    print("=" * 70)
    print("AGGREGATE across all", len(df), "random splits")
    print("=" * 70)
    tot = df[num_cols].sum()
    n_test_total = int(tot["n_test"])
    print(f"  total test predictions: {n_test_total}")
    print()
    print("  EXACT feature duplicates (test vector identical to >=1 train vector):")
    print(f"    n_exact total          : {int(tot['n_exact'])} "
          f"({100*tot['n_exact']/n_test_total:.2f}%)")
    print(f"    of which SAME label    : {int(tot['n_exact_same'])} "
          f"({100*tot['n_exact_same']/n_test_total:.2f}%)   <-- effective leak")
    print(f"    of which DIFF label    : {int(tot['n_exact_diff'])} "
          f"({100*tot['n_exact_diff']/n_test_total:.2f}%)")
    print()
    print("  NEAR duplicates (Jaccard >= 0.98 to nearest train vector):")
    print(f"    n_near098 total        : {int(tot['n_near098'])} "
          f"({100*tot['n_near098']/n_test_total:.2f}%)")
    print(f"    of which SAME label    : {int(tot['n_near098_same'])} "
          f"({100*tot['n_near098_same']/n_test_total:.2f}%)")
    print()
    print("  Jaccard thresholds (count of test genomes, fraction of all test):")
    for thr, col in [("0.99", "jac_099"), ("0.98", "jac_098"), ("0.95", "jac_095")]:
        print(f"    Jaccard >= {thr}: {int(tot[col]):5d} "
              f"({100*tot[col]/n_test_total:.2f}%)")
    print("  Hamming thresholds (count of test genomes, fraction of all test):")
    for thr, col in [("0", "ham_0"), ("<=2", "ham_2"), ("<=5", "ham_5"), ("<=10", "ham_10")]:
        print(f"    Hamming {thr:>4}: {int(tot[col]):5d} "
              f"({100*tot[col]/n_test_total:.2f}%)")
    print()

    print("=" * 70)
    print("PER-PHENOTYPE breakdown")
    print("=" * 70)
    grp = df.groupby("phenotype")[num_cols].sum()
    grp = grp.assign(
        pct_exact_same=(100 * grp["n_exact_same"] / grp["n_test"]).round(2),
        pct_near_same=(100 * grp["n_near098_same"] / grp["n_test"]).round(2),
    )
    show_cols = [
        "n_test", "n_exact", "n_exact_same", "n_exact_diff", "pct_exact_same",
        "n_near098", "n_near098_same", "pct_near_same",
        "ham_0", "ham_2", "jac_098",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(grp[show_cols].to_string())
    print()

    print("=" * 70)
    print("EXAMPLE exact same-label duplicate pairs (up to 25)")
    print("=" * 70)
    exact_same = [e for e in all_exact_examples if e["test_label"] == e["train_label"]]
    if not exact_same:
        print("  (none)")
    else:
        for e in exact_same[:25]:
            print(f"  {e['phenotype']:18s} split={e['split']:16s} "
                  f"test={e['test_id']:22s} train={e['train_id']:22s} "
                  f"label={e['test_label']} n_matches={e['n_train_matches']} "
                  f"mixed={e['mixed_train_labels']}")
    print()

    # mixed-label exact duplicates (same vector, different label in train) - ambiguous
    mixed = [e for e in all_exact_examples if e["mixed_train_labels"]]
    print(f"  exact-dup test genomes whose train matches have MIXED labels: {len(mixed)}")
    print()

    # Save full per-split table
    out = "data/outputs/leakage_feature_dup_per_split.csv"
    import os
    os.makedirs("data/outputs", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"per-split table written to {out}")


if __name__ == "__main__":
    main()
