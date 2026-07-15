#!/usr/bin/env python3
"""More data vs. concordant filtering: a head-to-head on held-out datasets.

For each held-out manuscript dataset D (leave-one-of-4-out) and each shared
phenotype, train CatBoost under five strategies along the concordance<->volume
axis and test on D's full labels:

    C    concordant samples of the other 3 datasets   (concordant filtering)
    F    full other 3 datasets                        (no filtering)
    FB   full other 3 + all BacDive                   (more data)
    B    BacDive only                                 (max data, diff. assay)
    CB   concordant other 3 + BacDive                 (filtering + volume)

Plus two controls:
    volume  F + BacDive(n) as n grows, vs the fixed C baseline (does volume
            ever reach concordant filtering?)
    nmatch  at matched N=|C|: C vs random-from-F vs BacDive-subsample
            (curation effect vs sheer sample count)

Resumable: results are checkpointed to ``data/outputs/bacdive/head_to_head.csv``
after every batch. Re-running skips fits already in the checkpoint, so an
interrupted run continues where it stopped. Use ``--fresh`` to start over.

Run:
    uv run python -m scripts.bacdive.head_to_head            # run / resume full sweep
    uv run python -m scripts.bacdive.head_to_head --fresh    # restart
    uv run python -m scripts.bacdive.head_to_head --smoke    # fast reduced grid

Parallelism / batching via env: BACDIVE_N_JOBS, BACDIVE_THREAD_COUNT,
BACDIVE_BATCH_SIZE (default 140; checkpoint frequency = once per batch).
"""

import argparse
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from scripts.bacdive.prepare_bacdive import KOFAM_REDUCED, strip_version
from scripts.bacdive.splits import load_bacdive_labels, load_group_a_labels
from scripts.bacdive.worker import load_matrix, run_mixed_job
from scripts.figure5.figure5a_data import (
    get_concordant_samples,
    load_experimental_phenotypes,
    load_gapmind_predictions,
)

MAN_KOFAM = "data/processed/features_reduced/combined_datasets/kofam.tsv"
BAC_KOFAM = str(KOFAM_REDUCED)
GAPMIND = Path("data/outputs/figure2/gapmind_phenotypes_loose.tsv")
PHENO_DIR = Path("data/processed/phenotypes")
OUTPUT_DIR = Path("data/outputs/bacdive")

DATASETS = ["atleaf", "lit", "marine", "pmi"]
PHENOTYPES = ["Glucose", "Maltose", "Sucrose", "Mannose", "Fructose", "Galactose",
              "Mannitol", "Cellobiose", "Glycerol", "Serine", "Alanine",
              "m-Inositol", "Histidine"]
SEEDS = range(5)
VOLUME_SEEDS = range(3)
VOLUME_GRID = [0, 100, 250, 500, 1000]  # + full appended per (phenotype, D)
MIN_TEST_MINORITY = 10
MIN_TRAIN_MINORITY = 5

N_JOBS = int(os.environ.get("BACDIVE_N_JOBS", os.cpu_count() or 4))
THREAD_COUNT = int(os.environ.get("BACDIVE_THREAD_COUNT", 1))


def _minority(y: pd.Series, ids: list[str]) -> int:
    """Size of the smaller class among ``ids`` (0 if only one class present)."""
    vc = y.loc[ids].value_counts()
    return int(vc.min()) if len(vc) == 2 else 0


def _train_val(
    ids: list[str], y: pd.Series, seed: int, val_frac: float = 0.15
) -> tuple[list[str], list[str]]:
    """Stratified train/val split of ids; falls back to no-val if too small."""
    sub = y.loc[ids]
    if sub.nunique() < 2 or sub.value_counts().min() < 2:
        return list(ids), []
    tr, va = train_test_split(np.asarray(ids), test_size=val_frac,
                              random_state=seed, stratify=sub.to_numpy())
    return list(tr), list(va)


def _subsample(ids: list[str], y: pd.Series, n: int, seed: int) -> list[str]:
    """Stratified subsample of ``ids`` to size ``n`` (all of ``ids`` if ``n`` exceeds it)."""
    if n >= len(ids):
        return list(ids)
    sub = y.loc[ids]
    if sub.nunique() < 2:
        return list(ids)[:n]
    keep, _ = train_test_split(np.asarray(ids), train_size=n, random_state=seed,
                               stratify=sub.to_numpy())
    return list(keep)


def build_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build every train/score job for the head-to-head sweep.

    Iterates phenotypes x held-out datasets and, per combination, emits the
    strategy fits (C/F/FB/B/CB), the volume curve (F + BacDive(n)), and the
    N-matched control at N=|C|. Leakage genomes shared between BacDive and the
    held-out test are dropped from the BacDive training pool.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(jobs, skips)`` where ``jobs`` are runnable job payloads for
        :func:`scripts.bacdive.worker.run_mixed_job` and ``skips`` records the
        (phenotype, held_out, strategy) combinations dropped for too few
        minority-class samples.
    """
    gm = load_gapmind_predictions(GAPMIND)
    exp = load_experimental_phenotypes(PHENO_DIR)
    man_idx = load_matrix(MAN_KOFAM).index
    bac_idx = load_matrix(BAC_KOFAM).index

    jobs: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []

    for ph in PHENOTYPES:
        man_y = load_group_a_labels(ph)
        man_y = man_y.loc[man_y.index.intersection(man_idx)]
        bac_y = load_bacdive_labels(ph)
        bac_y = bac_y.loc[bac_y.index.intersection(bac_idx)]
        concordant = set(get_concordant_samples(gm, exp, ph)) & set(man_y.index)

        # genome -> dataset membership (native ids)
        ds_of: dict[str, str] = {}
        for ds in DATASETS:
            f = PHENO_DIR / ds / f"{ph}.tsv"
            if f.exists():
                ids = pd.read_csv(f, sep="\t", index_col=0).index.astype(str)
                for g in ids:
                    ds_of[g] = ds

        for D in DATASETS:
            d_ids = [g for g in man_y.index if ds_of.get(g) == D]
            if _minority(man_y, d_ids) < MIN_TEST_MINORITY:
                skips.append({"phenotype": ph, "held_out": D,
                              "status": "skip: test minority"})
                continue
            other3 = [g for g in man_y.index if ds_of.get(g) not in (D, None)]
            c_ids = [g for g in other3 if g in concordant]
            # drop BacDive genomes that leak into the held-out test (stripped GCF)
            d_stripped = {strip_version(g) for g in d_ids}
            b_ids = [g for g in bac_y.index if strip_version(g) not in d_stripped]
            test_part = [MAN_KOFAM, "man", d_ids]

            strategies = {"C": [("man", c_ids)], "F": [("man", other3)],
                          "B": [("bac", b_ids)],
                          "FB": [("man", other3), ("bac", b_ids)],
                          "CB": [("man", c_ids), ("bac", b_ids)]}
            for strat, parts in strategies.items():
                if _minority(man_y if strat != "B" else bac_y,
                             parts[0][1]) < MIN_TRAIN_MINORITY:
                    skips.append({"phenotype": ph, "held_out": D, "strategy": strat,
                                  "status": "skip: train minority"})
                    continue
                for seed in SEEDS:
                    tp, vp = _build_parts(parts, man_y, bac_y, seed)
                    jobs.append(_job("strategy", ph, D, strat, seed, tp, vp,
                                     test_part, c_ids, b_ids, other3))

            # --- volume curve: F + BacDive(n) ---
            grid = sorted(set(VOLUME_GRID) | {len(b_ids)})
            for n in grid:
                for seed in VOLUME_SEEDS:
                    add = _subsample(b_ids, bac_y, n, seed) if n > 0 else []
                    parts = [("man", other3)] + ([("bac", add)] if add else [])
                    tp, vp = _build_parts(parts, man_y, bac_y, seed)
                    jobs.append(_job("volume", ph, D, f"FB_n{n}", seed, tp, vp,
                                     test_part, c_ids, b_ids, other3, n_added=n))

            # --- N-matched control at N=|C| ---
            n = len(c_ids)
            if n >= MIN_TRAIN_MINORITY * 2:
                for seed in SEEDS:
                    variants = {
                        "nmatch_C": [("man", c_ids)],
                        "nmatch_randF": [("man", _subsample(other3, man_y, n, seed))],
                        "nmatch_subB": [("bac", _subsample(b_ids, bac_y, n, seed))],
                    }
                    for tag, parts in variants.items():
                        tp, vp = _build_parts(parts, man_y, bac_y, seed)
                        jobs.append(_job("nmatch", ph, D, tag, seed, tp, vp,
                                         test_part, c_ids, b_ids, other3))
    return jobs, skips


def _build_parts(
    parts: list[tuple[str, list[str]]],
    man_y: pd.Series,
    bac_y: pd.Series,
    seed: int,
) -> tuple[list[list], list[list]]:
    """Turn [(group, ids)] into train_parts/val_parts with a stratified val carve."""
    train_parts, val_parts = [], []
    for group, ids in parts:
        path = MAN_KOFAM if group == "man" else BAC_KOFAM
        y = man_y if group == "man" else bac_y
        tr, va = _train_val(ids, y, seed)
        train_parts.append([path, "man" if group == "man" else "bacdive", tr])
        if va:
            val_parts.append([path, "man" if group == "man" else "bacdive", va])
    return train_parts, val_parts


def _job(
    analysis: str,
    ph: str,
    D: str,
    strat: str,
    seed: int,
    train_parts: list[list],
    val_parts: list[list],
    test_part: list,
    c_ids: list[str],
    b_ids: list[str],
    other3: list[str],
    n_added: int | None = None,
) -> dict[str, Any]:
    """Assemble one job payload (metadata + train/val/test parts) for a worker."""
    n_train = sum(len(p[2]) for p in train_parts)
    return {"analysis": analysis, "phenotype": ph, "held_out": D, "strategy": strat,
            "seed": seed, "n_added": n_added, "n_train": n_train,
            "n_test": len(test_part[2]), "n_concordant": len(c_ids),
            "n_bacdive": len(b_ids), "n_other3": len(other3),
            "train_parts": train_parts, "val_parts": val_parts, "test_part": test_part}


# A fit is uniquely identified by these columns (n_added is folded into the
# volume strategy tag "FB_n{n}", so the five fields below are sufficient).
KEY_COLS = ["analysis", "phenotype", "held_out", "strategy", "seed"]
BATCH_SIZE = int(os.environ.get("BACDIVE_BATCH_SIZE", 140))


def _job_key(job: dict[str, Any]) -> tuple:
    return tuple(str(job[k]) for k in KEY_COLS)


def _load_done_keys(path: Path) -> set[tuple]:
    """Keys of fits already saved in the checkpoint (resume support)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=KEY_COLS, dtype=str)
    return set(map(tuple, df.values))


def _append_results(path: Path, rows: list[dict]) -> None:
    """Append a batch to the checkpoint via read-concat-write (crash-safe enough)."""
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    tmp = path.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def _chunks(seq: list, n: int) -> Iterator[list]:
    """Yield successive ``n``-sized chunks of ``seq``."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main(smoke: bool = False, fresh: bool = False) -> None:
    """Run (or resume) the full sweep, checkpointing after every batch.

    Parameters
    ----------
    smoke : bool, optional
        Use a fast reduced grid (fewer phenotypes/seeds), by default False.
    fresh : bool, optional
        Ignore any existing checkpoint and start over, by default False.
    """
    global PHENOTYPES, SEEDS, VOLUME_SEEDS, DATASETS
    suffix = ""
    if smoke:
        PHENOTYPES = ["Histidine", "Glucose"]
        SEEDS = range(1)
        VOLUME_SEEDS = range(1)
        DATASETS = ["marine", "lit", "atleaf", "pmi"]
        suffix = "_smoke"
        print("[smoke] reduced grid")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUTPUT_DIR / f"head_to_head{suffix}.csv"
    skips_file = OUTPUT_DIR / f"head_to_head{suffix}_skips.csv"

    if fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[fresh] removed existing checkpoint {ckpt}")

    print("Building jobs ...")
    jobs, skips = build_jobs()
    if skips:
        pd.DataFrame(skips).to_csv(skips_file, index=False)

    done = _load_done_keys(ckpt)
    todo = [j for j in jobs if _job_key(j) not in done]
    print(f"  {len(jobs)} total fits | {len(done)} already done (resume) | "
          f"{len(todo)} to run | {len(skips)} build-skips")
    print(f"  n_jobs={N_JOBS}, thread_count={THREAD_COUNT}, batch_size={BATCH_SIZE}")
    if not todo:
        print("Nothing to run; checkpoint already complete.")
    else:
        t0 = time.time()
        for bi, batch in enumerate(_chunks(todo, BATCH_SIZE), 1):
            res = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(run_mixed_job)(job, THREAD_COUNT) for job in batch
            )
            _append_results(ckpt, list(res))
            elapsed = time.time() - t0
            n_done = min(bi * BATCH_SIZE, len(todo))
            rate = n_done / elapsed
            eta = (len(todo) - n_done) / rate / 60 if rate else 0
            print(f"  batch {bi}: {n_done}/{len(todo)} done, "
                  f"{elapsed / 60:.1f} min elapsed, ~{eta:.1f} min left "
                  f"-> checkpoint saved to {ckpt}")

    df = pd.read_csv(ckpt)
    ok = df[df.status == "ok"]
    print(f"\nsaved {ckpt} ({len(ok)} ok, {len(df) - len(ok)} error rows)")
    strat = ok[ok.analysis == "strategy"]
    if len(strat):
        print("\n=== mean balanced accuracy by strategy (held-out manuscript test) ===")
        print(strat.groupby("strategy")["balanced_accuracy"]
              .agg(["mean", "std", "count"]).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast reduced grid")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()
    main(smoke=args.smoke, fresh=args.fresh)
