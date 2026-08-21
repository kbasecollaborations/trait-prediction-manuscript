#!/usr/bin/env python3
"""Aggregate per_fit JSON files into comparison tables and figures.

Outputs (under data/outputs/ml_comparison/ and figures/ml_comparison/):

  data/outputs/ml_comparison/
    results.csv                (long-form, all fits)
    summary_by_model.csv       (per-model mean across all valid fits)
    summary_by_model_subset_split.csv  (model x subset x split_type summary)
    delta_vs_cb_noeval.csv     (per-phenotype Δ-BA vs the cb_noeval baseline)

  figures/ml_comparison/
    model_x_phenotype_BA_heatmap__{subset}_{split_type}.pdf
    delta_vs_baseline_bars.pdf
    cross_dataset_gap_by_model.pdf

Idempotent: the per_fit JSONs are re-read on every run.

Run with: ``uv run python -m scripts.ml_comparison.aggregate``
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_BASE = Path("data/outputs/ml_comparison")
FIG_BASE = Path("figures/ml_comparison")
BASELINE_MODEL = "cb_noeval"
PRIMARY_METRIC = "balanced_accuracy"


def _load_long() -> pd.DataFrame:
    rows = []
    root = OUTPUT_BASE / "per_fit"
    if not root.exists():
        return pd.DataFrame()
    for p in root.rglob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        try:
            with p.open() as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        row = {
            k: d.get(k)
            for k in (
                "model",
                "phenotype",
                "subset",
                "split_type",
                "repeat",
                "n_train",
                "n_test",
                "n_features",
                "fit_seconds",
                "accuracy",
                "balanced_accuracy",
                "matthews_corrcoef",
                "precision",
                "recall",
                "f1",
                "sensitivity",
                "specificity",
                "roc_auc",
            )
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(
        ["model", "subset", "split_type", "phenotype", "repeat"], inplace=True
    )
    return df


def write_summaries(df: pd.DataFrame) -> None:
    by_model = (
        df.groupby(["model", "subset", "split_type"], dropna=False)[PRIMARY_METRIC]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    by_model.to_csv(OUTPUT_BASE / "summary_by_model_subset_split.csv", index=False)

    by_model_global = (
        df.groupby("model")[PRIMARY_METRIC]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    by_model_global.to_csv(OUTPUT_BASE / "summary_by_model.csv", index=False)

    if BASELINE_MODEL not in df["model"].unique():
        print(
            f"[aggregate] WARN: baseline {BASELINE_MODEL!r} not present yet — skipping delta tables"
        )
        return

    pivot = df.pivot_table(
        index=["phenotype", "subset", "split_type", "repeat"],
        columns="model",
        values=PRIMARY_METRIC,
    )
    if BASELINE_MODEL not in pivot.columns:
        return
    base = pivot[BASELINE_MODEL]
    deltas = pivot.subtract(base, axis=0)
    deltas = deltas.drop(columns=[BASELINE_MODEL])
    deltas = deltas.reset_index().melt(
        id_vars=["phenotype", "subset", "split_type", "repeat"],
        var_name="model",
        value_name="delta_BA",
    )
    deltas.to_csv(OUTPUT_BASE / "delta_vs_cb_noeval.csv", index=False)

    delta_summary = (
        deltas.groupby(["model", "subset", "split_type"], dropna=False)["delta_BA"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    delta_summary.to_csv(OUTPUT_BASE / "delta_vs_cb_noeval_summary.csv", index=False)


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIG_BASE.mkdir(parents=True, exist_ok=True)

    # Per-(subset, split_type) heatmap of model x phenotype mean BA.
    for subset in df["subset"].unique():
        for split_type in df["split_type"].unique():
            sub = df[(df["subset"] == subset) & (df["split_type"] == split_type)]
            if sub.empty:
                continue
            mean_pivot = sub.pivot_table(
                index="model",
                columns="phenotype",
                values=PRIMARY_METRIC,
                aggfunc="mean",
            )
            mean_pivot = mean_pivot.loc[
                mean_pivot.mean(axis=1).sort_values(ascending=False).index
            ]
            fig, ax = plt.subplots(
                figsize=(
                    max(6, 0.6 * mean_pivot.shape[1]),
                    max(3, 0.45 * mean_pivot.shape[0]),
                )
            )
            sns.heatmap(
                mean_pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                vmin=0.3,
                vmax=1.0,
                cbar_kws={"label": "Balanced accuracy"},
                ax=ax,
            )
            ax.set_title(f"BA by model x phenotype — {subset} / {split_type}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out = FIG_BASE / f"model_x_phenotype_BA_heatmap__{subset}_{split_type}.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(f"[aggregate] wrote {out}")

    # Cross-dataset gap (random_split mean BA - dataset_split mean BA) per model x subset.
    if {"random_split", "dataset_split"}.issubset(set(df["split_type"].unique())):
        agg = (
            df.groupby(["model", "subset", "split_type"], dropna=False)[PRIMARY_METRIC]
            .mean()
            .unstack("split_type")
        )
        if "random_split" in agg.columns and "dataset_split" in agg.columns:
            agg["cross_dataset_gap"] = agg["random_split"] - agg["dataset_split"]
            agg = agg.reset_index().dropna(subset=["cross_dataset_gap"])
            agg.to_csv(OUTPUT_BASE / "cross_dataset_gap_by_model.csv", index=False)

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=agg, x="model", y="cross_dataset_gap", hue="subset", ax=ax)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel(
                "Random Holdout BA − Cross Dataset BA  (lower = better generalization)"
            )
            ax.set_xlabel("")
            ax.set_title("Cross-dataset generalization gap by model")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out = FIG_BASE / "cross_dataset_gap_by_model.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(f"[aggregate] wrote {out}")

    # ΔBA vs cb_noeval bars, one panel per split type.
    delta_path = OUTPUT_BASE / "delta_vs_cb_noeval.csv"
    if not delta_path.exists():
        return
    deltas = pd.read_csv(delta_path)
    if deltas.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, split_type in zip(axes, ("random_split", "dataset_split")):
        sub = deltas[deltas["split_type"] == split_type]
        if sub.empty:
            continue
        agg = (
            sub.groupby(["model", "subset"], dropna=False)["delta_BA"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        models_order = agg.groupby("model")["mean"].mean().sort_values().index.tolist()
        subsets = sorted(agg["subset"].dropna().unique().tolist())
        width = 0.4
        xpos = np.arange(len(models_order))
        for i, subset in enumerate(subsets):
            ai = agg[agg["subset"] == subset].set_index("model").reindex(models_order)
            offset = (i - (len(subsets) - 1) / 2) * width
            ax.bar(
                xpos + offset,
                ai["mean"].fillna(0),
                yerr=ai["std"].fillna(0),
                width=width,
                label=subset,
                capsize=2,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(models_order, rotation=30, ha="right")
        ax.set_title(f"ΔBA vs cb_noeval — {split_type}")
        ax.set_ylabel("ΔBA (positive = beats cb_noeval)")
        ax.legend(title="subset", fontsize=8)
    plt.tight_layout()
    out = FIG_BASE / "delta_vs_baseline_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[aggregate] wrote {out}")


def main() -> None:
    df = _load_long()
    if df.empty:
        print("[aggregate] no per_fit JSON files found — run runner first")
        return
    df.to_csv(OUTPUT_BASE / "results.csv", index=False)
    print(f"[aggregate] results.csv: {len(df)} rows")
    write_summaries(df)
    print("[aggregate] summaries written")
    make_figures(df)
    print("[aggregate] done")


if __name__ == "__main__":
    main()
