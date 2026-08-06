#!/usr/bin/env python3
"""Aggregate metric-sweep and per-phenotype delta plots for Figure 6 Panels B and C."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scripts.minority_filter import filter_by_minority, full_test_minority_counts

CONFIDENCE_CONFIGS: list[str] = [
    "free_balanced",
    "current",
    "high_mech",
    "very_high_mech",
]
W_GAP_VALUES: dict[str, float] = {
    "free_balanced": 0.0,
    "current": 0.3,
    "high_mech": 0.4,
    "very_high_mech": 0.5,
}

CONFIG_TO_LABEL: dict[str, str] = {
    "free_balanced": "Confidence (no GapMind)",
    "current": "Confidence (w=0.3)",
    "high_mech": "Confidence (w=0.4)",
    "very_high_mech": "Confidence (w=0.5)",
    "problematic_removed": "Problematic removed",
    "concordant": "Concordant",
}

CONDITION_COLORS: dict[str, str] = {
    "full_data": "#7F7F7F",
    "problematic_removed": "#8D6E63",
    "free_balanced": "#A6DBA0",
    "current": "#5AAE61",
    "high_mech": "#1B7837",
    "very_high_mech": "#00441B",
    "concordant": "#2E86AB",
    "gapmind": "#8B5CF6",
}

METRIC_COLORS: dict[str, str] = {
    "balanced_accuracy": "#1f77b4",
    "precision": "#E69F00",
    "recall": "#d62728",
}
METRIC_LABELS: dict[str, str] = {
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
}
METRIC_MARKERS: dict[str, str] = {
    "balanced_accuracy": "o",
    "precision": "s",
    "recall": "^",
}


def _load_long_form(data_dir: Path, phenotypes: list[str]) -> pd.DataFrame:
    """Assemble long-form data for the six filter conditions.

    Parameters
    ----------
    data_dir : Path
        Directory containing Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include.

    Returns
    -------
    pd.DataFrame
        Columns ``config``, ``phenotype``, ``balanced_accuracy``, ``precision``,
        ``recall``, ``n_train``, ``n_val``. The six configs are the four
        confidence-filter weight settings plus ``concordant`` and
        ``problematic_removed``.
    """
    full_minority = full_test_minority_counts()
    phenotype_set = set(phenotypes)
    metrics = ["balanced_accuracy", "precision", "recall"]
    cols = ["phenotype", *metrics, "n_train", "n_val"]
    rows: list[pd.DataFrame] = []

    sweep = pd.read_csv(data_dir / "figure6b_weight_sweep_combined.csv")
    sweep = sweep[sweep["split_type"] == "dataset_split"].copy()
    sweep = filter_by_minority(sweep, full_minority)
    sweep = sweep[sweep["phenotype"].isin(phenotype_set)]
    for cfg in CONFIDENCE_CONFIGS:
        sub = sweep[sweep["config"] == cfg][cols].assign(config=cfg)
        rows.append(sub)

    conc = pd.read_csv(
        Path("data/outputs/figure5/figure5c_concordant_train_different_test.csv")
    )
    conc = conc[
        (conc["split_type"] == "dataset_split") & (conc["test_type"] == "full")
    ].copy()
    conc = filter_by_minority(conc, full_minority)
    conc = conc[conc["phenotype"].isin(phenotype_set)]
    rows.append(conc[cols].assign(config="concordant"))

    prob_all = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    prob_all = filter_by_minority(prob_all, full_minority, key_column="split")
    prob_all = prob_all[prob_all["phenotype"].isin(phenotype_set)]
    prob = prob_all[prob_all["condition"] == "filtered"]
    rows.append(prob[cols].assign(config="problematic_removed"))
    full_data = prob_all[prob_all["condition"] == "full"]
    rows.append(full_data[cols].assign(config="full_data"))

    return pd.concat(rows, ignore_index=True)


def _gapmind_baseline(phenotypes: list[str]) -> pd.DataFrame:
    """Load GapMind cross-dataset metrics restricted to common phenotypes.

    Parameters
    ----------
    phenotypes : list[str]
        Phenotypes to retain.

    Returns
    -------
    pd.DataFrame
        Minority-filtered per-row metrics.
    """
    full_minority = full_test_minority_counts()
    gm = pd.read_csv("data/outputs/figure3/gapmind_dataset_split_metrics.tsv", sep="\t")
    test_col = "test_dataset" if "test_dataset" in gm.columns else None
    gm = filter_by_minority(gm, full_minority, test_dataset_column=test_col)
    return gm[gm["phenotype"].isin(set(phenotypes))]


def plot_metric_sweep(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
) -> pd.DataFrame:
    """Panel B: BA / precision / recall vs GapMind weight with reference endpoints.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data_dir : Path
        Directory containing Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include.

    Returns
    -------
    pd.DataFrame
        Per-phenotype-per-config metric means used for the plot.
    """
    rng = np.random.default_rng(0)
    long_df = _load_long_form(data_dir, phenotypes)
    long_df = long_df.assign(trainval=long_df["n_train"] + long_df["n_val"])
    ph_means = (
        long_df.groupby(["phenotype", "config"])[
            ["balanced_accuracy", "precision", "recall"]
        ]
        .mean()
        .reset_index()
    )

    # Equal categorical spacing for all seven columns; the x-tick labels carry
    # the literal w_gap values.
    column_order = [
        "full_data",
        "problematic_removed",
        *CONFIDENCE_CONFIGS,
        "concordant",
    ]
    column_step = 1.0
    config_to_x: dict[str, float] = {
        cfg: i * column_step for i, cfg in enumerate(column_order)
    }
    xs_sweep = [config_to_x[c] for c in CONFIDENCE_CONFIGS]
    ref_x = {
        "full_data": config_to_x["full_data"],
        "problematic_removed": config_to_x["problematic_removed"],
        "concordant": config_to_x["concordant"],
    }
    jitter_width = 0.12

    # Per-phenotype scatter underlay, jittered so the spread is visible behind
    # the mean +/- SEM markers.
    for metric, color in METRIC_COLORS.items():
        for cfg, x_centre in config_to_x.items():
            vals = ph_means[ph_means["config"] == cfg][metric].to_numpy()
            jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals))
            ax.scatter(
                np.full_like(vals, x_centre, dtype=float) + jitter,
                vals,
                s=10,
                color=color,
                alpha=0.22,
                edgecolors="none",
                zorder=2,
            )

    # Mean GapMind balanced accuracy over the same minority-filtered phenotypes
    # quoted in the Results text.
    gm = _gapmind_baseline(phenotypes)
    ax.axhline(
        float(gm.groupby("phenotype")["balanced_accuracy"].mean().mean()),
        linestyle=":",
        linewidth=1.2,
        color=METRIC_COLORS["balanced_accuracy"],
        alpha=0.8,
        zorder=1,
        label="GapMind (balanced accuracy)",
    )

    for metric, color in METRIC_COLORS.items():
        sweep_means = []
        sweep_sems = []
        for cfg in CONFIDENCE_CONFIGS:
            sub = ph_means[ph_means["config"] == cfg][metric]
            sweep_means.append(float(sub.mean()))
            sweep_sems.append(float(sub.sem()))
        ax.errorbar(
            xs_sweep,
            sweep_means,
            yerr=sweep_sems,
            color=color,
            linewidth=1.6,
            marker=METRIC_MARKERS[metric],
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
            elinewidth=0.9,
            capsize=2.5,
            label=METRIC_LABELS[metric],
            zorder=4,
        )
        for ref_cfg, ref_x_val in ref_x.items():
            sub = ph_means[ph_means["config"] == ref_cfg][metric]
            if sub.empty:
                continue
            ax.errorbar(
                ref_x_val,
                float(sub.mean()),
                yerr=float(sub.sem()),
                color=color,
                marker=METRIC_MARKERS[metric],
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                elinewidth=0.9,
                capsize=2.5,
                zorder=5,
            )

    # Dotted separators bracket the three filter groups.
    ref_full_x = config_to_x["full_data"]
    ref_prob_x = config_to_x["problematic_removed"]
    sweep_start = config_to_x[CONFIDENCE_CONFIGS[0]]
    sweep_end = config_to_x[CONFIDENCE_CONFIGS[-1]]
    ref_conc_x = config_to_x["concordant"]
    ax.axvline(
        (ref_prob_x + sweep_start) / 2.0,
        color="grey",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.axvline(
        (sweep_end + ref_conc_x) / 2.0,
        color="grey",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )

    config_n: dict[str, int] = {
        cfg: int(round(float(long_df[long_df["config"] == cfg]["trainval"].mean())))
        for cfg in column_order
    }
    xticks = [config_to_x[cfg] for cfg in column_order]
    base_labels = [
        "No filter",
        "Problematic\nremoved",
        "$w_{\\mathrm{gap}}\\!=\\!0$",
        "$w_{\\mathrm{gap}}\\!=\\!0.3$",
        "$w_{\\mathrm{gap}}\\!=\\!0.4$",
        "$w_{\\mathrm{gap}}\\!=\\!0.5$",
        "Concordant",
    ]
    xticklabels = [
        f"{label}\n$n$={config_n[cfg]}" for label, cfg in zip(base_labels, column_order)
    ]
    ax.set_xlabel("Training-data filter")
    ax.set_ylabel("Metric value across 15 phenotypes")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_xlim(ref_full_x - 0.6, ref_conc_x + 0.6)
    ax.set_ylim(0.2, 1.0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    return ph_means


MCNEMAR_FILES: dict[str, Path] = {
    "concordant": Path("data/outputs/stats/per_phenotype_mcnemar.tsv"),
    "mech_free": Path("data/outputs/stats/per_phenotype_mcnemar_mechfree.tsv"),
}
"""Per-phenotype sensitivity tests for the two Figure 6C arms, both scored on the
same held-out genomes by ``scripts/stats/manuscript_pvalues.py``."""


def _mcnemar_sensitivity(arm: str) -> pd.DataFrame | None:
    """Per-phenotype pooled sensitivity test for one Figure 6C arm.

    Parameters
    ----------
    arm : str
        ``"concordant"`` or ``"mech_free"``.

    Returns
    -------
    pd.DataFrame | None
        Frame indexed by phenotype with ``delta`` (ML minus GapMind sensitivity,
        pooled over genomes) and ``q_value_BH``, or ``None`` if the table is
        absent.
    """
    path = MCNEMAR_FILES[arm]
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["metric"] == "sensitivity"].set_index("phenotype")
    return frame[["delta", "q_value_BH"]]


def _pooled_sensitivity_deltas() -> pd.DataFrame | None:
    """Pooled sensitivity deltas and q-values for both arms, aligned by phenotype.

    Returns
    -------
    pd.DataFrame | None
        Columns ``concordant``, ``mech_free``, ``q_concordant``, ``q_mech_free``,
        or ``None`` when either arm's test table is missing.
    """
    conc = _mcnemar_sensitivity("concordant")
    free = _mcnemar_sensitivity("mech_free")
    if conc is None or free is None:
        return None
    common = conc.index.intersection(free.index)
    return pd.DataFrame(
        {
            "concordant": conc.loc[common, "delta"],
            "mech_free": free.loc[common, "delta"],
            "q_concordant": conc.loc[common, "q_value_BH"],
            "q_mech_free": free.loc[common, "q_value_BH"],
        }
    )


def plot_gapmind_delta_forest(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "recall",
) -> pd.DataFrame:
    """Panel C: per-phenotype $\\Delta$ metric vs the GapMind baseline.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data_dir : Path
        Directory containing Figure 6 output CSVs.
    phenotypes : list[str]
        Phenotypes to include (intersection with available data).
    metric : str
        Column name to compare; defaults to ``"recall"``.

    Returns
    -------
    pd.DataFrame
        Per-phenotype delta values for both filters.
    """
    long_df = _load_long_form(data_dir, phenotypes)
    long_df = long_df.assign(trainval=long_df["n_train"] + long_df["n_val"])
    gm = _gapmind_baseline(phenotypes)
    gm_means = gm.groupby("phenotype")[metric].mean()

    ml_means = long_df.groupby(["phenotype", "config"])[metric].mean().unstack("config")

    # Bars carry the pooled per-genome McNemar deltas, the quantity the markers
    # are tested on; averaging the per-split values instead would weight a
    # 33-genome split like a 235-genome one. Falls back to the per-split means
    # when the McNemar tables are absent.
    pooled = _pooled_sensitivity_deltas()
    if pooled is not None and metric == "recall":
        delta_df = pooled.sort_values("concordant")
        pooled_source = True
    else:
        delta_concordant = (ml_means["concordant"] - gm_means).dropna()
        delta_mechfree = (ml_means["free_balanced"] - gm_means).dropna()
        common = delta_concordant.index.intersection(delta_mechfree.index)
        delta_df = pd.DataFrame(
            {
                "concordant": delta_concordant.loc[common],
                "mech_free": delta_mechfree.loc[common],
            }
        ).sort_values("concordant")
        pooled_source = False

    n_conc = int(
        round(float(long_df[long_df["config"] == "concordant"]["trainval"].mean()))
    )
    n_mech = int(
        round(float(long_df[long_df["config"] == "free_balanced"]["trainval"].mean()))
    )

    y_positions = np.arange(len(delta_df))
    bar_height = 0.38

    ax.barh(
        y_positions - bar_height / 2,
        delta_df["concordant"].to_numpy(),
        height=bar_height,
        color=CONDITION_COLORS["concordant"],
        edgecolor="black",
        linewidth=0.5,
        label=f"Concordant ML ($n$={n_conc})",
        zorder=3,
    )
    ax.barh(
        y_positions + bar_height / 2,
        delta_df["mech_free"].to_numpy(),
        height=bar_height,
        color=CONDITION_COLORS["free_balanced"],
        edgecolor="black",
        linewidth=0.5,
        label=f"Confidence (no GapMind, $n$={n_mech})",
        zorder=3,
    )

    ax.axvline(0, color="black", linewidth=0.9, zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(delta_df.index, fontsize=9)
    ax.set_ylabel("Phenotype")
    metric_label = {
        "recall": "recall",
        "precision": "precision",
        "balanced_accuracy": "balanced accuracy",
        "f1": "F1",
        "f1_calc": "F1",
    }.get(metric, metric.replace("_", " "))
    ax.set_xlabel(f"$\\Delta$ {metric_label} (ML $-$ GapMind)")
    ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)

    # Per-phenotype significance markers, each drawn on the bar whose test
    # produced it.
    n_sig_better = n_sig_worse = None
    n_sig_better_mech = n_sig_worse_mech = None
    if pooled_source:
        n_sig_better = n_sig_worse = 0
        n_sig_better_mech = n_sig_worse_mech = 0
        for arm, q_col, offset_sign, counters in (
            ("concordant", "q_concordant", -1, "conc"),
            ("mech_free", "q_mech_free", +1, "mech"),
        ):
            for y, phenotype in zip(y_positions, delta_df.index):
                if delta_df.loc[phenotype, q_col] >= 0.05:
                    continue
                value = delta_df.loc[phenotype, arm]
                offset = 0.022 if value >= 0 else -0.022
                # A marker rather than a "*" text glyph, whose ink sits high in
                # its text box and so lands above the bar under va="center".
                ax.plot(
                    value + offset,
                    y + offset_sign * bar_height / 2,
                    marker=(6, 2, 0),
                    markersize=3.6,
                    markeredgewidth=0.7,
                    color="black",
                    linestyle="none",
                    zorder=4,
                )
                if counters == "conc":
                    if value > 0:
                        n_sig_better += 1
                    else:
                        n_sig_worse += 1
                else:
                    if value > 0:
                        n_sig_better_mech += 1
                    else:
                        n_sig_worse_mech += 1

        # Borders of the region containing no significant result, drawn only
        # when the significant and non-significant effects separate. They are
        # descriptive of this data, not a critical value: McNemar power also
        # depends on n and on the discordant-pair counts.
        magnitudes = pd.concat(
            [
                delta_df[["concordant", "q_concordant"]].rename(
                    columns={"concordant": "d", "q_concordant": "q"}
                ),
                delta_df[["mech_free", "q_mech_free"]].rename(
                    columns={"mech_free": "d", "q_mech_free": "q"}
                ),
            ]
        )
        magnitudes["absd"] = magnitudes["d"].abs()
        sig_min = magnitudes.loc[magnitudes["q"] < 0.05, "absd"].min()
        ns_max = magnitudes.loc[magnitudes["q"] >= 0.05, "absd"].max()
        if pd.notna(sig_min) and pd.notna(ns_max) and ns_max < sig_min:
            edge = float((sig_min + ns_max) / 2.0)
            for position, label in (
                (-edge, f"$|\\Delta|<{edge:.2f}$: no $q<0.05$"),
                (edge, None),
            ):
                ax.axvline(
                    position,
                    color="grey",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                    zorder=1,
                    label=label,
                )

    n_pos_conc = int((delta_df["concordant"] > 0).sum())
    n_pos_mech = int((delta_df["mech_free"] > 0).sum())
    n_total = len(delta_df)

    # How many phenotypes each filter significantly beats GapMind on. The
    # phenotype-level Wilcoxon tests live in
    # data/outputs/stats/manuscript_pvalues.tsv and are cited in the text.
    if n_sig_better is None:
        annotation = (
            f"$\\Delta>0$: {n_pos_conc}/{n_total} conc., {n_pos_mech}/{n_total} free"
        )
    else:
        annotation = (
            f"Significant vs GapMind:\n"
            f"  concordant {n_sig_better} better, {n_sig_worse} worse\n"
            f"  mech-free {n_sig_better_mech} better, {n_sig_worse_mech} worse"
        )
    # Drawn after the band so the band is included in the legend.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    # Bottom-right: rows sort ascending, so the positive half of the lowest
    # rows is clear.
    ax.text(
        0.98,
        0.03,
        annotation,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
    )

    return delta_df


def best_panel_b_config(
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "balanced_accuracy",
) -> tuple[str, str, pd.Series]:
    """Identify the highest-performing weight-sweep config on the test set.

    Parameters
    ----------
    data_dir : Path
        Directory containing the weight-sweep CSV.
    phenotypes : list[str]
        Phenotypes to score over.
    metric : str
        Metric column to rank by.

    Returns
    -------
    tuple[str, str, pd.Series]
        ``(config_name, condition_label, per_config_means)``.
    """
    full_minority = full_test_minority_counts()
    sweep = pd.read_csv(data_dir / "figure6b_weight_sweep_combined.csv")
    sweep = sweep[sweep["split_type"] == "dataset_split"].copy()
    sweep = filter_by_minority(sweep, full_minority)
    sweep = sweep[sweep["phenotype"].isin(set(phenotypes))]
    summary = (
        sweep.groupby(["config", "phenotype"])[metric]
        .mean()
        .groupby("config")
        .mean()
        .sort_values(ascending=False)
    )
    best_config = str(summary.idxmax())
    return best_config, CONFIG_TO_LABEL[best_config], summary
