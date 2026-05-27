#!/usr/bin/env python3
"""Aggregate visualisations for Figure 6 Panels B and C.

Panel B (``plot_metric_sweep``) plots cross-dataset balanced accuracy,
precision, and recall (mean across 15 phenotypes, with SEM error bars) as a
function of the GapMind weight $w_{\\mathrm{gap}}$ in the soft-label
confidence filter. The problematic-sample-removal and concordance filters are
anchored as marker columns at the left and right ends of the sweep so the
six conditions appear on a single axis.

Panel C (``plot_gapmind_delta_forest``) plots the per-phenotype difference
$\\Delta = \\mathrm{ML} - \\mathrm{GapMind}$ on a chosen metric (recall by
default) on the same full cross-dataset held-out test set, comparing two
representative ML filters (concordance-trained ML and the mechanism-free
confidence filter). Phenotypes are sorted by the concordant-ML $\\Delta$, and
the annotation reports the count of phenotypes with positive $\\Delta$ and
the paired one-sided Wilcoxon signed-rank $p$-value against GapMind.

``best_panel_b_config`` is a small utility that returns the highest-performing
confidence-sweep config for logging in ``figure6_plot.py``.
"""

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
    "precision": "#9467bd",
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
    rows.append(
        conc[cols].assign(config="concordant")
    )

    prob_all = pd.read_csv(data_dir / "figure6c_dataset_split_results.csv")
    prob_all = filter_by_minority(prob_all, full_minority, key_column="split")
    prob_all = prob_all[prob_all["phenotype"].isin(phenotype_set)]
    prob = prob_all[prob_all["condition"] == "filtered"]
    rows.append(
        prob[cols].assign(config="problematic_removed")
    )
    full_data = prob_all[prob_all["condition"] == "full"]
    rows.append(
        full_data[cols].assign(config="full_data")
    )

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
    gm = pd.read_csv(
        "data/outputs/figure3/gapmind_dataset_split_metrics.tsv", sep="\t"
    )
    test_col = "test_dataset" if "test_dataset" in gm.columns else None
    gm = filter_by_minority(gm, full_minority, test_dataset_column=test_col)
    return gm[gm["phenotype"].isin(set(phenotypes))]


def plot_metric_sweep(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
) -> pd.DataFrame:
    """Panel B: BA / precision / recall vs GapMind weight with reference endpoints.

    The confidence-filter sweep occupies the central x range. The
    problematic-sample-removal filter and the concordance filter are plotted
    as separated marker columns to the left and right of the sweep so the
    full filter family appears on a single axis.

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

    # Equal categorical spacing for all seven columns (Full data, Problematic,
    # four confidence-sweep settings, Concordant). Sweep positions are no
    # longer at the literal w_gap values; the x-tick labels carry the values
    # so the spacing remains uniform across the filter family.
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

    # Per-phenotype scatter underlay for each metric, lightly jittered around
    # the corresponding x-position so the spread across phenotypes is visible
    # behind the mean +/- SEM markers.
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

    # Dotted separators bracket the three filter groups (unfiltered references,
    # confidence sweep, hard concordance reference).
    ref_full_x = config_to_x["full_data"]
    ref_prob_x = config_to_x["problematic_removed"]
    sweep_start = config_to_x[CONFIDENCE_CONFIGS[0]]
    sweep_end = config_to_x[CONFIDENCE_CONFIGS[-1]]
    ref_conc_x = config_to_x["concordant"]
    ax.axvline((ref_prob_x + sweep_start) / 2.0,
               color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axvline((sweep_end + ref_conc_x) / 2.0,
               color="grey", linestyle=":", linewidth=0.6, alpha=0.5)

    # Mean train+val sample count per condition for the x-tick annotations.
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
        f"{label}\n$n$={config_n[cfg]}"
        for label, cfg in zip(base_labels, column_order)
    ]
    ax.set_xlabel("Training-data filter")
    ax.set_ylabel("Metric value across 15 phenotypes")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_xlim(ref_full_x - 0.6, ref_conc_x + 0.6)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    return ph_means


def plot_gapmind_delta_forest(
    ax: Axes,
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "recall",
) -> pd.DataFrame:
    """Panel C: per-phenotype $\\Delta$ metric vs the GapMind baseline.

    For each of the 15 phenotypes, two horizontal bars report
    $\\Delta = \\mathrm{ML} - \\mathrm{GapMind}$ on the same full cross-dataset
    held-out test set: one bar for concordance-trained ML and one bar for the
    mechanism-free confidence-filtered ML. Positive bars indicate phenotypes
    on which the ML filter exceeded GapMind; negative bars indicate phenotypes
    on which GapMind achieved a higher value of the metric. Phenotypes are
    sorted by the concordant-ML $\\Delta$.

    The annotation reports the count of phenotypes on which each ML filter
    exceeded GapMind and the paired one-sided Wilcoxon signed-rank $p$-value
    against the GapMind baseline.

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
    from scipy.stats import wilcoxon

    long_df = _load_long_form(data_dir, phenotypes)
    long_df = long_df.assign(trainval=long_df["n_train"] + long_df["n_val"])
    gm = _gapmind_baseline(phenotypes)
    gm_means = gm.groupby("phenotype")[metric].mean()

    ml_means = (
        long_df.groupby(["phenotype", "config"])[metric]
        .mean()
        .unstack("config")
    )

    delta_concordant = (ml_means["concordant"] - gm_means).dropna()
    delta_mechfree = (ml_means["free_balanced"] - gm_means).dropna()
    common = delta_concordant.index.intersection(delta_mechfree.index)
    delta_df = pd.DataFrame(
        {
            "concordant": delta_concordant.loc[common],
            "mech_free": delta_mechfree.loc[common],
        }
    ).sort_values("concordant")

    n_conc = int(round(float(long_df[long_df["config"] == "concordant"]["trainval"].mean())))
    n_mech = int(round(float(long_df[long_df["config"] == "free_balanced"]["trainval"].mean())))

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
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    n_pos_conc = int((delta_df["concordant"] > 0).sum())
    n_pos_mech = int((delta_df["mech_free"] > 0).sum())
    n_total = len(delta_df)
    _, p_conc = wilcoxon(
        ml_means.loc[common, "concordant"],
        gm_means.loc[common],
        alternative="greater",
    )
    _, p_mech = wilcoxon(
        ml_means.loc[common, "free_balanced"],
        gm_means.loc[common],
        alternative="greater",
    )
    ax.text(
        0.98,
        0.04,
        (
            f"Concordant: {n_pos_conc}/{n_total}, $p$={p_conc:.3f}\n"
            f"Mech-free: {n_pos_mech}/{n_total}, $p$={p_mech:.2f}"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
    )

    return delta_df


def best_panel_b_config(
    data_dir: Path,
    phenotypes: list[str],
    metric: str = "balanced_accuracy",
) -> tuple[str, str, pd.Series]:
    """Identify the highest-performing weight-sweep config on the test set.

    Used by ``figure6_plot.py`` to log the top-BA confidence-sweep config
    alongside whichever config is actually rendered in Panel C.

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
