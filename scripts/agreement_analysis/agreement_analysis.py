"""ML-GapMind agreement as a label-free applicability signal, with singleton and ensemble baselines.

Covers singleton baselines (ML alone, GapMind alone), agreement as an abstention
signal, voting and stacked ensembles, and a matched-coverage comparison of ML
confidence against ML-GapMind agreement, for both the concordant-trained and
full-data CatBoost models under the cross-dataset (leave-one-dataset-out)
protocol.

Reads the per-sample prediction tables in data/outputs/figure7/; no models are
trained here. Writes the a1_*.csv tables to data/outputs/agreement_analysis/.

Run with::

    uv run python -m scripts.agreement_analysis.agreement_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_DIR = Path("data/outputs/figure7")
OUT_DIR = Path("data/outputs/agreement_analysis")
CONCORDANT_FILE = DATA_DIR / "figure7_per_sample.tsv"
FULLDATA_FILE = DATA_DIR / "figure7_per_sample_fulldata.tsv"


def _ba(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Balanced accuracy guarding against single-class slices.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_pred : pd.Series
        Predicted binary labels.

    Returns
    -------
    float
        Balanced accuracy, or ``np.nan`` if fewer than two true classes are
        present.
    """
    if y_true.nunique() < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def _acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Plain accuracy of ``y_pred`` against ``y_true``."""
    return float(accuracy_score(y_true, y_pred))


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a per-sample prediction table and derive ensemble columns.

    Parameters
    ----------
    path : Path
        Path to a ``figure7_per_sample*.tsv`` file.

    Returns
    -------
    pd.DataFrame
        Predictions with added boolean ``agree`` column and ensemble rule
        predictions (``ens_*``).
    """
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["y_true", "y_pred", "gapmind_pred", "proba"]).copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["gapmind_pred"] = df["gapmind_pred"].astype(int)
    df["agree"] = df["y_pred"] == df["gapmind_pred"]

    # Ensemble rules, all label-free at deployment.
    df["ens_agree_else_gapmind"] = np.where(
        df["agree"], df["y_pred"], df["gapmind_pred"]
    )
    df["ens_agree_else_ml"] = np.where(df["agree"], df["y_pred"], df["y_pred"])
    df["soft_avg"] = 0.5 * df["proba"] + 0.5 * df["gapmind_pred"]
    df["ens_soft_avg"] = (df["soft_avg"] >= 0.5).astype(int)
    df["ens_or"] = ((df["y_pred"] == 1) | (df["gapmind_pred"] == 1)).astype(int)
    df["ens_and"] = ((df["y_pred"] == 1) & (df["gapmind_pred"] == 1)).astype(int)
    return df


def baselines_table(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Pooled and per-phenotype singleton baselines.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction table from :func:`load_predictions`.
    model : str
        Model label ("concordant" or "fulldata").

    Returns
    -------
    pd.DataFrame
        One row per (scope, predictor) with BA and accuracy.
    """
    rows: list[dict[str, object]] = []

    def add(scope: str, n: int, name: str, yt: pd.Series, yp: pd.Series) -> None:
        rows.append(
            {
                "model": model,
                "scope": scope,
                "n": n,
                "predictor": name,
                "balanced_accuracy": _ba(yt, yp),
                "accuracy": _acc(yt, yp),
            }
        )

    add("pooled", len(df), "ml_alone", df["y_true"], df["y_pred"])
    add("pooled", len(df), "gapmind_alone", df["y_true"], df["gapmind_pred"])
    for ph, g in df.groupby("phenotype"):
        add(ph, len(g), "ml_alone", g["y_true"], g["y_pred"])
        add(ph, len(g), "gapmind_alone", g["y_true"], g["gapmind_pred"])
    return pd.DataFrame(rows)


def agreement_table(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Agreement-as-selective-signal table (pooled + per-phenotype).

    Reports, for the full set, the agree subset, and the disagree subset, the
    coverage and BA/accuracy of the ML predictions on that subset.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction table from :func:`load_predictions`.
    model : str
        Model label.

    Returns
    -------
    pd.DataFrame
        Agreement breakdown rows.
    """
    rows: list[dict[str, object]] = []

    def add(scope: str, frame: pd.DataFrame) -> None:
        n_all = len(frame)
        agree = frame[frame["agree"]]
        disagree = frame[~frame["agree"]]
        rows.append(
            {
                "model": model,
                "scope": scope,
                "subset": "all",
                "n": n_all,
                "coverage": 1.0,
                "ml_balanced_accuracy": _ba(frame["y_true"], frame["y_pred"]),
                "ml_accuracy": _acc(frame["y_true"], frame["y_pred"]),
            }
        )
        rows.append(
            {
                "model": model,
                "scope": scope,
                "subset": "agree",
                "n": len(agree),
                "coverage": len(agree) / n_all if n_all else float("nan"),
                "ml_balanced_accuracy": _ba(agree["y_true"], agree["y_pred"]),
                "ml_accuracy": _acc(agree["y_true"], agree["y_pred"]),
            }
        )
        rows.append(
            {
                "model": model,
                "scope": scope,
                "subset": "disagree",
                "n": len(disagree),
                "coverage": len(disagree) / n_all if n_all else float("nan"),
                "ml_balanced_accuracy": _ba(disagree["y_true"], disagree["y_pred"]),
                "ml_accuracy": _acc(disagree["y_true"], disagree["y_pred"]),
            }
        )

    add("pooled", df)
    for ph, g in df.groupby("phenotype"):
        add(ph, g)
    return pd.DataFrame(rows)


def ensemble_table(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Voting / stacked ensemble predictors vs singletons.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction table from :func:`load_predictions`.
    model : str
        Model label.

    Returns
    -------
    pd.DataFrame
        One row per (scope, predictor) with BA and accuracy, covering both
        singletons and ensemble rules.
    """
    predictors = {
        "ml_alone": "y_pred",
        "gapmind_alone": "gapmind_pred",
        "agree_else_gapmind": "ens_agree_else_gapmind",
        "agree_else_ml": "ens_agree_else_ml",
        "soft_avg": "ens_soft_avg",
        "logical_or": "ens_or",
        "logical_and": "ens_and",
    }
    rows: list[dict[str, object]] = []

    def add(scope: str, frame: pd.DataFrame) -> None:
        for name, col in predictors.items():
            rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "n": len(frame),
                    "predictor": name,
                    "balanced_accuracy": _ba(frame["y_true"], frame[col]),
                    "accuracy": _acc(frame["y_true"], frame[col]),
                }
            )

    add("pooled", df)
    for ph, g in df.groupby("phenotype"):
        add(ph, g)
    return pd.DataFrame(rows)


def ensemble_win_summary(ens: pd.DataFrame) -> pd.DataFrame:
    """Count phenotypes where each ensemble beats both singletons on BA.

    Parameters
    ----------
    ens : pd.DataFrame
        Output of :func:`ensemble_table` for one model.

    Returns
    -------
    pd.DataFrame
        Per-ensemble pooled BA and the number of phenotypes where it strictly
        exceeds both ML-alone and GapMind-alone BA.
    """
    per_ph = ens[ens["scope"] != "pooled"]
    wide = per_ph.pivot_table(
        index=["model", "scope"], columns="predictor", values="balanced_accuracy"
    )
    ens_names = ["agree_else_gapmind", "soft_avg", "logical_or", "logical_and"]
    pooled = ens[ens["scope"] == "pooled"].set_index("predictor")
    out: list[dict[str, object]] = []
    for name in ens_names:
        wins = int(
            (
                (wide[name] > wide["ml_alone"]) & (wide[name] > wide["gapmind_alone"])
            ).sum()
        )
        n_ph = int(wide[name].notna().sum())
        out.append(
            {
                "model": ens["model"].iloc[0],
                "ensemble": name,
                "pooled_ba": float(pooled.loc[name, "balanced_accuracy"]),
                "pooled_ba_ml_alone": float(
                    pooled.loc["ml_alone", "balanced_accuracy"]
                ),
                "pooled_ba_gapmind_alone": float(
                    pooled.loc["gapmind_alone", "balanced_accuracy"]
                ),
                "n_phenotypes": n_ph,
                "wins_over_both_singletons": wins,
            }
        )
    return pd.DataFrame(out)


def retained_ba_at_coverage(
    df: pd.DataFrame, signal: str, target_coverage: float
) -> tuple[float, float, float]:
    """BA on the retained subset when keeping ``target_coverage`` by a signal.

    The confidence signal ranks samples by ``confidence`` (descending) and keeps
    the top fraction. The agreement signal keeps agreeing samples first and
    breaks ties by confidence to hit exactly the target coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled prediction table.
    signal : str
        Either ``"confidence"`` or ``"agreement"``.
    target_coverage : float
        Fraction of samples to retain.

    Returns
    -------
    tuple[float, float, float]
        ``(achieved_coverage, balanced_accuracy, accuracy)`` on the retained
        subset, evaluated using the ML prediction ``y_pred``.
    """
    n = len(df)
    k = int(round(target_coverage * n))
    if signal == "confidence":
        order = df.sort_values("confidence", ascending=False)
        kept = order.iloc[:k]
    elif signal == "agreement":
        df2 = df.assign(rank_key=df["agree"].astype(int))
        order = df2.sort_values(["rank_key", "confidence"], ascending=[False, False])
        kept = order.iloc[:k]
    else:
        raise ValueError(f"unknown signal: {signal}")
    return (
        len(kept) / n,
        _ba(kept["y_true"], kept["y_pred"]),
        _acc(kept["y_true"], kept["y_pred"]),
    )


def trust_head_to_head(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Head-to-head abstention signals at matched coverage.

    Compares ML confidence against ML-GapMind agreement at (i) the coverage that
    agreement naturally yields and (ii) 0.5 coverage, on the same pooled test
    set. The concordance meta-classifier is excluded because its saved artefact
    holds only AUC/AUPRC summaries, not per-sample P(concordant) scores.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled prediction table from :func:`load_predictions`.
    model : str
        Model label.

    Returns
    -------
    pd.DataFrame
        Retained-subset BA/accuracy for each (signal, target coverage).
    """
    natural_cov = float(df["agree"].mean())
    targets = {"agreement_natural": natural_cov, "half": 0.5}
    rows: list[dict[str, object]] = []
    for tname, cov in targets.items():
        for signal in ("confidence", "agreement"):
            ach, ba, acc = retained_ba_at_coverage(df, signal, cov)
            rows.append(
                {
                    "model": model,
                    "target": tname,
                    "target_coverage": cov,
                    "signal": signal,
                    "achieved_coverage": ach,
                    "retained_balanced_accuracy": ba,
                    "retained_accuracy": acc,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Build every table, write the a1_*.csv files to ``OUT_DIR``, and print a summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {"concordant": CONCORDANT_FILE, "fulldata": FULLDATA_FILE}

    base_all, agree_all, ens_all, win_all, h2h_all = [], [], [], [], []
    for model, path in models.items():
        df = load_predictions(path)
        base_all.append(baselines_table(df, model))
        agree_all.append(agreement_table(df, model))
        e = ensemble_table(df, model)
        ens_all.append(e)
        win_all.append(ensemble_win_summary(e))
        h2h_all.append(trust_head_to_head(df, model))

    baselines = pd.concat(base_all, ignore_index=True)
    agreement = pd.concat(agree_all, ignore_index=True)
    ensembles = pd.concat(ens_all, ignore_index=True)
    wins = pd.concat(win_all, ignore_index=True)
    h2h = pd.concat(h2h_all, ignore_index=True)

    baselines.to_csv(OUT_DIR / "a1_baselines.csv", index=False)
    agreement.to_csv(OUT_DIR / "a1_agreement.csv", index=False)
    ensembles.to_csv(OUT_DIR / "a1_ensembles.csv", index=False)
    wins.to_csv(OUT_DIR / "a1_ensemble_wins.csv", index=False)
    h2h.to_csv(OUT_DIR / "a1_trust_head_to_head.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n===== BASELINES (pooled) =====")
    print(baselines[baselines["scope"] == "pooled"].to_string(index=False))
    print("\n===== AGREEMENT (pooled) =====")
    print(agreement[agreement["scope"] == "pooled"].to_string(index=False))
    print("\n===== ENSEMBLE WIN SUMMARY =====")
    print(wins.to_string(index=False))
    print("\n===== ENSEMBLES (pooled) =====")
    print(ensembles[ensembles["scope"] == "pooled"].to_string(index=False))
    print("\n===== TRUST HEAD-TO-HEAD =====")
    print(h2h.to_string(index=False))


if __name__ == "__main__":
    main()
