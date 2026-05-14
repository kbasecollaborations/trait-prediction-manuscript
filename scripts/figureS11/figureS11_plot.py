#!/usr/bin/env python3
"""
Render Figure S11: SHAP beeswarm summaries for histidine and galactose.

The figure stacks one beeswarm panel per phenotype. Each panel uses the
canonical ``shap.summary_plot(plot_type="dot")`` representation, restricted
to the top 15 features by mean absolute SHAP value, so readers can see the
direction (and dispersion) in which each pathway gene drives the prediction
toward growth. Feature value (binary 0/1 = absence/presence) is encoded by
the colour bar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" matplotlib style)
import seaborn as sns
import shap

from scripts.visualization import configure_plot_style

plt.style.use(["science", "nature"])
sns.set_context("paper")
configure_plot_style()

PHENOTYPES: tuple[str, ...] = ("Histidine", "Galactose")
TOP_N: int = 15
PANEL_LABELS: tuple[str, ...] = ("A", "B")

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data/outputs/figureS11"
OUTPUT_PDF: Path = REPO_ROOT / "figures/figure_s11.pdf"


def load_shap_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    """
    Load SHAP arrays produced by ``figureS11_data.py``.

    Parameters
    ----------
    npz_path : Path
        Path to a ``<phenotype>_shap_values.npz`` archive containing
        ``shap_values``, ``feature_values``, ``feature_names``,
        ``predictions``, and ``y_true``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``shap_values`` (float, n_samples x n_features),
        ``feature_values`` (int, n_samples x n_features),
        ``feature_names`` (object array of feature labels),
        ``predictions`` and ``y_true`` (int arrays of length n_samples).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "shap_values": np.asarray(data["shap_values"], dtype=float),
            "feature_values": np.asarray(data["feature_values"], dtype=float),
            "feature_names": np.asarray(data["feature_names"], dtype=object),
            "predictions": np.asarray(data["predictions"], dtype=int),
            "y_true": np.asarray(data["y_true"], dtype=int),
        }


def render_beeswarm_panel(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    title: str,
    top_n: int,
    show_colorbar: bool,
) -> None:
    """
    Render a single SHAP beeswarm panel onto an existing matplotlib axis.

    Parameters
    ----------
    ax : plt.Axes
        Target matplotlib axis. ``shap.summary_plot`` paints onto the current
        figure, so this function makes ``ax`` current before plotting.
    arrays : dict[str, np.ndarray]
        Dictionary returned by :func:`load_shap_arrays`.
    title : str
        Panel title (typically the phenotype name).
    top_n : int
        Number of top features (by mean ``|SHAP|``) to display.
    show_colorbar : bool
        Whether to show the SHAP "feature value" colour bar for this panel.
    """
    plt.sca(ax)
    shap.summary_plot(
        arrays["shap_values"],
        arrays["feature_values"],
        feature_names=list(arrays["feature_names"]),
        plot_type="dot",
        max_display=top_n,
        show=False,
        color_bar=show_colorbar,
        plot_size=None,
    )
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("SHAP value (impact on prediction toward growth)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=10)


def make_figure(
    phenotypes: tuple[str, ...],
    data_dir: Path,
    output_pdf: Path,
    top_n: int,
) -> None:
    """
    Build and save the two-panel Figure S11 PDF.

    Parameters
    ----------
    phenotypes : tuple[str, ...]
        Phenotype names to render, one panel per entry.
    data_dir : Path
        Directory containing ``<phenotype>_shap_values.npz`` files.
    output_pdf : Path
        Path to the output PDF.
    top_n : int
        Number of top features to display per panel.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(phenotypes), 1, figsize=(12, 12))
    if len(phenotypes) == 1:
        axes = [axes]

    for ax, phenotype, label in zip(axes, phenotypes, PANEL_LABELS, strict=False):
        npz_path = data_dir / f"{phenotype}_shap_values.npz"
        arrays = load_shap_arrays(npz_path)
        render_beeswarm_panel(
            ax=ax,
            arrays=arrays,
            title=f"{label}. {phenotype}",
            top_n=top_n,
            show_colorbar=True,
        )
        report_top_features(phenotype, arrays, top_n=5)

    fig.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"\nWrote {output_pdf}")
    plt.close(fig)


def report_top_features(
    phenotype: str, arrays: dict[str, np.ndarray], top_n: int
) -> None:
    """
    Print the top ``top_n`` features by mean absolute SHAP value.

    Parameters
    ----------
    phenotype : str
        Phenotype name (printed in the report header).
    arrays : dict[str, np.ndarray]
        Dictionary returned by :func:`load_shap_arrays`.
    top_n : int
        Number of features to report.
    """
    mean_abs = np.abs(arrays["shap_values"]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    print(f"\n[{phenotype}] Top-{top_n} features by mean |SHAP|:")
    for rank, idx in enumerate(order, start=1):
        print(
            f"  {rank}. {arrays['feature_names'][idx]} "
            f"(mean |SHAP|={mean_abs[idx]:.4f})"
        )


def main() -> None:
    """Entry point for the Figure S11 plotting script."""
    make_figure(
        phenotypes=PHENOTYPES,
        data_dir=DATA_DIR,
        output_pdf=OUTPUT_PDF,
        top_n=TOP_N,
    )


if __name__ == "__main__":
    main()
