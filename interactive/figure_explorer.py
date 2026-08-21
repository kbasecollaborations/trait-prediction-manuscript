"""
Interactive Figure Explorer for Trait Prediction Manuscript.

A marimo notebook for exploring manuscript figures interactively using Altair visualizations.
Run with: marimo run figure_explorer.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def imports():
    """Import required libraries."""
    import marimo as mo
    import altair as alt
    import pandas as pd
    from pathlib import Path

    alt.data_transformers.enable("default")

    return alt, mo, pd, Path


@app.cell
def constants(Path):
    """Define constants and paths."""
    # Base data directory (relative to notebook location)
    DATA_DIR = Path(__file__).parent.parent / "data" / "outputs"

    # Phenotypes in display order
    PHENOTYPES = [
        "Alanine",
        "Arginine",
        "Cellobiose",
        "Fructose",
        "Galactose",
        "Galacturonic-Acid",
        "Glucose",
        "Glycerol",
        "Histidine",
        "m-Inositol",
        "Maltose",
        "Mannose",
        "Mannitol",
        "Serine",
        "Sucrose",
    ]

    DATASETS = ["atleaf", "lit", "marine", "pmi"]

    # Dataset display names
    DATASET_DISPLAY = {
        "atleaf": "AtLeaf",
        "lit": "Biolog",
        "marine": "Marine",
        "pmi": "Populus",
    }

    # Colorblind-friendly palette (seaborn colorblind)
    DATASET_COLORS = {
        "atleaf": "#0173b2",  # Blue
        "lit": "#de8f05",  # Orange
        "marine": "#029e73",  # Green
        "pmi": "#d55e00",  # Red
    }

    # Metrics available in ML results
    METRICS = [
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "accuracy",
    ]

    return (
        DATA_DIR,
        PHENOTYPES,
        DATASETS,
        DATASET_DISPLAY,
        DATASET_COLORS,
        METRICS,
    )


@app.cell
def data_loaders(DATA_DIR, pd):
    """Data loading functions for each figure."""

    def load_figure1b_data() -> pd.DataFrame:
        """Load genome counts data for Figure 1B."""
        return pd.read_csv(DATA_DIR / "figure1" / "figure1b_data.csv")

    def load_figure2_gapmind_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load GapMind metrics data for Figure 2."""
        strict = pd.read_csv(
            DATA_DIR / "figure2" / "gapmind_strict_metrics.tsv", sep="\t"
        )
        loose = pd.read_csv(
            DATA_DIR / "figure2" / "gapmind_loose_metrics.tsv", sep="\t"
        )
        return strict, loose

    def load_figure2_baselines_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load baseline model results for Figure 2."""
        random_baselines = pd.read_csv(
            DATA_DIR / "figure2" / "random_split_baselines.tsv", sep="\t"
        )
        phylo_baselines = pd.read_csv(
            DATA_DIR / "figure2" / "out_of_clade_split_baselines.tsv", sep="\t"
        )
        return random_baselines, phylo_baselines

    def load_figure3_data() -> pd.DataFrame:
        """Load ML results for Figure 3."""
        return pd.read_csv(DATA_DIR / "figure3" / "ml_results.csv")

    def load_figure3_gapmind_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load GapMind reference data for Figure 3."""
        random_gapmind = pd.read_csv(
            DATA_DIR / "figure3" / "gapmind_random_split_metrics.tsv", sep="\t"
        )
        dataset_gapmind = pd.read_csv(
            DATA_DIR / "figure3" / "gapmind_dataset_split_metrics.tsv", sep="\t"
        )
        return random_gapmind, dataset_gapmind

    def load_figure5a_data() -> pd.DataFrame:
        """Load concordant ML results for Figure 5A."""
        return pd.read_csv(
            DATA_DIR / "figure5" / "figure5a_concordant_ml_results.csv"
        )

    def load_figure5b_data() -> pd.DataFrame:
        """Load feature comparison data for Figure 5B."""
        return pd.read_csv(
            DATA_DIR / "figure5" / "figure5b_feature_comparison_summary.csv"
        )

    def load_figure5c_data() -> pd.DataFrame:
        """Load concordant train different test data for Figure 5C."""
        return pd.read_csv(
            DATA_DIR / "figure5" / "figure5c_concordant_train_different_test.csv"
        )

    def load_figure6b_data() -> pd.DataFrame:
        """Load confident ML results for Figure 6B."""
        return pd.read_csv(
            DATA_DIR / "figure6" / "figure6b_confident_ml_results.csv"
        )

    def load_figure6c_data() -> pd.DataFrame:
        """Load dataset split results for Figure 6C."""
        return pd.read_csv(
            DATA_DIR / "figure6" / "figure6c_dataset_split_results.csv"
        )

    def load_figure6d_data() -> pd.DataFrame:
        """Load all results for Figure 6D."""
        return pd.read_csv(DATA_DIR / "figure6" / "figure6d_all_results.csv")

    def load_figure7_data() -> pd.DataFrame:
        """Load data requirements results for Figure 7."""
        return pd.read_csv(
            DATA_DIR / "figure7" / "figure7_data_requirements_gapmind.csv"
        )

    return (
        load_figure1b_data,
        load_figure2_gapmind_data,
        load_figure2_baselines_data,
        load_figure3_data,
        load_figure3_gapmind_data,
        load_figure5a_data,
        load_figure5b_data,
        load_figure5c_data,
        load_figure6b_data,
        load_figure6c_data,
        load_figure6d_data,
        load_figure7_data,
    )


# Figure 1B: Genome Counts per Phenotype


@app.cell
def figure1b_widgets(mo, PHENOTYPES, DATASETS, DATASET_DISPLAY):
    """Widgets for Figure 1B."""
    fig1b_phenotypes = mo.ui.multiselect(
        options=PHENOTYPES,
        value=PHENOTYPES,
        label="Phenotypes",
    )
    fig1b_datasets = mo.ui.multiselect(
        options={DATASET_DISPLAY[d]: d for d in DATASETS},
        value=[DATASET_DISPLAY[d] for d in DATASETS],
        label="Datasets",
    )
    fig1b_stacked = mo.ui.switch(value=True, label="Stacked bars")

    return fig1b_phenotypes, fig1b_datasets, fig1b_stacked


@app.cell
def figure1b_chart(
    alt,
    pd,
    load_figure1b_data,
    fig1b_phenotypes,
    fig1b_datasets,
    fig1b_stacked,
    DATASET_COLORS,
    DATASET_DISPLAY,
    PHENOTYPES,
):
    """Create Figure 1B chart."""
    _df = load_figure1b_data()

    _selected_phenotypes = fig1b_phenotypes.value
    _selected_datasets = fig1b_datasets.value

    _df_filtered = _df[
        (_df["phenotype"].isin(_selected_phenotypes))
        & (_df["dataset"].isin(_selected_datasets))
    ].copy()

    _df_melted = _df_filtered.melt(
        id_vars=["phenotype", "dataset"],
        value_vars=["positive_count", "negative_count"],
        var_name="status",
        value_name="count",
    )
    _df_melted["status"] = _df_melted["status"].str.replace("_count", "").str.title()
    _df_melted["dataset_display"] = _df_melted["dataset"].map(DATASET_DISPLAY)

    _domain = [DATASET_DISPLAY[d] for d in _selected_datasets if d in DATASET_DISPLAY]
    _range_colors = [DATASET_COLORS[d] for d in _selected_datasets if d in DATASET_COLORS]

    if fig1b_stacked.value:
        chart_1b = (
            alt.Chart(_df_melted)
            .mark_bar()
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y("count:Q", title="Number of Genomes", stack="zero"),
                color=alt.Color(
                    "dataset_display:N",
                    title="Dataset",
                    scale=alt.Scale(domain=_domain, range=_range_colors),
                ),
                opacity=alt.condition(
                    alt.datum.status == "Positive",
                    alt.value(0.9),
                    alt.value(0.5),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("dataset_display:N", title="Dataset"),
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(width=700, height=400, title="Genome Counts by Phenotype")
        )
    else:
        chart_1b = (
            alt.Chart(_df_melted)
            .mark_bar()
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                xOffset="dataset_display:N",
                y=alt.Y("count:Q", title="Number of Genomes"),
                color=alt.Color(
                    "dataset_display:N",
                    title="Dataset",
                    scale=alt.Scale(domain=_domain, range=_range_colors),
                ),
                opacity=alt.condition(
                    alt.datum.status == "Positive",
                    alt.value(0.9),
                    alt.value(0.5),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("dataset_display:N", title="Dataset"),
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(width=700, height=400, title="Genome Counts by Phenotype")
        )

    return (chart_1b,)


@app.cell
def figure1b_tab(mo, fig1b_phenotypes, fig1b_datasets, fig1b_stacked, chart_1b):
    """Assemble Figure 1B tab content."""
    fig1b_controls = mo.hstack(
        [fig1b_phenotypes, fig1b_datasets, fig1b_stacked],
        justify="start",
        gap=2,
    )
    fig1b_content = mo.vstack([fig1b_controls, chart_1b])
    return (fig1b_content,)


# Figure 2: GapMind Comparison & Baseline Models


@app.cell
def figure2_widgets(mo, PHENOTYPES):
    """Widgets for Figure 2."""
    fig2_phenotypes = mo.ui.multiselect(
        options=PHENOTYPES,
        value=PHENOTYPES,
        label="Phenotypes",
    )
    fig2_panel = mo.ui.dropdown(
        options={"GapMind Comparison": "gapmind", "Baseline Models": "baselines"},
        value="GapMind Comparison",
        label="Panel",
    )
    fig2_split_type = mo.ui.dropdown(
        options={"Random Split": "random", "Out-of-Clade": "phylo"},
        value="Random Split",
        label="Split Type (Baselines)",
    )
    fig2_models = mo.ui.multiselect(
        options={
            "Identity": "identity",
            "Bernoulli": "bernoulli",
            "Nearest Neighbor": "nearest_neighbor",
        },
        value=["Identity", "Bernoulli", "Nearest Neighbor"],
        label="Models (Baselines)",
    )
    fig2_show_mean = mo.ui.switch(value=True, label="Show mean lines")

    return fig2_phenotypes, fig2_panel, fig2_split_type, fig2_models, fig2_show_mean


@app.cell
def figure2_chart(
    alt,
    pd,
    load_figure2_gapmind_data,
    load_figure2_baselines_data,
    fig2_phenotypes,
    fig2_panel,
    fig2_split_type,
    fig2_models,
    fig2_show_mean,
    PHENOTYPES,
):
    """Create Figure 2 charts."""
    _selected_phenotypes = fig2_phenotypes.value

    if fig2_panel.value == "gapmind":
        # GapMind comparison chart
        _strict_df, _loose_df = load_figure2_gapmind_data()

        _strict_df = _strict_df[_strict_df["phenotype"].isin(_selected_phenotypes)].copy()
        _loose_df = _loose_df[_loose_df["phenotype"].isin(_selected_phenotypes)].copy()

        _strict_df["confidence"] = "Strict"
        _loose_df["confidence"] = "Permissive"
        _combined = pd.concat([_strict_df, _loose_df], ignore_index=True)

        _bars = (
            alt.Chart(_combined)
            .mark_bar()
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                xOffset="confidence:N",
                y=alt.Y(
                    "balanced_accuracy:Q",
                    title="Balanced Accuracy",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "confidence:N",
                    title="Confidence",
                    scale=alt.Scale(
                        domain=["Strict", "Permissive"],
                        range=["#2E86AB", "#A23B72"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("confidence:N", title="Confidence"),
                    alt.Tooltip("balanced_accuracy:Q", title="Balanced Accuracy", format=".3f"),
                ],
            )
        )

        if fig2_show_mean.value:
            _mean_strict = _strict_df["balanced_accuracy"].mean()
            _mean_loose = _loose_df["balanced_accuracy"].mean()
            _mean_data = pd.DataFrame(
                {
                    "confidence": ["Strict", "Permissive"],
                    "mean": [_mean_strict, _mean_loose],
                }
            )
            _rules = (
                alt.Chart(_mean_data)
                .mark_rule(strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    y="mean:Q",
                    color=alt.Color(
                        "confidence:N",
                        scale=alt.Scale(
                            domain=["Strict", "Permissive"],
                            range=["#2E86AB", "#A23B72"],
                        ),
                    ),
                )
            )
            chart_2 = (_bars + _rules).properties(
                width=700, height=400, title="GapMind: Strict vs Permissive Confidence"
            )
        else:
            chart_2 = _bars.properties(
                width=700, height=400, title="GapMind: Strict vs Permissive Confidence"
            )

    else:
        # Baseline models chart
        _random_df, _phylo_df = load_figure2_baselines_data()

        if fig2_split_type.value == "random":
            _df = _random_df.copy()
            _title = "Baseline Models (Random Split)"
        else:
            _df = _phylo_df.copy()
            _title = "Baseline Models (Out-of-Clade Split)"

        _selected_models = fig2_models.value
        _df = _df[
            (_df["phenotype"].isin(_selected_phenotypes))
            & (_df["model"].isin(_selected_models))
        ]

        _model_colors = {
            "identity": "#A0A0A0",
            "bernoulli": "#707070",
            "nearest_neighbor": "#000000",
        }
        _model_shapes = {
            "identity": "circle",
            "bernoulli": "square",
            "nearest_neighbor": "triangle-up",
        }

        _points = (
            alt.Chart(_df)
            .mark_point(size=80, opacity=0.7)
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y(
                    "balanced_accuracy:Q",
                    title="Balanced Accuracy",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "model:N",
                    title="Model",
                    scale=alt.Scale(
                        domain=list(_model_colors.keys()),
                        range=list(_model_colors.values()),
                    ),
                ),
                shape=alt.Shape(
                    "model:N",
                    scale=alt.Scale(
                        domain=list(_model_shapes.keys()),
                        range=list(_model_shapes.values()),
                    ),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("balanced_accuracy:Q", title="Balanced Accuracy", format=".3f"),
                ],
            )
        )

        if fig2_show_mean.value:
            _mean_lines = (
                alt.Chart(_df)
                .mark_line(strokeWidth=2, opacity=0.7)
                .encode(
                    x=alt.X("phenotype:N"),
                    y=alt.Y("mean(balanced_accuracy):Q"),
                    color=alt.Color("model:N"),
                )
            )
            chart_2 = (_points + _mean_lines).properties(width=700, height=400, title=_title)
        else:
            chart_2 = _points.properties(width=700, height=400, title=_title)

    return (chart_2,)


@app.cell
def figure2_tab(
    mo,
    fig2_phenotypes,
    fig2_panel,
    fig2_split_type,
    fig2_models,
    fig2_show_mean,
    chart_2,
):
    """Assemble Figure 2 tab content."""
    fig2_controls = mo.hstack(
        [fig2_phenotypes, fig2_panel, fig2_split_type, fig2_models, fig2_show_mean],
        justify="start",
        gap=2,
    )
    fig2_content = mo.vstack([fig2_controls, chart_2])
    return (fig2_content,)


# Figure 3: ML Performance Across Split Types


@app.cell
def figure3_widgets(mo, PHENOTYPES, METRICS, DATASETS, DATASET_DISPLAY):
    """Widgets for Figure 3."""
    fig3_phenotypes = mo.ui.multiselect(
        options=PHENOTYPES,
        value=PHENOTYPES,
        label="Phenotypes",
    )
    fig3_split_type = mo.ui.dropdown(
        options={"Random Split": "random_split", "Dataset Split": "dataset_split"},
        value="Random Split",
        label="Split Type",
    )
    fig3_metric = mo.ui.dropdown(
        options={m.replace("_", " ").title(): m for m in METRICS},
        value="Balanced Accuracy",
        label="Metric",
    )
    fig3_show_gapmind = mo.ui.switch(value=True, label="Show GapMind reference")
    fig3_datasets = mo.ui.multiselect(
        options={DATASET_DISPLAY[d]: d for d in DATASETS if d != "pmi"},
        value=[DATASET_DISPLAY[d] for d in DATASETS if d != "pmi"],
        label="Test Datasets (Dataset Split)",
    )

    return fig3_phenotypes, fig3_split_type, fig3_metric, fig3_show_gapmind, fig3_datasets


@app.cell
def figure3_chart(
    alt,
    pd,
    load_figure3_data,
    load_figure3_gapmind_data,
    fig3_phenotypes,
    fig3_split_type,
    fig3_metric,
    fig3_show_gapmind,
    fig3_datasets,
    PHENOTYPES,
    DATASET_COLORS,
    DATASET_DISPLAY,
):
    """Create Figure 3 chart."""
    _df = load_figure3_data()
    _random_gapmind, _dataset_gapmind = load_figure3_gapmind_data()

    _selected_phenotypes = fig3_phenotypes.value
    _metric = fig3_metric.value
    _split_type = fig3_split_type.value

    _df_filtered = _df[
        (_df["split_type"] == _split_type)
        & (_df["phenotype"].isin(_selected_phenotypes))
    ].copy()

    if _split_type == "random_split":
        # Box plot for random split
        chart_3 = (
            alt.Chart(_df_filtered)
            .mark_boxplot(color="#06A77D", opacity=0.7)
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y(
                    f"{_metric}:Q",
                    title=_metric.replace("_", " ").title(),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                ],
            )
            .properties(width=700, height=400, title="ML Performance (Random Split)")
        )

        if fig3_show_gapmind.value and _metric == "balanced_accuracy":
            _gapmind_data = _random_gapmind[
                _random_gapmind["phenotype"].isin(_selected_phenotypes)
            ]
            _gapmind_ref = (
                alt.Chart(_gapmind_data)
                .mark_tick(color="#A23B72", thickness=3, size=30)
                .encode(
                    x=alt.X("phenotype:N"),
                    y=alt.Y("balanced_accuracy:Q"),
                )
            )
            chart_3 = chart_3 + _gapmind_ref

    else:
        # Extract test dataset from key column
        _df_filtered["test_dataset"] = _df_filtered["key"].apply(
            lambda x: x.split("test(")[1].split(")")[0] if "test(" in str(x) else "unknown"
        )

        _selected_test_datasets = fig3_datasets.value
        _df_filtered = _df_filtered[_df_filtered["test_dataset"].isin(_selected_test_datasets)]
        _df_filtered["test_dataset_display"] = _df_filtered["test_dataset"].map(DATASET_DISPLAY)

        _domain = [DATASET_DISPLAY[d] for d in _selected_test_datasets if d in DATASET_DISPLAY]
        _range_colors = [DATASET_COLORS[d] for d in _selected_test_datasets if d in DATASET_COLORS]

        chart_3 = (
            alt.Chart(_df_filtered)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y(
                    f"{_metric}:Q",
                    title=_metric.replace("_", " ").title(),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "test_dataset_display:N",
                    title="Test Dataset",
                    scale=alt.Scale(domain=_domain, range=_range_colors),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("test_dataset_display:N", title="Test Dataset"),
                    alt.Tooltip(f"{_metric}:Q", title=_metric.replace("_", " ").title(), format=".3f"),
                ],
            )
            .properties(width=700, height=400, title="ML Performance (Dataset Split)")
        )

        if fig3_show_gapmind.value and _metric == "balanced_accuracy":
            _gapmind_data = _dataset_gapmind[
                _dataset_gapmind["phenotype"].isin(_selected_phenotypes)
            ]
            _gapmind_ref = (
                alt.Chart(_gapmind_data)
                .mark_tick(color="#A23B72", thickness=3, size=30)
                .encode(
                    x=alt.X("phenotype:N"),
                    y=alt.Y("balanced_accuracy:Q"),
                )
            )
            chart_3 = chart_3 + _gapmind_ref

    return (chart_3,)


@app.cell
def figure3_tab(
    mo,
    fig3_phenotypes,
    fig3_split_type,
    fig3_metric,
    fig3_show_gapmind,
    fig3_datasets,
    chart_3,
):
    """Assemble Figure 3 tab content."""
    fig3_controls = mo.hstack(
        [fig3_phenotypes, fig3_split_type, fig3_metric, fig3_show_gapmind, fig3_datasets],
        justify="start",
        gap=2,
    )
    fig3_content = mo.vstack([fig3_controls, chart_3])
    return (fig3_content,)


# Figure 5: Concordant Samples Analysis


@app.cell
def figure5_widgets(mo, PHENOTYPES, METRICS, DATASETS, DATASET_DISPLAY):
    """Widgets for Figure 5."""
    fig5_phenotypes = mo.ui.multiselect(
        options=PHENOTYPES,
        value=PHENOTYPES,
        label="Phenotypes",
    )
    fig5_panel = mo.ui.dropdown(
        options={
            "Dataset Split Performance": "5a",
            "Feature Comparison": "5b",
            "Concordant Train on Discordant": "5c",
        },
        value="Dataset Split Performance",
        label="Panel",
    )
    fig5_metric = mo.ui.dropdown(
        options={m.replace("_", " ").title(): m for m in METRICS},
        value="Balanced Accuracy",
        label="Metric",
    )
    fig5_datasets = mo.ui.multiselect(
        options={DATASET_DISPLAY[d]: d for d in DATASETS if d != "pmi"},
        value=[DATASET_DISPLAY[d] for d in DATASETS if d != "pmi"],
        label="Datasets",
    )

    return fig5_phenotypes, fig5_panel, fig5_metric, fig5_datasets


@app.cell
def figure5_chart(
    alt,
    pd,
    load_figure5a_data,
    load_figure5b_data,
    load_figure5c_data,
    fig5_phenotypes,
    fig5_panel,
    fig5_metric,
    fig5_datasets,
    PHENOTYPES,
    DATASET_COLORS,
    DATASET_DISPLAY,
):
    """Create Figure 5 charts."""
    _selected_phenotypes = fig5_phenotypes.value
    _metric = fig5_metric.value
    _panel = fig5_panel.value

    if _panel == "5a":
        # Panel A: Dataset split performance on concordant samples
        _df = load_figure5a_data()
        _df_filtered = _df[
            (_df["split_type"] == "random_split")
            & (_df["phenotype"].isin(_selected_phenotypes))
        ]

        chart_5 = (
            alt.Chart(_df_filtered)
            .mark_boxplot(color="#2E86AB", opacity=0.7)
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y(
                    f"{_metric}:Q",
                    title=_metric.replace("_", " ").title(),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                ],
            )
            .properties(
                width=700,
                height=400,
                title="ML Performance on Concordant Samples (Random Split)",
            )
        )

    elif _panel == "5b":
        # Panel B: Feature comparison
        _df = load_figure5b_data()
        _selected_datasets = fig5_datasets.value

        _df_filtered = _df[
            (_df["phenotype"].isin(_selected_phenotypes))
            & (_df["test_dataset"].isin(_selected_datasets))
        ].copy()
        _df_filtered["test_dataset_display"] = _df_filtered["test_dataset"].map(
            DATASET_DISPLAY
        )

        # Melt for stacking
        _df_melted = _df_filtered.melt(
            id_vars=["phenotype", "test_dataset", "test_dataset_display"],
            value_vars=["n_intersection", "n_unique_to_individual"],
            var_name="feature_type",
            value_name="count",
        )
        _df_melted["feature_type"] = _df_melted["feature_type"].map(
            {
                "n_intersection": "Common Features",
                "n_unique_to_individual": "Unique Features",
            }
        )

        _domain = [DATASET_DISPLAY[d] for d in _selected_datasets if d in DATASET_DISPLAY]
        _range_colors = [DATASET_COLORS[d] for d in _selected_datasets if d in DATASET_COLORS]

        chart_5 = (
            alt.Chart(_df_melted)
            .mark_bar()
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                xOffset="test_dataset_display:N",
                y=alt.Y("count:Q", title="Number of Stable Features", stack="zero"),
                color=alt.Color(
                    "test_dataset_display:N",
                    title="Test Dataset",
                    scale=alt.Scale(domain=_domain, range=_range_colors),
                ),
                opacity=alt.condition(
                    alt.datum.feature_type == "Common Features",
                    alt.value(0.9),
                    alt.value(0.5),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("test_dataset_display:N", title="Dataset"),
                    alt.Tooltip("feature_type:N", title="Feature Type"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(width=700, height=400, title="Feature Comparison (Concordant Samples)")
        )

    else:
        # Panel C: Concordant train on discordant test
        _df = load_figure5c_data()
        _df_filtered = _df[_df["phenotype"].isin(_selected_phenotypes)].copy()

        if "test_type" in _df_filtered.columns:
            _df_filtered = _df_filtered[_df_filtered["test_type"] == "discordant"]

        chart_5 = (
            alt.Chart(_df_filtered)
            .mark_boxplot()
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                xOffset="split_type:N",
                y=alt.Y(
                    f"{_metric}:Q",
                    title=_metric.replace("_", " ").title(),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "split_type:N",
                    title="Split Type",
                    scale=alt.Scale(
                        domain=["random_split", "dataset_split"],
                        range=["#06A77D", "#2E86AB"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("split_type:N", title="Split Type"),
                ],
            )
            .properties(
                width=700,
                height=400,
                title="Concordant Training -> Discordant Testing",
            )
        )

    return (chart_5,)


@app.cell
def figure5_tab(mo, fig5_phenotypes, fig5_panel, fig5_metric, fig5_datasets, chart_5):
    """Assemble Figure 5 tab content."""
    fig5_controls = mo.hstack(
        [fig5_phenotypes, fig5_panel, fig5_metric, fig5_datasets],
        justify="start",
        gap=2,
    )
    fig5_content = mo.vstack([fig5_controls, chart_5])
    return (fig5_content,)


# Figure 6: Confident Samples & Precision-Recall


@app.cell
def figure6_widgets(mo, PHENOTYPES, METRICS):
    """Widgets for Figure 6."""
    fig6_phenotypes = mo.ui.multiselect(
        options=PHENOTYPES,
        value=PHENOTYPES,
        label="Phenotypes",
    )
    fig6_panel = mo.ui.dropdown(
        options={
            "Confident Samples Performance": "6b",
            "Precision-Recall Scatter": "6c",
            "Combined vs Filtered Features": "6d",
        },
        value="Confident Samples Performance",
        label="Panel",
    )
    fig6_split_type = mo.ui.dropdown(
        options={"Random Split": "random_split", "Dataset Split": "dataset_split"},
        value="Random Split",
        label="Split Type",
    )
    fig6_metric = mo.ui.dropdown(
        options={m.replace("_", " ").title(): m for m in METRICS},
        value="Balanced Accuracy",
        label="Metric",
    )

    return fig6_phenotypes, fig6_panel, fig6_split_type, fig6_metric


@app.cell
def figure6_chart(
    alt,
    pd,
    load_figure6b_data,
    load_figure6c_data,
    load_figure6d_data,
    fig6_phenotypes,
    fig6_panel,
    fig6_split_type,
    fig6_metric,
    PHENOTYPES,
):
    """Create Figure 6 charts."""
    _selected_phenotypes = fig6_phenotypes.value
    _panel = fig6_panel.value
    _split_type = fig6_split_type.value
    _metric = fig6_metric.value

    if _panel == "6b":
        # Panel B: Confident samples performance
        _df = load_figure6b_data()
        _df_filtered = _df[
            (_df["split_type"] == _split_type)
            & (_df["phenotype"].isin(_selected_phenotypes))
        ]

        chart_6 = (
            alt.Chart(_df_filtered)
            .mark_boxplot(color="#2E86AB", opacity=0.7)
            .encode(
                x=alt.X(
                    "phenotype:N",
                    title="Phenotype",
                    sort=[p for p in PHENOTYPES if p in _selected_phenotypes],
                ),
                y=alt.Y(
                    f"{_metric}:Q",
                    title=_metric.replace("_", " ").title(),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                ],
            )
            .properties(
                width=700, height=400, title=f"Performance on Confident Samples ({_split_type})"
            )
        )

    elif _panel == "6c":
        # Panel C: Precision-Recall scatter
        _df = load_figure6c_data()
        _df_filtered = _df[
            (_df["split_type"] == _split_type)
            & (_df["phenotype"].isin(_selected_phenotypes))
        ]

        _df_agg = (
            _df_filtered.groupby("phenotype")
            .agg({"precision": "mean", "recall": "mean"})
            .reset_index()
        )

        _scatter = (
            alt.Chart(_df_agg)
            .mark_circle(size=150, opacity=0.7)
            .encode(
                x=alt.X("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("phenotype:N", title="Phenotype"),
                tooltip=[
                    alt.Tooltip("phenotype:N", title="Phenotype"),
                    alt.Tooltip("precision:Q", title="Precision", format=".3f"),
                    alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                ],
            )
        )

        _line_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        _diagonal = (
            alt.Chart(_line_data)
            .mark_line(strokeDash=[5, 5], color="gray", strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        chart_6 = (_scatter + _diagonal).properties(
            width=500, height=500, title=f"Precision vs Recall ({_split_type})"
        )

    else:
        # Panel D: Combined vs filtered features
        _df = load_figure6d_data()
        _df_filtered = _df[_df["phenotype"].isin(_selected_phenotypes)]

        if "feature_set" in _df_filtered.columns:
            _scatter = (
                alt.Chart(_df_filtered)
                .mark_circle(size=100, opacity=0.7)
                .encode(
                    x=alt.X("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("feature_set:N", title="Feature Set"),
                    shape=alt.Shape("split_type:N", title="Split Type"),
                    tooltip=[
                        alt.Tooltip("phenotype:N", title="Phenotype"),
                        alt.Tooltip("feature_set:N", title="Feature Set"),
                        alt.Tooltip("precision:Q", format=".3f"),
                        alt.Tooltip("recall:Q", format=".3f"),
                    ],
                )
            )
        else:
            _scatter = (
                alt.Chart(_df_filtered)
                .mark_circle(size=100, opacity=0.7)
                .encode(
                    x=alt.X("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("phenotype:N", title="Phenotype"),
                    tooltip=[
                        alt.Tooltip("phenotype:N", title="Phenotype"),
                        alt.Tooltip("precision:Q", format=".3f"),
                        alt.Tooltip("recall:Q", format=".3f"),
                    ],
                )
            )

        _line_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        _diagonal = (
            alt.Chart(_line_data)
            .mark_line(strokeDash=[5, 5], color="gray", strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        chart_6 = (_scatter + _diagonal).properties(
            width=500, height=500, title="Combined vs Phenotype-Filtered Features"
        )

    return (chart_6,)


@app.cell
def figure6_tab(mo, fig6_phenotypes, fig6_panel, fig6_split_type, fig6_metric, chart_6):
    """Assemble Figure 6 tab content."""
    fig6_controls = mo.hstack(
        [fig6_phenotypes, fig6_panel, fig6_split_type, fig6_metric],
        justify="start",
        gap=2,
    )
    fig6_content = mo.vstack([fig6_controls, chart_6])
    return (fig6_content,)


# Figure 7: Data Requirements (Sample Size Effects)


@app.cell
def figure7_widgets(mo, load_figure7_data):
    """Widgets for Figure 7."""
    # Get phenotypes available in Figure 7 data (limited dataset)
    _df7 = load_figure7_data()
    _fig7_phenotypes = sorted(_df7["phenotype"].unique().tolist())

    # Limit to 4 phenotypes for faceting
    fig7_phenotypes = mo.ui.multiselect(
        options=_fig7_phenotypes,
        value=_fig7_phenotypes[:4] if len(_fig7_phenotypes) >= 4 else _fig7_phenotypes,
        label="Phenotypes (max 4 recommended)",
    )
    fig7_split_types = mo.ui.multiselect(
        options={
            "Random Split": "random_split",
            "Dataset Split": "dataset_split",
            "Out-of-Clade": "phylo_ooc",
        },
        value=["Random Split", "Dataset Split"],
        label="Split Types",
    )
    fig7_training_types = mo.ui.multiselect(
        options={"Full": "full", "Concordant": "concordant"},
        value=["Full", "Concordant"],
        label="Training Types",
    )
    fig7_test_subset = mo.ui.dropdown(
        options={
            "Full Test": "full",
            "Concordant Test": "concordant",
            "Discordant Test": "discordant",
        },
        value="Full Test",
        label="Test Subset",
    )

    return fig7_phenotypes, fig7_split_types, fig7_training_types, fig7_test_subset


@app.cell
def figure7_chart(
    alt,
    load_figure7_data,
    fig7_phenotypes,
    fig7_split_types,
    fig7_training_types,
    fig7_test_subset,
):
    """Create Figure 7 chart."""
    _df = load_figure7_data()

    _selected_phenotypes = fig7_phenotypes.value[:4]  # Limit to 4
    _selected_split_types = fig7_split_types.value
    _selected_training_types = fig7_training_types.value
    _test_subset = fig7_test_subset.value

    _split_display = {
        "random_split": "Random Split",
        "dataset_split": "Dataset Split",
        "phylo_ooc": "Out-of-Clade",
    }
    _training_display = {"full": "Full", "concordant": "Concordant"}

    _df_filtered = _df[
        (_df["phenotype"].isin(_selected_phenotypes))
        & (_df["split_type"].isin(_selected_split_types))
        & (_df["training_type"].isin(_selected_training_types))
        & (_df["test_subset"] == _test_subset)
    ].copy()

    _df_filtered["split_display"] = _df_filtered["split_type"].map(_split_display)
    _df_filtered["training_display"] = _df_filtered["training_type"].map(_training_display)

    _line = (
        alt.Chart(_df_filtered)
        .mark_line(opacity=0.4)
        .encode(
            x=alt.X("n_train_samples:Q", title="Training Samples"),
            y=alt.Y(
                "balanced_accuracy:Q",
                title="Balanced Accuracy",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "training_display:N",
                title="Training Type",
                scale=alt.Scale(
                    domain=["Full", "Concordant"], range=["#1f77b4", "#ff7f0e"]
                ),
            ),
            detail="key:N",
        )
    )

    _points = (
        alt.Chart(_df_filtered)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("n_train_samples:Q", title="Training Samples"),
            y=alt.Y("balanced_accuracy:Q", title="Balanced Accuracy"),
            color=alt.Color("training_display:N", title="Training Type"),
            tooltip=[
                alt.Tooltip("phenotype:N", title="Phenotype"),
                alt.Tooltip("split_display:N", title="Split Type"),
                alt.Tooltip("training_display:N", title="Training Type"),
                alt.Tooltip("n_train_samples:Q", title="Training Samples"),
                alt.Tooltip("balanced_accuracy:Q", title="Balanced Accuracy", format=".3f"),
            ],
        )
    )

    chart_7 = (
        (_line + _points)
        .properties(width=180, height=150)
        .facet(column=alt.Column("phenotype:N", title="Phenotype"), row=alt.Row("split_display:N", title="Split Type"))
        .resolve_scale(x="independent")
    )

    return (chart_7,)


@app.cell
def figure7_tab(
    mo,
    fig7_phenotypes,
    fig7_split_types,
    fig7_training_types,
    fig7_test_subset,
    chart_7,
):
    """Assemble Figure 7 tab content."""
    fig7_controls = mo.hstack(
        [fig7_phenotypes, fig7_split_types, fig7_training_types, fig7_test_subset],
        justify="start",
        gap=2,
    )
    fig7_content = mo.vstack([fig7_controls, chart_7])
    return (fig7_content,)


# Main App: Tab Assembly


@app.cell
def main_app(
    mo,
    fig1b_content,
    fig2_content,
    fig3_content,
    fig5_content,
    fig6_content,
    fig7_content,
):
    """Assemble the main application with tabs."""
    tabs = mo.ui.tabs(
        {
            "Figure 1B: Genome Counts": fig1b_content,
            "Figure 2: GapMind & Baselines": fig2_content,
            "Figure 3: ML Performance": fig3_content,
            "Figure 5: Concordant Samples": fig5_content,
            "Figure 6: Confident Samples": fig6_content,
            "Figure 7: Data Requirements": fig7_content,
        }
    )

    app_layout = mo.vstack(
        [
            mo.md("# Trait Prediction Manuscript - Interactive Figure Explorer"),
            mo.md(
                "Explore the manuscript figures interactively. Use the controls in each tab to filter and customize the visualizations."
            ),
            tabs,
        ]
    )

    return (app_layout,)


@app.cell
def display(app_layout):
    """Display the application."""
    app_layout


if __name__ == "__main__":
    app.run()
