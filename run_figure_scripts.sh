#!/usr/bin/env bash
#
# Render the figure PDFs and LaTeX tables listed below from the data files
# under data/outputs/. Run run_data_scripts.sh first if those are stale.
# Figures not listed here are built by running their plotting module directly.
#
# Run with: uv run bash run_figure_scripts.sh (from the repository root)

echo "========================================"
echo "Running: figure1b_plot"
echo "========================================"
python -m scripts.figure1.figure1b_plot

echo "========================================"
echo "Running: figure2_plot"
echo "========================================"
python -m scripts.figure2.figure2_plot

echo ""
echo "========================================"
echo "Running: figure3_plot"
echo "========================================"
python -m scripts.figure3.figure3_plot

echo ""
echo "========================================"
echo "Running: figure4_plot"
echo "========================================"
python -m scripts.figure4.figure4_plot

echo ""
echo "========================================"
echo "Running: figure5_plot"
echo "========================================"
# python -m scripts.figure5.figure5_plot

echo ""
echo "========================================"
echo "Running: histidine_feature_table"
echo "========================================"
python -m scripts.tables.histidine_feature_table

echo ""
echo "========================================"
echo "All figure scripts completed!"
echo "========================================"
