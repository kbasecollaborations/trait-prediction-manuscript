#!/usr/bin/env bash

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
echo "Running: table1"
echo "========================================"
python -m scripts.tables.table1

echo ""
echo "========================================"
echo "Running: table2"
echo "========================================"
python -m scripts.tables.table2

echo ""
echo "========================================"
echo "All figure scripts completed!"
echo "========================================"
