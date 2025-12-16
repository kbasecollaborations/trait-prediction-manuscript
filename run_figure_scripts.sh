#!/usr/bin/env bash

# echo "========================================"
# echo "Running: scripts.figure1.figure1b_plot"
# echo "========================================"
# python -m scripts.figure1.figure1b_plot

echo "========================================"
echo "Running: scripts.figure1.figure2_plot"
echo "========================================"
python -m scripts.figure1.figure2_plot

echo ""
echo "========================================"
echo "Running: scripts.figure1.figure3_plot"
echo "========================================"
python -m scripts.figure1.figure3_plot

echo ""
echo "========================================"
echo "Running: scripts.figure1.figure4_plot"
echo "========================================"
python -m scripts.figure1.figure4_plot

echo ""
echo "========================================"
echo "Running: scripts.figure1.figure5_plot"
echo "========================================"
python -m scripts.figure1.figure5_plot

echo ""
echo "========================================"
echo "All figure scripts completed!"
echo "========================================"
