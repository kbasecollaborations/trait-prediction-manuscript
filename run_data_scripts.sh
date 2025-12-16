#!/usr/bin/env bash

echo "========================================"
echo "Running: scripts.figure1.figure1b_data"
echo "========================================"
python -m scripts.figure1.figure1b_data

echo ""
echo "========================================"
echo "Running: scripts.figure2.figure2a_data"
echo "========================================"
python -m scripts.figure2.figure2a_data

echo ""
echo "========================================"
echo "Running: scripts.figure2.figure2b_data"
echo "========================================"
python -m scripts.figure2.figure2b_data

echo ""
echo "========================================"
echo "Running: scripts.figure3.figure3a_gapmind_random_split"
echo "========================================"
python -m scripts.figure3.figure3a_gapmind_random_split

echo ""
echo "========================================"
echo "Running: scripts.figure3.figure3a_gapmind_dataset_split"
echo "========================================"
python -m scripts.figure3.figure3b_gapmind_dataset_split

echo ""
echo "========================================"
echo "Running: scripts.figure3.figure3ab_data"
echo "========================================"
python -m scripts.figure3.figure3ab_data

echo ""
echo "========================================"
echo "Running: scripts.figure3.figure3c_data"
echo "========================================"
python -m scripts.figure3.figure3c_data

echo ""
echo "========================================"
echo "Running: scripts.figure4.figure4c_data"
echo "========================================"
python -m scripts.figure4.figure4c_data

echo ""
echo "========================================"
echo "Running: scripts.figure5.figure5a_data"
echo "========================================"
python -m scripts.figure5.figure5a_data

echo ""
echo "========================================"
echo "Running: scripts.figure5.figure5b_data"
echo "========================================"
python -m scripts.figure5.figure5b_data

echo ""
echo "========================================"
echo "Running: scripts.figure5.figure5cd_data"
echo "========================================"
python -m scripts.figure5.figure5cd_data

echo ""
echo "========================================"
echo "All data scripts completed!"
echo "========================================"
