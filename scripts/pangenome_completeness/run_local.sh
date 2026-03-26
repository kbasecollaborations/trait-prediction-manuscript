#!/bin/bash
# =============================================================================
# Pangenome Completeness - Local Run Script
# =============================================================================
# This script runs the pangenome completeness calculation on local data.
#
# Usage:
#   ./run_local.sh
#
# Prerequisites:
#   - pixi installed (https://prefix.dev/docs/pixi/overview)
#   - Run 'pixi install' first to create the environment
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration - paths relative to project root
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Input directories
ALL_SEQS_DIR="${PROJECT_ROOT}/data/raw/all_seqs"
CORE_GENES_DIR="${PROJECT_ROOT}/data/processed/unique_core_faas"
MAPPING_FILE="${PROJECT_ROOT}/data/processed/pangenome/assignments.ani.merged_mmseqs90.tsv"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/data/outputs/pangenome_completeness"
OUTPUT_FILE="${OUTPUT_DIR}/pangenome_completeness.tsv"

# Default number of parallel jobs (adjust based on your CPU)
JOBS="20"

# MMseqs2 thresholds (matching NERSC script)
MIN_IDENTITY=0.90
MIN_COVERAGE=0.80
EVALUE=1e-3

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Pangenome Completeness - Local Run"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "All seqs directory: ${ALL_SEQS_DIR}"
echo "Core genes directory: ${CORE_GENES_DIR}"
echo "Mapping file: ${MAPPING_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "Min identity: ${MIN_IDENTITY}"
echo "Min coverage: ${MIN_COVERAGE}"
echo "E-value: ${EVALUE}"
echo "Parallel jobs: ${JOBS}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------
if [[ ! -d "${ALL_SEQS_DIR}" ]]; then
    echo "ERROR: All seqs directory does not exist: ${ALL_SEQS_DIR}"
    exit 1
fi

if [[ ! -d "${CORE_GENES_DIR}" ]]; then
    echo "ERROR: Core genes directory does not exist: ${CORE_GENES_DIR}"
    exit 1
fi

if [[ ! -f "${MAPPING_FILE}" ]]; then
    echo "ERROR: Mapping file does not exist: ${MAPPING_FILE}"
    exit 1
fi

# Count input files
NUM_GENOME_FILES=$(find "${ALL_SEQS_DIR}" -maxdepth 1 -name "*.faa" -type f | wc -l)
NUM_CORE_GENE_FILES=$(find "${CORE_GENES_DIR}" -maxdepth 1 -name "*.faa" -type f | wc -l)
echo ""
echo "Found ${NUM_GENOME_FILES} genome .faa files"
echo "Found ${NUM_CORE_GENE_FILES} core gene .faa files"

if [[ "${NUM_GENOME_FILES}" -eq 0 ]]; then
    echo "ERROR: No .faa files found in ${ALL_SEQS_DIR}"
    exit 1
fi

if [[ "${NUM_CORE_GENE_FILES}" -eq 0 ]]; then
    echo "ERROR: No .faa files found in ${CORE_GENES_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Check for pixi environment or system mmseqs
# -----------------------------------------------------------------------------
if command -v pixi &>/dev/null && [[ -f "${SCRIPT_DIR}/pixi.toml" ]]; then
    echo ""
    echo "Using pixi environment..."
    cd "${SCRIPT_DIR}"

    # Check if environment exists, if not install
    if ! pixi run which mmseqs &>/dev/null 2>&1; then
        echo "Installing pixi environment..."
        pixi install
    fi

    # Run with pixi
    echo ""
    echo "Running pangenome completeness calculation..."
    pixi run python pangenome_completeness.py \
        --all-seqs "${ALL_SEQS_DIR}" \
        --core-genes "${CORE_GENES_DIR}" \
        --mapping "${MAPPING_FILE}" \
        --output "${OUTPUT_FILE}" \
        --min-identity "${MIN_IDENTITY}" \
        --min-coverage "${MIN_COVERAGE}" \
        --evalue "${EVALUE}" \
        --jobs "${JOBS}"
else
    # Check for system mmseqs
    if ! command -v mmseqs &>/dev/null; then
        echo "ERROR: mmseqs not found. Please either:"
        echo "  1. Install pixi and run 'pixi install' in ${SCRIPT_DIR}"
        echo "  2. Install mmseqs2 system-wide"
        exit 1
    fi

    echo ""
    echo "Using system mmseqs..."
    echo "Running pangenome completeness calculation..."
    python3 "${SCRIPT_DIR}/pangenome_completeness.py" \
        --all-seqs "${ALL_SEQS_DIR}" \
        --core-genes "${CORE_GENES_DIR}" \
        --mapping "${MAPPING_FILE}" \
        --output "${OUTPUT_FILE}" \
        --min-identity "${MIN_IDENTITY}" \
        --min-coverage "${MIN_COVERAGE}" \
        --evalue "${EVALUE}" \
        --jobs "${JOBS}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Pangenome Completeness Complete"
echo "=============================================="
echo "Output file: ${OUTPUT_FILE}"
echo ""
echo "Done!"
