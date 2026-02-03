#!/bin/bash
#SBATCH --job-name=pangenome_completeness
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=pangenome_completeness_%j.out
#SBATCH --error=pangenome_completeness_%j.err

# =============================================================================
# Pangenome Completeness SLURM Script for NERSC Perlmutter
# =============================================================================
# This script calculates genome completeness based on core gene presence from
# species pangenomes using MMseqs2 for sequence similarity search.
#
# Usage:
#   sbatch run_pangenome_completeness_nersc.sh
#
# Configuration:
#   Edit ALL_SEQS_DIR, CORE_GENES_DIR, MAPPING_FILE, OUTPUT_DIR, and
#   PIXI_PROJECT_DIR below before running.
#
# Notes:
#   - Perlmutter CPU nodes have 128 cores (2x AMD EPYC 7763)
#   - Uses --jobs for parallel genome processing
#   - Each genome is processed independently using MMseqs2
#   - For ~1000 genomes, expect ~1-2 hours runtime depending on pangenome sizes
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration - modify these as needed
# -----------------------------------------------------------------------------
# Directory containing genome protein .faa files
ALL_SEQS_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/gapmind_analysis/all_seqs"

# Directory containing core gene .faa files (named by gtdb_species_clade_id)
CORE_GENES_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/pangenome/core_genes"

# Path to genome-to-species mapping TSV file
MAPPING_FILE="/global/cfs/cdirs/kbase/ke_prototype/traits/pangenome/assignments.ani.merged.tsv"

# Output directory
OUTPUT_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/pangenome_completeness_output"

# Number of parallel jobs (matches allocated CPUs)
JOBS="${SLURM_CPUS_PER_TASK:-128}"

# MMseqs2 thresholds
MIN_IDENTITY=0.90
MIN_COVERAGE=0.80
EVALUE=1e-3

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Pangenome Completeness Job Started: $(date)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs allocated: ${JOBS}"
echo "All seqs directory: ${ALL_SEQS_DIR}"
echo "Core genes directory: ${CORE_GENES_DIR}"
echo "Mapping file: ${MAPPING_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Min identity: ${MIN_IDENTITY}"
echo "Min coverage: ${MIN_COVERAGE}"
echo "E-value: ${EVALUE}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Load environment using pixi
# -----------------------------------------------------------------------------
# Set path to pixi project directory (where pixi.toml lives)
# Modify this to point to your pixi project location
PIXI_PROJECT_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/pangenome_completeness"

if [[ ! -f "${PIXI_PROJECT_DIR}/pixi.toml" ]]; then
	echo "ERROR: pixi.toml not found in ${PIXI_PROJECT_DIR}"
	echo "Set PIXI_PROJECT_DIR to the directory containing your pixi.toml"
	exit 1
fi

echo "Using pixi project: ${PIXI_PROJECT_DIR}"

# Activate pixi environment by adding it to PATH
eval "$(pixi shell-hook --manifest-path "${PIXI_PROJECT_DIR}/pixi.toml")"

# -----------------------------------------------------------------------------
# Verify dependencies
# -----------------------------------------------------------------------------
if ! command -v mmseqs &>/dev/null; then
	echo "ERROR: mmseqs command not found. Check your pixi environment."
	exit 1
fi

echo "MMseqs2 version: $(mmseqs version 2>&1 || echo 'version check failed')"

if ! python -c "import pandas" &>/dev/null; then
	echo "ERROR: Python with pandas not found. Check your pixi environment."
	exit 1
fi

echo "Python version: $(python --version)"

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
# Run pangenome completeness calculation
# -----------------------------------------------------------------------------
echo ""
echo "Starting pangenome completeness calculation..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/pangenome_completeness.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
	echo "ERROR: Python script not found at ${PYTHON_SCRIPT}"
	exit 1
fi

# Use local scratch for temporary files if available (faster I/O)
if [[ -n "${PSCRATCH:-}" ]]; then
	export TMPDIR="${PSCRATCH}/pangenome_tmp_${SLURM_JOB_ID}"
	mkdir -p "${TMPDIR}"
	echo "Using PSCRATCH for temp files: ${TMPDIR}"
fi

OUTPUT_FILE="${OUTPUT_DIR}/pangenome_completeness.csv"

echo ""
echo "Command: python ${PYTHON_SCRIPT} \\"
echo "    --all-seqs ${ALL_SEQS_DIR} \\"
echo "    --core-genes ${CORE_GENES_DIR} \\"
echo "    --mapping ${MAPPING_FILE} \\"
echo "    --output ${OUTPUT_FILE} \\"
echo "    --min-identity ${MIN_IDENTITY} \\"
echo "    --min-coverage ${MIN_COVERAGE} \\"
echo "    --evalue ${EVALUE} \\"
echo "    --jobs ${JOBS}"
echo ""

python "${PYTHON_SCRIPT}" \
	--all-seqs "${ALL_SEQS_DIR}" \
	--core-genes "${CORE_GENES_DIR}" \
	--mapping "${MAPPING_FILE}" \
	--output "${OUTPUT_FILE}" \
	--min-identity "${MIN_IDENTITY}" \
	--min-coverage "${MIN_COVERAGE}" \
	--evalue "${EVALUE}" \
	--jobs "${JOBS}"

# -----------------------------------------------------------------------------
# Cleanup and summary
# -----------------------------------------------------------------------------
# Clean up temp directory if created
if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
	rm -rf "${TMPDIR}"
fi

echo ""
echo "=============================================="
echo "Pangenome Completeness Job Completed: $(date)"
echo "=============================================="
echo "Output files:"
echo "  - Completeness results: ${OUTPUT_FILE}"
echo ""
echo "Done!"
