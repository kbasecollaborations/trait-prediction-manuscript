#!/bin/bash
#SBATCH --job-name=checkm2
#SBATCH --account=kbase
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=checkm2_%j.out
#SBATCH --error=checkm2_%j.err

# CheckM2 SLURM script for NERSC Perlmutter.
#
# Runs CheckM2 on a folder of .faa protein sequence files, then runs
# aggregate_checkm2_results.py on the resulting quality_report.tsv.
#
# Usage:
#   sbatch run_checkm2_nersc.sh
#
# Configuration:
#   Edit INPUT_DIR, OUTPUT_DIR, and PIXI_PROJECT_DIR below before running.
#
# Notes:
#   - Perlmutter CPU nodes have 128 cores (2x AMD EPYC 7763)
#   - CheckM2 parallelises internally via --threads
#   - Uses --genes because the input is protein sequences (.faa)
#   - For ~1000 genomes with pre-computed proteins, expect ~30-60 min runtime
#   - The 4-hour limit provides safety margin; can reduce to 2 hours if needed

set -euo pipefail

# Configuration - modify these as needed
INPUT_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/gapmind_analysis/all_seqs"
OUTPUT_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/gapmind_analysis/checkm2_output"

# Number of threads (matches allocated CPUs)
THREADS="${SLURM_CPUS_PER_TASK:-128}"

# File extension for protein sequences
EXTENSION="faa"

# Environment setup
echo "=============================================="
echo "CheckM2 Job Started: $(date)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs allocated: ${THREADS}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

# Pixi project directory (where pixi.toml lives); modify as needed
PIXI_PROJECT_DIR="/global/cfs/cdirs/kbase/ke_prototype/traits/gapmind_analysis/checkm2"

if [[ ! -f "${PIXI_PROJECT_DIR}/pixi.toml" ]]; then
	echo "ERROR: pixi.toml not found in ${PIXI_PROJECT_DIR}"
	echo "Set PIXI_PROJECT_DIR to the directory containing your pixi.toml"
	exit 1
fi

echo "Using pixi project: ${PIXI_PROJECT_DIR}"

# Activate the pixi environment by adding it to PATH
eval "$(pixi shell-hook --manifest-path "${PIXI_PROJECT_DIR}/pixi.toml")"

# Verify CheckM2 is available
if ! command -v checkm2 &>/dev/null; then
	echo "ERROR: checkm2 command not found. Check your pixi environment."
	exit 1
fi

echo "CheckM2 version: $(checkm2 --version 2>&1 || echo 'version check failed')"

# Verify Python with pandas is available (for aggregation script)
if ! python -c "import pandas" &>/dev/null; then
	echo "WARNING: Python with pandas not found. Aggregation script will use awk fallback."
fi

# Validate inputs
if [[ ! -d "${INPUT_DIR}" ]]; then
	echo "ERROR: Input directory does not exist: ${INPUT_DIR}"
	exit 1
fi

# Count input files
NUM_FILES=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.${EXTENSION}" -type f | wc -l)
echo "Found ${NUM_FILES} .${EXTENSION} files in input directory"

if [[ "${NUM_FILES}" -eq 0 ]]; then
	echo "ERROR: No .${EXTENSION} files found in ${INPUT_DIR}"
	exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run CheckM2
echo ""
echo "Starting CheckM2 predict..."
echo "Command: checkm2 predict --threads ${THREADS} --genes --input ${INPUT_DIR} --output-directory ${OUTPUT_DIR} -x ${EXTENSION}"
echo ""

# Use local scratch for temporary files if available (faster I/O)
if [[ -n "${PSCRATCH:-}" ]]; then
	TMPDIR="${PSCRATCH}/checkm2_tmp_${SLURM_JOB_ID}"
	mkdir -p "${TMPDIR}"
	export TMPDIR
	echo "Using PSCRATCH for temp files: ${TMPDIR}"
fi

# --genes: input files are protein sequences rather than assemblies
checkm2 predict \
	--threads "${THREADS}" \
	--genes \
	--input "${INPUT_DIR}" \
	--output-directory "${OUTPUT_DIR}" \
	-x "${EXTENSION}"

# Clean up temp directory if created
if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
	rm -rf "${TMPDIR}"
fi

echo ""
echo "=============================================="
echo "CheckM2 Job Completed: $(date)"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Quality report: ${OUTPUT_DIR}/quality_report.tsv"

# Run the Python aggregation script that sits alongside this one
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_checkm2_results.py"

if [[ -f "${OUTPUT_DIR}/quality_report.tsv" ]]; then
	echo ""
	echo "Running Python aggregation script..."

	if [[ -f "${AGGREGATE_SCRIPT}" ]]; then
		# Run aggregation with output files in the CheckM2 output directory
		python "${AGGREGATE_SCRIPT}" \
			"${OUTPUT_DIR}" \
			--output "${OUTPUT_DIR}/checkm2_aggregated.tsv" \
			--failed-list "${OUTPUT_DIR}/failed_genomes.txt" \
			--passed-list "${OUTPUT_DIR}/passed_genomes.txt"
	else
		echo "WARNING: Aggregation script not found at ${AGGREGATE_SCRIPT}"
		echo "Falling back to basic summary..."

		# Basic summary using awk
		echo ""
		echo "Results summary:"
		echo "----------------"
		TOTAL=$(tail -n +2 "${OUTPUT_DIR}/quality_report.tsv" | wc -l)
		echo "Total genomes processed: ${TOTAL}"

		awk -F'\t' 'NR>1 {
            if ($2 >= 90 && $3 < 5) high++;
            else if ($2 >= 50 && $3 < 10) med++;
            else low++;
        }
        END {
            print "High quality (>=90% complete, <5% contamination): " high+0;
            print "Medium quality (>=50% complete, <10% contamination): " med+0;
            print "Low quality (other): " low+0;
        }' "${OUTPUT_DIR}/quality_report.tsv"

		# Generate simple genome lists using awk
		echo ""
		echo "Generating genome lists..."
		awk -F'\t' 'NR>1 && ($2 < 50 || $3 > 10) {print $1}' "${OUTPUT_DIR}/quality_report.tsv" >"${OUTPUT_DIR}/failed_genomes.txt"
		awk -F'\t' 'NR>1 && ($2 >= 50 && $3 <= 10) {print $2}' "${OUTPUT_DIR}/quality_report.tsv" >"${OUTPUT_DIR}/passed_genomes.txt"
		echo "Failed genomes: ${OUTPUT_DIR}/failed_genomes.txt"
		echo "Passed genomes: ${OUTPUT_DIR}/passed_genomes.txt"
	fi
else
	echo "ERROR: quality_report.tsv not found in ${OUTPUT_DIR}"
	exit 1
fi

echo ""
echo "=============================================="
echo "All processing complete: $(date)"
echo "=============================================="
echo "Output files:"
echo "  - Quality report: ${OUTPUT_DIR}/quality_report.tsv"
echo "  - Aggregated results: ${OUTPUT_DIR}/checkm2_aggregated.tsv"
echo "  - Failed genomes: ${OUTPUT_DIR}/failed_genomes.txt"
echo "  - Passed genomes: ${OUTPUT_DIR}/passed_genomes.txt"
echo ""
echo "Done!"
