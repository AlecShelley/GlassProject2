#!/bin/bash
#SBATCH --job-name=fire_reset
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/reset_%j.out
#SBATCH --error=logs/reset_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu

# ============================================================
# Reset Frequency Analysis — reset_analysis.py
#
# Tracks per-step reset statistics for Global and Async FIRE.
# Quantifies the "slowest boat in the convoy" problem.
#
# Typical runtime: 1-2 hours with 16 CPUs, N=1M
#
# Default: 6 phi values × (Global + Async K=7) = 12 runs × 300s
# ============================================================

set -euo pipefail

mkdir -p logs

# --- Environment setup ---
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-numba/0.59.0_py312

# --- Configuration ---
CPUS=${SLURM_CPUS_PER_TASK:-16}
N_ATOMS=1000000
WALL_TIME=300
DELAYED_RESET=3
SEED=42
PHI="0.85 0.86 0.87 0.88 0.89 0.90"
OUTPUT_DIR="${SCRATCH}/fire_results/reset_analysis"

echo "=================================================="
echo "Reset Frequency Analysis"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      ${SLURM_NODELIST}"
echo "  CPUs:      ${CPUS}"
echo "  N_ATOMS:   ${N_ATOMS}"
echo "  Phi:       ${PHI}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date)"
echo "=================================================="

python reset_analysis.py \
    --cpus ${CPUS} \
    --n-atoms ${N_ATOMS} \
    --wall-time ${WALL_TIME} \
    --delayed-reset ${DELAYED_RESET} \
    --seed ${SEED} \
    --phi ${PHI} \
    --output-dir ${OUTPUT_DIR}

echo "Completed at $(date)"
