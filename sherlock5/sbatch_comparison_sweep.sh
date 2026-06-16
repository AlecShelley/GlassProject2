#!/bin/bash
#SBATCH --job-name=fire_sweep
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu

# ============================================================
# FIRE Parameter Sweep — comparison_run_compute.py
#
# Runs Global FIRE baseline + Async FIRE across 5 gamma targets
# for each packing fraction. Default: 2 phi × 6 runs = 12 runs,
# each up to 600s wall time.
#
# Typical runtime: 3-6 hours with 16 CPUs, N=1M
#
# Variations:
#   Full 5-density sweep:  --phi 0.80 0.82 0.84 0.86 0.90
#   Quick test:            --n-atoms 100000 --wall-time 60
#   Delayed reset sweep:   --delayed-reset 3 --output-dir sweep_delayed
#   High tolerance:        --tol 1e-8
# ============================================================

set -euo pipefail

mkdir -p logs

# --- Environment setup ---
# Option A: Sherlock modules
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-numba/0.59.0_py312

# Option B: Conda (uncomment if using conda instead)
# source activate fire_env

# --- Configuration ---
CPUS=${SLURM_CPUS_PER_TASK:-16}
N_ATOMS=1000000
WALL_TIME=600
TOL=1e-4
SEED=42
DELAYED_RESET=0
OUTPUT_DIR="${SCRATCH}/fire_results/parameter_sweep"

echo "=================================================="
echo "FIRE Parameter Sweep"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      ${SLURM_NODELIST}"
echo "  CPUs:      ${CPUS}"
echo "  N_ATOMS:   ${N_ATOMS}"
echo "  Wall time: ${WALL_TIME}s per run"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date)"
echo "=================================================="

python comparison_run_compute.py \
    --cpus ${CPUS} \
    --n-atoms ${N_ATOMS} \
    --wall-time ${WALL_TIME} \
    --tol ${TOL} \
    --seed ${SEED} \
    --delayed-reset ${DELAYED_RESET} \
    --output-dir ${OUTPUT_DIR}

echo "Completed at $(date)"
