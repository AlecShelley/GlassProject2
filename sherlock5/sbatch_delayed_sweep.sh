#!/bin/bash
#SBATCH --job-name=fire_delay
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/delay_%A_%a.out
#SBATCH --error=logs/delay_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu

# ============================================================
# Delayed-Reset (D,K) Sweep — SLURM Array Job
#
# Runs comparison_run_compute.py with 6 delayed-reset values
# IN PARALLEL. Each array task handles one D value.
# This maps the (D,K) energy landscape at phi = 0.84, 0.90.
#
# Array index → D mapping:
#   0 → D=2,  1 → D=3,  2 → D=5,  3 → D=7,  4 → D=8,  5 → D=10
#
# Submit: sbatch sbatch_delayed_sweep.sh
# ============================================================

set -euo pipefail

mkdir -p logs

# --- Environment setup ---
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-numba/0.59.0_py312

# --- Delayed-reset array ---
D_VALUES=(2 3 5 7 8 10)
D=${D_VALUES[$SLURM_ARRAY_TASK_ID]}

# --- Configuration ---
CPUS=${SLURM_CPUS_PER_TASK:-16}
N_ATOMS=1000000
WALL_TIME=600
TOL=1e-4
SEED=42
OUTPUT_DIR="${SCRATCH}/fire_results/sweep_delayed_${D}"

echo "=================================================="
echo "Delayed-Reset Sweep — D=${D} (array task ${SLURM_ARRAY_TASK_ID})"
echo "  Job ID:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node:      ${SLURM_NODELIST}"
echo "  CPUs:      ${CPUS}"
echo "  D:         ${D}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date)"
echo "=================================================="

python comparison_run_compute.py \
    --cpus ${CPUS} \
    --n-atoms ${N_ATOMS} \
    --wall-time ${WALL_TIME} \
    --tol ${TOL} \
    --seed ${SEED} \
    --delayed-reset ${D} \
    --output-dir ${OUTPUT_DIR}

echo "Completed at $(date)"
