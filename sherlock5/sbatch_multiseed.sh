#!/bin/bash
#SBATCH --job-name=fire_seed
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu

# ============================================================
# Multi-Seed Robustness Test — SLURM Array Job
#
# Runs time_to_target.py with 5 different seeds IN PARALLEL.
# Each array task handles one seed. All 5 run simultaneously
# if resources are available.
#
# Array index → seed mapping:
#   0 → 42,  1 → 137,  2 → 256,  3 → 512,  4 → 1024
#
# Tests boundary packing fractions where Global FIRE is
# unreliable: phi = 0.77, 0.78, 0.88, 0.89
#
# Submit: sbatch sbatch_multiseed.sh
# Monitor: squeue -u $USER
# ============================================================

set -euo pipefail

mkdir -p logs

# --- Environment setup ---
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-numba/0.59.0_py312

# --- Seed array ---
SEEDS=(42 137 256 512 1024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# --- Configuration ---
CPUS=${SLURM_CPUS_PER_TASK:-16}
N_ATOMS=1000000
WALL_TIME=300
ENERGY_TARGET=1e-4
DELAYED_RESET=3
PHI="0.77 0.78 0.88 0.89"
OUTPUT_DIR="${SCRATCH}/fire_results/ttt_seed${SEED}"

echo "=================================================="
echo "Multi-Seed TTT — Seed ${SEED} (array task ${SLURM_ARRAY_TASK_ID})"
echo "  Job ID:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node:      ${SLURM_NODELIST}"
echo "  CPUs:      ${CPUS}"
echo "  Phi:       ${PHI}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date)"
echo "=================================================="

python time_to_target.py \
    --cpus ${CPUS} \
    --n-atoms ${N_ATOMS} \
    --wall-time ${WALL_TIME} \
    --energy-target ${ENERGY_TARGET} \
    --delayed-reset ${DELAYED_RESET} \
    --seed ${SEED} \
    --phi ${PHI} \
    --grids 7 \
    --output-dir ${OUTPUT_DIR}

echo "Completed at $(date)"
