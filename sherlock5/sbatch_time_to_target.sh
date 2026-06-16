#!/bin/bash
#SBATCH --job-name=fire_ttt
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/ttt_%j.out
#SBATCH --error=logs/ttt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@stanford.edu

# ============================================================
# Time-to-Target Experiment — time_to_target.py
#
# Measures wall-clock time for Global and Async FIRE to reach
# an energy threshold. Logs energy every step for high-resolution
# energy-vs-time curves.
#
# Typical runtime: 1-3 hours with 16 CPUs, N=1M
#
# Common experiment configurations:
#
#   High-density TTT (above jamming):
#     --phi 0.85 0.86 0.87 0.88 0.89 0.90
#     --energy-target 1.0 --wall-time 300
#
#   Low-density boundary pinpointing:
#     --phi 0.77 0.78 0.79
#     --energy-target 1e-4 --wall-time 300 --grids 3 7
#
#   N-dependence (low density):
#     --phi 0.77 0.78 --n-atoms 500000
#     --energy-target 1e-4 --wall-time 300 --grids 7
#
#   N-dependence (high density):
#     --phi 0.88 0.89 --n-atoms 500000
#     --energy-target 1e-4 --wall-time 300 --grids 7
#
#   Best-config validation:
#     --phi 0.88 0.89 0.90 --grids 21
#     --delayed-reset 8 --energy-target 1e-4 --wall-time 300
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
ENERGY_TARGET=0.01
DELAYED_RESET=3
SEED=42
PHI="0.84 0.90"
OUTPUT_DIR="${SCRATCH}/fire_results/time_to_target"

echo "=================================================="
echo "Time-to-Target Experiment"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      ${SLURM_NODELIST}"
echo "  CPUs:      ${CPUS}"
echo "  N_ATOMS:   ${N_ATOMS}"
echo "  Target:    E < ${ENERGY_TARGET}"
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
    --output-dir ${OUTPUT_DIR}

echo "Completed at $(date)"
