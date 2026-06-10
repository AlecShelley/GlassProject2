#!/bin/bash
#
# FIRE Parameter Sweep — Sherlock SBATCH Script
# N_ATOMS = 10,000,000 | 5x5 density × grid sweep
#
# Submit with:  sbatch cell_listfix.sbatch
# Monitor with: squeue -u $USER
# Cancel with:  scancel <job_id>
#

#SBATCH --job-name=fire_sweep_10M
#SBATCH --partition=YOUR_PI_PARTITION   # <-- Replace with your PI's partition name
#SBATCH --time=08:00:00                 # 8 hours wall time
#SBATCH --cpus-per-task=16              # Numba parallel kernels (prange)
#SBATCH --mem=32G                       # 32 GB RAM for 10M atoms
#SBATCH --output=fire_sweep_%j.out      # stdout → fire_sweep_<jobid>.out
#SBATCH --error=fire_sweep_%j.err       # stderr → fire_sweep_<jobid>.err
#SBATCH --mail-type=END,FAIL            # Email on completion or failure
#SBATCH --mail-user=YOUR_EMAIL@stanford.edu  # <-- Replace with your email

# ---- Environment Setup ----
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-numba/0.59.0_py312
module load py-scipy/1.12.0_py312

# If using a conda environment instead, comment out the modules above and use:
# module load conda
# conda activate your_env_name

# Numba threading — match SLURM allocation
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Output directory on $SCRATCH for fast I/O and large storage
OUTPUT_DIR="${SCRATCH}/fire_sweep_10M/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
export OUTPUT_DIR

echo "=============================================="
echo "FIRE Parameter Sweep — Sherlock HPC"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "Memory:      ${SLURM_MEM_PER_NODE}MB"
echo "Partition:   ${SLURM_JOB_PARTITION}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Start time:  $(date)"
echo "=============================================="

# ---- Run Computation ----
python cell_listfix_compute.py

echo ""
echo "Compute finished at: $(date)"
echo "Results in: ${OUTPUT_DIR}"
echo ""

# ---- Generate Plots (headless) ----
python cell_listfix_plot.py "${OUTPUT_DIR}"

echo ""
echo "All done. Plots saved to: ${OUTPUT_DIR}"
echo "To copy results to your home directory:"
echo "  cp -r ${OUTPUT_DIR} ~/fire_sweep_results/"
