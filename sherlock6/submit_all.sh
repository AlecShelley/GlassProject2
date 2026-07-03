#!/bin/bash
# Master submission script — run from the sweep directory
# Submits all async FIRE and LAMMPS jobs as SLURM arrays

set -e

# ── Async FIRE N=1,000,000 (180 jobs) ──
FIRE_N1000000=$(sbatch \
    --array=1-180 \
    --cpus-per-task=16 \
    --mem=32G \
    --time=24:00:00 \
    --export=PARAM_FILE=params_fire_N1000000.txt \
    submit_fire.sbatch | awk '{print $NF}')
echo "Submitted async FIRE N=1,000,000: $FIRE_N1000000"

# ── Async FIRE N=10,000,000 (180 jobs) ──
FIRE_N10000000=$(sbatch \
    --array=1-180 \
    --cpus-per-task=20 \
    --mem=64G \
    --time=48:00:00 \
    --export=PARAM_FILE=params_fire_N10000000.txt \
    submit_fire.sbatch | awk '{print $NF}')
echo "Submitted async FIRE N=10,000,000: $FIRE_N10000000"

# ── LAMMPS baselines (20 jobs) ──
LAMMPS=$(sbatch \
    --array=1-20 \
    submit_lammps.sbatch | awk '{print $NF}')
echo "Submitted LAMMPS baselines: $LAMMPS"

echo "All jobs submitted. Monitor with: squeue -u $USER"
