#!/bin/bash
# One-time setup for the async FIRE sweep on Sherlock.
#
# Usage:
#   1. Copy this directory to Sherlock:
#      scp -r sherlock/ <sunetid>@login.sherlock.stanford.edu:$SCRATCH/kd_sweep/
#
#   2. SSH to Sherlock and run:
#      cd $SCRATCH/kd_sweep
#      bash setup.sh
#
#   3. Submit all jobs:
#      bash submit_all.sh
#
#   4. When jobs finish, collect results:
#      python collect_results.py

set -e

echo "=== Async FIRE K-D Sweep — Sherlock Setup ==="
echo ""

# ── 1. Create conda environment ─────────────────────────────────────
if conda env list 2>/dev/null | grep -q fire_env; then
    echo "Conda env 'fire_env' already exists."
else
    echo "Creating conda environment 'fire_env'..."
    module load conda
    conda create -n fire_env python=3.11 numpy numba matplotlib -y
    echo "Done."
fi

echo ""

# ── 2. Generate sweep files ────────────────────────────────────────
echo "Generating parameter files, LAMMPS inputs, and submission script..."
source activate fire_env 2>/dev/null || conda activate fire_env
python gen_sweep.py --outdir .

echo ""

# ── 3. Create directory structure ────────────────────────────────────
mkdir -p logs results figures

echo ""
echo "=== Setup complete ==="
echo ""
echo "To edit the sweep parameters (N, phi, K, D values):"
echo "  Edit the top of gen_sweep.py and re-run: python gen_sweep.py"
echo ""
echo "To submit all jobs:"
echo "  bash submit_all.sh"
echo ""
echo "To check progress:"
echo "  squeue -u \$USER"
echo "  ls results/ | wc -l"
echo ""
echo "When all jobs finish, generate plots:"
echo "  python collect_results.py --resultsdir results/ --outdir figures/"
