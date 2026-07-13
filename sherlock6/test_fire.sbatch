#!/bin/bash
#SBATCH --job-name=afire_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs_test/test_%j.out
#SBATCH --error=logs_test/test_%j.err

# Smoke test: run the REAL async_fire_hpc.py unchanged, just at small N,
# to confirm the module env + code + JSON output all work on Sherlock.
set -e

ml load devel py-numba/0.63.1_py314
echo "python: $(which python3)"
python3 -c "import numpy, numba; print('numpy', numpy.__version__, '| numba', numba.__version__)"

echo "Start: $(date)"
python3 async_fire_hpc.py --N 20000 --phi 0.84 --K 2 --D 1 --outdir results_test/
echo "End: $(date)"
echo "=== output json ==="
ls -la results_test/
