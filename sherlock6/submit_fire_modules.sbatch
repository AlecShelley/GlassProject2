#!/bin/bash
#SBATCH --job-name=afire
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/fire_%A_%a.out
#SBATCH --error=logs/fire_%A_%a.err

# Module-based variant of submit_fire.sbatch (no conda on this Sherlock account).
# Resources (--cpus-per-task, --mem, --time, --array) are set at submit time.
# PARAM_FILE is passed via --export. Core science code is unchanged.
set -e

ml load devel py-numba/0.63.1_py314

PARAM_FILE=${PARAM_FILE:-params_fire_N1000000.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")
N=$(echo "$LINE" | awk '{print $1}')
PHI=$(echo "$LINE" | awk '{print $2}')
K=$(echo "$LINE" | awk '{print $3}')
D=$(echo "$LINE" | awk '{print $4}')

echo "Task ${SLURM_ARRAY_TASK_ID}: N=$N phi=$PHI K=$K D=$D"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK, Mem: $SLURM_MEM_PER_NODE"
echo "Start: $(date)"

python3 async_fire_hpc.py \
    --N "$N" --phi "$PHI" --K "$K" --D "$D" \
    --outdir results/

echo "End: $(date)"
