#!/bin/bash
#SBATCH --job-name=afire
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/fire_%A_%a.out
#SBATCH --error=logs/fire_%A_%a.err

# Resources (--cpus-per-task, --mem, --time, --array) are set by submit_all.sh
# PARAM_FILE is passed via --export

set -e

# ── Environment ─────────────────────────────────────────────────────
# Option A: conda (recommended — install once with setup.sh)
source activate fire_env 2>/dev/null || conda activate fire_env

# Option B: if using system python + pip install --user
# module load python/3.9.0
# export PATH=$HOME/.local/bin:$PATH

# ── Read parameters for this array task ─────────────────────────────
PARAM_FILE=${PARAM_FILE:-params_fire.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")
N=$(echo "$LINE" | awk '{print $1}')
PHI=$(echo "$LINE" | awk '{print $2}')
K=$(echo "$LINE" | awk '{print $3}')
D=$(echo "$LINE" | awk '{print $4}')

echo "Task ${SLURM_ARRAY_TASK_ID}: N=$N phi=$PHI K=$K D=$D"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK, Mem: $SLURM_MEM_PER_NODE"
echo "Start: $(date)"

# ── Run ─────────────────────────────────────────────────────────────
python async_fire_hpc.py \
    --N "$N" --phi "$PHI" --K "$K" --D "$D" \
    --outdir results/

echo "End: $(date)"
