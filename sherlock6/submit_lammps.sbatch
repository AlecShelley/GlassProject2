#!/bin/bash
#SBATCH --job-name=lammps_base
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/lammps_%A_%a.out
#SBATCH --error=logs/lammps_%A_%a.err

# Array size is set by submit_all.sh

set -e

# ── Environment ─────────────────────────────────────────────────────
# Adjust module name to match your Sherlock LAMMPS installation.
# Run `module spider lammps` to find available versions.
module load lammps

# ── Read parameters ─────────────────────────────────────────────────
PARAM_FILE=${PARAM_FILE:-params_lammps.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")
INFILE=$(echo "$LINE" | awk '{print $1}')
LOGFILE=$(echo "$LINE" | awk '{print $2}')
NTASKS=$(echo "$LINE" | awk '{print $3}')

#SBATCH --ntasks=$NTASKS   # note: can't use variable here; override via submit_all.sh

echo "Task ${SLURM_ARRAY_TASK_ID}: $INFILE → $LOGFILE (np=$NTASKS)"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd lammps_inputs/

# Skip if result exists
if [ -f "../results/$LOGFILE" ] && grep -q "FINAL_ENERGY" "../results/$LOGFILE"; then
    echo "SKIP: ../results/$LOGFILE already has FINAL_ENERGY"
    exit 0
fi

# ── Run ─────────────────────────────────────────────────────────────
mpirun -np "$NTASKS" lmp -in "$INFILE" -log "../results/$LOGFILE"

echo "End: $(date)"
