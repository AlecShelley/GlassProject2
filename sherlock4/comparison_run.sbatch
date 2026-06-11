#!/bin/bash
#
# Comparison Run — FIRE Parameter Sweep on Sherlock
# Global FIRE vs Async FIRE with Lipschitz clock synchronization
#
# Submit with:  sbatch comparison_run.sbatch
# Custom cores: sbatch --cpus-per-task=8 comparison_run.sbatch
# Monitor with: squeue -u $USER
# Cancel with:  scancel <job_id>
#

#SBATCH --job-name=fire_comparison
#SBATCH --partition=YOUR_PI_PARTITION        # <-- Replace with your PI's partition
#SBATCH --time=12:00:00                      # 12 hours (25 runs × 600s/run + overhead)
#SBATCH --cpus-per-task=16                   # Numba prange threads — configurable
#SBATCH --mem=32G
#SBATCH --output=fire_comparison_%j.out
#SBATCH --error=fire_comparison_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@stanford.edu  # <-- Replace with your email

# ---- Pre-flight Validation ----
ERRORS=0

echo "=============================================="
echo "Pre-flight checks..."
echo "=============================================="

# Check $SCRATCH is set and writable
if [ -z "${SCRATCH}" ]; then
    echo "FAIL: \$SCRATCH is not set. Cannot create output directory."
    ERRORS=$((ERRORS + 1))
elif [ ! -d "${SCRATCH}" ]; then
    echo "FAIL: \$SCRATCH directory does not exist: ${SCRATCH}"
    ERRORS=$((ERRORS + 1))
elif [ ! -w "${SCRATCH}" ]; then
    echo "FAIL: \$SCRATCH is not writable: ${SCRATCH}"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: \$SCRATCH = ${SCRATCH}"
fi

# Check Python scripts exist in the submission directory
SCRIPT_DIR="$(dirname "$(scontrol show job ${SLURM_JOB_ID} -o | grep -oP 'Command=\K\S+')")"
if [ -z "${SCRIPT_DIR}" ]; then
    SCRIPT_DIR="$(pwd)"
fi

for script in comparison_run_compute.py comparison_run_plot.py; do
    if [ ! -f "${SCRIPT_DIR}/${script}" ]; then
        echo "FAIL: ${script} not found in ${SCRIPT_DIR}"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: ${script} found"
    fi
done

# Load modules and validate each one
module purge

REQUIRED_MODULES=(
    "python/3.12.1"
    "py-numpy/1.26.3_py312"
    "py-numba/0.59.0_py312"
    "py-scipy/1.12.0_py312"
)

for mod in "${REQUIRED_MODULES[@]}"; do
    if module load "${mod}" 2>/dev/null; then
        echo "  OK: module ${mod}"
    else
        echo "FAIL: module '${mod}' not available. Run 'module spider ${mod%%/*}' to find valid versions."
        ERRORS=$((ERRORS + 1))
    fi
done

# If using a conda environment instead, comment out the modules above and use:
# module load conda
# conda activate your_env_name

# Verify Python can import required packages
for pkg in numpy numba; do
    if python -c "import ${pkg}" 2>/dev/null; then
        echo "  OK: python import ${pkg}"
    else
        echo "FAIL: python cannot import '${pkg}'. Check module versions or use a conda env."
        ERRORS=$((ERRORS + 1))
    fi
done

# Abort if any checks failed
if [ ${ERRORS} -gt 0 ]; then
    echo ""
    echo "ABORTING: ${ERRORS} pre-flight check(s) failed. Fix the issues above and resubmit."
    exit 1
fi

echo "All pre-flight checks passed."
echo ""

# ---- Configurable Parameters ----
# Override these at submit time with environment variables:
#   sbatch --export=N_ATOMS=5000000,WALL_TIME=300 comparison_run.sbatch
N_ATOMS=${N_ATOMS:-1000000}
WALL_TIME=${WALL_TIME:-600}
CPUS=${SLURM_CPUS_PER_TASK}

OUTPUT_DIR="${SCRATCH}/fire_comparison/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ABORTING: Failed to create output directory: ${OUTPUT_DIR}"
    exit 1
fi

echo "=============================================="
echo "Comparison Run — FIRE Parameter Sweep"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "CPUs:          ${CPUS}"
echo "Memory:        ${SLURM_MEM_PER_NODE}MB"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "N_ATOMS:       ${N_ATOMS}"
echo "Wall time/run: ${WALL_TIME}s"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Start time:    $(date)"
echo "=============================================="

# ---- Run Computation ----
python comparison_run_compute.py \
    --cpus "${CPUS}" \
    --n-atoms "${N_ATOMS}" \
    --wall-time "${WALL_TIME}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Compute finished at: $(date)"

# ---- Generate Plots (headless) ----
python comparison_run_plot.py "${OUTPUT_DIR}" --n-atoms "${N_ATOMS}"

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"
echo "To copy locally:"
echo "  scp -r sherlock:${OUTPUT_DIR} ./comparison_results/"
