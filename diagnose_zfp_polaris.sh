#!/bin/bash -l
#PBS -l select=4:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -o diagnose_zfp_polaris.%j.out
#PBS -e diagnose_zfp_polaris.%j.err
#PBS -N diagnose-zfp

cd ${PBS_O_WORKDIR}

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript3.sh

# Hook timing not needed for the diagnostic.
export DDP_HOOK_TIMING=0

export MASTER_ADDR="$(head -n 1 "${PBS_NODEFILE}")"
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

PPN=4
TRIALS=30

# Run both scales so we can compare 8 vs 16 directly.
NUM_PROCS_LIST=(
    "8"
    "16"
)

ZFP_RATES=(
    "8"
    "16"
)

OVERALL_STATUS=0

for NUM_PROCS in "${NUM_PROCS_LIST[@]}"; do
for ZFP_RATE in "${ZFP_RATES[@]}"; do

    echo "=== Starting diagnostic: ranks=${NUM_PROCS}, ppn=${PPN}, zfp_rate=${ZFP_RATE}, trials=${TRIALS} at $(date) ==="

    MPI_ENV_ARGS=(
        -env MASTER_ADDR="${MASTER_ADDR}"
        -env MASTER_PORT="${MASTER_PORT}"
        -env MPICH_GPU_SUPPORT_ENABLED=1
        -env LD_PRELOAD="${LD_PRELOAD}"
        -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
        -env PYTHONPATH="${PYTHONPATH}"
        -env DDP_HOOK_TIMING=0
    )

    mpiexec -np ${NUM_PROCS} --ppn ${PPN} --depth=8 --cpu-bind depth \
        "${MPI_ENV_ARGS[@]}" \
        python diagnose_zfp_hooks.py --rate ${ZFP_RATE} --trials ${TRIALS}

    status=$?
    echo "=== Finished diagnostic: ranks=${NUM_PROCS}, zfp_rate=${ZFP_RATE}, status=${status} at $(date) ==="

    # Keep going so one failing config doesn't hide the others.
    if [[ ${status} -ne 0 ]]; then
        OVERALL_STATUS=${status}
    fi

done   # ZFP_RATES
done   # NUM_PROCS_LIST

echo "=== Job finished: $(date) (overall status=${OVERALL_STATUS}) ==="
exit ${OVERALL_STATUS}