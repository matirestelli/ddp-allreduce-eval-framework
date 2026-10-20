#!/bin/bash -l
#PBS -l select=8:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -o ddp_gpt_polaris_nccl_32gpu.out
#PBS -e ddp_gpt_polaris_nccl_32gpu.err
#PBS -N ddp-gpt_polaris_nccl_32gpu

# GPT (nanoGPT) DDP runs on Polaris with the NCCL backend.
# Same experiment grid as trainJobPolaris_gpt.sh (MPI), but:
#   - envScript_nccl.sh instead of envScript3.sh (NCCL / aws-ofi-nccl setup)
#   - no MPICH GPU (GTL) variables: NCCL does the GPU communication
#   - per-job log file via tee (PBS does not expand %j in #PBS -o)
#   - only hooks that exist for the nccl backend in communication_strategy.py

cd ${PBS_O_WORKDIR}

# PBS does NOT substitute %j in #PBS -o. Use the runtime job id instead.
# Keep "gpt", "polaris" and the GPU count in the name: plot_gpt.py uses them.
LOG="ddp_gpt_polaris_nccl_64gpu.${PBS_JOBID%%.*}.out"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript_nccl.sh

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export DDP_ITER_LOG=0        # per-rank JSONL off (space)
export DDP_PROFILE_BARRIER=0 # MUST stay 0 for timing runs — it kills bwd/comm overlap

export MASTER_ADDR="$(head -n 1 "${PBS_NODEFILE}")"
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Edit these lists to run multiple experiments inside one queued job.
# Leave only one active value in each list for a single run.

# Strong scaling: the GLOBAL work per iteration is fixed by GRAD_ACCUM_GLOBALS
# below; each rank does GRAD_ACCUM_GLOBAL / NUM_PROCS micro-batches.
# NUM_PROCS must divide GRAD_ACCUM_GLOBAL. Keep #PBS select = NUM_PROCS / PPN.
#   16 GPUs -> select=4  (debug-scaling) -> 4 micro-steps per rank
#   32 GPUs -> select=8  (debug-scaling) -> 2 micro-steps per rank
#   64 GPUs -> select=16 (prod queue)    -> 1 micro-step  per rank
NUM_PROCS=32
PPN=4
# processes per node

# Stop the whole job at the first failed run (1), or log it and move on (0).
# 0 is handy while trying new hooks with NCCL: one hang/crash doesn't cost
# the rest of the grid.
STOP_ON_FAILURE=1

# Sanity checks: enough nodes for NUM_PROCS, and NUM_PROCS divides accumulation.
NODES_AVAILABLE=$(sort -u "${PBS_NODEFILE}" | wc -l)
NODES_NEEDED=$(( NUM_PROCS / PPN ))
if (( NODES_AVAILABLE < NODES_NEEDED )); then
    echo "=== ERROR: NUM_PROCS=${NUM_PROCS} with PPN=${PPN} needs ${NODES_NEEDED} nodes, PBS gave ${NODES_AVAILABLE} ==="
    exit 1
fi

# Where the token files live: <DATA_DIR>/<DATASET>/train.bin and val.bin
DATA_DIR="/lus/eagle/projects/UIC-HPC/mrest/mpiAllReduceCompressed/data"

# Paper workloads (Chen et al. ICPP'25, Fig. 10): n_layer x n_head x n_embd
MODELS=(
    "gpt2_12x12x768"     # GPT-2 small, nanoGPT default: 124.4M params, ~498 MB grads/iter
    # "gpt2_24x24x768"   # paper workload 2: 209.3M params, ~838 MB grads/iter
)

DATASETS=(
    "tinystories"        # small, fast to stage; same iteration time as OWT
    # "openwebtext"      # nanoGPT / paper default corpus
)

BLOCK_SIZES=(
    "1024"     # nanoGPT default sequence length
    # "512"
)

NUM_EPOCHS_LIST=(
    "3"        # timing runs: epoch 1 = warm-up (discarded), 2-3 measured
    # "10"
)

# GPT has no natural epoch: "epoch" = this many optimizer steps.
ITERS_PER_EPOCH_LIST=(
    "50"       # timing runs
    # "100"
    # "200"
)

# Micro-batch per rank (sequences of BLOCK_SIZE tokens)
BATCH_SIZES=(
    "12"       # nanoGPT default
    # "6"      # if 24x24x768 runs out of memory (use GRAD_ACCUM_GLOBAL=128)
    # "8"
)

# Micro-batches per optimizer step summed over ALL ranks.
# tokens/iteration = GRAD_ACCUM_GLOBAL x BATCH_SIZE x BLOCK_SIZE
GRAD_ACCUM_GLOBALS=(
    "64"       # nanoGPT default 40 -> 64 (40 not divisible by 32/64): 786,432 tokens/iter
    # "128"    # with BATCH_SIZE=6 (same tokens/iter), or 2x the work with BATCH_SIZE=12
    # "none"   # no accumulation = weak scaling (1 micro-batch per rank)
)

LEARNING_RATES=(
    "6e-4"     # nanoGPT default
    # "3e-4"
)

OPTIMIZERS=(
    "adamw"    # GPT always uses nanoGPT's configure_optimizers (fused AdamW)
)

ADAM_BETA2S=(
    "0.95"     # nanoGPT default
    # "0.999"  # PyTorch default
)

WEIGHT_DECAYS=(
    "0.1"      # nanoGPT default (2-D weights only)
    # "0.0"
)

GRAD_CLIPS=(
    "1.0"      # nanoGPT default
    # "none"
)

SCHEDULERS=(
    "cosine"   # nanoGPT: warmup + cosine to MIN_LR_RATIO x lr
    # "constant"
)

WARMUP_ITERS_LIST=(
    "20"       # shortened for timing runs
    # "2000"   # nanoGPT default (full training)
)

MIN_LR_RATIOS=(
    "0.1"      # nanoGPT: min_lr = 6e-5 = 0.1 x 6e-4
    # "0.0"
)

AMPS=(
    "bf16"     # nanoGPT default dtype (params + grads stay fp32)
    # "none"   # pure fp32 (with ALLOW_TF32=true)
)

ALLOW_TF32_VALUES=(
    "true"     # nanoGPT default
    # "false"
)

DROP_LAST_VALUES=(
    # "false"
    "true"     # required: every rank must run the same number of iterations
)

BACKENDS=(
    "nccl"
)

# Hooks registered for the nccl backend in communication_strategy.py.
# "none" = built-in DDP allreduce = NCCL's own ring/tree (the baseline to beat).
#
# NOTE: on nccl, ring_zfp_online_coll and recursive_doubling_zfp_online_coll
# BOTH map to the same hierarchical leader hook (_hier_zfp_online_coll_hook),
# so only one of them is listed per rate; the run is the hierarchical hook.
#
# kring_* needs the patched communication_strategy.py with the k-ring hook,
# and test_kring.py passing with --backend nccl first.
EXPERIMENTS=(
    "none:"
    #"default_sync:"
    #"default_clone:"
    #"default_cpu_stage:"
    #"default:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"          # = hierarchical leader ZFP hook on nccl
    "ring_zfp_online_coll:8"           # = hierarchical leader ZFP hook on nccl
    "recursive_doubling:"
    "recursive_doubling_zfp_naive:16"
    "direct_t4:"
    "direct_t8:"
    "direct_t16:"
    "direct_t32:"
    #"direct_t64:"                      # = all peers at once on 64 GPUs
    #"kring_k1:"                        # k-ring with k=1 = plain ring (sanity check)
    "kring_k4:"                        # k = ranks per node on Polaris
    #"kring_k2:"
)

for MODEL_NAME in "${MODELS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
for NUM_EPOCHS in "${NUM_EPOCHS_LIST[@]}"; do
for ITERS_PER_EPOCH in "${ITERS_PER_EPOCH_LIST[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for GRAD_ACCUM_GLOBAL in "${GRAD_ACCUM_GLOBALS[@]}"; do
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
for OPTIMIZER in "${OPTIMIZERS[@]}"; do
for ADAM_BETA2 in "${ADAM_BETA2S[@]}"; do
for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
for SCHEDULER in "${SCHEDULERS[@]}"; do
for WARMUP_ITERS in "${WARMUP_ITERS_LIST[@]}"; do
for MIN_LR_RATIO in "${MIN_LR_RATIOS[@]}"; do
for AMP in "${AMPS[@]}"; do
for ALLOW_TF32 in "${ALLOW_TF32_VALUES[@]}"; do
for DROP_LAST in "${DROP_LAST_VALUES[@]}"; do
for BACKEND in "${BACKENDS[@]}"; do
    if [[ "${GRAD_ACCUM_GLOBAL}" != "none" ]] && (( GRAD_ACCUM_GLOBAL % NUM_PROCS != 0 )); then
        echo "=== ERROR: NUM_PROCS=${NUM_PROCS} must divide GRAD_ACCUM_GLOBAL=${GRAD_ACCUM_GLOBAL} ==="
        exit 1
    fi

    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        COMM_ALGORITHM="${EXPERIMENT%%:*}"
        ZFP_RATE="${EXPERIMENT#*:}"

        echo "=== Starting training: model=${MODEL_NAME}, dataset=${DATASET}, procs=${NUM_PROCS}, batch=${BATCH_SIZE}, accum=${GRAD_ACCUM_GLOBAL}, block=${BLOCK_SIZE}, lr=${LEARNING_RATE}, amp=${AMP}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none} at $(date) ==="

        MPI_ENV_ARGS=(
            # multi-node NCCL: every rank needs the rendezvous address/port
            -env MASTER_ADDR="${MASTER_ADDR}"
            -env MASTER_PORT="${MASTER_PORT}"
            -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
            -env PYTHONPATH="${PYTHONPATH}"
            # ranks per node for the hierarchical / k-ring hooks (fallback when
            # hostnames are not used)
            -env PPN="${PPN}"
            # turn NCCL errors/hangs into a failed run instead of a stuck job
            -env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
            -env MODEL_NAME="${MODEL_NAME}"
            -env DATASET="${DATASET}"
            -env BLOCK_SIZE="${BLOCK_SIZE}"
            -env NUM_EPOCHS="${NUM_EPOCHS}"
            -env ITERS_PER_EPOCH="${ITERS_PER_EPOCH}"
            -env VAL_MAX_BATCHES="10"
            -env BATCH_SIZE="${BATCH_SIZE}"
            -env GRAD_ACCUM_GLOBAL="${GRAD_ACCUM_GLOBAL}"
            -env LEARNING_RATE="${LEARNING_RATE}"
            -env OPTIMIZER="${OPTIMIZER}"
            -env ADAM_BETA1="0.9"
            -env ADAM_BETA2="${ADAM_BETA2}"
            -env WEIGHT_DECAY="${WEIGHT_DECAY}"
            -env GRAD_CLIP="${GRAD_CLIP}"
            -env SCHEDULER="${SCHEDULER}"
            -env WARMUP_ITERS="${WARMUP_ITERS}"
            -env MIN_LR_RATIO="${MIN_LR_RATIO}"
            -env AMP="${AMP}"
            -env ALLOW_TF32="${ALLOW_TF32}"
            -env NUM_WORKERS="4"
            -env PIN_MEMORY="true"
            -env DROP_LAST="${DROP_LAST}"
            -env PRETRAINED="false"
            -env CIFAR_STEM="false"
            -env DATA_DIR="${DATA_DIR}"
            -env CHECKPOINT_DIR="./checkpoints"
            -env SAVE_CHECKPOINT="0"
            -env SEED="42"
            -env BACKEND="${BACKEND}"
            -env COMM_ALGORITHM="${COMM_ALGORITHM}"
        )

        # Only forward LD_PRELOAD if the NCCL env script set one.
        if [[ -n "${LD_PRELOAD:-}" ]]; then
            MPI_ENV_ARGS+=(-env LD_PRELOAD="${LD_PRELOAD}")
        fi

        if [[ -n "${ZFP_RATE}" ]]; then
            MPI_ENV_ARGS+=(-env ZFP_RATE="${ZFP_RATE}")
        fi

        mpiexec -np ${NUM_PROCS} --ppn ${PPN} --depth=8 --cpu-bind depth \
            "${MPI_ENV_ARGS[@]}" \
            python interface.py

        status=$?
        echo "=== Finished training: model=${MODEL_NAME}, dataset=${DATASET}, procs=${NUM_PROCS}, batch=${BATCH_SIZE}, accum=${GRAD_ACCUM_GLOBAL}, lr=${LEARNING_RATE}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

        if [[ ${status} -ne 0 ]]; then
            if [[ ${STOP_ON_FAILURE} -eq 1 ]]; then
                echo "=== Stopping job because training failed ==="
                exit ${status}
            fi
            echo "=== Run failed (status=${status}); continuing with the next experiment ==="
        fi
    done   # EXPERIMENTS
done       # BACKEND
done       # DROP_LAST
done       # ALLOW_TF32
done       # AMP
done       # MIN_LR_RATIO
done       # WARMUP_ITERS
done       # SCHEDULER
done       # GRAD_CLIP
done       # WEIGHT_DECAY
done       # ADAM_BETA2
done       # OPTIMIZER
done       # LEARNING_RATE
done       # GRAD_ACCUM_GLOBAL
done       # BATCH_SIZE
done       # ITERS_PER_EPOCH
done       # NUM_EPOCHS
done       # BLOCK_SIZE
done       # DATASET
done       # MODELS

echo "=== Job finished: $(date) ==="

#to request an interactive node
# qsub -I -l select=1:ngpus=4 -l walltime=01:00:00 -l filesystems=home:eagle -q debug -A UIC-HPC