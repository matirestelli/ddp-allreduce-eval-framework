#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J ddp-gpt-frontier_64gpus
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -C nvme
#SBATCH -o ddp_gpt_frontier_64gpus.%j.out
#SBATCH -e ddp_gpt_frontier_64gpus.%j.err


#ask for a compute node
# salloc -A gen243 -p batch -J trygpt -N 1 -t 00:30:00 -C nvme

cd "${SLURM_SUBMIT_DIR}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source envScriptFrontier.sh

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export DDP_ITER_LOG=0        # per-rank JSONL off (space)
export DDP_PROFILE_BARRIER=0 # MUST stay 0 for timing runs — it kills bwd/comm overlap

# Fix change cray mpich policy from NUMA for multiple nodes with the ranks
export MPICH_OFI_NIC_POLICY=GPU


# Edit these lists to run multiple experiments inside one queued job.
# Leave only one active value in each list for a single run.

# Strong scaling: the GLOBAL work per iteration is fixed by GRAD_ACCUM_GLOBALS
# below; each rank does GRAD_ACCUM_GLOBAL / NUM_PROCS micro-batches.
# NUM_PROCS must divide GRAD_ACCUM_GLOBAL. Frontier: 8 GCDs ("GPUs") per node.
#   16 GPUs -> -N 2 -> 4 micro-steps per rank
#   32 GPUs -> -N 4 -> 2 micro-steps per rank
#   64 GPUs -> -N 8 -> 1 micro-step  per rank
NUM_PROCS=64
PPN=8
# processes per node

# Where the token files live: <DATA_DIR>/<DATASET>/train.bin and val.bin
# Prepare once on a Frontier login node with prepare_tokens.py (same command as Polaris).
DATA_DIR="/lustre/orion/gen243/proj-shared/matilderestelli/data"

# Paper workloads (Chen et al. ICPP'25, Fig. 10): n_layer x n_head x n_embd
MODELS=(
    "gpt2_12x12x768"     # GPT-2 small, nanoGPT default: 124.4M params, ~498 MB grads/iter
    # "gpt2_24x24x768"   # paper workload 2: 209.3M params, ~838 MB grads/iter (fits in 64 GB)
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
    # "2"      # shorter: epoch 1 warm-up, epoch 2 measured
    # "10"
)

# GPT has no natural epoch: "epoch" = this many optimizer steps.
ITERS_PER_EPOCH_LIST=(
    "50"       # timing runs (same as Polaris)
    # "25"
    # "100"
)

# Micro-batch per rank (sequences of BLOCK_SIZE tokens)
BATCH_SIZES=(
    "12"       # nanoGPT default
    # "6"      # with GRAD_ACCUM_GLOBAL=128 (same tokens/iter)
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
    "20"       # shortened for timing runs (use "10" with ITERS_PER_EPOCH=25)
    # "2000"   # nanoGPT default (full training)
)

MIN_LR_RATIOS=(
    "0.1"      # nanoGPT: min_lr = 6e-5 = 0.1 x 6e-4
    # "0.0"
)

AMPS=(
    "bf16"     # nanoGPT default dtype (params + grads stay fp32); MI250X supports bf16
    # "none"   # pure fp32
)

ALLOW_TF32_VALUES=(
    "false"    # MI250X has no TF32: the flag has no effect on ROCm
    # "true"
)

DROP_LAST_VALUES=(
    # "false"
    "true"     # required: every rank must run the same number of iterations
)

BACKENDS=(
    "mpi"
    # "nccl"   # = RCCL on Frontier
)

# "none"          = no custom communication hook (built-in DDP allreduce).
# "default"       = built-in DDP allreduce wrapped with NVTX timing.
# "direct_t<N>"   = Direct Sendrecv allreduce (Chen et al. ICPP'25), throttle N
#                   (N peers per batch; t = 32 = all peers at once on 32 GPUs).
#                   Paper Fig. 4 (64 GPUs): best t=32 @8 MB, 16 @16 MB, 8 @32-512 MB, 4 @1 GB.
EXPERIMENTS=(
    #"none:"
    #"default_sync:"
    #"default_clone:"
    #"default_cpu_stage:"
    #"default:"
    #"ring:"
    ##"ring_zfp_naive:16"
    #"ring_zfp_online_coll:16"
    #"ring_zfp_online_coll:8"

    #"recursive_doubling:"
    #"recursive_doubling_zfp_naive:16"
    #"recursive_doubling_zfp_online_coll:16"
    #"recursive_doubling_zfp_online_coll:8"

    "hierarchical_zfp_online_coll:8"

    #"direct_t4:"
    #"direct_t8:"
    #"direct_t16:"
    #"direct_t32:"
    #"direct_t64:"
    #"kring_k1:"                        # k-ring with k=1 = plain ring (sanity check)
    #"kring_k8:"                        # k = ranks per node on Polaris
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
    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        COMM_ALGORITHM="${EXPERIMENT%%:*}"
        ZFP_RATE="${EXPERIMENT#*:}"

        echo "=== Starting: model=${MODEL_NAME}, dataset=${DATASET}, ws=${NUM_PROCS}, bs=${BATCH_SIZE}, accum=${GRAD_ACCUM_GLOBAL}, block=${BLOCK_SIZE}, lr=${LEARNING_RATE}, amp=${AMP}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp=${ZFP_RATE:-none} at $(date) ==="

        export MODEL_NAME="${MODEL_NAME}"
        export DATASET="${DATASET}"
        export BLOCK_SIZE="${BLOCK_SIZE}"
        export NUM_EPOCHS="${NUM_EPOCHS}"
        export ITERS_PER_EPOCH="${ITERS_PER_EPOCH}"
        export VAL_MAX_BATCHES="10"
        export BATCH_SIZE="${BATCH_SIZE}"
        export GRAD_ACCUM_GLOBAL="${GRAD_ACCUM_GLOBAL}"
        export LEARNING_RATE="${LEARNING_RATE}"
        export OPTIMIZER="${OPTIMIZER}"
        export ADAM_BETA1="0.9"
        export ADAM_BETA2="${ADAM_BETA2}"
        export WEIGHT_DECAY="${WEIGHT_DECAY}"
        export GRAD_CLIP="${GRAD_CLIP}"
        export SCHEDULER="${SCHEDULER}"
        export WARMUP_ITERS="${WARMUP_ITERS}"
        export MIN_LR_RATIO="${MIN_LR_RATIO}"
        export AMP="${AMP}"
        export ALLOW_TF32="${ALLOW_TF32}"
        export NUM_WORKERS="4"
        export PIN_MEMORY="true"
        export DROP_LAST="${DROP_LAST}"
        export PRETRAINED="false"
        export CIFAR_STEM="false"
        export DATA_DIR="${DATA_DIR}"
        export CHECKPOINT_DIR="./checkpoints"
        export SAVE_CHECKPOINT="0"
        export SEED="42"
        export BACKEND="${BACKEND}"
        export COMM_ALGORITHM="${COMM_ALGORITHM}"

        unset MIOPEN_FIND_MODE
        unset MIOPEN_FIND_ENFORCE

        export TORCH_DISTRIBUTED_DEBUG=OFF
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

        if [[ -n "${ZFP_RATE}" ]]; then
            export ZFP_RATE="${ZFP_RATE}"
        else
            unset ZFP_RATE
        fi

        unset SRUN_CPUS_PER_TASK
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

        srun --exact -n "${NUM_PROCS}" --ntasks-per-node="${PPN}" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            --gpus-per-task=1 \
            python -u interface.py

        status=$?
        echo "=== Finished training: model=${MODEL_NAME}, dataset=${DATASET}, ws=${NUM_PROCS}, batch=${BATCH_SIZE}, accum=${GRAD_ACCUM_GLOBAL}, lr=${LEARNING_RATE}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

        if [[ ${status} -ne 0 ]]; then
            echo "=== Stopping job because training failed ==="
            exit ${status}
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

# salloc -A gen243 -p batch -J torch-build-and-pack -N 1 -t 00:30:00 -c 16 -C nvme
# squeue -j <jobid> -o "%.18i %.9P %.10T %.8u %.10M %.20S %.6D %R"