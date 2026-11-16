# FABRIC: Framework for Architecture, Backend, and Reduction Interchange Comparison

FABRIC is a PyTorch DistributedDataParallel (DDP) evaluation framework for
studying how communication algorithms and backends affect end-to-end training.
It exposes gradient synchronization through interchangeable communication hooks,
so built-in, custom, and compressed AllReduce algorithms can be compared within
the same training loop.

This README describes the scope presented in the poster **"Evaluating
All-Reduce Operations in DDP Training with FABRIC"**. The project builds on the
earlier EuroMPI work, **"Accelerating MPI AllReduce Communication with Efficient
GPU-Based Compression Schemes on Modern GPU Clusters"**, and extends the focus
to comparisons across GPU architectures, backends, and scaling regimes.

## Framework Architecture

FABRIC separates experiment choices into four tiers:

| Tier | Role |
|---|---|
| ML frontend | Model, dataset, optimizer, and training hyperparameters |
| DDP training loop | Forward pass, backward pass, and gradient synchronization |
| Backend | MPI, NCCL on NVIDIA GPUs, or RCCL on AMD GPUs |
| Communication algorithm | Built-in AllReduce, ring, recursive doubling (RD), and naive or online ZFP-compressed variants |

Experiments are configured through a Python interface or a batch/job script.
New communication algorithms can be integrated by implementing and registering
a DDP communication hook. The frontend is designed to accommodate models
trainable with DDP; WideResNet, BERT, and NanoGPT appear in the poster's
architecture diagram. The performance figures shown in the poster evaluate
WideResNet.

## Repository Contents

```text
.
|-- interface.py                         # User-facing experiment configuration
|-- config.py                            # Models, data loaders, and TrainingConfig
|-- ddp_training.py                      # DDP training loop, logging, timing
|-- communication_strategy.py            # DDP communication hooks
|-- zfp_cuda.py                          # Python wrapper for cuZFP extension
|-- zfp_cuda_extension.cpp               # C++/CUDA-facing ZFP extension
|-- build_zfp_cuda.sh                    # PBS build script for modified cuZFP
|-- envScript3.sh                        # Polaris environment for CUDA-aware MPI
|-- trainJob.sh                          # PBS sweep template
|-- experiments_results_lr001_GlobalBS128/
|-- experiments_results_lr001_LocalBS128/
`-- plotting/                            # Plotting utilities for saved experiment logs
```

Auxiliary notes:

- `BATCH_SIZE.md` explains fixed-global versus fixed-local batch size.
- `TRAINING_STABILITY.md` explains warmup, gradient clipping, and pretrained
  weights for deep CIFAR-10 experiments.
- `logs/` contains debugging notes from hook development.

## Evaluated Systems and Backends

The poster reports development and evaluation on two HPC platforms:

| System | Facility | GPU architecture | Backends compared |
|---|---|---|---|
| Polaris | ALCF | NVIDIA A100; four GPUs per node | GPU-aware MPI and NCCL |
| Frontier | OLCF | AMD Instinct MI250X; four GPUs per node | GPU-aware MPI and RCCL |

Supporting MPI required a source build of PyTorch with corrected GPU-aware MPI
detection for Cray MPICH. ZFP was integrated with CUDA and HIP support, including
stream-aware compression and decompression launches.

The setup commands below cover the Polaris/CUDA files available in this
checkout: PBS, Cray MPICH with CUDA-aware MPI/GTL, CUDA 12.9, MPI-enabled
PyTorch, `mpi4py`, and the modified CUDA-enabled ZFP library. Frontier requires
the corresponding ROCm/HIP software stack and system-specific launch and build
configuration. The CUDA scripts below are specific to Polaris.

RCCL is the AMD collective library; PyTorch ROCm builds expose it through the
`nccl` distributed backend name. The poster's RCCL label should therefore not
be interpreted as a literal `BACKEND=rccl` setting in this checkout.

## Future Work coming very soon

Explore upcoming models, communication hooks, and experiments</strong></summary>

- **Larger models and new model families:** GPT-2 training with NanoGPT.
- **More AllReduce hooks:** Direct Sendrecv, generalized k-ring, and hierarchical ZFP-compressed AllReduce for MPI and NCCL/RCCL.
- **Understanding the built-in MPI baseline:** experiments with instrumented open-source MPICH to track which AllReduce algorithms execute when PyTorch DDP uses the MPI backend, across message sizes and process counts.


## Quick Start

From the repository root:

```bash
cd ddp-allreduce-eval-framework
source envScript3.sh
```

Check that the environment sees CUDA, MPI, and the ZFP install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch.distributed as dist; print('MPI available:', dist.is_mpi_available())"
python -c "import zfp_cuda; print('zfp_cuda import OK')"
```

For an initial 4-GPU training check, request one Polaris node and set
`NUM_EPOCHS=1`, `MODEL_NAME=wide_resnet50_2`, and `BATCH_SIZE=32` in the job
environment before launching:

```bash
mpiexec -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=default \
    -env NUM_EPOCHS=1 \
    -env MODEL_NAME=wide_resnet50_2 \
    -env BATCH_SIZE=32 \
    python interface.py
```

Rank 0 writes one `.log` file in the current directory and one CSV file under
`results/`.

## Data and Weights

Experiments use CIFAR-10 by default.  The data loader uses
`download=False`, so the dataset must already exist under:

```text
ddp-allreduce-eval-framework/data/cifar/cifar-10-batches-py
```

If the dataset is missing, download CIFAR-10 once in an interactive or login
environment where network access is allowed, or temporarily change the
`download` argument in `config.py` to `True` for setup only.

Some saved experiment configurations use `pretrained=True` in `interface.py`, especially
for `resnext101_32x8d`.  Make sure the corresponding torchvision weights are
available in the local torch cache before submitting jobs on compute nodes
without internet access.

## Build ZFP for the Target System

Build and install the modified ZFP sources separately for each system. The
compressed hooks need GPU support; a CPU-only ZFP build is insufficient.

For Polaris, `build_zfp_cuda.sh` provides a PBS build template with CUDA enabled
and A100 architecture settings. It rebuilds `zfp/build-cuda/` and installs into
`zfp-install/`; adjust the allocation and paths for your environment before
submitting:

```bash
qsub build_zfp_cuda.sh
```

For a separate CUDA installation prefix, the equivalent CMake configuration is:

```bash
cmake -S zfp -B zfp/build-newSystem \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/zfp-install-newSystem" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DZFP_WITH_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build zfp/build-newSystem -j 8
cmake --install zfp/build-newSystem
```

The architecture value `80` targets Polaris A100 GPUs. Frontier needs the
HIP-enabled integration and an AMD-specific build configuration.

Point your environment script to the installation used by the Python extension:

```bash
export ZFP_HOME="/path/to/ddp-allreduce-eval-framework/zfp-install-newSystem"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"
```

If `import zfp_cuda` fails, check `ZFP_HOME`, the installed `libzfp.so`, and the
extension's build and runtime library paths in the launched job environment.

## Configure an Experiment

Most runs are configured in `interface.py` through `TrainingConfig`.

The most important fields are:

```python
TrainingConfig(
    model_name="wide_resnet50_2",      # or "resnext101_32x8d"
    dataset="cifar10",
    num_epochs=20,
    batch_size=32,                    # per-rank batch size
    learning_rate=0.001,
    backend="mpi",
    comm_algorithm=None,               # built-in baseline; "default" adds timing wrapper
    zfp_rate=16.0,
    pretrained=True,
    cifar_stem=True,
)
```

Job scripts can override the backend, hook, and ZFP rate without editing
`interface.py`:

```bash
BACKEND=mpi
COMM_ALGORITHM=ring_zfp_online_coll
ZFP_RATE=8
```

These environment variables are read by `apply_job_overrides()` in
`interface.py`.

## Communication Algorithms

Use these values for `COMM_ALGORITHM` or `config.comm_algorithm`:

| Value | Meaning |
|---|---|
| `none` / `None` | No custom hook; DDP built-in AllReduce |
| `default` | DDP built-in AllReduce wrapped for timing/NVTX only |
| `ring` | Custom ring AllReduce using reduce-scatter plus allgather |
| `recursive_doubling` | Custom recursive-doubling AllReduce |
| `ring_zfp_naive` | Ring AllReduce with ZFP compression at each exchange step |
| `recursive_doubling_zfp_naive` | Recursive doubling with ZFP compression |
| `ring_zfp_online_coll` | Ring with collective-level online ZFP compression |
| `recursive_doubling_zfp_online_coll` | Recursive doubling with collective-level online ZFP compression |

For ZFP hooks, set `ZFP_RATE` or `config.zfp_rate` (bits per value). The
poster compares naive compression at rate 16 and online compression at rates
16 and 8 for both ring and recursive doubling.

Hook availability depends on the selected backend and checkout. In this
checkout, online ZFP hooks are registered for MPI; NCCL has the default,
uncompressed, and naive ZFP hooks. Inspect the registry before choosing a pair:

```bash
python -c "from communication_strategy import list_available_hooks; print(list_available_hooks())"
```

## Poster Experiment Design

Hyperparameters were tuned once per model and then fixed across backends,
algorithms, and both systems, with the learning-rate scaling rule below applied
for weak scaling. The WideResNet comparisons use 8, 16, and 32 GPUs.

| Regime | Batch-size rule | Learning-rate rule |
|---|---|---|
| Strong scaling | Fixed global batch size 128 | Keep the tuned learning rate fixed |
| Weak scaling | Fixed local batch size 8 per GPU | Increase learning rate with GPU count |

`TrainingConfig.batch_size` is always the per-rank batch size, with one rank
per GPU in these experiments:

| GPUs | Strong: batch per GPU | Strong: global batch | Weak: batch per GPU | Weak: global batch |
|---:|---:|---:|---:|---:|
| 8 | 16 | 128 | 8 | 64 |
| 16 | 8 | 128 | 8 | 128 |
| 32 | 4 | 128 | 8 | 256 |

Set the weak-scaling learning rate explicitly for each GPU count; the poster
does not specify its reference value or exact scaling formula. The interface
does not automatically derive it from `world_size`.

Primary metrics are epoch time and iteration time. Instrumentation also records
hook work time and communication tail time to help isolate communication
overhead. Figure 4 compares communication tail and train-only epoch time for
16-GPU strong scaling with global batch 128. Figure 5 compares WideResNet
training time across systems, backends, algorithms, and scaling regimes.

The results show that backend and algorithm choices strongly affect training
performance, and their effects depend on the system and scaling regime.
Compression should be evaluated through end-to-end training time alongside
communication timing.

### Existing Result Directories

The checkout also contains earlier experiment sets:

| Directory | Batch-size regime |
|---|---|
| `experiments_results_lr001_GlobalBS128/` | Fixed global batch size 128 |
| `experiments_results_lr001_LocalBS128/` | Fixed per-rank batch size 128 |

The `LocalBS128` results use a different local batch from the poster's weak
scaling. These directories and the plotting examples below should not be
assumed to reproduce the full Frontier/Polaris poster figures.

## Launch Jobs

### 4 GPUs

Request one Polaris node:

```bash
#PBS -l select=1:ngpus=4
```

Launch:

```bash
mpiexec -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=ring \
    python interface.py
```

### 8 GPUs

Request two Polaris nodes:

```bash
#PBS -l select=2:ngpus=4
```

Launch:

```bash
mpiexec -np 8 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=ring_zfp_online_coll \
    -env ZFP_RATE=8 \
    python interface.py
```

### 16 GPUs

Request four Polaris nodes:

```bash
#PBS -l select=4:ngpus=4
```

Launch:

```bash
mpiexec -np 16 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=recursive_doubling_zfp_online_coll \
    -env ZFP_RATE=8 \
    python interface.py
```

### 32 GPUs (Poster Strong Scaling)

On Polaris, request eight nodes and launch 32 ranks with four ranks per node.
Set the local batch size to 4 to keep the global batch size at 128:

```bash
#PBS -l select=8:ngpus=4
```

```bash
mpiexec -np 32 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env MODEL_NAME=wide_resnet50_2 \
    -env BATCH_SIZE=4 \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=ring_zfp_online_coll \
    -env ZFP_RATE=8 \
    python interface.py
```

For each node count, choose a PBS queue and allocation that permit the request.

## Run an Algorithm Sweep

Use `trainJob.sh` as the PBS template.  Set the following before each sweep:

1. The node request, `NUM_PROCS`, and `PPN` for the GPU count.
2. The model, per-rank batch size, learning rate, and other training settings in
   `interface.py` or the corresponding lists in `trainJob.sh`. Job-script
   environment overrides take precedence over Python defaults.
3. `BACKENDS` and `EXPERIMENTS`, using only registered backend/hook pairs.

MPI hook sweep matching the poster's algorithm and rate choices:

```bash
EXPERIMENTS=(
    "none:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:8"
    "recursive_doubling:"
    "recursive_doubling_zfp_naive:16"
    "recursive_doubling_zfp_online_coll:16"
    "recursive_doubling_zfp_online_coll:8"
)
```

Use `none` for the built-in baseline.  It produces log filenames containing
`builtin`, which the plotting scripts label as `Baseline`.  Use `default` only
when you specifically want the built-in DDP AllReduce wrapped as a hook for
timing or NVTX instrumentation.

Submit:

```bash
qsub trainJob.sh
```

After each run, move the generated `.log` files into the matching result
directory.  The plotting scripts infer model, GPU count, hook name, and ZFP
rate from the log filenames.

Expected filename pattern:

```text
<model>_<epochs>_<backend>_<algorithm>[_rate<rate>].log
```

Examples:

```text
wide_resnet50_2_20_mpi_builtin.log
wide_resnet50_2_20_mpi_ring_zfp_online_coll_rate8.log
resnext101_32x8d_20_mpi_recursive_doubling_zfp_online_coll_rate8.log
```

## Plot Existing Logs

The following commands plot the earlier result directories listed above.
They are not a complete poster reproduction pipeline. Run them from the
repository root unless noted otherwise.

Fixed 8-GPU Wide_ResNet50_2 comparison:

```bash
python plotting/plot_fixed_gpu_hooks.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --gpu-dir 8GPUs_2nodes \
    --model-dir wideresnet50 \
    --run-dir run1 \
    --model-title Wide_ResNet50_2 \
    --gpus-label "8 GPUs" \
    --metric epoch_mean \
    --out plotting/wide_resnet50_2_8gpu_epoch_mean.pdf
```

Wide_ResNet50_2 fixed-global scaling:

```bash
python plotting/plot_hook_scaling.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --model-dir wideresnet \
    --model-title Wide_ResNet50_2 \
    --metric epoch_mean \
    --out plotting/wide_resnet50_2_scaling_epoch_mean_global.pdf
```

Wide_ResNet50_2 fixed-local scaling:

```bash
python plotting/plot_hook_scaling.py \
    --results-root experiments_results_lr001_LocalBS128 \
    --model-dir wideresnet \
    --model-title Wide_ResNet50_2 \
    --metric epoch_mean \
    --out plotting/wide_resnet50_2_scaling_epoch_mean_local.pdf
```

ResNeXt101_32x8d fixed-global scaling:

```bash
python plotting/plot_hook_scaling.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --model-dir resnext101_32x8d \
    --model-title ResNeXt101_32x8d \
    --metric epoch_mean \
    --out plotting/resnext101_32x8d_scaling_epoch_mean_global.pdf
```

ResNeXt101_32x8d fixed-local scaling:

```bash
python plotting/plot_hook_scaling.py \
    --results-root experiments_results_lr001_LocalBS128 \
    --model-dir resnext101_32x8d \
    --model-title ResNeXt101_32x8d \
    --metric epoch_mean \
    --out plotting/resnext101_32x8d_scaling_epoch_mean_local.pdf
```

Hook-work timing comparison:

```bash
python plotting/plot_hook_work_mean_comparison.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --model-dir wideresnet \
    --model-title Wide_ResNet50_2 \
    --gpus 8 \
    --out plotting/wide_resnet50_2_8gpu_hook_work_mean.pdf
```

Communication accounting breakdown:

```bash
python plotting/plot_epoch_comm_timing_breakdown.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --model-dir wideresnet \
    --model-title Wide_ResNet50_2 \
    --gpus 8 \
    --select-by combined \
    --out plotting/wide_resnet50_2_worst_comm_accounting.pdf
```

The plotting code skips the first epoch by default for epoch-time metrics
because the first epoch includes warmup and one-time setup effects.  Use
`--skip-first-epoch 0` if you intentionally want to include it.

## Hook Timing and NVTX

Enable per-epoch hook timing summaries:

```bash
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
```

Pass them through MPI if launching manually:

```bash
-env DDP_HOOK_TIMING="${DDP_HOOK_TIMING}" \
-env DDP_HOOK_TIMING_RANK0_ONLY="${DDP_HOOK_TIMING_RANK0_ONLY}"
```

Enable NVTX ranges for profiling:

```bash
export DDP_PROFILE_NVTX=1
```

When debugging hangs, first disable hook timing and NVTX:

```bash
export DDP_HOOK_TIMING=0
export DDP_PROFILE_NVTX=0
```

## Data Loader Requirement for Large Per-Rank Batches

For 16 GPUs with per-rank `batch_size=128`, CIFAR-10 produces a partial final
global batch:

```text
50000 / (16 x 128) = 24.4
```

Built-in DDP can handle this, but custom hooks such as `ring` can deadlock if
the final training step is uneven.  For fixed-local 16-GPU runs, set
`drop_last=True` in the training data loader:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

With `drop_last=True`, the 16-GPU `batch_size=128` run reports:

```text
Train batches: 24
```

instead of:

```text
Train batches: 25
```

Keep validation unchanged unless doing timing-only validation.

## Training Stability Settings

For earlier CIFAR-10 runs (these are not a complete specification of the
poster hyperparameters):

| Model | Recommended settings |
|---|---|
| `wide_resnet50_2` | `pretrained=True`, `cifar_stem=False`, `warmup_epochs=1`, `grad_clip=None` |
| `resnext101_32x8d` | `pretrained=True`, `cifar_stem=True`, `warmup_epochs=5`, `grad_clip=0.5` or `1.0` |

The communication hooks benchmark gradient synchronization behavior; pretrained
weights are used so large ImageNet-scale models train stably on CIFAR-10.
See `TRAINING_STABILITY.md` for details.

## Known Constraints

### Fixed GPU Buffer Addresses

Cray MPICH caches GPU memory to RDMA registration mappings.  MPI-facing tensors
should be preallocated and reused.  Avoid allocating new GPU tensors inside hook
bodies.

### GPU Stream Ordering

The poster integration adds stream-aware ZFP launches for CUDA and HIP.
Compression, communication, and consumption of decompressed tensors must
respect GPU stream dependencies. For older cuZFP builds, ensure the required
synchronization before MPI or PyTorch consumes a buffer.

### `static_graph=True`

DDP uses `static_graph=True` so bucket indices remain stable across backward
passes.

### Delete `__pycache__` After Editing Python

Before submitting after code edits:

```bash
rm -rf __pycache__
```

This avoids stale bytecode on compute nodes.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `MPI available: False` | PyTorch was not built with MPI | Use the MPI-enabled PyTorch build and set `PYTHONPATH` as in `envScript3.sh` |
| `import zfp_cuda` fails | ZFP install or library path missing | Run `build_zfp_cuda.sh`, then source `envScript3.sh` |
| Hang on final batch | Uneven last training batch | Use `drop_last=True` for large per-rank fixed-local runs |
| NaN loss in ResNeXt | Unstable deep-model training on CIFAR-10 | Use `pretrained=True`, warmup, and gradient clipping |
| ZFP hook crashes but uncompressed hook works | ZFP rate/build/runtime mismatch | Check `ZFP_RATE`, `ZFP_HOME`, and `LD_LIBRARY_PATH` |
| Multi-node MPI hook fails | CUDA-aware MPI/GTL not active | Confirm `MPICH_GPU_SUPPORT_ENABLED=1` and `LD_PRELOAD` includes `libmpi_gtl_cuda.so` |

## Future Work at the Poster Stage

The poster identifies larger models, compute-intensive workloads, and
hierarchical communication for HPC as future evaluation directions. These are
outside the evaluated scope documented here.

## Related Work and Attribution

**Poster:** *Evaluating All-Reduce Operations in DDP Training with FABRIC*.
Matilde Restelli, Ke Fan, Aadirupa Saha, John K. Holmen, and Sidharth Kumar.
Affiliations: University of Illinois Chicago, Oak Ridge National Laboratory,
and Temple University.
