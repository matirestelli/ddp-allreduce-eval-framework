# Note 03 — Experiment setup: configuration, data, and job scripts for nanoGPT

## Configuration path

- Experiments are configured entirely through environment variables set in the PBS job script.
- `interface.py` reads them and overrides the matching `TrainingConfig` fields.
- A variable that is not set leaves the default untouched. This is why existing CNN job scripts run
  unchanged after the nanoGPT extension.
- The launcher prints every setting at the start of each run, so each log records exactly what it
  ran with.
- GPT-specific variables:
  - **Model:** `MODEL_NAME` (`gpt2_<layers>x<heads>x<width>`), `DATASET`, `BLOCK_SIZE`.
  - **Accumulation and precision:** `GRAD_ACCUM_GLOBAL`, `AMP`, `ALLOW_TF32`.
  - **Optimizer and schedule:** `ADAM_BETA1/2`, `WARMUP_ITERS`, `MIN_LR_RATIO`.
  - **Run length:** `ITERS_PER_EPOCH`, `VAL_MAX_BATCHES`.
  - **Checkpointing:** `SAVE_CHECKPOINT`.

## Model provenance

- **Source:** nanoGPT `model.py`, downloaded unmodified from https://github.com/karpathy/nanoGPT at
  commit `3adf61e` and stored as `nanogpt_model.py`.
- **Verification:** a model built through the framework has 124.37M parameters for 12×12×768, equal
  to nanoGPT. nanoGPT's own printed figure of 123.59M excludes position embeddings.

## Data preparation

- **Corpus:** TinyStories (Eldan & Li, 2023), from the HuggingFace hub (`roneneldan/TinyStories`).
- **Tokenization:** GPT-2 BPE via `tiktoken`, with an end-of-text token appended to each document.
- **Format:** flat `uint16` files `train.bin` / `val.bin`, identical to nanoGPT's OpenWebText
  preparation. The corpus is interchangeable with OpenWebText through `DATASET` alone.
- **When and where:** run once on a login node (compute nodes are not used for downloads). The
  output and the HuggingFace cache are placed on the Eagle project filesystem rather than the home
  filesystem, for capacity.
- **Why TinyStories:**
  - Time per iteration depends only on tensor shapes: micro-batch × sequence length, model size,
    vocabulary. It does not depend on corpus content.
  - The gradient volume per iteration is set by the model and the 50304-token vocabulary, not by the
    corpus.
  - TinyStories therefore yields the same communication workload as OpenWebText while needing
    ~1 GB of storage instead of ~17 GB (plus ~54 GB of download cache).
  - OpenWebText remains the faithful choice for convergence studies (e.g. effect of lossy gradient
    compression on validation loss).

## Strong-scaling experiment design

- **What is fixed:** the global work per optimizer step.
  - Micro-batch: 12 sequences × 1024 tokens.
  - Global micro-batches per step: 64.
  - Tokens per optimizer step: 786,432, at every GPU count.
- **How it is split:** each rank processes 64 / P micro-batches per step (P = number of GPUs):
  4 at 16 GPUs, 2 at 32, 1 at 64.
- **Communication:** one allreduce of the full gradient per step regardless of P (≈ 498 MB fp32 for
  12×12×768).
- **Interpretation:** increasing P reduces per-GPU compute but not per-step communication volume, so
  the communication share of each iteration grows with scale. This is the regime in which the
  communication hooks are evaluated.
- **Job layout:** one GPU count per job, with all hooks run back to back in the same allocation (same
  nodes and network placement), which keeps hook-to-hook comparisons fair.

## Hooks compared

| Hook | Role |
|---|---|
| `none` | Baseline: PyTorch DDP's built-in allreduce through the MPI backend |
| `ring`, `recursive_doubling` | Uncompressed custom allreduce algorithms |
| `*_zfp_naive`, `*_zfp_online_coll` | ZFP-compressed variants at rates 16 and 8 bits/value |

## Training recipe (nanoGPT `train.py` defaults unless noted)

- **Optimizer:** AdamW, learning rate 6 × 10⁻⁴, betas (0.9, 0.95), weight decay 0.1 (2-D weights
  only), gradient clipping at 1.0.
- **Numerics:** bf16 autocast with fp32 parameters and gradients; TF32 matmuls.
- **Schedule:** cosine decay to 0.1 × lr after a linear warmup.
- **Shortened for timing runs:** warmup is 20 iterations instead of 2000.
- **Run length:** each run is 3 epochs of 50 optimizer steps. The first epoch is excluded as warm-up
  (allocator, cuDNN/cuBLAS autotuning, hook buffer allocation). The remaining 100 iterations give the
  median time per iteration.
- **Effect on results:** the learning-rate settings affect convergence only, not time per iteration.

## Validation procedure before sweeps

1. One-node smoke test (4 GPUs, 2 micro-steps per rank). Checks:
   - the synchronized warm-up backward runs;
   - the accumulation configuration printed at startup is correct;
   - the initial loss is ≈ ln(50304) ≈ 10.8 and decreasing;
   - peak memory is within the 40 GB A100 limit.
2. Gradient-consistency check across ranks with a custom hook (`verify_gradient_sync`).
3. Full sweep.