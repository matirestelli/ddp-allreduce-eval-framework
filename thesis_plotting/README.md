# Thesis and paper plotting tools

This folder is self-contained. Copy it next to `experiments_frontier`,
`experiments_frontier_rccl`, `experiments_polaris`, and
`experiments_polaris_nccl`, or pass their absolute paths from anywhere.
The original plotting tools and experiment logs are not modified.

## Install and run

Requires Python 3.10+ and matplotlib:

```bash
python3 -m pip install -r /path/to/thesis_plotting/requirements.txt
```

From the directory containing the experiment folders:

```bash
python3 /path/to/thesis_plotting/plot_all_frontier.py . --output-root plots_thesis
python3 /path/to/thesis_plotting/plot_all_polaris.py . --output-root plots_thesis
```

The same commands from anywhere, using this session's source directory:

```bash
python3 plot_all_frontier.py /Users/matilderestelli/Desktop/Thesis/Fall26/code_oct26 --output-root ../plots_thesis
python3 plot_all_polaris.py /Users/matilderestelli/Desktop/Thesis/Fall26/code_oct26 --output-root ../plots_thesis
```

Run the commands above from inside `thesis_plotting`, or supply each script's
absolute path. Paths with spaces must be quoted. All scripts show their full
options and commented usage examples with `--help`.

## Choose a script

| Script | Default figure | Other supported layouts |
|---|---|---|
| `plot.py` | General metric/layout selector | Same options as every preset |
| `plot_time.py` | Shaded family bars for training/iteration time | `--layout lines` or `hbar` |
| `plot_scaling.py` | Lines emphasizing strong/weak scaling | `--metric hook`, `tail`, `tail_pct` also supported |
| `plot_hook.py` | Cumulative instrumented hook work | Bars, lines, horizontal bars |
| `plot_tail.py` | Paired horizontal train-only time and exposed-tail bars (select one GPU count) | Optional tail-only bars/lines; optional tail percentage |
| `plot_compare.py` | Separate backend/system panels | Bars, lines, horizontal comparisons |
| `parse_results.py` | Audit/normalized CSV, without plotting | All raw files and partial runs included in audit |
| `plot_all_frontier.py` | Entire available Frontier matrix | All filters and batch controls |
| `plot_all_polaris.py` | Entire available Polaris matrix | Identical behavior, partial data supported |

## Input formats

Every plotting script accepts **one or more** folders, individual `.out`/`.log`
files, per-epoch training CSV files, or exported `parsed_runs.csv` files. Folder
scanning is recursive. Existing `plots`, `plots_thesis`, calibration tuning,
`old`, `archive`, and `.git` directories are not scanned recursively. To inspect
an archived individual file, pass its file path explicitly. No arbitrary CSV
column mapping is guessed: use the training column names below or normalized CSV.

Supported per-epoch CSV names include `epoch`, `model`, `dataset`, `backend`,
`algorithm`, `zfp_rate`, `world_size`/`ranks`, `batch_size`/`batch_per_rank`,
`global_batch_size`/`global_batch`, `epoch_train_time_s`/`train_s`,
`t_iter_median_ms`/`iter_ms`, `hook_work_total_ms`, `hook_work_mean_ms`,
`hook_calls`, `tail_total_ms`, `tail_mean_ms`, `tail_steps`/`steps`.
Use `num_epochs` to allow detection of truncated CSVs. GPT CSVs should also supply
`effective_batch`, `accum_global`, `block_size` and/or `tokens_per_iter`.
Without `num_epochs`, completion is inferred only through the last CSV epoch.
For detailed hook/step metrics the corresponding measured totals and step counts
must actually exist; absent columns do not become zero.

Examples:

```bash
python3 parse_results.py /path/to/code_oct26 --out ../plots_thesis/audit
python3 plot_time.py /path/to/a.out --mode strong --assume-system frontier --global-batch 128 --gpus 16
python3 plot_time.py /path/to/a.log --mode weak --assume-system polaris --batch-per-rank 32
python3 plot_time.py ../plots_thesis/audit/parsed_runs.csv --mode strong --system frontier --model gpt --backend mpi
```

A raw file without identifiable system/folder metadata needs `--assume-system`;
a raw file outside a strong/weak folder needs `--mode`. Header rank counts take
precedence over filenames. The audit flags mismatches and lists incomplete runs.
Files whose measured backend contradicts an explicitly named `_rccl`/`_nccl`
experiment folder are quarantined from default plots and preserved in the audit.

## Typical configurations

### Strong scaling: fixed global batch, all GPU counts and hooks

```bash
python3 plot_time.py /path/to/code_oct26 --system frontier --mode strong --model wideresnet32 --backend mpi --global-batch 128 --prefer-rate8 --no-diagnostics
python3 plot_time.py /path/to/code_oct26 --system frontier --mode strong --model wideresnet32 --backend mpi --global-batch 512
python3 plot_time.py /path/to/code_oct26 --system frontier --mode strong --model wideresnet224 --backend rccl --global-batch 128
python3 plot_time.py /path/to/code_oct26 --system frontier --mode strong --model gpt --backend mpi --global-batch 768 --direct gpu
```

For GPT, `--global-batch` means the effective **accumulated** global batch,
not simply ranks × microbatch. Block length must also match. Example `768`
corresponds to batch/rank12 × global accumulation64 in the supplied GPT logs.

### One configuration: one GPU count

```bash
python3 plot_time.py experiments_frontier/wideresnet/cifar10/strongScaling --global-batch 128 --gpus 16
python3 plot_time.py experiments_frontier/gpt/strongScaling --gpus 32 --layout hbar
```

### Weak scaling: fixed local batch

```bash
python3 plot_time.py experiments_frontier/wideresnet/cifar10/weakScaling --batch-per-rank 8
python3 plot_time.py experiments_frontier/wideresnet/cifar10/weakScaling --batch-per-rank 32 --gpus 16
python3 plot_scaling.py experiments_frontier/wideresnet/cifar10/weakScaling --batch-per-rank 128
```

Weak scaling always uses **iteration time**, not epoch duration. A decreasing
number of iterations per epoch is not a weak-scaling speedup.

### Scaling-focused lines, speedup and efficiency

```bash
python3 plot_scaling.py experiments_frontier/wideresnet/cifar10/strongScaling --global-batch 128
python3 plot_scaling.py experiments_frontier/gpt/strongScaling --direct gpu --normalize speedup --reference-gpus 16
python3 plot_scaling.py experiments_frontier/gpt/strongScaling --direct gpu --normalize efficiency --reference-gpus 16
python3 plot_scaling.py experiments_frontier/gpt/strongScaling --direct all --families Direct Baseline --xscale log2
```

Speedup is `time(reference)/time(P)` for the **same method**. Efficiency divides
that by `P/reference`. These are relative to the selected reference count, not a
claim of single-GPU efficiency. A missing reference measurement is an error,
not an invented value. Missing intermediate measurements break a line.
`--normalize baseline` instead divides by the matching built-in baseline at each
rank count. It measures hook overhead, not parallel scaling efficiency.

### Vary batch at fixed GPU count

```bash
python3 plot_time.py experiments_frontier/wideresnet/cifar10/strongScaling --gpus 16 --x global_batch
python3 plot_time.py experiments_frontier/wideresnet/cifar10/weakScaling --gpus 16 --x batch_per_rank
python3 plot_hook.py experiments_frontier/wideresnet/cifar10/weakScaling --gpus 16 --x batch_per_rank --unit iter
```

### Hook work and exposed tail

```bash
python3 plot_hook.py experiments_frontier/wideresnet/cifar10/strongScaling --global-batch 128 --unit epoch
python3 plot_hook.py experiments_frontier/wideresnet/cifar10/weakScaling --batch-per-rank 32 --unit iter --layout lines
python3 plot_hook.py experiments_frontier/gpt/strongScaling --gpus 32 --unit call --layout hbar
python3 plot_tail.py experiments_frontier/gpt/strongScaling --gpus 32 --unit iter --layout hbar
python3 plot_tail.py experiments_frontier/gpt/strongScaling --unit epoch --layout lines
python3 plot_tail.py experiments_frontier/gpt/strongScaling --gpus 32 --metric tail_pct --layout hbar
```

| Metric/unit | Definition after excluding epoch1 |
|---|---|
| Strong time | Reported training-only epoch median; per-epoch median if no summary |
| Weak time | Median of the per-epoch iteration medians; not a pooled all-iteration median |
| Hook, `epoch` | Median cumulative `work_total_ms / 1000`, seconds/epoch |
| Hook, `iter` | Median `work_total_ms / steps`, milliseconds/optimizer step |
| Hook, `call` | Median `work_mean_ms`, milliseconds/hook call or bucket |
| Tail, `epoch` | Median `tail_total_ms / 1000`, seconds/epoch |
| Tail, `iter` | Median `tail_mean_ms`, milliseconds/optimizer step |
| `tail_pct` | Median per-epoch `100 × tail_total_ms / (train_s × 1000)` |

`--unit auto` uses epoch units for strong mode and iteration units for weak mode.
Cumulative hook work can overlap; **do not interpret it as a disjoint part of
wall time or stack it with compute**. Exposed tail is not all communication time.
A built-in baseline or native single-node fallback may have no instrumentation;
that series is omitted from hook/tail figures, with a reason in the JSON sidecar.
No missing measurement is set to zero, and unexpectedly large tails are not
silently clipped. No error bars pretend that epochs are independent job repeats.

### Backends and systems

```bash
python3 plot_compare.py experiments_frontier experiments_frontier_rccl --mode strong --model wideresnet32 --global-batch 128
python3 plot_compare.py experiments_frontier experiments_polaris --mode weak --model wideresnet32 --backend mpi --batch-per-rank 32 --layout lines
python3 plot_compare.py experiments_frontier experiments_frontier_rccl experiments_polaris experiments_polaris_nccl --mode strong --model gpt --global-batch 768 --normalize baseline
```

Default comparison panels stack vertically, preserving readable text. Use
`--columns 2 --width 12` for a wider two-column figure; choose dimensions appropriate
to your actual publication/page. `--share-y` is on by default; disable with
`--no-share-y`. Different models, resolutions, fixed workloads or GPT block sizes
are rejected rather than mixed. Strong rank sweeps also reject different learning
rates. Comparing different batch sizes is intentional only with batch on x.
The software does not establish that all other training settings are identical:
review the original logs for precision, optimizer, model implementation and
profiling changes before making scientific cross-system claims.

## Methods and families

Canonical method IDs for `--methods`:

```text
none
ring
ring_zfp_naive:16
ring_zfp_online_coll:16
ring_zfp_online_coll:10
ring_zfp_online_coll:8
recursive_doubling
recursive_doubling_zfp_naive:16
recursive_doubling_zfp_online_coll:16
recursive_doubling_zfp_online_coll:8
hierarchical_zfp_online_coll:16
hierarchical_zfp_online_coll:8
direct_p
direct_t4 direct_t8 direct_t16 direct_t32 direct_t64
kring_k8
default default_sync default_clone default_cpu_stage
```

```bash
python3 plot_time.py experiments_frontier/gpt/strongScaling --methods none ring recursive_doubling kring_k8
python3 plot_time.py experiments_frontier/gpt/strongScaling --families Baseline Ring "Recursive doubling"
python3 plot_time.py experiments_frontier_rccl/gpt/strongScaling --rates 8 --direct omit
```

- `--direct gpu` (default): `direct_t16` at16 ranks, `direct_t32` at32, etc. One
  **Direct t=GPU count** series, explicitly not a fixed-t scaling experiment.
- `--direct all`: every measured fixed-t configuration retains its own series.
- `--direct omit`: hide Direct.
- RCCL online launch aliases are grouped as Hierarchical **only when the actual
  internal hierarchical label or explicit native-fallback message establishes
  that implementation**. The raw launch hook stays in provenance.
- Diagnostic hooks are included by default. `--no-diagnostics` hides them.
- All measured ZFP rates are distinct. `--prefer-rate8` removes rate10 only where
  a corresponding rate8 measurement exists. A rate10 measurement is never called
  rate8. `--rates 8` explicitly selects rate8 plus uncompressed methods.
- Ring uses green shades; RD blue; Hierarchical purple; baseline gray; Direct
  orange and k-ring brown. White hatches, family labels and different line markers
  supplement hue. `--palette solid` uses one dark fill per family instead.

## Batch runners: generate everything

Default output includes all observed strong/weak model/backend/workload cells,
all-rank bars and lines, each fixed-rank bar/horizontal comparison, matching
MPI/native-backend comparison panels, and varying-
batch figures where multiple batches exist. Metrics: time, hook and tail. Tail defaults to paired horizontal comparisons for each fixed GPU count; tail-only bars/curves are not generated in the default batch.
Missing metrics are recorded as `no_metric` in the manifest; unfinished runs are
excluded by default. A fresh invocation rereads the current downloaded results.

```bash
python3 plot_all_frontier.py /path/to/code_oct26 --dry-run
python3 plot_all_polaris.py /path/to/code_oct26 --scopes all --metrics time hook tail --layouts bars lines
python3 plot_all_frontier.py /path/to/code_oct26 --model gpt --direct all --no-diagnostics
python3 plot_all_frontier.py /path/to/code_oct26 --normalizations absolute baseline --prefer-rate8
python3 plot_all_polaris.py /path/to/code_oct26 --max-plots 12 --formats png
python3 plot_all_polaris.py /path/to/code_oct26 --skip-existing
```

`--no-comparisons` omits the extra backend-comparison figures. Cross-system
comparisons use `plot_compare.py` with both systems as input.

`--include-partial` is an explicit exploratory override; it does not make a run
complete. Completion and source references remain in parsed data. `--skip-existing`
is for intentional resume, **not** automatic freshness detection; rerun normally
after updating results. `--dry-run` inventories all planned figures without making
images. `--max-plots` truncation is stated in the manifest. Inputs whose scaling
mode is unknown are not included by batch runners.

Output structure:

```text
plots_thesis/
├── frontier/
│   ├── index.html
│   ├── parsed_runs.csv
│   ├── input_issues.json
│   ├── manifest.json
│   ├── wideresnet32/
│   │   ├── strongScaling/mpi/
│   │   │   ├── normal/    # shaded training-time bars and horizontal comparisons
│   │   │   ├── lines/     # training-time scaling curves
│   │   │   ├── hooks/     # hook bars, curves, horizontal comparisons
│   │   │   └── tails/     # paired training-time / exposed-tail horizontal comparisons
│   │   ├── strongScaling/backend_comparison/...
│   │   └── weakScaling/nccl/...
│   ├── wideresnet224/...
│   └── gpt/...
└── polaris/              # same organization, only available configurations
```

When different models/datasets/block sizes share a short filename tag, a stable
`__cfgXXXXXXXX` suffix prevents overwriting; the JSON identifies its configuration.

Every figure stem produces `.png`, `.pdf` by default, plus:

- `.csv`: the values actually plotted, with repeat counts.
- `.json`: sources/line numbers, raw hooks, values, metric units, omissions,
  normalization, dimensions and color-contrast checks.
- `.description.md`: short alternative and detailed-description **drafts**.
- `.html`: the figure with its alternative, a detailed description and a structured
  table of all plotted data. Each system's index links these pages.

Default repeat policy is the median across differing records with the same
configuration. Exact mirrored log/output metric traces are deduplicated.
Aliases with different measured timings are repeated measurements of the same
implementation, not separate algorithms. `--repeats latest` instead chooses by
source modification time, path and line; timestamps of copied logs may be equal,
so use only when this selection policy is appropriate. Nothing chooses the fastest
run. Differences in epoch count remain in per-point provenance.

## Style/export configuration

```bash
python3 plot_time.py experiments_frontier/gpt/strongScaling --width 9 --font-size 11 --formats png pdf svg --dpi 300 --out ../plots_thesis/custom/gpt_mpi
python3 plot_time.py experiments_frontier/gpt/strongScaling --width 6.5 --no-family-labels --no-diagnostics --methods none kring_k8 direct_p
python3 plot_time.py experiments_frontier/gpt/strongScaling --config profiles/paper.json
```

`--out` is a stem, not a directory. Relative paths are relative to your current
working directory. `--output-root` is the root directory for batch runners.
Command-line flags override JSON profile defaults. `--width`, `--height`,
`--font-size`, `--ymax`, `--title`, `--columns`, `--palette`, `--xscale`, and
`--family-labels/--no-family-labels` are available on every plotting script.

## UIC accessibility workflow

The scripts implement checked color pairs, white patterns and dark outlines,
non-color family cues, sans-serif text and 11-point default text at the exported
physical size. Lines use family-specific markers and line styles. Contrast results
are written alongside each figure. Very dense selections still require visual
review: filter methods, use separate panels, or change the physical dimensions.
Do not shrink a 9-inch landscape export into a 6.5-inch portrait slot without
regenerating it at that size. Some all-hook figures are taller than one thesis
page; use a narrower method selection or separate figures as appropriate.

These are **figure packages, not a certification of the thesis**. Insert and
review the generated description drafts in the authoring document, add figure
numbering/captions and correct reading order, preserve semantic structure, and
check the final PDF/EPUB. Do not rely on palette contrast checks alone.

The [UIC policy](https://grad.uic.edu/academic-support/exams-defense/theses-dissertations-accessibility/)
and linked [complex-image guidance](https://www.w3.org/WAI/tutorials/images/complex/)
are the basis for this workflow. For the linked UIC LaTeXML-to-EPUB route, also
export PNG: its template documents limitations for PDF/SVG alt text in that
conversion workflow. Keep vector PDF/SVG masters for suitable paper/PDF workflows.
The script's PNG/SVG/HTML alternatives do not automatically tag your thesis PDF.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

Tests cover real hook aliases, partial jobs, exit status, filename/rank mismatch,
per-call vs per-step units, warmup exclusion, mirrored files, accumulated GPT batch,
native fallbacks, CSV roundtripping, and palette contrast. Source metric provenance
is retained so publication values can be independently checked.


### Corrected exposed-tail comparison (26 October)

The default `plot_tail.py` view now matches the paired horizontal example: for each hook, a gray dotted train-only bar above a family-colored/patterned exposed-tail bar. Both start at zero and are **not stacked**. Select one configuration/GPU count. Matching comparison panels share horizontal limits. Labels identify every method; green/blue/purple shades and white patterns remain consistent with the other plots.

```bash
# One configuration (strong: seconds per epoch)
python3 plot_tail.py /path/to/code_oct26 --system frontier --model wideresnet32 --mode strong --backend mpi --global-batch 128 --gpus 16 --no-diagnostics --rates 8 16
# Frontier and Polaris side by side; 18 inches is the actual export width
python3 plot_tail.py /path/to/code_oct26 --model wideresnet32 --mode strong --backend mpi --global-batch 128 --gpus 16 --no-diagnostics --rates 8 16 --columns 2 --width 18
# Weak scaling: paired means per measured optimizer step (milliseconds)
python3 plot_tail.py /path/to/code_oct26 --system frontier --model wideresnet32 --mode weak --backend mpi --batch-per-rank 32 --gpus 16
# Optional tail-only scaling curve
python3 plot_tail.py /path/to/code_oct26 --system frontier --model wideresnet32 --mode strong --backend mpi --global-batch 128 --layout lines
# Legacy tail-only horizontal view
python3 plot_tail.py /path/to/code_oct26 --system frontier --model wideresnet32 --mode strong --backend mpi --global-batch 128 --gpus 16 --no-tail-context
```

Paired bars use exactly the same recorded post-warmup epochs and contributing runs for both quantities. Epoch context is median train-only epoch seconds and median exposed-tail seconds. In `iter` units, each epoch contributes `train_s * 1000 / steps` and `tail_total_ms / steps`; both are **means per measured optimizer step**. The gray bar does not use the separate median iteration latency statistic. Repeats follow `--repeats`. Missing context or tail values are omitted, never filled with zero. CSV/JSON/HTML companions include `training_value` alongside the exposed-tail `value`.

System batch defaults now generate only paired fixed-configuration horizontal figures for `tail`. `--scopes all` alone does not produce these because no single GPU configuration was requested; retain the default `--scopes all fixed` or use `--scopes fixed`. Explicit `--no-tail-context` restores tail-only layouts in the batch runner. Hook-work figures are unchanged. Earlier tail-only outputs are retained in `_previous_tail_plots` when the corrected output tree is installed, and are excluded from the current indexes.

Keep the exported physical size in mind: an 18-inch two-panel example must not be shrunk into a portrait thesis text block. For insertion at ordinary thesis width, use the default single-column panels or separate figures, retaining the minimum font size.
