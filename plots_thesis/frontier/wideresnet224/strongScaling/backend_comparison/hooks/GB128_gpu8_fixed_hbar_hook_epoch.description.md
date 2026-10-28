# WideResNet 224×224 · strong scaling · effective global batch 128

## Short alternative

WideResNet 224×224 · strong scaling · effective global batch 128. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses hbar. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 8: DDP diagnostics · default sync has the lowest plotted value, 7.579. Frontier · RCCL, GPU count (ranks) 8: DDP diagnostics · default sync has the lowest plotted value, 0.347. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
