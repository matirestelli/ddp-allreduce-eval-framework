# WideResNet 224×224 · strong scaling · effective global batch 128

## Short alternative

WideResNet 224×224 · strong scaling · effective global batch 128. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8, 16, 32. Frontier · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 8: DDP diagnostics · default sync has the lowest plotted value, 7.579. Frontier · MPI, GPU count (ranks) 16: DDP diagnostics · default sync has the lowest plotted value, 14.650. Frontier · MPI, GPU count (ranks) 32: DDP diagnostics · default sync has the lowest plotted value, 19.315. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
