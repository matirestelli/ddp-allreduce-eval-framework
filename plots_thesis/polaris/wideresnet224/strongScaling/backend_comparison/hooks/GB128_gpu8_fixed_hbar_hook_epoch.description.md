# WideResNet 224×224 · strong scaling · effective global batch 128

## Short alternative

WideResNet 224×224 · strong scaling · effective global batch 128. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses hbar. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Polaris · MPI, GPU count (ranks) 8: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 21.214. Polaris · NCCL, GPU count (ranks) 8: Hierarchical · ZFP online · rate8 has the lowest plotted value, 33.840. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
