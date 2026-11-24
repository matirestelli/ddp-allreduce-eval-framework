# WideResNet 32×32 · weak scaling · local batch 8

## Short alternative

WideResNet 32×32 · weak scaling · local batch 8. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 32. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / optimizer step (ms). Polaris · MPI, GPU count (ranks) 32: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 81.615. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 108.158. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
