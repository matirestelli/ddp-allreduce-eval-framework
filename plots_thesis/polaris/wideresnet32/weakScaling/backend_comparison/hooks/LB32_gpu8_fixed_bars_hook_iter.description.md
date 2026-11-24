# WideResNet 32×32 · weak scaling · local batch 32

## Short alternative

WideResNet 32×32 · weak scaling · local batch 32. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 8. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / optimizer step (ms). Polaris · MPI, GPU count (ranks) 8: Ring · ZFP online · rate10 has the lowest plotted value, 37.812. Polaris · NCCL, GPU count (ranks) 8: Hierarchical · ZFP online · rate8 has the lowest plotted value, 81.658. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
