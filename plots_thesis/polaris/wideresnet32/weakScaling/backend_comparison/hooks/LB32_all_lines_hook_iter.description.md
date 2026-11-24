# WideResNet 32×32 · weak scaling · local batch 32

## Short alternative

WideResNet 32×32 · weak scaling · local batch 32. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 8, 16, 32. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / optimizer step (ms). Polaris · MPI, GPU count (ranks) 8: Ring · ZFP online · rate10 has the lowest plotted value, 37.812. Polaris · MPI, GPU count (ranks) 16: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 62.310. Polaris · MPI, GPU count (ranks) 32: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 94.126. Polaris · NCCL, GPU count (ranks) 8: Hierarchical · ZFP online · rate8 has the lowest plotted value, 81.658. Polaris · NCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 98.530. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 116.401. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
