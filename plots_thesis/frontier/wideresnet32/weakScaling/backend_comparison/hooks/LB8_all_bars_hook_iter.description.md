# WideResNet 32×32 · weak scaling · local batch 8

## Short alternative

WideResNet 32×32 · weak scaling · local batch 8. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 8, 16, 32. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / optimizer step (ms). Frontier · MPI, GPU count (ranks) 8: Recursive doubling · Uncompressed has the lowest plotted value, 29.498. Frontier · MPI, GPU count (ranks) 16: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 45.517. Frontier · MPI, GPU count (ranks) 32: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 57.292. Frontier · RCCL, GPU count (ranks) 8: Ring · Uncompressed has the lowest plotted value, 65.261. Frontier · RCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 147.902. Frontier · RCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 215.942. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
