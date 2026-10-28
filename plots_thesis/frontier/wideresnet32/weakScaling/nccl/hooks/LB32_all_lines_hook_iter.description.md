# WideResNet 32×32 · weak scaling · local batch 32

## Short alternative

WideResNet 32×32 · weak scaling · local batch 32. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 8, 16, 32. Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / optimizer step (ms). Frontier · RCCL, GPU count (ranks) 8: Ring · Uncompressed has the lowest plotted value, 105.688. Frontier · RCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 178.495. Frontier · RCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 252.563. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
