# WideResNet 32×32 · strong scaling · effective global batch 512

## Short alternative

WideResNet 32×32 · strong scaling · effective global batch 512. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8, 16, 32. Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Frontier · RCCL, GPU count (ranks) 8: DDP diagnostics · default has the lowest plotted value, 0.085. Frontier · RCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 17.302. Frontier · RCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 21.844. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
