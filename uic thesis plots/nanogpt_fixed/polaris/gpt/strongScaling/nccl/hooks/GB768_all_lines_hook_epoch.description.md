# NanoGPT · Polaris · strong scaling

## Short alternative

NanoGPT · Polaris · strong scaling. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 16, 32, 64. Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Polaris · NCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 14.565. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 14.024. Polaris · NCCL, GPU count (ranks) 64: Hierarchical · ZFP online · rate8 has the lowest plotted value, 7.596. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
