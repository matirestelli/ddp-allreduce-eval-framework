# WideResNet 32×32 · weak scaling · fixed GPU count 32

## Short alternative

WideResNet 32×32 · weak scaling · fixed GPU count 32. Cumulative hook work / optimizer step (ms) is compared across batch / rank values 8, 32, 128. Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / optimizer step (ms). Polaris · NCCL, Batch / rank 8: Hierarchical · ZFP online · rate8 has the lowest plotted value, 108.158. Polaris · NCCL, Batch / rank 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 116.401. Polaris · NCCL, Batch / rank 128: Hierarchical · ZFP online · rate8 has the lowest plotted value, 172.732. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
