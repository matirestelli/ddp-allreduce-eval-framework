# WideResNet 32×32 · strong scaling · fixed GPU count 8

## Short alternative

WideResNet 32×32 · strong scaling · fixed GPU count 8. Cumulative hook work / epoch (s) is compared across effective global batch values 128, 512. Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Frontier · RCCL, Effective global batch 128: DDP diagnostics · default has the lowest plotted value, 0.332. Frontier · RCCL, Effective global batch 512: DDP diagnostics · default has the lowest plotted value, 0.085. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
