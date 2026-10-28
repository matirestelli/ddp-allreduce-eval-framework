# WideResNet 32×32 · strong scaling · fixed GPU count 16

## Short alternative

WideResNet 32×32 · strong scaling · fixed GPU count 16. Cumulative hook work / epoch (s) is compared across effective global batch values 128, 512. Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Frontier · RCCL, Effective global batch 128: Hierarchical · ZFP online · rate8 has the lowest plotted value, 58.250. Frontier · RCCL, Effective global batch 512: Hierarchical · ZFP online · rate8 has the lowest plotted value, 17.302. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
