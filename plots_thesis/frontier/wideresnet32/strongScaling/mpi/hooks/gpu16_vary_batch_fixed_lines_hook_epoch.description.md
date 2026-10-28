# WideResNet 32×32 · strong scaling · fixed GPU count 16

## Short alternative

WideResNet 32×32 · strong scaling · fixed GPU count 16. Cumulative hook work / epoch (s) is compared across effective global batch values 128, 512. Frontier · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Frontier · MPI, Effective global batch 128: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 19.847. Frontier · MPI, Effective global batch 512: Ring · Uncompressed has the lowest plotted value, 5.903. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
