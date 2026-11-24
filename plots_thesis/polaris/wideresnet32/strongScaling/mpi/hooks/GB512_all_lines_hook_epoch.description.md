# WideResNet 32×32 · strong scaling · effective global batch 512

## Short alternative

WideResNet 32×32 · strong scaling · effective global batch 512. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8, 16, 32. Polaris · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Polaris · MPI, GPU count (ranks) 8: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 5.973. Polaris · MPI, GPU count (ranks) 16: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 6.104. Polaris · MPI, GPU count (ranks) 32: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 8.454. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
