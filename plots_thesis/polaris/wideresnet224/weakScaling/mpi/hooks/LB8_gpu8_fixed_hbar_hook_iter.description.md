# WideResNet 224×224 · weak scaling · local batch 8

## Short alternative

WideResNet 224×224 · weak scaling · local batch 8. Cumulative hook work / optimizer step (ms) is compared across gpu count (ranks) values 8. Polaris · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses hbar. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / optimizer step (ms). Polaris · MPI, GPU count (ranks) 8: Ring · ZFP online · rate16 has the lowest plotted value, 39.128. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
