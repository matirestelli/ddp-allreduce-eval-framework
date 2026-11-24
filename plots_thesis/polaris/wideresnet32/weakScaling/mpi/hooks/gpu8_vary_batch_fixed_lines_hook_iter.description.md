# WideResNet 32×32 · weak scaling · fixed GPU count 8

## Short alternative

WideResNet 32×32 · weak scaling · fixed GPU count 8. Cumulative hook work / optimizer step (ms) is compared across batch / rank values 8, 32, 128. Polaris · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / optimizer step (ms). Polaris · MPI, Batch / rank 8: Ring · ZFP online · rate10 has the lowest plotted value, 26.942. Polaris · MPI, Batch / rank 32: Ring · ZFP online · rate10 has the lowest plotted value, 37.812. Polaris · MPI, Batch / rank 128: Ring · ZFP online · rate10 has the lowest plotted value, 86.182. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
