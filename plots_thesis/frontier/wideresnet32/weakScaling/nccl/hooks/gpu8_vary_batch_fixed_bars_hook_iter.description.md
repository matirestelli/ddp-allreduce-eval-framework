# WideResNet 32×32 · weak scaling · fixed GPU count 8

## Short alternative

WideResNet 32×32 · weak scaling · fixed GPU count 8. Cumulative hook work / optimizer step (ms) is compared across batch / rank values 8, 32, 128. Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / optimizer step (ms). Frontier · RCCL, Batch / rank 8: Ring · Uncompressed has the lowest plotted value, 65.261. Frontier · RCCL, Batch / rank 32: Ring · Uncompressed has the lowest plotted value, 105.688. Frontier · RCCL, Batch / rank 128: Recursive doubling · Uncompressed has the lowest plotted value, 178.037. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
