# WideResNet 32×32 · weak scaling · local batch 32

## Short alternative

WideResNet 32×32 · weak scaling · local batch 32. Mean time per optimizer step (ms) is compared across gpu count (ranks) values 8. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses hbar. Families and methods are identified by text and patterns in addition to color. The metric is Mean time per optimizer step (ms). Frontier · MPI, GPU count (ranks) 8: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 20.308. Frontier · RCCL, GPU count (ranks) 8: Hierarchical · ZFP online · rate16 has the lowest plotted value, 0.184. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Exposed tail is the instrumented tail, not total communication time.  Each method has two separate horizontal bars starting at zero: a light gray dotted bar for train-only time and a family-colored patterned bar for exposed tail. They are not stacked. Both bars use the same post-warmup epochs and repeat selection. In iteration units, both are means per measured optimizer step, not median iteration latency. Shared horizontal limits support comparison across panels. The baseline is omitted when tail instrumentation is unavailable.
