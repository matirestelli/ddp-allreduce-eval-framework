# WideResNet 32×32 · weak scaling · local batch 128

## Short alternative

WideResNet 32×32 · weak scaling · local batch 128. Median iteration time (ms) is compared across gpu count (ranks) values 8, 16, 32. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Median iteration time (ms). Frontier · MPI, GPU count (ranks) 8: Built-in baseline has the lowest plotted value, 151.625. Frontier · MPI, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 151.970. Frontier · MPI, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 136.490. Frontier · RCCL, GPU count (ranks) 8: Built-in baseline has the lowest plotted value, 154.755. Frontier · RCCL, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 214.310. Frontier · RCCL, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 247.110. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. 
