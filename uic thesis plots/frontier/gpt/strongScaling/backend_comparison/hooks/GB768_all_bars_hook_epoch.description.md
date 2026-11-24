# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 16, 32, 64. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 16: Ring · Uncompressed has the lowest plotted value, 56.868. Frontier · MPI, GPU count (ranks) 32: Ring · Uncompressed has the lowest plotted value, 33.672. Frontier · MPI, GPU count (ranks) 64: Ring · Uncompressed has the lowest plotted value, 21.099. Frontier · RCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 72.857. Frontier · RCCL, GPU count (ranks) 32: Ring · Uncompressed has the lowest plotted value, 55.796. Frontier · RCCL, GPU count (ranks) 64: Ring · Uncompressed has the lowest plotted value, 49.403. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
