# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 64. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 64: k-ring · k = 8 has the lowest plotted value, 20.241. Frontier · RCCL, GPU count (ranks) 64: k-ring · k = 8 has the lowest plotted value, 40.696. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
