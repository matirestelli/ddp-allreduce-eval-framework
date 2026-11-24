# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 16, 32, 64. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 16: k-ring · k = 8 has the lowest plotted value, 56.592. Frontier · MPI, GPU count (ranks) 32: k-ring · k = 8 has the lowest plotted value, 33.105. Frontier · MPI, GPU count (ranks) 64: k-ring · k = 8 has the lowest plotted value, 20.241. Frontier · RCCL, GPU count (ranks) 16: k-ring · k = 8 has the lowest plotted value, 67.332. Frontier · RCCL, GPU count (ranks) 32: k-ring · k = 8 has the lowest plotted value, 46.560. Frontier · RCCL, GPU count (ranks) 64: k-ring · k = 8 has the lowest plotted value, 40.696. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
