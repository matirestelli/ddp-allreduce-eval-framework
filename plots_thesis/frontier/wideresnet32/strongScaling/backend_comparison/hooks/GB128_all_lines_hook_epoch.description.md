# WideResNet 32×32 · strong scaling · effective global batch 128

## Short alternative

WideResNet 32×32 · strong scaling · effective global batch 128. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 8, 16, 32. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Frontier · MPI, GPU count (ranks) 8: Ring · Uncompressed has the lowest plotted value, 14.213. Frontier · MPI, GPU count (ranks) 16: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 19.847. Frontier · MPI, GPU count (ranks) 32: Recursive doubling · ZFP online · rate8 has the lowest plotted value, 26.313. Frontier · RCCL, GPU count (ranks) 8: DDP diagnostics · default has the lowest plotted value, 0.332. Frontier · RCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 58.250. Frontier · RCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 83.497. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
