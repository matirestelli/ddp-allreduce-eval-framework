# WideResNet 32×32 · strong scaling · effective global batch 128

## Short alternative

WideResNet 32×32 · strong scaling · effective global batch 128. Train-only epoch time (s) is compared across gpu count (ranks) values 8, 16, 32. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Train-only epoch time (s). Polaris · MPI, GPU count (ranks) 8: Ring · ZFP online · rate10 has the lowest plotted value, 15.900. Polaris · MPI, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 20.150. Polaris · MPI, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 22.250. Polaris · NCCL, GPU count (ranks) 8: Hierarchical · ZFP online · rate8 has the lowest plotted value, 33.435. Polaris · NCCL, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 40.080. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 46.830. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. 
