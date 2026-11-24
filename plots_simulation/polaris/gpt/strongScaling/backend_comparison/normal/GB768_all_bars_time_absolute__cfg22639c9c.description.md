# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Train-only epoch time (s) is compared across gpu count (ranks) values 16, 32, 64. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Train-only epoch time (s). Polaris · MPI, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 41.670. Polaris · MPI, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 21.100. Polaris · MPI, GPU count (ranks) 64: Built-in baseline has the lowest plotted value, 10.550. Polaris · NCCL, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 27.100. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate16 has the lowest plotted value, 22.900. Polaris · NCCL, GPU count (ranks) 64: Built-in baseline has the lowest plotted value, 10.900. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. 
