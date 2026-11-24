# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Train-only epoch time (s) is compared across gpu count (ranks) values 16, 32, 64. Panels compare Frontier · MPI, Frontier · RCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Train-only epoch time (s). Frontier · MPI, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 63.660. Frontier · MPI, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 32.050. Frontier · MPI, GPU count (ranks) 64: Built-in baseline has the lowest plotted value, 16.170. Frontier · RCCL, GPU count (ranks) 16: Built-in baseline has the lowest plotted value, 70.890. Frontier · RCCL, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 43.880. Frontier · RCCL, GPU count (ranks) 64: Built-in baseline has the lowest plotted value, 27.790. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. 
