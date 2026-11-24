# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Train-only epoch time (s) is compared across gpu count (ranks) values 32. Panels compare Polaris · MPI, Polaris · NCCL. A detailed description and all plotted values follow.

## Detailed description

The figure uses bars. Families and methods are identified by text and patterns in addition to color. The metric is Train-only epoch time (s). Polaris · MPI, GPU count (ranks) 32: Built-in baseline has the lowest plotted value, 21.100. Polaris · NCCL, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 15.530. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero.  Explicit predictions are included as estimates, not experimental measurements. Open markers and dashed connecting segments identify predictions in line plots; Bars use normal solid outlines; the figure note and data table disclose estimated values without letter annotations. The data table labels every point as measured or predicted.
