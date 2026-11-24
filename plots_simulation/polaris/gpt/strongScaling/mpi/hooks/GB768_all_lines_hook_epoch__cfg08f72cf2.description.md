# GPT · strong scaling · effective global batch 768

## Short alternative

GPT · strong scaling · effective global batch 768. Cumulative hook work / epoch (s) is compared across gpu count (ranks) values 16, 32, 64. Polaris · MPI. A detailed description and all plotted values follow.

## Detailed description

The figure uses lines. Methods use colors, line styles and markers. The metric is Cumulative hook work / epoch (s). Polaris · MPI, GPU count (ranks) 16: Hierarchical · ZFP online · rate8 has the lowest plotted value, 16.527. Polaris · MPI, GPU count (ranks) 32: Hierarchical · ZFP online · rate8 has the lowest plotted value, 8.934. Polaris · MPI, GPU count (ranks) 64: Hierarchical · ZFP online · rate8 has the lowest plotted value, 5.371. Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is median. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. 
