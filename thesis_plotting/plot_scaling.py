#!/usr/bin/env python3
"""Scaling-focused line plots. Missing x values remain gaps, never zeros.

  python plot_scaling.py experiments_frontier --mode strong --model wideresnet32 --global-batch 128 --backend mpi
  python plot_scaling.py experiments_frontier --mode weak --model wideresnet32 --batch-per-rank 32 --backend mpi
  python plot_scaling.py experiments_frontier/gpt/strongScaling --direct gpu --normalize speedup --reference-gpus 16
  python plot_scaling.py experiments_frontier/gpt/strongScaling --direct all --families Direct Baseline
  python plot_scaling.py a.out --mode strong --metric tail --unit iter

Use --metric hook/tail/tail_pct for communication scaling; --normalize efficiency
for strong scaling. Read --help and README for every input/filter/layout option.
"""
from thesisplots.cli import main
if __name__=='__main__':main('time','lines',__doc__)
