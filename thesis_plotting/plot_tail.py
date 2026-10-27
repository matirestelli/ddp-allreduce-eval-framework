#!/usr/bin/env python3
"""Paired train-only / exposed-tail horizontal comparisons (default), or optional tail scaling lines.

  python plot_tail.py experiments_frontier --mode strong --model wideresnet32 --global-batch 128 --backend mpi --gpus 16
  python plot_tail.py a.out --mode strong --gpus 16 --layout hbar --unit iter
  python plot_tail.py a.out --mode strong --layout lines --unit epoch
  python plot_tail.py a.out --mode strong --metric tail_pct --layout hbar --gpus 16

Tail is per optimizer step (iter) or total exposed tail per epoch (epoch).
It is not total communication time. Baselines without instrumentation and
single-node native fallbacks are omitted rather than set to zero. No clipping
of implausible values: inspect provenance. See --help and README for all cases.
"""
from thesisplots.cli import main
if __name__=='__main__':main('tail','hbar',__doc__)
