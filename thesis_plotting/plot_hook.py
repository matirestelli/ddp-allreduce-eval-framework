#!/usr/bin/env python3
"""Cumulative hook-work comparisons. Missing baseline timing is omitted.

  python plot_hook.py experiments_frontier --mode strong --model wideresnet32 --global-batch 128 --backend mpi --unit epoch
  python plot_hook.py experiments_frontier --mode weak --model wideresnet32 --batch-per-rank 32 --backend mpi --unit iter
  python plot_hook.py a.out --mode strong --gpus 16 --layout hbar --unit call
  python plot_hook.py a.out --mode strong --layout lines --unit iter

Epoch=median cumulative hook work/epoch (seconds); iter=median work_total/steps
(milliseconds per optimizer step); call=median work_mean (milliseconds per hook
call). Calls/buckets are not steps. Cumulative work can overlap and is not an
additive wall-time component. See --help/README for all input/filter options.
"""
from thesisplots.cli import main
if __name__=='__main__':main('hook','bars',__doc__)
