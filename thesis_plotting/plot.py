#!/usr/bin/env python3
"""General entry point: choose metric, layout and all configuration filters.

  python plot.py a.out --mode strong --metric time --layout bars
  python plot.py experiments_frontier/gpt/strongScaling --metric tail --unit iter --layout lines
  python plot.py parsed_runs.csv --mode weak --model wideresnet32 --backend mpi --batch-per-rank 32 --gpus 16 --metric hook --unit call --layout hbar

The named plot_time/plot_hook/plot_tail/plot_scaling scripts are convenient
presets of this same engine. All options and input cases: --help and README.md.
"""
from thesisplots.cli import main
if __name__=='__main__':main('time','bars',__doc__)
