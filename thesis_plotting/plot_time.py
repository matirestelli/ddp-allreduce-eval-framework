#!/usr/bin/env python3
"""Shaded family bars for one configuration or a fixed-workload GPU sweep.

Examples (run from the directory containing experiments_*):
  python /path/to/thesis_plotting/plot_time.py experiments_frontier --mode strong --model wideresnet32 --global-batch 128 --backend mpi
  python /path/to/thesis_plotting/plot_time.py experiments_frontier/wideresnet/cifar10/weakScaling --batch-per-rank 32 --gpus 16
  python /path/to/thesis_plotting/plot_time.py a.out --mode strong --assume-system frontier --global-batch 128 --gpus 8
  python /path/to/thesis_plotting/plot_time.py parsed_runs.csv --mode strong --model gpt --direct gpu
  python /path/to/thesis_plotting/plot_time.py experiments_frontier --mode strong --model wideresnet32 --gpus 16 --x global_batch

All cases/options, including file/folder/CSV inputs, filters, normalization,
comparison panels, output formats, dimensions and rates, are in --help and README.
"""
from thesisplots.cli import main
if __name__=='__main__':main('time','bars',__doc__)
