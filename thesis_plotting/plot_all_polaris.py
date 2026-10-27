#!/usr/bin/env python3
"""Generate every available Polaris experiment/configuration.

  python plot_all_polaris.py /path/to/code_oct26 --output-root /path/to/plots_thesis
  python plot_all_polaris.py experiments_polaris experiments_polaris_nccl
  python plot_all_polaris.py /path/to/code_oct26 --dry-run
  python plot_all_polaris.py /path/to/code_oct26 --scopes all --metrics time hook tail --layouts bars lines
  python plot_all_polaris.py /path/to/code_oct26 --backend nccl --no-diagnostics
  python plot_all_polaris.py /path/to/code_oct26 --include-partial --max-plots 10

This uses exactly the same parsing/style/options as Frontier. Unfinished or
not-yet-produced configurations stay absent and are recorded in the audit,
not replaced with measurements from a different model/system/rate. See --help
and README for all cases, paths and interpretation.
"""
from thesisplots.batch import run
if __name__=='__main__':run('polaris')
