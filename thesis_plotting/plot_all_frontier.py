#!/usr/bin/env python3
"""Generate every available Frontier experiment/configuration.

  python plot_all_frontier.py /path/to/code_oct26 --output-root /path/to/plots_thesis
  python plot_all_frontier.py experiments_frontier experiments_frontier_rccl
  python plot_all_frontier.py /path/to/code_oct26 --dry-run
  python plot_all_frontier.py /path/to/code_oct26 --scopes all --metrics time tail --layouts bars lines
  python plot_all_frontier.py /path/to/code_oct26 --model gpt --direct all --no-diagnostics
  python plot_all_frontier.py /path/to/code_oct26 --gpus 16 --rates 8 --formats png svg

Same common --help as individual scripts plus batch controls. This scans only
available runs, excludes incomplete runs by default, preserves missing metrics,
and writes one canonical shaded family series for actual hierarchical aliases.
Every output has a CSV, source/contrast JSON, description draft and HTML data
companion. See README for the full matrix and output tree.
"""
from thesisplots.batch import run
if __name__=='__main__':run('frontier')
