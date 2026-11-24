#!/usr/bin/env python3
"""Compare backend/system panels at the same model and workload.

  python plot_compare.py experiments_frontier experiments_frontier_rccl --mode strong --model wideresnet32 --global-batch 128
  python plot_compare.py experiments_frontier experiments_polaris --mode weak --model wideresnet32 --batch-per-rank 32 --backend mpi
  python plot_compare.py experiments_frontier experiments_frontier_rccl experiments_polaris experiments_polaris_nccl --mode strong --model gpt --layout lines --direct gpu
  python plot_compare.py first.out second.out --mode strong --gpus 16 --global-batch 128 --layout hbar

Use --share-y/--no-share-y. Methods retain real implementation labels, so RCCL
hierarchical is not falsely paired with MPI ring-online. A baseline-normalized
panel compares overhead relative to each panel's own native baseline; absolute
time compares actual duration. GPT block length/effective batch must match.
See --help and README for all file/folder/CSV/filter/export options.
"""
from thesisplots.cli import main
if __name__=='__main__':main('time','bars',__doc__)
