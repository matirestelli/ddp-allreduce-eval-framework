#!/bin/bash
export LD_PRELOAD=/opt/cray/pe/mpich/default/ofi/gnu/12.3/lib/libmpi_gtl_cuda.so.0:$LD_PRELOAD
export MPICH_GPU_SUPPORT_ENABLED=1
exec "$@"
