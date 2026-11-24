#!/bin/bash
# =============================================================================
# Polaris DDP + patched PyTorch runtime + CUDA-aware upstream MPICH 5.0.2
#
# Usage: source envScript_upstream_mpich.sh
# =============================================================================

# --- 1. Clear stale settings -------------------------------------------------
# The original training job explicitly forwards LD_PRELOAD.
export LD_PRELOAD=""

unset LD_LIBRARY_PATH PYTHONPATH CPATH LIBRARY_PATH CMAKE_PREFIX_PATH
unset GTL_LIB MPICH_COMPAT MPI_HOME
unset MPICH_GPU_SUPPORT_LEVEL MPICH_MAX_THREAD_SAFETY CRAY_ACCEL_TARGET

# --- 2. Modules --------------------------------------------------------------
module purge
module use /soft/modulefiles
module load PrgEnv-gnu
module load cuda/12.9
module load libfabric/2.3.1
module load cray-pmi/6.1.17
module load cray-pals/1.8.0

[[ -d "${HOME}/my_modulefiles" ]] && module use "${HOME}/my_modulefiles"

CONDA_MODULE=""
for cand in conda/2025-09-28 conda/2025-09-26 conda/2025-09-25 conda; do
    if module load "${cand}" 2>/dev/null; then
        CONDA_MODULE="${cand}"
        break
    fi
done

if [[ -z "${CONDA_MODULE}" ]]; then
    echo "[ERROR] No conda module could be loaded. Check: module avail conda"
    return 1 2>/dev/null || exit 1
fi

echo "[INFO] conda module: ${CONDA_MODULE}"

# Disable profiling preloads associated with the Cray setup.
if module is-loaded darshan; then
    module unload darshan
fi
if module is-loaded xalt; then
    module unload xalt
fi
export LD_PRELOAD=""

# --- 3. CUDA libs ------------------------------------------------------------
HPCSDK=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9

CUDA12_DIRS=(
    "${HPCSDK}/cuda/12.9/targets/x86_64-linux/lib"
    "${HPCSDK}/cuda/12.9/extras/CUPTI/lib64"
    "${HPCSDK}/math_libs/12.9/targets/x86_64-linux/lib"
    "${HPCSDK}/math_libs/12.9/lib64"
)

_cuda12=""
for d in "${CUDA12_DIRS[@]}"; do
    [[ -d "$d" ]] && _cuda12="${_cuda12:+${_cuda12}:}${d}"
done

if [[ -z "${_cuda12}" ]]; then
    echo "[ERROR] No CUDA 12.9 lib dirs found under ${HPCSDK}"
    return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${_cuda12}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[INFO] CUDA 12.9 libs: ${_cuda12}"

# No CUDA 13 addition: upstream MPI does not use Cray GTL.

# --- 4. ZFP / cuZFP ----------------------------------------------------------
PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export ZFP_HOME="${PROJECT_ROOT}/zfp-install"
export PATH="${ZFP_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ZFP_HOME}/lib64:${LD_LIBRARY_PATH}"
export CPATH="${ZFP_HOME}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${ZFP_HOME}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ZFP_HOME}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# --- 5. Proxy ----------------------------------------------------------------
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*,grand.alcf.anl.gov"

# --- 6. Conda + patched copy of the existing PyTorch build ---------------------
conda activate base

PYTORCH_BUILD="/lus/eagle/projects/UIC-HPC/mrest/mpi-experiments/pytorch-runtime-upstream"
export PYTHONPATH="${PYTORCH_BUILD}:${PROJECT_ROOT}"
export LD_LIBRARY_PATH="${PYTORCH_BUILD}/torch/lib:${LD_LIBRARY_PATH}"

# --- 7. CUDA-aware upstream MPICH --------------------------------------------
# Existing PyTorch CUDA-awareness check:
export MPICH_GPU_SUPPORT_ENABLED=1

# Upstream MPICH GPU control:
export MPIR_CVAR_ENABLE_GPU=1
export FI_PROVIDER=cxi

export MPI_HOME="/lus/eagle/projects/UIC-HPC/mrest/mpi-experiments/install/mpich-5.0.2-cxi-cuda12.9"
MPI_LIB="${MPI_HOME}/lib/libmpi.so.12"

if [[ ! -e "${MPI_LIB}" ]]; then
    echo "[ERROR] ${MPI_LIB} not found"
    return 1 2>/dev/null || exit 1
fi

# Use upstream compiler wrappers and the existing PALS launcher.
# This MPICH installation was configured with --with-pm=no.
export PATH="${MPI_HOME}/bin:/opt/cray/pals/1.8/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH}"

echo "[INFO] MPI_HOME: ${MPI_HOME} -> $(readlink -f "${MPI_HOME}")"

_missing="$(ldd "${PYTORCH_BUILD}/torch/lib/libtorch_cpu.so" \
                "${PYTORCH_BUILD}/torch/lib/libtorch_cuda.so" 2>/dev/null \
            | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"

if [[ -n "${_missing}" ]]; then
    echo "[ERROR] Unresolved libraries: ${_missing}"
    return 1 2>/dev/null || exit 1
fi

echo "[INFO] Upstream MPI and copied PyTorch dependencies resolve."

# --- 8. Threading ------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- 9. Slingshot tuning -----------------------------------------------------
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software

# --- 10. Sanity check --------------------------------------------------------
_py() { python -c "$1" 2>&1 | tail -n 1; }

echo "============================================="
echo "  Polaris DDP + copied PyTorch + upstream MPICH"
echo "============================================="
echo "Node              : $(hostname)"
echo "conda module      : ${CONDA_MODULE}"
echo "ZFP_HOME          : ${ZFP_HOME}"
echo "MPI_HOME          : ${MPI_HOME}"
echo ""
echo "Python            : $(which python)"
echo "PyTorch version   : $(_py 'import torch; print(torch.__version__)')"
echo "PyTorch location  : $(_py 'import torch; print(torch.__file__)')"
echo "CUDA available    : $(_py 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "CUDA built with   : $(_py 'import torch; print(torch.version.cuda)')"
echo "MPI available     : $(_py 'import torch.distributed as d; print(d.is_mpi_available())')"
echo "MPI version       : $("${MPI_HOME}/bin/mpichversion" | head -n 1)"
echo "============================================="