#!/bin/bash
# =============================================================================
# Polaris DDP + custom PyTorch (Eagle build) + CUDA-aware Cray MPICH
# For the 2026-08-19 Polaris upgrade: PE 26.03, MPICH 9.1.0 (only version
# installed), NVIDIA driver 580.65.06, hpc_sdk 25.9 (ships CUDA 12.9 and 13.0).
#
# What the existing PyTorch build needs (from ldd on torch/lib):
#   libcudart.so.12, libcublas.so.12, ...   -> CUDA 12.9
#   libmpi_gnu_123.so.12                    -> MPICH ofi/gnu/12.3 (NOT nvidia/23.3)   libmpi_gnu_123.so.12                    -> renamed libmpi_gnu.so.12 in MPICH 9.1.0;
#                                              the script creates a compat link (section 7)
#   libmpi_gtl_cuda.so.0 (linked directly)  -> MPICH 9.1.0 GTL, which needs libcudart.so.13
#
# So this build runs with CUDA 12.9 libs first on the path and the CUDA 13.0
# lib dir appended at the end (only the GTL uses it; library sonames differ,
# so nothing collides). That is a bridge, not a final state: rebuild PyTorch
# and zfp against cuda/13.0 to get back to a single CUDA runtime.
#
# Usage:   source envScript_polaris_2026-08.sh
# Launch:  mpiexec -n N --ppn P ./gpu_wrapper.sh python train.py
# =============================================================================

# --- 1. Clear stale preloads -------------------------------------------------
unset LD_PRELOAD

# --- 2. Modules --------------------------------------------------------------
module purge
module use /soft/modulefiles
module load PrgEnv-gnu               # brings in its default gcc-native
module load cuda/12.9                # NOT the 13.0 default: PyTorch was built with 12.9
module load cray-mpich
module load craype-accel-nvidia80

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

# --- 3. CUDA libs ------------------------------------------------------------
HPCSDK=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9

# CUDA 12.9 first (what PyTorch and cuZFP were built with). The math libs and
# CUPTI live in separate directories inside hpc_sdk; only existing ones are added.
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

# CUDA 13.0 runtime, APPENDED: only needed by the MPICH 9.1.0 GTL.
CUDA13_LIB="${HPCSDK}/cuda/13.0/targets/x86_64-linux/lib"
if ! ls "${CUDA13_LIB}"/libcudart.so.13* >/dev/null 2>&1; then
    echo "[ERROR] libcudart.so.13 not found in ${CUDA13_LIB}"
    echo "        Find it with: find ${HPCSDK}/cuda/13.0 -name 'libcudart.so.13*'"
    return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA13_LIB}"
echo "[INFO] CUDA 13.0 runtime (for GTL): ${CUDA13_LIB}"

# --- 4. ZFP / cuZFP ----------------------------------------------------------
# Built against pre-upgrade CUDA 12.x; still fine with the 12.9 libs above.
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

# --- 6. Conda + custom PyTorch build ----------------------------------------
conda activate base

PYTORCH_BUILD="/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/src/pytorch"
export PYTHONPATH="${PYTORCH_BUILD}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${PYTORCH_BUILD}/torch/lib:${PYTORCH_BUILD}/build/lib:${LD_LIBRARY_PATH}"

# --- 7. CUDA-aware Cray MPICH ------------------------------------------------
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_SUPPORT_LEVEL=1
export MPICH_MAX_THREAD_SAFETY=multiple
export CRAY_ACCEL_TARGET=nvidia80

# PyTorch links libmpi_gnu_123.so.12 -> the GNU-built MPICH.
export MPI_HOME="/opt/cray/pe/mpich/default/ofi/gnu/12.3"
MPI_LIB="${MPI_HOME}/lib/libmpi_gnu.so.12"
if [[ ! -e "${MPI_LIB}" ]]; then
    echo "[ERROR] ${MPI_LIB} not found"
    return 1 2>/dev/null || exit 1
fi

# MPICH 9.1.0 renamed libmpi_gnu_123.so.12 -> libmpi_gnu.so.12 (same ABI number).
# The PyTorch build still asks for the old name, so provide it with a per-user link.
# Created automatically, safe to repeat, skipped if the system has the old name again.
if [[ -e "${MPI_HOME}/lib/libmpi_gnu_123.so.12" ]]; then
    export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH}"
else
    MPICH_COMPAT="${HOME}/.cache/mpich_compat"
    mkdir -p "${MPICH_COMPAT}"
    ln -sfn "${MPI_LIB}" "${MPICH_COMPAT}/libmpi_gnu_123.so.12"
    if [[ ! -e "${MPICH_COMPAT}/libmpi_gnu_123.so.12" ]]; then
        echo "[ERROR] could not create ${MPICH_COMPAT}/libmpi_gnu_123.so.12"
        return 1 2>/dev/null || exit 1
    fi
    export LD_LIBRARY_PATH="${MPI_HOME}/lib:${MPICH_COMPAT}:${LD_LIBRARY_PATH}"
fi
echo "[INFO] MPI_HOME: ${MPI_HOME} -> $(readlink -f "${MPI_HOME}")"

export GTL_LIB="${MPI_HOME}/lib/libmpi_gtl_cuda.so.0"
if [[ ! -e "${GTL_LIB}" ]]; then
    echo "[ERROR] GTL not found at ${GTL_LIB}"
    return 1 2>/dev/null || exit 1
fi

# Every dependency of the GTL and of PyTorch must resolve BEFORE anything is
# preloaded. A preloaded library with a missing dependency breaks every
# command in the shell (that was the libcudart.so.13 error).
_missing="$(ldd "${GTL_LIB}" \
                "${PYTORCH_BUILD}/torch/lib/libtorch_cpu.so" \
                "${PYTORCH_BUILD}/torch/lib/libtorch_cuda.so" 2>/dev/null \
            | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
if [[ -n "${_missing}" ]]; then
    echo "[ERROR] Unresolved libraries: ${_missing}"
    echo "        Locate each with: find ${HPCSDK} /opt/cray/pe -name '<libname>*'"
    echo "        and add its directory to LD_LIBRARY_PATH in section 3."
    return 1 2>/dev/null || exit 1
fi
echo "[INFO] GTL and PyTorch dependencies all resolve."
echo "[INFO] GTL_LIB=${GTL_LIB} (preloaded only in MPI ranks via gpu_wrapper.sh)"

# --- 8. Threading ------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- 9. Slingshot tuning (SHS 14.0.1) ----------------------------------------
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software

# --- 10. Sanity check --------------------------------------------------------
_py() { LD_PRELOAD="${GTL_LIB}" python -c "$1" 2>&1 | tail -n 1; }
echo "============================================="
echo "  Polaris DDP + Eagle PyTorch + Cray MPICH"
echo "============================================="
echo "Node              : $(hostname)"
echo "conda module      : ${CONDA_MODULE}"
echo "ZFP_HOME          : ${ZFP_HOME}"
echo "MPI_HOME          : ${MPI_HOME}"
echo "GTL_LIB           : ${GTL_LIB}"
echo ""
echo "Python            : $(which python)"
echo "PyTorch version   : $(_py 'import torch; print(torch.__version__)')"
echo "PyTorch location  : $(_py 'import torch; print(torch.__file__)')"
echo "CUDA available    : $(_py 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "CUDA built with   : $(_py 'import torch; print(torch.version.cuda)')"
echo "MPI available     : $(_py 'import torch.distributed as d; print(d.is_mpi_available())')"
echo "mpi4py            : $(_py 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])')"
echo "============================================="

# Interactive node:
# qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A UIC-HPC

# request interactive node:
# qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A UIC-HPC