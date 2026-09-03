#!/usr/bin/env bash
# Source this file before any uv, training, or evaluation command.
# Every mutable cache is kept below the repository root.
set -euo pipefail

ARIS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARIS_PROJECT_ROOT
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ARIS_PROJECT_ROOT/.venv}"

# By default, use standard user caches unless ARIS_ISOLATE_CACHE=1 is explicitly requested.
if [ "${ARIS_ISOLATE_CACHE:-0}" = "1" ]; then
  export UV_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/uv"
  export UV_PYTHON_INSTALL_DIR="$ARIS_PROJECT_ROOT/.cache/uv/python"
  export PIP_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/pip"
  export XDG_CACHE_HOME="$ARIS_PROJECT_ROOT/.cache/xdg"
  export TORCH_HOME="$ARIS_PROJECT_ROOT/.cache/torch"
  export TORCH_EXTENSIONS_DIR="$ARIS_PROJECT_ROOT/.cache/torch_extensions"
  export NUMBA_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/numba"
  export HF_HOME="$ARIS_PROJECT_ROOT/.cache/huggingface"
  export MPLCONFIGDIR="$ARIS_PROJECT_ROOT/.cache/matplotlib"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$NUMBA_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR"
fi

# Auto-detect CUDA toolkit if not already specified
if [ -z "${CUDA_HOME:-}" ]; then
  for candidate in "/easybuild/software/CUDA/12.4.0" "/easybuild/software/CUDA/12.1.1" "/usr/local/cuda" "/opt/cuda"; do
    if [ -d "$candidate/lib64" ]; then
      export CUDA_HOME="$candidate"
      break
    fi
  done
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/nvvm/lib64:${LD_LIBRARY_PATH:-}"
  if [ -d "${CUDA_HOME}/nvvm/libdevice" ] && [ -z "${NUMBA_CUDA_LIBDEVICE:-}" ]; then
    export NUMBA_CUDA_LIBDEVICE="${CUDA_HOME}/nvvm/libdevice"
  fi
fi
