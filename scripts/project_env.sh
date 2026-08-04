#!/usr/bin/env bash
# Source this file before any uv, training, or evaluation command.
# Every mutable cache is kept below the repository root.
set -euo pipefail

ARIS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARIS_PROJECT_ROOT
export UV_PROJECT_ENVIRONMENT="$ARIS_PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ARIS_PROJECT_ROOT/.cache/uv/python"
export UV_MANAGED_PYTHON=true
export PIP_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/pip"
export XDG_CACHE_HOME="$ARIS_PROJECT_ROOT/.cache/xdg"
export TORCH_HOME="$ARIS_PROJECT_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$ARIS_PROJECT_ROOT/.cache/torch_extensions"
export NUMBA_CACHE_DIR="$ARIS_PROJECT_ROOT/.cache/numba"
export HF_HOME="$ARIS_PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MPLCONFIGDIR="$ARIS_PROJECT_ROOT/.cache/matplotlib"
export WANDB_DIR="$ARIS_PROJECT_ROOT/artifacts/wandb"
export SWANLAB_LOG_DIR="$ARIS_PROJECT_ROOT/artifacts/swanlab"
# GOLF's torchlpc path uses Numba JIT and therefore needs libNVVM from a full
# CUDA toolkit on GPU nodes. The Slurm launcher loads CUDA/12.4.0 explicitly.

mkdir -p \
  "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$NUMBA_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$SWANLAB_LOG_DIR"
