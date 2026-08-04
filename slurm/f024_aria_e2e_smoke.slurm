#!/usr/bin/env bash
# Real ARIA-GOLF F024 checkpoint and manipulation acceptance.
# Submit from the repository root: sbatch slurm/f024_aria_e2e_smoke.slurm
#SBATCH --job-name=phonlab-f024-aria
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/f024_aria_smoke/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
module load ALICE/default CUDA/12.4.0
source scripts/project_env.sh
export CUDA_MODULE_LOADING=LAZY

dataset="artifacts/f024_smoke/dataset"
run="artifacts/f024_aria_smoke/run-${SLURM_JOB_ID}"
experiment="$run/experiment"
checkpoint="$experiment/runs/checkpoints/last.ckpt"
baseline="$run/reconstruction"
manipulations="$run/manipulations"

test -f "$dataset/dataset.json"
test ! -e "$run"
mkdir -p "$run"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "run=$run"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/python - <<'PY'
import os
from pathlib import Path

import torch

root = Path(os.environ["PHONLAB_PROJECT_ROOT"]).resolve()
for name in (
    "UV_CACHE_DIR",
    "PIP_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "NUMBA_CACHE_DIR",
    "HF_HOME",
):
    value = Path(os.environ[name]).resolve()
    if root not in value.parents:
        raise RuntimeError(f"{name} escapes project: {value}")
if not torch.cuda.is_available():
    raise RuntimeError("Slurm job has no visible CUDA device")
print("python", os.sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY

.venv/bin/phonlab validate "$dataset"
.venv/bin/phonlab init-experiment \
  "$dataset" "$experiment" \
  --model aria-golf \
  --batch-size 8 \
  --max-steps 400 \
  --workers 4 \
  --f0-min 80 \
  --f0-max 500 \
  --partition gpu-short \
  --gres gpu:l4:1 \
  --time 01:00:00 \
  --cpus 4 \
  --memory 24G \
  --exclude ''

.venv/bin/phonlab train "$experiment"
test -s "$checkpoint"

.venv/bin/phonlab synthesize "$experiment" "$checkpoint" "$baseline"
.venv/bin/phonlab compare "$dataset" "$baseline" --output "$run/reconstruction.html"
.venv/bin/phonlab manipulate \
  "$experiment" "$checkpoint" "$manipulations" \
  --variant 'f1_down:f1_scale=0.9' \
  --variant 'f1_up:f1_scale=1.1' \
  --variant 'f2_down:f2_scale=0.9' \
  --variant 'f2_up:f2_scale=1.1' \
  --variant 'tilt_down:tilt_alpha_delta=-0.1' \
  --variant 'tilt_up:tilt_alpha_delta=0.1' \
  --baseline "$baseline" \
  --report "$run/manipulation.html"
.venv/bin/phonlab metrics "$experiment" --output "$run/metrics.html"

.venv/bin/python tools/check_aria_manipulation.py \
  "$baseline" "$manipulations" \
  --output "$run/aria-acceptance.json"
.venv/bin/python tools/check_webui.py \
  --workspace . \
  --result "$run" \
  --check-export \
  --output "$run/webui-acceptance.json"

echo "PHONLAB_F024_ARIA_E2E_OK"
