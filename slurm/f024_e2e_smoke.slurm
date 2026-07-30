#!/usr/bin/env bash
# Real F024 end-to-end acceptance test. Submit from the repository root.
#SBATCH --job-name=phonlab-f024-e2e
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --exclude=node857
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/f024_smoke/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
module load ALICE/default CUDA/12.4.0
source scripts/project_env.sh
export CUDA_MODULE_LOADING=LAZY

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "project=$PHONLAB_PROJECT_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/python - <<'PY'
from pathlib import Path
import os
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
print("cuda_home", os.environ.get("CUDA_HOME"))
PY

.venv/bin/python -m pytest tests/test_workflow.py -q
.venv/bin/phonlab validate artifacts/f024_smoke/dataset
.venv/bin/phonlab train artifacts/f024_smoke/experiment

checkpoint="artifacts/f024_smoke/experiment/runs/checkpoints/last.ckpt"
test -s "$checkpoint"

if [ -d artifacts/f024_smoke/reconstructions ]; then
  mv artifacts/f024_smoke/reconstructions \
    "artifacts/f024_smoke/reconstructions.failed-${SLURM_JOB_ID}"
fi
.venv/bin/phonlab synthesize \
  artifacts/f024_smoke/experiment \
  "$checkpoint" \
  artifacts/f024_smoke/reconstructions
.venv/bin/phonlab compare \
  artifacts/f024_smoke/dataset \
  artifacts/f024_smoke/reconstructions \
  --output artifacts/f024_smoke/comparison.html

.venv/bin/python tools/check_f024_e2e.py artifacts/f024_smoke
