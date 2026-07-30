#!/usr/bin/env bash
# End-to-end acceptance run for the documented CMU ARCTIC SLT example.
# Data acquisition and preparation happen before submission; all neural-network
# training and checkpoint inference happen on the allocated GPU node.
#SBATCH --job-name=phonlab-slt-e2e
#SBATCH --partition=testing
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
# Submit from the repository root so this relative output path stays portable.
#SBATCH --output=artifacts/cmu_arctic_slt_demo/experiment/slurm-%j.log

set -Eeuo pipefail

PHONLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_ROOT="$PHONLAB_ROOT/artifacts/cmu_arctic_slt_demo"
EXPERIMENT="$PIPELINE_ROOT/experiment"
CHECKPOINT="$EXPERIMENT/runs/checkpoints/last.ckpt"
STATUS_FILE="$PIPELINE_ROOT/training_job.json"

cd "$PHONLAB_ROOT"
module load ALICE/default CUDA/12.4.0
source scripts/project_env.sh
export CUDA_MODULE_LOADING=LAZY

record_status() {
  local state="$1"
  local stage="$2"
  local exit_code="${3:-0}"
  PHONLAB_JOB_STATE="$state" \
  PHONLAB_JOB_STAGE="$stage" \
  PHONLAB_JOB_EXIT_CODE="$exit_code" \
  PHONLAB_STATUS_FILE="$STATUS_FILE" \
  PHONLAB_CHECKPOINT="$CHECKPOINT" \
  .venv/bin/python - <<'PY'
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["PHONLAB_STATUS_FILE"])
payload = {}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
payload.update(
    {
        "schema_version": "1.0",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "state": os.environ["PHONLAB_JOB_STATE"],
        "stage": os.environ["PHONLAB_JOB_STAGE"],
        "exit_code": int(os.environ["PHONLAB_JOB_EXIT_CODE"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
)
gpu = os.environ.get("PHONLAB_GPU")
if gpu:
    payload["gpu"] = gpu
checkpoint = Path(os.environ["PHONLAB_CHECKPOINT"])
if checkpoint.is_file():
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    payload["checkpoint"] = str(checkpoint.resolve())
    payload["checkpoint_sha256"] = digest.hexdigest()
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

on_error() {
  local exit_code=$?
  trap - ERR
  record_status FAILED failed "$exit_code" || true
  exit "$exit_code"
}
trap on_error ERR

record_status RUNNING environment
test -x .venv/bin/python
test "$UV_CACHE_DIR" = "$PHONLAB_ROOT/.cache/uv"
export PHONLAB_GPU
PHONLAB_GPU="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
record_status RUNNING dataset-validation
.venv/bin/phonlab validate "$PIPELINE_ROOT/dataset" --json

record_status RUNNING training
.venv/bin/phonlab train "$EXPERIMENT"
test -s "$CHECKPOINT"

record_status RUNNING metrics
.venv/bin/phonlab metrics "$EXPERIMENT" --output "$PIPELINE_ROOT/metrics.html"

record_status RUNNING reconstruction
.venv/bin/phonlab synthesize \
  "$EXPERIMENT" "$CHECKPOINT" "$PIPELINE_ROOT/reconstruction"
.venv/bin/phonlab compare \
  "$PIPELINE_ROOT/dataset" "$PIPELINE_ROOT/reconstruction" \
  --output "$PIPELINE_ROOT/comparison.html"

record_status RUNNING manipulation
.venv/bin/phonlab manipulate \
  "$EXPERIMENT" "$CHECKPOINT" "$PIPELINE_ROOT/manipulations" \
  --semitones -4 4 \
  --baseline "$PIPELINE_ROOT/reconstruction" \
  --report "$PIPELINE_ROOT/manipulation.html"

record_status COMPLETED pipeline-complete
.venv/bin/python tools/check_complete_pipeline.py "$PIPELINE_ROOT"
