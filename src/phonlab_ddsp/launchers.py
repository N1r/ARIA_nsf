"""Generate auditable Slurm job bundles for checkpoint post-processing."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .manipulation import pitch_directory_name, semitones_to_scale


def create_postprocess_job(
    experiment: Path,
    checkpoint: Path,
    output: Path,
    semitone_shifts: list[float],
    *,
    partition: str = "gpu-short",
    gres: str = "gpu:1",
    time_limit: str = "00:30:00",
    cpus: int = 4,
    memory: str = "24G",
    exclude: str = "",
) -> Path:
    """Create a directory containing a submit-ready ``train.slurm``.

    Job bundles use the same filename as training experiments so the safe
    :mod:`phonlab_ddsp.jobs` backend can submit, query, and tail either one.
    """
    experiment = Path(experiment).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if not (experiment / "experiment.json").is_file():
        raise FileNotFoundError(f"Experiment metadata does not exist: {experiment}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Post-processing output already exists: {output}")
    shifts = [float(value) for value in semitone_shifts]
    if not shifts:
        raise ValueError("At least one semitone shift is required")
    if len(set(shifts)) != len(shifts):
        raise ValueError("Semitone shifts must be unique")
    for shift in shifts:
        semitones_to_scale(shift)
    if cpus <= 0:
        raise ValueError("cpus must be positive")
    for value, name in (
        (partition, "partition"),
        (gres, "gres"),
        (time_limit, "time_limit"),
        (memory, "memory"),
    ):
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"{name} must be a non-empty Slurm token")
    if exclude and any(character.isspace() for character in exclude):
        raise ValueError("exclude must be a comma-separated Slurm token")

    metadata = json.loads((experiment / "experiment.json").read_text())
    dataset = Path(metadata["dataset"]).resolve()
    project = Path(__file__).resolve().parents[2]
    checkpoint_hash = _sha256(checkpoint)
    shift_slug = "-".join(pitch_directory_name(value) for value in shifts)
    bundle = experiment / "jobs" / f"postprocess-{checkpoint_hash[:10]}-{shift_slug}"
    if bundle.exists():
        raise FileExistsError(f"Post-processing job already exists: {bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = bundle.parent / f".{bundle.name}.creating-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        script = _postprocess_script(
            project=project,
            experiment=experiment,
            dataset=dataset,
            checkpoint=checkpoint,
            output=output,
            shifts=shifts,
            partition=partition,
            gres=gres,
            time_limit=time_limit,
            cpus=cpus,
            memory=memory,
            exclude=exclude,
        )
        (staging / "train.slurm").write_text(script, encoding="utf-8")
        (staging / "job.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "kind": "checkpoint-postprocess",
                    "experiment": str(experiment),
                    "dataset": str(dataset),
                    "dataset_fingerprint": metadata["dataset_fingerprint"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "output": str(output),
                    "semitone_shifts": shifts,
                    "slurm": {
                        "partition": partition,
                        "gres": gres,
                        "time": time_limit,
                        "cpus": cpus,
                        "memory": memory,
                        "exclude": exclude or None,
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(bundle)
        return bundle
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _postprocess_script(**values) -> str:
    project = shlex.quote(str(values["project"]))
    experiment = shlex.quote(str(values["experiment"]))
    dataset = shlex.quote(str(values["dataset"]))
    checkpoint = shlex.quote(str(values["checkpoint"]))
    output = shlex.quote(str(values["output"]))
    shifts = " ".join(shlex.quote(str(value)) for value in values["shifts"])
    exclude = f"#SBATCH --exclude={values['exclude']}\n" if values["exclude"] else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=phonlab-postprocess
#SBATCH --partition={values["partition"]}
#SBATCH --gres={values["gres"]}
{exclude}#SBATCH --cpus-per-task={values["cpus"]}
#SBATCH --mem={values["memory"]}
#SBATCH --time={values["time_limit"]}
#SBATCH --output=slurm-%j.log
set -euo pipefail
cd {project}
module load ALICE/default CUDA/12.4.0
source scripts/project_env.sh
export CUDA_MODULE_LOADING=LAZY
test -s {checkpoint}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/phonlab synthesize {experiment} {checkpoint} {output}/reconstruction
.venv/bin/phonlab compare {dataset} {output}/reconstruction \\
  --output {output}/reconstruction.html
.venv/bin/phonlab manipulate {experiment} {checkpoint} {output}/manipulations \\
  --semitones {shifts} --baseline {output}/reconstruction \\
  --report {output}/manipulation.html
.venv/bin/phonlab metrics {experiment} --output {output}/metrics.html
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
