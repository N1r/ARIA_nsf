"""Generate auditable Slurm job bundles for checkpoint post-processing."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .controls.specs import ControlVariant, pitch_variants, validate_controls
from .manipulation import pitch_directory_name, semitones_to_scale


def create_postprocess_job(
    experiment: Path,
    checkpoint: Path,
    output: Path,
    semitone_shifts: list[float],
    *,
    control_variants: Optional[list[ControlVariant]] = None,
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
    variants = list(control_variants or [])
    if not shifts and not variants:
        raise ValueError("At least one semitone shift or control variant is required")
    if len(set(shifts)) != len(shifts):
        raise ValueError("Semitone shifts must be unique")
    pitch_conditions = pitch_variants(shifts)
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
    if variants:
        model = metadata.get("model")
        if not model:
            raise ValueError("Experiment metadata has no model capability declaration")
        variants = [
            ControlVariant(item.name, validate_controls(model, item.controls)) for item in variants
        ]
        names = [item.name for item in variants]
        if len(names) != len(set(names)):
            raise ValueError("Control variant names must be unique")
    all_names = [item.name for item in pitch_conditions] + [item.name for item in variants]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Pitch and control variant output names must be unique")
    dataset = Path(metadata["dataset"]).resolve()
    project = Path(__file__).resolve().parents[2]
    checkpoint_hash = _sha256(checkpoint)
    shift_slug = "-".join(pitch_directory_name(value) for value in shifts) or "controls"
    if variants:
        variant_hash = hashlib.sha256(
            json.dumps(
                [{"name": item.name, "controls": item.controls} for item in variants],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:10]
        shift_slug += f"-{variant_hash}"
    output_hash = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    bundle = experiment / "jobs" / f"postprocess-{checkpoint_hash[:10]}-{shift_slug}-{output_hash}"
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
            checkpoint_hash=checkpoint_hash,
            output=output,
            shifts=shifts,
            variants=variants,
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
                    "control_variants": [
                        {"name": item.name, "controls": item.controls} for item in variants
                    ],
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
    checkpoint_hash = shlex.quote(str(values["checkpoint_hash"]))
    output = shlex.quote(str(values["output"]))
    manipulation_args = []
    if values["shifts"]:
        manipulation_args.extend(["--semitones", *map(str, values["shifts"])])
    for variant in values["variants"]:
        encoded = ",".join(f"{name}={value!r}" for name, value in sorted(variant.controls.items()))
        manipulation_args.extend(["--variant", f"{variant.name}:{encoded}"])
    manipulation_args.extend(
        [
            "--baseline",
            f"{values['output']}/reconstruction",
            "--report",
            f"{values['output']}/manipulation.html",
        ]
    )
    manipulation_args_text = " ".join(shlex.quote(value) for value in manipulation_args)
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

verify_checkpoint() {{
  local actual
  actual="$(sha256sum -- {checkpoint})"
  actual="${{actual%% *}}"
  if [[ "$actual" != {checkpoint_hash} ]]; then
    echo "Checkpoint hash changed after this job bundle was created" >&2
    echo "Expected: {checkpoint_hash}" >&2
    echo "Actual:   $actual" >&2
    exit 1
  fi
}}

verify_checkpoint
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/phonlab synthesize {experiment} {checkpoint} {output}/reconstruction
verify_checkpoint
.venv/bin/phonlab compare {dataset} {output}/reconstruction \\
  --output {output}/reconstruction.html
.venv/bin/phonlab manipulate {experiment} {checkpoint} {output}/manipulations {manipulation_args_text}
verify_checkpoint
.venv/bin/phonlab metrics {experiment} --output {output}/metrics.html
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
