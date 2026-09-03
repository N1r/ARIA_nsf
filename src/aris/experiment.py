"""Create and launch self-contained training experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .controls.specs import validate_controls
from .manifest import DatasetManifest, validate_manifest

MODELS = {
    "golf": "golf_16k.yaml",
    "ddsp": "ddsp_16k.yaml",
    "aria-golf": "aria_golf_16k.yaml",
}


def create_experiment(
    dataset: Path,
    output: Path,
    *,
    model: str = "golf",
    batch_size: int = 32,
    max_steps: int = 40000,
    seed: int = 42,
    f0_min: Optional[float] = None,
    f0_max: float = 500,
    workers: int = 4,
    learning_rate: float = 0.0002,
    slurm_partition: str = "gpu",
    slurm_gres: str = "gpu:1",
    slurm_time: str = "04:00:00",
    slurm_cpus: int = 8,
    slurm_memory: str = "32G",
    slurm_exclude: str = "",
    overwrite: bool = False,
) -> Path:
    """Create a self-contained experiment directory bound to a validated dataset.

    Writes config.yaml, decoder.yaml, experiment.json (with content hashes and
    the dataset fingerprint), train.sh, and train.slurm.
    """
    manifest = DatasetManifest.load(dataset)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors[:20]))
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODELS)}")
    if model == "aria-golf":
        missing_formants = [record.id for record in manifest.records if not record.formant_path]
        if missing_formants:
            raise ValueError(
                "aria-golf requires F1/F2 supervision targets; rerun `aris prepare` "
                "with --extract-formants before creating the experiment"
            )
    if f0_min is None:
        f0_min = 100.0 if model == "aria-golf" else 60.0
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    sample_rates = {record.sample_rate for record in manifest.records}
    if sample_rates != {16000}:
        raise ValueError(
            "The bundled presets currently require 16 kHz audio; run aris prepare "
            "with --sample-rate 16000"
        )
    slurm = _validate_slurm_settings(
        partition=slurm_partition,
        gres=slurm_gres,
        time=slurm_time,
        cpus=slurm_cpus,
        memory=slurm_memory,
        exclude=slurm_exclude,
    )
    output = Path(output).resolve()
    # Refuse non-empty targets so config hashes always describe a single run.
    if output.exists() and any(output.iterdir()):
        if overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"Experiment directory is not empty: {output}; set overwrite=True to replace"
            )
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "runs"
    config = _training_config(
        manifest,
        run_dir,
        model=model,
        batch_size=batch_size,
        max_steps=max_steps,
        seed=seed,
        f0_min=f0_min,
        f0_max=f0_max,
        workers=workers,
        learning_rate=learning_rate,
    )
    config_path = output / "config.yaml"
    config_path.write_text(config, encoding="utf-8")
    shutil.copyfile(_preset(model), output / "decoder.yaml")
    decoder_bytes = (output / "decoder.yaml").read_bytes()
    try:
        # Relative to the experiment dir so the pair stays portable if moved
        # or copied together; only an unrelated location (e.g. another drive)
        # falls back to an absolute path.
        dataset_value = os.path.relpath(manifest.root, output)
    except ValueError:
        dataset_value = str(manifest.root)
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_value,
        "dataset_fingerprint": manifest.fingerprint,
        "model": model,
        "decoder_config": "decoder.yaml",
        "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "decoder_sha256": hashlib.sha256(decoder_bytes).hexdigest(),
        "command": _command(output, model),
        "slurm": slurm,
    }
    (output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    project_root = _repo_root()
    (output / "train.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"PROJECT_ROOT={shlex.quote(str(project_root))}\n"
        'if [ -f "$PROJECT_ROOT/scripts/project_env.sh" ]; then\n'
        '  source "$PROJECT_ROOT/scripts/project_env.sh"\n'
        "fi\n"
        'PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python)}"\n'
        'if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then\n'
        '  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"\n'
        "fi\n"
        '"$PYTHON_BIN" -m aris.cli train "$EXPERIMENT_DIR"\n',
        encoding="utf-8",
    )
    os.chmod(output / "train.sh", 0o755)
    (output / "train.slurm").write_text(
        _slurm_script(output, **slurm),
        encoding="utf-8",
    )
    return output


def train(
    experiment: Path, extra_args: Optional[list[str]] = None, dry_run: bool = False
) -> list[str]:
    """Re-verify the dataset and run (or return) the training command."""
    from .cuda_env import auto_configure_cuda

    auto_configure_cuda()
    experiment = Path(experiment).resolve()
    metadata = json.loads((experiment / "experiment.json").read_text())
    manifest = DatasetManifest.load(_resolve_dataset_path(metadata["dataset"], experiment))
    # The fingerprint ties results to the exact data; refuse silent drift.
    if manifest.fingerprint != metadata["dataset_fingerprint"]:
        raise RuntimeError("Dataset fingerprint changed after this experiment was created")
    command = _command(experiment, metadata["model"]) + list(extra_args or [])
    if dry_run:
        return command
    subprocess.run(command, cwd=_repo_root(), check=True)
    return command


def synthesize(
    experiment: Path,
    checkpoint: Path,
    output: Path,
    *,
    dry_run: bool = False,
    f0_scale: float = 1.0,
    controls: Optional[Mapping[str, float]] = None,
    split: str = "test",
    item_ids: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> list[str]:
    """Render selected manifest audio from a checkpoint, optionally with controls."""
    from .cuda_env import auto_configure_cuda

    auto_configure_cuda()
    experiment = Path(experiment).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if not 0 < f0_scale < 16:
        raise ValueError("f0_scale must be in (0, 16)")
    metadata = json.loads((experiment / "experiment.json").read_text())
    control_values = validate_controls(metadata["model"], {} if controls is None else controls)
    # Pitch is applied through the F0 conditioning path, not the decoder hook,
    # so it is folded into f0_scale here and removed from the runtime controls.
    pitch_semitones = control_values.pop("pitch_semitones", None)
    if pitch_semitones is not None:
        pitch_scale = 2 ** (pitch_semitones / 12)
        if not math.isclose(f0_scale, 1.0) and not math.isclose(
            f0_scale, pitch_scale, rel_tol=1e-12
        ):
            raise ValueError("Conflicting pitch controls: f0_scale does not match pitch_semitones")
        f0_scale = pitch_scale
    manifest = DatasetManifest.load(_resolve_dataset_path(metadata["dataset"], experiment))
    if manifest.fingerprint != metadata["dataset_fingerprint"]:
        raise RuntimeError("Dataset fingerprint changed after this experiment was created")
    requested_ids = list(item_ids or [])
    if len(set(requested_ids)) != len(requested_ids) or not all(
        isinstance(item_id, str) and item_id for item_id in requested_ids
    ):
        raise ValueError("item_ids must contain unique non-empty strings")
    eligible_ids = {
        record.id for record in manifest.records if split == "all" or record.split == split
    }
    missing_ids = [item_id for item_id in requested_ids if item_id not in eligible_ids]
    if missing_ids:
        raise ValueError(f"Requested item IDs are not in split {split!r}: {', '.join(missing_ids)}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint} — has training completed? "
            "(expected e.g. experiments/<name>/runs/checkpoints/last.ckpt)"
        )
    # Never write into an existing directory: outputs must stay attributable
    # to exactly one checkpoint and control set.
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"Synthesis output already exists: {output}; set overwrite=True to replace"
            )
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "aris.engine",
        "predict",
        "--config",
        str(experiment / "config.yaml"),
        "--model",
        str(experiment / "decoder.yaml"),
        "--ckpt_path",
        str(checkpoint),
        "--trainer.logger",
        "false",
        "--trainer.enable_model_summary",
        "false",
        "--trainer.callbacks+=aris.controls.lightning.ControlledPredictionWriter",
        "--trainer.callbacks.output_dir",
        str(output),
        "--trainer.callbacks.controls_json",
        json.dumps(control_values, sort_keys=True, separators=(",", ":")),
        "--data.init_args.predict_split",
        str(split),
        "--data.init_args.predict_item_ids",
        json.dumps(requested_ids, separators=(",", ":")),
        "--data.init_args.predict_f0_scale",
        str(f0_scale),
    ]
    if not dry_run:
        subprocess.run(command, cwd=_repo_root(), check=True)
    return command


def _command(experiment: Path, model: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "aris.engine",
        "fit",
        "--config",
        str(experiment / "config.yaml"),
        "--model",
        str(experiment / "decoder.yaml"),
    ]


def _resolve_dataset_path(dataset: str, experiment: Path) -> Path:
    """Resolve experiment.json's ``dataset`` field to an existing directory.

    A relative path is tried against the experiment's own directory first —
    so a copied experiment+dataset pair keeps working from any working
    directory — then against the current working directory for backward
    compatibility with older experiment.json files that stored an absolute
    or cwd-relative path.
    """
    candidate = Path(dataset)
    if candidate.is_absolute():
        return candidate
    experiment_relative = (experiment / candidate).resolve()
    if experiment_relative.exists():
        return experiment_relative
    cwd_relative = candidate.resolve()
    if cwd_relative.exists():
        return cwd_relative
    raise FileNotFoundError(
        f"Dataset {dataset!r} from {experiment / 'experiment.json'} not found; tried:\n"
        f"  {experiment_relative} (relative to the experiment directory)\n"
        f"  {cwd_relative} (relative to the current directory)"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preset(model: str) -> Path:
    return Path(__file__).with_name("presets") / MODELS[model]


def _training_config(manifest, run_dir, **settings) -> str:
    root = manifest.root.as_posix()
    run = run_dir.as_posix()
    aria = settings["model"] == "aria-golf"
    aria_losses = (
        """    formant_loss_weight: 2.0
    residual_reg_weight: 0.02
    formant_smooth_weight: 0.05
"""
        if aria
        else ""
    )
    return f"""# Generated by ARIS. Edit a copy; experiment.json records its hash.
seed_everything: {settings["seed"]}
trainer:
  max_steps: {settings["max_steps"]}
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 1
  enable_progress_bar: false
  gradient_clip_val: 0.5
  strategy: auto
  accelerator: auto
  devices: 1
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: {run}
      name: metrics
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: {run}/checkpoints
        save_last: true
        monitor: val_loss
        mode: min
        filename: "{{epoch}}-{{step}}-{{val_loss:.3f}}"
        save_top_k: 3
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: aris.training.progress.TrainingProgressPrinter
      init_args:
        every_n_steps: 50
model:
  class_path: aris.training.autoencoder.VoiceAutoEncoder
  init_args:
    encoder_class_path: aris.models.enc.VocoderParameterEncoderInterface
    encoder_init_args:
      f0_min: {settings["f0_min"]}
      f0_max: {settings["f0_max"]}
      backbone_type: aris.models.unet.UNetEncoder
      n_fft: 512
      hop_length: 160
      channels: [32, 64, 128, 256]
      strides: [4, 4, 4, 4]
      lstm_hidden_size: 256
      num_layers: 3
      dropout: 0.1
      learn_voicing: false
      learn_f0: false
    decoder: null
    criterion:
      class_path: aris.losses.spec.MSSLoss
      init_args:
        n_ffts: [127, 255, 511, 1023]
        alpha: 1.0
        window: hanning
        center: true
    sample_rate: 16000
    detach_voicing: true
    detach_f0: true
    train_with_true_f0: true
{aria_losses.rstrip()}
data:
  class_path: aris.lightning.ManifestDataModule
  init_args:
    manifest_path: {root}/manifest.csv
    batch_size: {settings["batch_size"]}
    duration: 1.5
    overlap: 1.0
    num_workers: {settings["workers"]}
    load_formants: {str(aria).lower()}
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: {settings["learning_rate"]}
"""


def _validate_slurm_settings(**values) -> dict:
    for name in ("partition", "gres", "time", "memory"):
        value = values[name]
        if (
            not isinstance(value, str)
            or not value
            or any(character.isspace() for character in value)
        ):
            raise ValueError(f"slurm_{name} must be a non-empty token")
    exclude = values["exclude"]
    if not isinstance(exclude, str) or any(character.isspace() for character in exclude):
        raise ValueError("slurm_exclude must be an optional comma-separated token")
    cpus = values["cpus"]
    if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus <= 0:
        raise ValueError("slurm_cpus must be a positive integer")
    return dict(values)


def _slurm_script(
    output: Path,
    *,
    partition: str,
    gres: str,
    time: str,
    cpus: int,
    memory: str,
    exclude: str,
) -> str:
    exclude_directive = f"#SBATCH --exclude={exclude}\n" if exclude else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=aris
#SBATCH --partition={partition}
#SBATCH --gres={gres}
{exclude_directive}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={output.as_posix()}/slurm-%j.log
set -euo pipefail
# module load <your CUDA module>
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" {(output / "train.sh").as_posix()}
"""
