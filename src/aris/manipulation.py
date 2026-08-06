"""Auditable checkpoint manipulations and listening reports."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .controls.specs import ControlVariant, pitch_variants, validate_controls
from .experiment import synthesize
from .manifest import DatasetManifest
from .report import _relative_url


def semitones_to_scale(semitones: float) -> float:
    """Convert a semitone shift into a multiplicative F0 scale."""
    semitones = float(semitones)
    if not math.isfinite(semitones) or not -36 <= semitones <= 36:
        raise ValueError("semitones must be finite and between -36 and 36")
    return 2 ** (semitones / 12)


def pitch_directory_name(semitones: float) -> str:
    """Return the filesystem-safe directory name for a pitch shift."""
    value = f"{abs(float(semitones)):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    sign = "plus" if semitones > 0 else "minus" if semitones < 0 else "zero"
    return f"pitch_{sign}_{value}st"


def manipulate_pitch(
    experiment: Path,
    checkpoint: Path,
    output: Path,
    semitone_shifts: list[float],
    *,
    dry_run: bool = False,
) -> list[list[str]]:
    """Backward-compatible wrapper for a pitch-only control sweep."""
    return manipulate_controls(
        experiment,
        checkpoint,
        output,
        pitch_variants(semitone_shifts),
        dry_run=dry_run,
    )


def manipulate_controls(
    experiment: Path,
    checkpoint: Path,
    output: Path,
    variants: list[ControlVariant],
    *,
    dry_run: bool = False,
) -> list[list[str]]:
    """Render named control variants and write checkpoint-bound provenance."""
    experiment = Path(experiment).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint} — has training completed? "
            "(expected e.g. experiments/<name>/runs/checkpoints/last.ckpt)"
        )
    if not variants:
        raise ValueError("At least one manipulation variant is required")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("Manipulation variant names must be unique")
    experiment_metadata = json.loads((experiment / "experiment.json").read_text())
    model = experiment_metadata["model"]
    normalized = [
        ControlVariant(variant.name, validate_controls(model, variant.controls))
        for variant in variants
    ]
    if output.exists():
        raise FileExistsError(f"Manipulation output already exists: {output}")

    if dry_run:
        return [
            synthesize(
                experiment,
                checkpoint,
                output / variant.name,
                dry_run=True,
                f0_scale=semitones_to_scale(variant.controls.get("pitch_semitones", 0.0)),
                controls=variant.controls,
            )
            for variant in normalized
        ]

    # Hash the checkpoint once up front; it is re-verified around every render
    # so all variants provably come from the same model state.
    checkpoint_hash = _sha256(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    # PID-suffixed staging keeps concurrent runs from colliding; the final
    # rename makes the output appear atomically or not at all.
    staging = output.parent / f".{output.name}.manipulating-{os.getpid()}"
    if staging.exists():
        # ignore_errors: NFS silly-rename (.nfs*) files can block deletion.
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    commands = []
    try:
        outputs = []
        for variant in normalized:
            _verify_checkpoint(checkpoint, checkpoint_hash)
            semitones = float(variant.controls.get("pitch_semitones", 0.0))
            scale = semitones_to_scale(semitones)
            command = synthesize(
                experiment,
                checkpoint,
                staging / variant.name,
                f0_scale=scale,
                controls=variant.controls,
            )
            _verify_checkpoint(checkpoint, checkpoint_hash)
            commands.append(command)
            render_metadata = json.loads(
                (staging / variant.name / "_render.json").read_text(encoding="utf-8")
            )
            item = {
                "name": variant.name,
                "directory": variant.name,
                "controls": variant.controls,
                "f0_scale": scale,
                "label": _variant_label(variant.controls),
                "render_audit": {
                    "metadata": f"{variant.name}/_render.json",
                    "runtime_capabilities": render_metadata["runtime_capabilities"],
                    "decoder_control_calls": render_metadata["decoder_control_calls"],
                    "files_written": render_metadata["files_written"],
                    "clipped_fraction": render_metadata["clipped_fraction"],
                },
            }
            if "pitch_semitones" in variant.controls:
                item["semitones"] = semitones
            outputs.append(item)
        metadata = {
            "schema_version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operation": "ddsp-multi-parameter-control",
            "unvoiced_policy": "preserve-zero",
            "model": model,
            "experiment": str(experiment),
            "dataset_fingerprint": experiment_metadata["dataset_fingerprint"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "outputs": outputs,
        }
        (staging / "manipulation.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return commands
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def build_manipulation_report(
    experiment: Path,
    baseline: Path,
    manipulations: Path,
    output: Path,
) -> Path:
    """Compare held-out originals, baseline reconstruction, and control variants."""
    experiment = Path(experiment).resolve()
    baseline = Path(baseline).resolve()
    manipulations = Path(manipulations).resolve()
    output = Path(output).resolve()
    experiment_metadata = json.loads((experiment / "experiment.json").read_text())
    manifest = DatasetManifest.load(Path(experiment_metadata["dataset"]))
    manipulation_metadata = json.loads((manipulations / "manipulation.json").read_text())
    variants = manipulation_metadata["outputs"]
    rows = []
    missing = []
    for record in manifest.records:
        if record.split != "test":
            continue
        files = [
            ("Original", manifest.root / record.audio_path),
            ("Reconstruction", baseline / f"{record.id}.wav"),
        ]
        files.extend(
            (
                _metadata_variant_label(variant),
                manipulations / variant["directory"] / f"{record.id}.wav",
            )
            for variant in variants
        )
        if any(not path.is_file() for _, path in files):
            missing.append(record.id)
            continue
        players = "".join(
            "<td>"
            f'<div class="variant">{html.escape(label)}</div>'
            f'<audio controls preload="none" src="{html.escape(_relative_url(output.parent, path))}"></audio>'
            "</td>"
            for label, path in files
        )
        rows.append(f"<tr><th>{html.escape(record.id)}</th>{players}</tr>")
    if not rows:
        raise ValueError("No complete original/baseline/manipulation rows were found")
    headers = "".join(
        f"<th>{html.escape(label)}</th>"
        for label in ["Item", "Original", "Reconstruction"]
        + [_metadata_variant_label(variant) for variant in variants]
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARIS parameter manipulation</title>
<style>
body{{font:15px system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#172129}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}
th,td{{padding:.65rem;border-bottom:1px solid #d9d3c8;text-align:left}}
audio{{width:230px}}code{{background:#eee;padding:.15rem .3rem}}
.variant{{font-size:.78rem;color:#66727a;margin-bottom:.2rem}}
</style></head><body><h1>Held-out DDSP parameter manipulation</h1>
<p>{html.escape(_operation_description(manipulation_metadata))}</p>
<p>Checkpoint SHA-256: <code>{html.escape(manipulation_metadata["checkpoint_sha256"])}</code></p>
<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>Incomplete rows omitted: {len(missing)}</p>
<h2>Provenance</h2><pre>{html.escape(json.dumps(manipulation_metadata, indent=2))}</pre>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint(path: Path, expected_sha256: str) -> None:
    """Fail if the checkpoint file changed since the manipulation started."""
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            "Checkpoint changed during manipulation; refusing to mix outputs from "
            f"different model states ({expected_sha256} != {actual})"
        )


def _variant_label(controls: dict[str, float]) -> str:
    """Short human-readable label for a control combination."""
    if set(controls) == {"pitch_semitones"}:
        return f"{controls['pitch_semitones']:+g} st"
    return ", ".join(f"{name}={value:g}" for name, value in sorted(controls.items()))


def _metadata_variant_label(variant: dict) -> str:
    if variant.get("label"):
        return str(variant["label"])
    if "semitones" in variant:
        return f"{float(variant['semitones']):+g} st"
    if isinstance(variant.get("controls"), dict):
        return _variant_label(variant["controls"])
    return str(variant.get("name", variant["directory"]))


def _operation_description(metadata: dict) -> str:
    if metadata.get("operation") == "voiced-f0-scale":
        return "Operation: voiced F0 scaling; unvoiced frames remain zero."
    return (
        "Operation: capability-checked DDSP controls. Pitch changes affect voiced F0 "
        "conditioning; every requested control and checkpoint hash is recorded below."
    )
