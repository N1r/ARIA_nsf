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

from .experiment import synthesize
from .manifest import DatasetManifest
from .report import _relative_url


def semitones_to_scale(semitones: float) -> float:
    semitones = float(semitones)
    if not math.isfinite(semitones) or not -36 <= semitones <= 36:
        raise ValueError("semitones must be finite and between -36 and 36")
    return 2 ** (semitones / 12)


def pitch_directory_name(semitones: float) -> str:
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
    """Run one predict pass per pitch shift and record its provenance."""
    experiment = Path(experiment).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    shifts = [float(value) for value in semitone_shifts]
    if not shifts:
        raise ValueError("At least one semitone shift is required")
    if len(set(shifts)) != len(shifts):
        raise ValueError("Semitone shifts must be unique")
    scales = [semitones_to_scale(value) for value in shifts]
    if output.exists():
        raise FileExistsError(f"Manipulation output already exists: {output}")

    if dry_run:
        return [
            synthesize(
                experiment,
                checkpoint,
                output / pitch_directory_name(shift),
                dry_run=True,
                f0_scale=scale,
            )
            for shift, scale in zip(shifts, scales)
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.manipulating-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    commands = []
    try:
        outputs = []
        for shift, scale in zip(shifts, scales):
            name = pitch_directory_name(shift)
            command = synthesize(
                experiment,
                checkpoint,
                staging / name,
                f0_scale=scale,
            )
            commands.append(command)
            outputs.append(
                {
                    "semitones": shift,
                    "f0_scale": scale,
                    "directory": name,
                }
            )
        experiment_metadata = json.loads((experiment / "experiment.json").read_text())
        metadata = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operation": "voiced-f0-scale",
            "unvoiced_policy": "preserve-zero",
            "experiment": str(experiment),
            "dataset_fingerprint": experiment_metadata["dataset_fingerprint"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
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
            shutil.rmtree(staging)
        raise


def build_manipulation_report(
    experiment: Path,
    baseline: Path,
    manipulations: Path,
    output: Path,
) -> Path:
    """Compare held-out originals, baseline reconstruction, and pitch shifts."""
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
                f"{variant['semitones']:+g} st",
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
        + [f"{variant['semitones']:+g} st" for variant in variants]
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonLab-DDSP pitch manipulation</title>
<style>
body{{font:15px system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#172129}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}
th,td{{padding:.65rem;border-bottom:1px solid #d9d3c8;text-align:left}}
audio{{width:230px}}code{{background:#eee;padding:.15rem .3rem}}
.variant{{font-size:.78rem;color:#66727a;margin-bottom:.2rem}}
</style></head><body><h1>Held-out pitch manipulation</h1>
<p>Operation: voiced F0 scaling; unvoiced frames remain zero.</p>
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
