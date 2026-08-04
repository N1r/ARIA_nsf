#!/usr/bin/env python3
"""Strict acoustic acceptance check for schema-v2 control manipulations.

The checker is intentionally independent of the training/runtime package.  It
only needs NumPy and SoundFile, reads paths contained in ``manipulation.json``
through a traversal-safe resolver, and emits a deterministic JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

SUCCESS_TOKEN = "PHONLAB_CONTROL_MANIPULATION_OK"
REPORT_SCHEMA_VERSION = "1.0"
MANIPULATION_SCHEMA_VERSION = "2.0"
RENDER_SCHEMA_VERSION = "1.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
GAIN_RATIO_RELATIVE_TOLERANCE = 0.02
NOISE_ENERGY_RATIO_RELATIVE_TOLERANCE = 0.05

# Kept local so this acceptance program can verify artifacts even when the
# project itself is not installed in the checking environment.
CONTROL_SPECS: Dict[str, Tuple[float, float, float, Tuple[str, ...]]] = {
    "pitch_semitones": (-36.0, 36.0, 0.0, ("golf", "ddsp", "aria-golf")),
    "output_gain_db": (-24.0, 12.0, 0.0, ("golf", "ddsp", "aria-golf")),
    "noise_gain_db": (-24.0, 24.0, 0.0, ("golf", "ddsp", "aria-golf")),
    "glottal_rd_scale": (0.5, 2.0, 1.0, ("golf", "aria-golf")),
    "f1_scale": (0.7, 1.3, 1.0, ("aria-golf",)),
    "f2_scale": (0.7, 1.3, 1.0, ("aria-golf",)),
    "tilt_alpha_delta": (-0.25, 0.25, 0.0, ("aria-golf",)),
}


@dataclass
class Audio:
    samples: np.ndarray
    sample_rate: int


def check_control_manipulation(baseline: Path, manipulations: Path) -> Dict[str, Any]:
    """Return a machine-readable report for one baseline/control render set."""

    baseline = Path(baseline).expanduser().resolve()
    manipulations = Path(manipulations).expanduser().resolve()
    issues: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": "check_control_manipulation",
        "success": False,
        "success_token": None,
        "inputs": {
            "baseline": str(baseline),
            "manipulations": str(manipulations),
        },
        "metadata": {},
        "baseline": {},
        "conditions": [],
        "noise_symmetric_pairs": [],
        "issues": issues,
    }

    baseline_audio, baseline_names = _load_baseline(baseline, issues)
    report["baseline"] = {
        "files": len(baseline_names),
        "readable_files": len(baseline_audio),
        "sample_rates": sorted({item.sample_rate for item in baseline_audio.values()}),
        "total_samples": int(sum(item.samples.size for item in baseline_audio.values())),
    }

    metadata_path = manipulations / "manipulation.json"
    metadata = _load_json(metadata_path, "metadata.json", issues)
    if metadata is None:
        _finish_report(report)
        return report

    model = _validate_metadata(metadata, report, issues)
    raw_outputs = metadata.get("outputs")
    if not isinstance(raw_outputs, list):
        _finish_report(report)
        return report

    seen_names: set[str] = set()
    seen_directories: set[str] = set()
    noise_difference_energies: Dict[str, Dict[str, float]] = {}
    normalized_controls: Dict[str, Dict[str, float]] = {}

    for index, raw_variant in enumerate(raw_outputs):
        condition, difference_energies, controls = _check_condition(
            index=index,
            raw_variant=raw_variant,
            model=model,
            baseline_root=baseline,
            baseline_audio=baseline_audio,
            baseline_names=baseline_names,
            manipulations=manipulations,
            seen_names=seen_names,
            seen_directories=seen_directories,
            issues=issues,
        )
        report["conditions"].append(condition)
        name = condition.get("name")
        if isinstance(name, str) and controls is not None:
            normalized_controls[name] = controls
            if difference_energies is not None:
                noise_difference_energies[name] = difference_energies

    report["noise_symmetric_pairs"] = _measure_noise_pairs(
        normalized_controls,
        noise_difference_energies,
        issues,
    )
    _finish_report(report)
    return report


def _load_baseline(
    root: Path,
    issues: List[Dict[str, Any]],
) -> Tuple[Dict[str, Audio], set[str]]:
    if not root.is_dir():
        _add_issue(issues, "baseline.missing", "Baseline directory does not exist", path=root)
        return {}, set()
    if root.is_symlink():
        _add_issue(
            issues, "baseline.symlink", "Baseline directory must not be a symlink", path=root
        )
        return {}, set()

    names: set[str] = set()
    audio: Dict[str, Audio] = {}
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        _add_issue(issues, "baseline.read", f"Cannot list baseline directory: {error}", path=root)
        return {}, set()
    for path in entries:
        if path.suffix.lower() != ".wav":
            continue
        if path.is_symlink() or not path.is_file():
            _add_issue(
                issues,
                "baseline.unsafe_file",
                "Baseline WAV must be a regular non-symlink file",
                path=path,
            )
            continue
        names.add(path.name)
        loaded = _read_audio(path, "baseline.audio", issues)
        if loaded is not None:
            audio[path.name] = loaded
    if not names:
        _add_issue(issues, "baseline.empty", "Baseline directory contains no WAV files", path=root)
    return audio, names


def _validate_metadata(
    metadata: Mapping[str, Any],
    report: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Optional[str]:
    if metadata.get("schema_version") != MANIPULATION_SCHEMA_VERSION:
        _add_issue(
            issues,
            "metadata.schema_version",
            "manipulation.json schema_version must be '2.0'",
        )
    if metadata.get("operation") != "ddsp-multi-parameter-control":
        _add_issue(
            issues,
            "metadata.operation",
            "operation must be 'ddsp-multi-parameter-control'",
        )
    if metadata.get("unvoiced_policy") != "preserve-zero":
        _add_issue(
            issues,
            "metadata.unvoiced_policy",
            "unvoiced_policy must be 'preserve-zero'",
        )

    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or SHA256_PATTERN.fullmatch(checkpoint_sha) is None:
        _add_issue(
            issues,
            "metadata.checkpoint_sha256",
            "checkpoint_sha256 must be 64 lowercase hexadecimal characters",
        )

    model = metadata.get("model")
    if model not in {"golf", "ddsp", "aria-golf"}:
        _add_issue(
            issues,
            "metadata.model",
            "model must be one of golf, ddsp, or aria-golf",
        )
        normalized_model = None
    else:
        normalized_model = str(model)

    for key in ("created_at", "experiment", "checkpoint"):
        if not isinstance(metadata.get(key), str) or not str(metadata[key]).strip():
            _add_issue(issues, f"metadata.{key}", f"{key} must be a non-empty string")

    dataset_fingerprint = metadata.get("dataset_fingerprint")
    if (
        not isinstance(dataset_fingerprint, str)
        or SHA256_PATTERN.fullmatch(dataset_fingerprint) is None
    ):
        _add_issue(
            issues,
            "metadata.dataset_fingerprint",
            "dataset_fingerprint must be 64 lowercase hexadecimal characters",
        )

    outputs = metadata.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        _add_issue(
            issues,
            "metadata.outputs",
            "outputs must be a non-empty list of control conditions",
        )

    report["metadata"] = {
        "schema_version": metadata.get("schema_version"),
        "operation": metadata.get("operation"),
        "model": metadata.get("model"),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_fingerprint": dataset_fingerprint,
        "conditions_declared": len(outputs) if isinstance(outputs, list) else 0,
    }
    return normalized_model


def _check_condition(
    *,
    index: int,
    raw_variant: Any,
    model: Optional[str],
    baseline_root: Path,
    baseline_audio: Mapping[str, Audio],
    baseline_names: set[str],
    manipulations: Path,
    seen_names: set[str],
    seen_directories: set[str],
    issues: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    label = f"outputs[{index}]"
    condition: Dict[str, Any] = {
        "index": index,
        "name": None,
        "directory": None,
        "controls": {},
        "is_default": None,
        "files": 0,
        "changed_files": 0,
        "changed_samples": 0,
        "max_abs_difference": None,
        "mean_difference_rms": None,
        "output_gain": None,
    }
    if not isinstance(raw_variant, dict):
        _add_issue(issues, "variant.type", f"{label} must be an object")
        return condition, None, None

    name = raw_variant.get("name")
    directory = raw_variant.get("directory")
    condition["name"] = name
    condition["directory"] = directory
    if not isinstance(name, str) or VARIANT_NAME_PATTERN.fullmatch(name) is None:
        _add_issue(
            issues,
            "variant.name",
            f"{label}.name must be a safe ASCII slug",
            condition=_display_condition(name, index),
        )
    elif name in seen_names:
        _add_issue(
            issues, "variant.name_duplicate", f"Duplicate condition name: {name}", condition=name
        )
    else:
        seen_names.add(name)

    safe_directory = _safe_condition_directory(
        manipulations,
        directory,
        issues,
        condition=_display_condition(name, index),
    )
    if isinstance(directory, str):
        if directory in seen_directories:
            _add_issue(
                issues,
                "variant.directory_duplicate",
                f"Duplicate condition directory: {directory}",
                condition=_display_condition(name, index),
            )
        else:
            seen_directories.add(directory)
    if isinstance(name, str) and isinstance(directory, str) and name != directory:
        _add_issue(
            issues,
            "variant.name_directory",
            "Condition name and directory must be identical",
            condition=name,
        )

    controls = _validate_controls(
        raw_variant.get("controls"),
        model,
        label,
        _display_condition(name, index),
        issues,
    )
    if controls is not None:
        condition["controls"] = controls
        condition["is_default"] = _controls_are_default(controls)

    _check_f0_scale(raw_variant, controls, _display_condition(name, index), issues)
    if not isinstance(raw_variant.get("label"), str) or not raw_variant["label"].strip():
        _add_issue(
            issues,
            "variant.label",
            f"{label}.label must be a non-empty string",
            condition=_display_condition(name, index),
        )

    if safe_directory is None:
        return condition, None, controls

    actual_names = _condition_wav_names(
        safe_directory,
        baseline_names,
        _display_condition(name, index),
        issues,
    )
    condition["files"] = len(actual_names)
    render = _check_render_metadata(
        raw_variant=raw_variant,
        controls=controls,
        condition_dir=safe_directory,
        manipulations=manipulations,
        actual_names=actual_names,
        condition=_display_condition(name, index),
        issues=issues,
    )

    changed_files = 0
    changed_samples = 0
    maximum_difference = 0.0
    difference_rms_values: List[float] = []
    gain_ratios: List[float] = []
    difference_energies: Dict[str, float] = {}

    for filename in sorted(baseline_names & actual_names):
        baseline_item = baseline_audio.get(filename)
        if baseline_item is None:
            continue
        candidate = _read_audio(
            safe_directory / filename,
            "audio.condition_read",
            issues,
            condition=_display_condition(name, index),
        )
        if candidate is None:
            continue
        if candidate.sample_rate != baseline_item.sample_rate:
            _add_issue(
                issues,
                "audio.sample_rate",
                (
                    f"{filename}: sample rate {candidate.sample_rate} does not match "
                    f"baseline {baseline_item.sample_rate}"
                ),
                condition=_display_condition(name, index),
                path=safe_directory / filename,
            )
            continue
        if candidate.samples.shape != baseline_item.samples.shape:
            _add_issue(
                issues,
                "audio.shape",
                (
                    f"{filename}: shape {list(candidate.samples.shape)} does not match "
                    f"baseline {list(baseline_item.samples.shape)}"
                ),
                condition=_display_condition(name, index),
                path=safe_directory / filename,
            )
            continue

        difference = candidate.samples - baseline_item.samples
        different = np.not_equal(candidate.samples, baseline_item.samples)
        file_changed_samples = int(np.count_nonzero(different))
        if file_changed_samples:
            changed_files += 1
            changed_samples += file_changed_samples
        file_maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
        maximum_difference = max(maximum_difference, file_maximum)
        difference_energy = float(np.mean(np.square(difference, dtype=np.float64)))
        difference_energies[filename] = difference_energy
        difference_rms_values.append(math.sqrt(difference_energy))

        if controls is not None and set(controls) == {"output_gain_db"}:
            denominator = _rms(baseline_item.samples)
            if denominator > 0.0:
                gain_ratios.append(_rms(candidate.samples) / denominator)

    condition["changed_files"] = changed_files
    condition["changed_samples"] = changed_samples
    condition["max_abs_difference"] = maximum_difference
    condition["mean_difference_rms"] = (
        float(np.mean(difference_rms_values)) if difference_rms_values else None
    )

    if controls is not None and not _controls_are_default(controls):
        if changed_samples == 0:
            _add_issue(
                issues,
                "audio.unchanged",
                "Non-default condition does not change any comparable baseline sample",
                condition=_display_condition(name, index),
            )
        elif changed_files != len(baseline_names):
            _add_issue(
                issues,
                "audio.partly_unchanged",
                (
                    f"Non-default condition changes only {changed_files}/"
                    f"{len(baseline_names)} baseline WAVs"
                ),
                condition=_display_condition(name, index),
            )

    if controls is not None and set(controls) == {"output_gain_db"}:
        condition["output_gain"] = _measure_output_gain(
            controls["output_gain_db"],
            gain_ratios,
            _display_condition(name, index),
            issues,
        )

    if render is None:
        difference_energies = {}
    return condition, difference_energies, controls


def _validate_controls(
    raw_controls: Any,
    model: Optional[str],
    label: str,
    condition: str,
    issues: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    if not isinstance(raw_controls, dict) or not raw_controls:
        _add_issue(
            issues,
            "variant.controls",
            f"{label}.controls must be a non-empty object",
            condition=condition,
        )
        return None

    normalized: Dict[str, float] = {}
    for raw_name, raw_value in raw_controls.items():
        if not isinstance(raw_name, str) or raw_name not in CONTROL_SPECS:
            _add_issue(
                issues,
                "variant.control_unknown",
                f"Unknown control: {raw_name!r}",
                condition=condition,
            )
            continue
        number = _finite_number(raw_value)
        if number is None:
            _add_issue(
                issues,
                "variant.control_value",
                f"Control {raw_name} must be a finite number",
                condition=condition,
            )
            continue
        minimum, maximum, _, supported_models = CONTROL_SPECS[raw_name]
        if not minimum <= number <= maximum:
            _add_issue(
                issues,
                "variant.control_range",
                f"Control {raw_name}={number:g} is outside [{minimum:g}, {maximum:g}]",
                condition=condition,
            )
            continue
        if model is not None and model not in supported_models:
            _add_issue(
                issues,
                "variant.control_model",
                f"Control {raw_name} is not supported by model {model}",
                condition=condition,
            )
            continue
        normalized[raw_name] = number
    return normalized if len(normalized) == len(raw_controls) else None


def _check_f0_scale(
    variant: Mapping[str, Any],
    controls: Optional[Mapping[str, float]],
    condition: str,
    issues: List[Dict[str, Any]],
) -> None:
    value = _finite_number(variant.get("f0_scale"))
    if value is None or value <= 0.0:
        _add_issue(
            issues,
            "variant.f0_scale",
            "f0_scale must be a finite positive number",
            condition=condition,
        )
        return
    if controls is None:
        return
    expected = 2.0 ** (controls.get("pitch_semitones", 0.0) / 12.0)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        _add_issue(
            issues,
            "variant.f0_scale_mismatch",
            f"f0_scale {value:g} does not match controls (expected {expected:g})",
            condition=condition,
        )


def _safe_condition_directory(
    root: Path,
    raw_directory: Any,
    issues: List[Dict[str, Any]],
    *,
    condition: str,
) -> Optional[Path]:
    if not isinstance(raw_directory, str) or VARIANT_NAME_PATTERN.fullmatch(raw_directory) is None:
        _add_issue(
            issues,
            "variant.directory",
            "Condition directory must be one safe relative ASCII slug",
            condition=condition,
        )
        return None
    candidate = root / raw_directory
    if not _is_beneath(candidate, root):
        _add_issue(
            issues,
            "variant.directory_escape",
            "Condition directory escapes the manipulations root",
            condition=condition,
            path=candidate,
        )
        return None
    if candidate.is_symlink() or not candidate.is_dir():
        _add_issue(
            issues,
            "variant.directory_missing",
            "Condition directory is missing or is a symlink",
            condition=condition,
            path=candidate,
        )
        return None
    return candidate


def _condition_wav_names(
    directory: Path,
    baseline_names: set[str],
    condition: str,
    issues: List[Dict[str, Any]],
) -> set[str]:
    names: set[str] = set()
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as error:
        _add_issue(
            issues,
            "audio.directory_read",
            f"Cannot list condition directory: {error}",
            condition=condition,
            path=directory,
        )
        return names
    for path in entries:
        if path.suffix.lower() != ".wav":
            continue
        if path.is_symlink() or not path.is_file():
            _add_issue(
                issues,
                "audio.unsafe_file",
                "Condition WAV must be a regular non-symlink file",
                condition=condition,
                path=path,
            )
            continue
        names.add(path.name)
    missing = sorted(baseline_names - names)
    extra = sorted(names - baseline_names)
    if missing:
        _add_issue(
            issues,
            "audio.files_missing",
            f"Condition is missing {len(missing)} baseline WAV(s): {_short_names(missing)}",
            condition=condition,
        )
    if extra:
        _add_issue(
            issues,
            "audio.files_extra",
            f"Condition has {len(extra)} extra WAV(s): {_short_names(extra)}",
            condition=condition,
        )
    return names


def _check_render_metadata(
    *,
    raw_variant: Mapping[str, Any],
    controls: Optional[Mapping[str, float]],
    condition_dir: Path,
    manipulations: Path,
    actual_names: set[str],
    condition: str,
    issues: List[Dict[str, Any]],
) -> Optional[Mapping[str, Any]]:
    audit = raw_variant.get("render_audit")
    if not isinstance(audit, dict):
        _add_issue(
            issues,
            "variant.render_audit",
            "render_audit must be an object",
            condition=condition,
        )
        return None

    expected_relative = f"{condition_dir.name}/_render.json"
    recorded_relative = audit.get("metadata")
    if recorded_relative != expected_relative:
        _add_issue(
            issues,
            "render.metadata_path",
            f"render_audit.metadata must be exactly {expected_relative!r}",
            condition=condition,
        )
        return None
    render_path = _safe_relative_file(manipulations, recorded_relative)
    if render_path is None or render_path != condition_dir / "_render.json":
        _add_issue(
            issues,
            "render.metadata_path_unsafe",
            "Recorded _render.json path is unsafe",
            condition=condition,
        )
        return None
    render = _load_json(render_path, "render.json", issues, condition=condition)
    if render is None:
        return None
    if render.get("schema_version") != RENDER_SCHEMA_VERSION:
        _add_issue(
            issues,
            "render.schema_version",
            "_render.json schema_version must be '1.0'",
            condition=condition,
        )

    expected_runtime_controls = (
        {key: value for key, value in controls.items() if key != "pitch_semitones"}
        if controls is not None
        else None
    )
    render_controls = _numeric_mapping(render.get("controls"))
    if expected_runtime_controls is None or render_controls != expected_runtime_controls:
        _add_issue(
            issues,
            "render.controls",
            "_render.json controls do not exactly match runtime condition controls",
            condition=condition,
        )

    capabilities = render.get("runtime_capabilities")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(item, str) for item in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        _add_issue(
            issues,
            "render.runtime_capabilities",
            "runtime_capabilities must be a unique list of strings",
            condition=condition,
        )
    elif expected_runtime_controls is not None:
        missing_capabilities = sorted(set(expected_runtime_controls) - set(capabilities))
        if missing_capabilities:
            _add_issue(
                issues,
                "render.runtime_capabilities_missing",
                "Runtime capabilities omit requested control(s): "
                + ", ".join(missing_capabilities),
                condition=condition,
            )

    files_written = _nonnegative_integer(render.get("files_written"))
    if files_written is None or files_written != len(actual_names):
        _add_issue(
            issues,
            "render.files_written",
            (f"_render.json files_written must equal actual WAV count ({len(actual_names)})"),
            condition=condition,
        )
    audit_files_written = _nonnegative_integer(audit.get("files_written"))
    if audit_files_written is None or audit_files_written != files_written:
        _add_issue(
            issues,
            "render.audit_files_written",
            "render_audit.files_written does not match _render.json",
            condition=condition,
        )

    decoder_calls = _nonnegative_integer(render.get("decoder_control_calls"))
    audit_decoder_calls = _nonnegative_integer(audit.get("decoder_control_calls"))
    if decoder_calls is None or audit_decoder_calls != decoder_calls:
        _add_issue(
            issues,
            "render.decoder_control_calls",
            "render_audit decoder_control_calls does not match _render.json",
            condition=condition,
        )
    decoder_controls = (
        set(expected_runtime_controls) - {"output_gain_db"}
        if expected_runtime_controls is not None
        else set()
    )
    if decoder_controls and (decoder_calls is None or decoder_calls == 0):
        _add_issue(
            issues,
            "render.decoder_control_not_applied",
            (
                "Decoder-internal controls were requested but decoder_control_calls "
                "is zero: " + ", ".join(sorted(decoder_controls))
            ),
            condition=condition,
        )

    audit_capabilities = audit.get("runtime_capabilities")
    if capabilities != audit_capabilities:
        _add_issue(
            issues,
            "render.audit_capabilities",
            "render_audit runtime_capabilities does not match _render.json",
            condition=condition,
        )

    _check_clipping(render, audit, actual_names, condition_dir, condition, issues)
    return render


def _check_clipping(
    render: Mapping[str, Any],
    audit: Mapping[str, Any],
    actual_names: set[str],
    condition_dir: Path,
    condition: str,
    issues: List[Dict[str, Any]],
) -> None:
    clipped_samples = _nonnegative_integer(render.get("clipped_samples"))
    samples = _positive_integer(render.get("samples"))
    clipped_fraction = _fraction(render.get("clipped_fraction"))
    audit_fraction = _fraction(audit.get("clipped_fraction"))
    file_rows = render.get("files")
    file_clipped = 0
    file_samples = 0
    recorded_names: List[str] = []

    if not isinstance(file_rows, list):
        _add_issue(
            issues,
            "render.files",
            "_render.json files must be a list",
            condition=condition,
        )
    else:
        for index, row in enumerate(file_rows):
            if not isinstance(row, dict):
                _add_issue(
                    issues,
                    "render.file_entry",
                    f"files[{index}] must be an object",
                    condition=condition,
                )
                continue
            raw_path = row.get("path")
            if (
                not isinstance(raw_path, str)
                or Path(raw_path).name != raw_path
                or raw_path not in actual_names
                or _safe_relative_file(condition_dir, raw_path) is None
            ):
                _add_issue(
                    issues,
                    "render.file_path",
                    f"files[{index}].path is unsafe or is not an output WAV",
                    condition=condition,
                )
            else:
                recorded_names.append(raw_path)
                try:
                    info = sf.info(str(condition_dir / raw_path))
                except (OSError, RuntimeError) as error:
                    _add_issue(
                        issues,
                        "render.file_info",
                        f"Cannot inspect {raw_path}: {error}",
                        condition=condition,
                    )
                else:
                    row_samples = _positive_integer(row.get("samples"))
                    expected_samples = int(info.frames * info.channels)
                    if row_samples != expected_samples:
                        _add_issue(
                            issues,
                            "render.file_samples",
                            (
                                f"{raw_path}: recorded samples must equal decoded sample "
                                f"count {expected_samples}"
                            ),
                            condition=condition,
                        )
            for peak_name in ("peak_before_gain", "peak_after_gain_unclipped"):
                peak = _finite_number(row.get(peak_name))
                if peak is None or peak < 0.0:
                    _add_issue(
                        issues,
                        "render.file_peak",
                        f"files[{index}].{peak_name} must be a finite non-negative number",
                        condition=condition,
                    )
            row_clipped = _nonnegative_integer(row.get("clipped_samples"))
            row_samples = _positive_integer(row.get("samples"))
            if row_clipped is None:
                _add_issue(
                    issues,
                    "render.file_clipping",
                    f"files[{index}].clipped_samples must be a non-negative integer",
                    condition=condition,
                )
            else:
                file_clipped += row_clipped
            if row_samples is None:
                _add_issue(
                    issues,
                    "render.file_samples",
                    f"files[{index}].samples must be a positive integer",
                    condition=condition,
                )
            else:
                file_samples += row_samples
        if len(recorded_names) != len(set(recorded_names)):
            _add_issue(
                issues,
                "render.file_duplicate",
                "_render.json files contains duplicate paths",
                condition=condition,
            )
        if set(recorded_names) != actual_names:
            _add_issue(
                issues,
                "render.file_set",
                "_render.json files does not exactly enumerate condition WAVs",
                condition=condition,
            )

    if clipped_samples is None or clipped_samples != file_clipped:
        _add_issue(
            issues,
            "render.clipped_samples",
            "Global clipped_samples does not equal per-file sum",
            condition=condition,
        )
    if samples is None or samples != file_samples:
        _add_issue(
            issues,
            "render.samples",
            "Global samples does not equal per-file sum",
            condition=condition,
        )
    expected_fraction = (
        clipped_samples / samples
        if clipped_samples is not None and samples is not None and samples > 0
        else None
    )
    if (
        clipped_fraction is None
        or expected_fraction is None
        or not math.isclose(clipped_fraction, expected_fraction, abs_tol=1e-15)
    ):
        _add_issue(
            issues,
            "render.clipped_fraction",
            "clipped_fraction does not match clipped_samples / samples",
            condition=condition,
        )
    if (
        audit_fraction is None
        or clipped_fraction is None
        or not math.isclose(audit_fraction, clipped_fraction, abs_tol=1e-15)
    ):
        _add_issue(
            issues,
            "render.audit_clipping",
            "render_audit clipped_fraction does not match _render.json",
            condition=condition,
        )
    if clipped_samples is not None and clipped_samples > 0:
        _add_issue(
            issues,
            "render.clipping_nonzero",
            f"Condition reports {clipped_samples} clipped sample(s)",
            condition=condition,
        )


def _measure_output_gain(
    gain_db: float,
    ratios: Sequence[float],
    condition: str,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected = 10.0 ** (gain_db / 20.0)
    if not ratios:
        _add_issue(
            issues,
            "metric.output_gain_unmeasurable",
            "No non-silent, shape-compatible WAV can measure output gain",
            condition=condition,
        )
        return {
            "gain_db": gain_db,
            "files_measured": 0,
            "expected_median_rms_ratio": expected,
            "measured_median_rms_ratio": None,
            "absolute_error": None,
            "relative_error": None,
            "within_tolerance": False,
        }
    measured = float(np.median(np.asarray(ratios, dtype=np.float64)))
    absolute_error = abs(measured - expected)
    relative_error = absolute_error / expected
    within_tolerance = relative_error <= GAIN_RATIO_RELATIVE_TOLERANCE
    if not within_tolerance:
        _add_issue(
            issues,
            "metric.output_gain_ratio",
            (
                f"Measured median RMS ratio {measured:.9g} differs from theoretical "
                f"{expected:.9g} by {relative_error:.3%}"
            ),
            condition=condition,
        )
    return {
        "gain_db": gain_db,
        "files_measured": len(ratios),
        "expected_median_rms_ratio": expected,
        "measured_median_rms_ratio": measured,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "relative_tolerance": GAIN_RATIO_RELATIVE_TOLERANCE,
        "within_tolerance": within_tolerance,
    }


def _measure_noise_pairs(
    controls_by_name: Mapping[str, Mapping[str, float]],
    energies_by_name: Mapping[str, Mapping[str, float]],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    negative: List[Tuple[str, float]] = []
    positive: List[Tuple[str, float]] = []
    for name, controls in controls_by_name.items():
        if set(controls) != {"noise_gain_db"}:
            continue
        gain = controls["noise_gain_db"]
        if gain < 0.0:
            negative.append((name, gain))
        elif gain > 0.0:
            positive.append((name, gain))

    pairs: List[Dict[str, Any]] = []
    for negative_name, negative_gain in sorted(negative):
        matches = [
            (positive_name, positive_gain)
            for positive_name, positive_gain in positive
            if math.isclose(positive_gain, -negative_gain, abs_tol=1e-12)
        ]
        for positive_name, positive_gain in sorted(matches):
            common_names = sorted(
                set(energies_by_name.get(negative_name, {}))
                & set(energies_by_name.get(positive_name, {}))
            )
            ratios = [
                energies_by_name[positive_name][filename]
                / energies_by_name[negative_name][filename]
                for filename in common_names
                if energies_by_name[negative_name][filename] > 0.0
            ]
            expected = 10.0 ** (positive_gain / 10.0)
            measured = float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None
            absolute_error = abs(measured - expected) if measured is not None else None
            relative_error = absolute_error / expected if absolute_error is not None else None
            within_tolerance = (
                relative_error is not None
                and relative_error <= NOISE_ENERGY_RATIO_RELATIVE_TOLERANCE
            )
            pair = {
                "negative_condition": negative_name,
                "positive_condition": positive_name,
                "symmetric_gain_db": positive_gain,
                "files_measured": len(ratios),
                "expected_difference_energy_ratio": expected,
                "measured_difference_energy_ratio": measured,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "relative_tolerance": NOISE_ENERGY_RATIO_RELATIVE_TOLERANCE,
                "within_tolerance": within_tolerance,
            }
            pairs.append(pair)
            if not ratios:
                _add_issue(
                    issues,
                    "metric.noise_pair_unmeasurable",
                    "Symmetric noise pair has no comparable non-zero differences",
                    condition=f"{negative_name}/{positive_name}",
                )
            elif not within_tolerance:
                _add_issue(
                    issues,
                    "metric.noise_energy_ratio",
                    (
                        f"Symmetric noise difference-energy ratio {measured:.9g} "
                        f"differs from theoretical {expected:.9g} by "
                        f"{relative_error:.3%}"
                    ),
                    condition=f"{negative_name}/{positive_name}",
                )
    return pairs


def _load_json(
    path: Path,
    code: str,
    issues: List[Dict[str, Any]],
    *,
    condition: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _add_issue(
            issues,
            f"{code}.missing",
            "Required JSON file is missing or is a symlink",
            condition=condition,
            path=path,
        )
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _add_issue(
            issues,
            f"{code}.invalid",
            f"Cannot read valid JSON: {error}",
            condition=condition,
            path=path,
        )
        return None
    if not isinstance(value, dict):
        _add_issue(
            issues,
            f"{code}.type",
            "JSON top level must be an object",
            condition=condition,
            path=path,
        )
        return None
    return value


def _read_audio(
    path: Path,
    code: str,
    issues: List[Dict[str, Any]],
    *,
    condition: Optional[str] = None,
) -> Optional[Audio]:
    try:
        samples, sample_rate = sf.read(
            str(path),
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError) as error:
        _add_issue(
            issues,
            code,
            f"Cannot decode WAV: {error}",
            condition=condition,
            path=path,
        )
        return None
    if samples.size == 0:
        _add_issue(
            issues,
            code,
            "WAV contains no audio samples",
            condition=condition,
            path=path,
        )
        return None
    if not np.isfinite(samples).all():
        _add_issue(
            issues,
            code,
            "WAV contains NaN or infinite samples",
            condition=condition,
            path=path,
        )
        return None
    return Audio(samples=samples, sample_rate=int(sample_rate))


def _safe_relative_file(root: Path, raw_path: Any) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink() or not _is_beneath(candidate, root):
        return None
    return candidate


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _numeric_mapping(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    result: Dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        number = _finite_number(item)
        if number is None:
            return None
        result[key] = number
    return result


def _controls_are_default(controls: Mapping[str, float]) -> bool:
    return all(
        math.isclose(value, CONTROL_SPECS[name][2], abs_tol=1e-12)
        for name, value in controls.items()
    )


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_integer(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _positive_integer(value: Any) -> Optional[int]:
    number = _nonnegative_integer(value)
    return number if number is not None and number > 0 else None


def _fraction(value: Any) -> Optional[float]:
    number = _finite_number(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def _rms(samples: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))


def _display_condition(name: Any, index: int) -> str:
    return name if isinstance(name, str) and name else f"outputs[{index}]"


def _short_names(names: Sequence[str], limit: int = 5) -> str:
    visible = list(names[:limit])
    suffix = "" if len(names) <= limit else f", ... (+{len(names) - limit})"
    return ", ".join(visible) + suffix


def _add_issue(
    issues: List[Dict[str, Any]],
    code: str,
    message: str,
    *,
    condition: Optional[str] = None,
    path: Optional[Path] = None,
) -> None:
    issue: Dict[str, Any] = {"code": code, "message": message}
    if condition is not None:
        issue["condition"] = condition
    if path is not None:
        issue["path"] = str(path)
    issues.append(issue)


def _finish_report(report: Dict[str, Any]) -> None:
    report["success"] = not report["issues"]
    report["success_token"] = SUCCESS_TOKEN if report["success"] else None


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly verify schema-v2 DDSP control renders against baseline WAVs "
            "and emit a machine-readable acoustic report"
        )
    )
    parser.add_argument("baseline", type=Path, help="baseline reconstruction directory")
    parser.add_argument(
        "manipulations",
        type=Path,
        help="directory containing manipulation.json and condition subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path; stdout is the success token when supplied",
    )
    args = parser.parse_args(argv)

    report = check_control_manipulation(args.baseline, args.manipulations)
    if args.output is not None:
        try:
            _write_json_atomic(args.output, report)
        except OSError as error:
            print(f"Could not write JSON report: {error}", file=sys.stderr)
            return 2
        if report["success"]:
            print(SUCCESS_TOKEN)
        else:
            for issue in report["issues"]:
                print(
                    f"{issue['code']}: {issue['message']}",
                    file=sys.stderr,
                )
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
