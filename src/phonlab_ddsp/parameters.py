"""Portable acoustic-parameter exports for dataset inspection and analysis."""

from __future__ import annotations

import csv
import math
import os
import statistics
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .manifest import DatasetManifest, DatasetRecord, validate_manifest

PARAMETER_SCHEMA_VERSION = "1.0"
PARAMETER_SCHEMA: Tuple[str, ...] = (
    "id",
    "split",
    "duration_s",
    "sample_rate",
    "rms_dbfs",
    "peak",
    "clipped_fraction",
    "dc_offset",
    "median_f0_hz",
    "voiced_fraction",
    "f0_backend",
    "audio_path",
    "f0_path",
)

DatasetInput = Union[str, os.PathLike, DatasetManifest]
StatisticValue = Optional[Union[int, float]]


def export_parameters(dataset_or_manifest: DatasetInput, output: Union[str, os.PathLike]) -> Path:
    """Export one deterministic CSV row per dataset record.

    ``output`` is created atomically and is never overwritten. Paths remain relative to the
    dataset root so that the export stays portable with the prepared dataset.
    """

    output_path = Path(output).expanduser().absolute()
    if os.path.lexists(str(output_path)):
        raise FileExistsError(f"Parameter export already exists: {output_path}")

    manifest = _load_and_validate(dataset_or_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = _new_staging_path(output_path)
    try:
        with staging.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PARAMETER_SCHEMA)
            writer.writeheader()
            for record in sorted(manifest.records, key=lambda item: item.id):
                writer.writerow(_record_row(record))
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(staging, output_path)
        except FileExistsError:
            raise FileExistsError(f"Parameter export already exists: {output_path}") from None
    finally:
        if staging.exists():
            staging.unlink()
    return output_path


def parameter_summary(dataset_or_manifest: DatasetInput) -> Dict[str, object]:
    """Return dependency-free descriptive statistics for a validated dataset."""

    manifest = _load_and_validate(dataset_or_manifest)
    records = sorted(manifest.records, key=lambda item: item.id)
    duration_s = math.fsum(record.duration_s for record in records)
    voiced_f0 = [record.median_f0_hz for record in records if record.median_f0_hz > 0]
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
    }
    return {
        "schema_version": PARAMETER_SCHEMA_VERSION,
        "files": len(records),
        "duration_minutes": duration_s / 60.0,
        "splits": split_counts,
        "sample_rates": sorted({record.sample_rate for record in records}),
        "median_f0_hz": _descriptive_statistics(voiced_f0),
        "voiced_fraction": _descriptive_statistics(record.voiced_fraction for record in records),
        "rms_dbfs": _descriptive_statistics(record.rms_dbfs for record in records),
    }


def _load_and_validate(dataset_or_manifest: DatasetInput) -> DatasetManifest:
    if isinstance(dataset_or_manifest, DatasetManifest):
        manifest = dataset_or_manifest
    else:
        try:
            dataset_path = Path(dataset_or_manifest)
        except TypeError as error:
            raise TypeError(
                "dataset_or_manifest must be a dataset path, manifest.csv path, or DatasetManifest"
            ) from error
        try:
            manifest = DatasetManifest.load(dataset_path)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid dataset manifest at {dataset_path}: {error}") from error

    errors = validate_manifest(manifest)
    if not manifest.records:
        errors.append("manifest contains no records")
    errors.extend(_numeric_validation_errors(manifest.records))
    if errors:
        detail = "\n".join(f"- {message}" for message in errors)
        raise ValueError(f"Dataset manifest validation failed:\n{detail}")
    return manifest


def _numeric_validation_errors(records: Iterable[DatasetRecord]) -> List[str]:
    errors: List[str] = []
    for record in records:
        numeric = {
            "duration_s": record.duration_s,
            "rms_dbfs": record.rms_dbfs,
            "peak": record.peak,
            "clipped_fraction": record.clipped_fraction,
            "dc_offset": record.dc_offset,
            "median_f0_hz": record.median_f0_hz,
            "voiced_fraction": record.voiced_fraction,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                errors.append(f"{record.id}: non-finite {name}")
        if record.sample_rate <= 0:
            errors.append(f"{record.id}: sample_rate must be positive")
        if record.peak < 0:
            errors.append(f"{record.id}: peak must be non-negative")
        if not 0 <= record.clipped_fraction <= 1:
            errors.append(f"{record.id}: clipped_fraction must be in [0, 1]")
        if record.median_f0_hz < 0:
            errors.append(f"{record.id}: median_f0_hz must be non-negative")
        if not 0 <= record.voiced_fraction <= 1:
            errors.append(f"{record.id}: voiced_fraction must be in [0, 1]")
        if record.voiced_fraction == 0 and record.median_f0_hz != 0:
            errors.append(f"{record.id}: unvoiced record has non-zero median_f0_hz")
        if record.voiced_fraction > 0 and record.median_f0_hz <= 0:
            errors.append(f"{record.id}: voiced record has no positive median_f0_hz")
        if not record.f0_backend.strip():
            errors.append(f"{record.id}: f0_backend is empty")
    return errors


def _record_row(record: DatasetRecord) -> Mapping[str, object]:
    return {
        "id": record.id,
        "split": record.split,
        "duration_s": _format_float(record.duration_s),
        "sample_rate": record.sample_rate,
        "rms_dbfs": _format_float(record.rms_dbfs),
        "peak": _format_float(record.peak),
        "clipped_fraction": _format_float(record.clipped_fraction),
        "dc_offset": _format_float(record.dc_offset),
        "median_f0_hz": _format_float(record.median_f0_hz),
        "voiced_fraction": _format_float(record.voiced_fraction),
        "f0_backend": record.f0_backend,
        "audio_path": record.audio_path,
        "f0_path": record.f0_path,
    }


def _format_float(value: float) -> str:
    return format(value, ".12g")


def _descriptive_statistics(values: Iterable[float]) -> Dict[str, StatisticValue]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None,
        }
    if len(ordered) == 1:
        q25 = q75 = ordered[0]
    else:
        q25, _, q75 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "q25": q25,
        "median": statistics.median(ordered),
        "q75": q75,
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _new_staging_path(output: Path) -> Path:
    for _ in range(100):
        candidate = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
        if not os.path.lexists(str(candidate)):
            return candidate
    raise FileExistsError(f"Could not allocate a staging file beside {output}")
