"""Portable dataset manifests and deterministic preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .audio import (
    SUPPORTED_SUFFIXES,
    audio_statistics,
    estimate_f0,
    read_audio,
    resample,
    write_wav,
)

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetRecord:
    """One prepared utterance: file paths, provenance hash, and audio statistics."""

    id: str
    split: str
    audio_path: str
    f0_path: str
    source_path: str
    source_sha256: str
    sample_rate: int
    samples: int
    duration_s: float
    peak: float
    rms_dbfs: float
    clipped_fraction: float
    dc_offset: float
    f0_backend: str
    median_f0_hz: float
    voiced_fraction: float


@dataclass
class DatasetManifest:
    """A prepared dataset: its root directory, records, and dataset.json metadata."""

    root: Path
    records: list[DatasetRecord]
    metadata: dict

    @classmethod
    def load(cls, dataset: Path) -> "DatasetManifest":
        """Load manifest.csv (given the dataset directory or the CSV itself)."""
        dataset = Path(dataset).resolve()
        manifest_path = dataset if dataset.name == "manifest.csv" else dataset / "manifest.csv"
        root = manifest_path.parent
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        records = [
            DatasetRecord(
                **{
                    field: (
                        int(row[field])
                        if field in {"sample_rate", "samples"}
                        else float(row[field])
                        if field
                        in {
                            "duration_s",
                            "peak",
                            "rms_dbfs",
                            "clipped_fraction",
                            "dc_offset",
                            "median_f0_hz",
                            "voiced_fraction",
                        }
                        else row[field]
                    )
                    for field in DatasetRecord.__dataclass_fields__
                }
            )
            for row in rows
        ]
        metadata_path = root / "dataset.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return cls(root=root, records=records, metadata=metadata)

    def save(self) -> None:
        """Write manifest.csv and dataset.json under the manifest root."""
        self.root.mkdir(parents=True, exist_ok=True)
        fields = list(DatasetRecord.__dataclass_fields__)
        with (self.root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in self.records)
        (self.root / "dataset.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @property
    def fingerprint(self) -> str:
        """Order-independent SHA-256 over (id, source hash, split, sample rate) of every record.

        Experiments record this value and refuse to run if it changes.
        """
        digest = hashlib.sha256()
        for record in sorted(self.records, key=lambda item: item.id):
            digest.update(record.id.encode())
            digest.update(record.source_sha256.encode())
            digest.update(record.split.encode())
            digest.update(str(record.sample_rate).encode())
        return digest.hexdigest()


def discover_audio(root: Path) -> list[Path]:
    """Recursively list supported audio files in stable sorted order."""
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def prepare_dataset(
    source: Path,
    output: Path,
    *,
    sample_rate: int = 16000,
    f0_method: str = "auto",
    f0_floor: float = 60.0,
    f0_ceiling: float = 500.0,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_duration: float = 0.25,
    normalize_peak: Optional[float] = None,
    overwrite: bool = False,
) -> DatasetManifest:
    """Resample, extract F0, split deterministically, and write a manifest.

    The dataset is built in a staging directory and renamed into place only on
    success, so an existing output directory is never partially overwritten.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Audio source is not a directory: {source}")
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"{output} already exists; choose another output directory or set overwrite=True"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be >= 0 and sum to < 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < f0_floor < f0_ceiling:
        raise ValueError("F0 limits must satisfy 0 < floor < ceiling")
    if normalize_peak is not None and not 0 < normalize_peak <= 1:
        raise ValueError("normalize_peak must be in (0, 1]")
    files = discover_audio(source)
    if not files:
        raise ValueError(f"No supported audio files found below {source}")

    # PID-suffixed staging avoids collisions between concurrent preparations.
    staging = output.parent / f".{output.name}.preparing-{os.getpid()}"
    if staging.exists():
        # ignore_errors: NFS silly-rename (.nfs*) files can block deletion.
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    records: list[DatasetRecord] = []
    skipped_invalid: list[dict] = []
    skipped_too_short: list[str] = []
    try:
        splits = _split_map(files, source, seed, validation_ratio, test_ratio)
        for path in files:
            relative_source = path.relative_to(source).as_posix()
            try:
                source_hash = _sha256(path)
                split = splits[path]
                safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-") or "audio"
                # The hash suffix keeps ids unique when different files share a stem.
                item_id = f"{safe_stem}-{source_hash[:10]}"
                relative_audio = Path("audio") / f"{item_id}.wav"
                relative_f0 = Path("f0") / f"{item_id}.f0.txt"

                audio, original_rate = read_audio(path)
                if audio.ndim != 1 or not np.isfinite(audio).all():
                    raise ValueError(f"Invalid samples in {path}")
                audio = resample(audio, original_rate, sample_rate)
                if len(audio) / sample_rate < min_duration:
                    skipped_too_short.append(relative_source)
                    continue
                if normalize_peak is not None:
                    peak = float(np.max(np.abs(audio), initial=0))
                    if peak > 0:
                        audio = audio * (normalize_peak / peak)
                statistics = audio_statistics(audio, sample_rate)
                if f0_method == "sidecar":
                    sidecar = path.with_suffix(".pv")
                    if not sidecar.is_file():
                        raise FileNotFoundError(f"F0 sidecar is missing: {sidecar}")
                    f0 = np.loadtxt(sidecar, dtype=np.float32, ndmin=1)
                    if not np.isfinite(f0).all() or np.any(f0 < 0):
                        raise ValueError(f"Invalid F0 values in {sidecar}")
                    # Matches the 5 ms hop lightning.py assumes when upsampling F0.
                    f0_hop_seconds = 0.005
                    audio_seconds = len(audio) / sample_rate
                    sidecar_seconds = len(f0) * f0_hop_seconds
                    if abs(audio_seconds - sidecar_seconds) > max(0.1, 2 * f0_hop_seconds):
                        expected_frames = audio_seconds / f0_hop_seconds
                        raise ValueError(
                            f"F0 sidecar frame count does not match audio duration in "
                            f"{sidecar}: expected ~{expected_frames:.1f} frames "
                            f"({audio_seconds:.3f}s audio) but got {len(f0)} frames "
                            f"({sidecar_seconds:.3f}s) at a {f0_hop_seconds * 1000:.0f} ms hop"
                        )
                    backend = "sidecar-pv"
                else:
                    f0, backend = estimate_f0(
                        audio, sample_rate, f0_floor, f0_ceiling, method=f0_method
                    )
                voiced = f0[f0 > 0]
                write_wav(staging / relative_audio, audio, sample_rate)
                (staging / relative_f0).parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(staging / relative_f0, f0, fmt="%.5f")
                records.append(
                    DatasetRecord(
                        id=item_id,
                        split=split,
                        audio_path=relative_audio.as_posix(),
                        f0_path=relative_f0.as_posix(),
                        source_path=relative_source,
                        source_sha256=source_hash,
                        sample_rate=sample_rate,
                        samples=len(audio),
                        duration_s=statistics["duration_s"],
                        peak=statistics["peak"],
                        rms_dbfs=statistics["rms_dbfs"],
                        clipped_fraction=statistics["clipped_fraction"],
                        dc_offset=statistics["dc_offset"],
                        f0_backend=backend,
                        median_f0_hz=float(np.median(voiced)) if voiced.size else 0.0,
                        voiced_fraction=float(np.mean(f0 > 0)),
                    )
                )
            except Exception as error:
                # One corrupt/mismatched file should not discard the work already
                # done on every other file in the source tree.
                skipped_invalid.append({"file": relative_source, "error": str(error)})
        if not records:
            raise ValueError(
                f"No usable audio files below {source}: {len(skipped_too_short)} too short, "
                f"{len(skipped_invalid)} invalid/unreadable"
            )
        manifest = DatasetManifest(
            root=staging,
            records=records,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source),
                "sample_rate": sample_rate,
                "f0_method_requested": f0_method,
                "f0_floor_hz": f0_floor,
                "f0_ceiling_hz": f0_ceiling,
                "split_seed": seed,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "min_duration_s": min_duration,
                "normalize_peak": normalize_peak,
                "tool": "aris",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "skipped": {"invalid": skipped_invalid, "too_short": skipped_too_short},
            },
        )
        manifest.metadata["dataset_fingerprint"] = manifest.fingerprint
        manifest.save()
        staging.rename(output)
        manifest.root = output
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _split_map(files, root, seed, validation_ratio, test_ratio):
    """Assign deterministic splits and keep small corpora trainable."""
    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root).as_posix()}".encode()
        ).digest(),
    )
    count = len(ranked)
    test_count = max(1, round(count * test_ratio)) if test_ratio and count >= 3 else 0
    validation_count = (
        max(1, round(count * validation_ratio)) if validation_ratio and count >= 3 else 0
    )
    while test_count + validation_count >= count:
        if test_count >= validation_count and test_count:
            test_count -= 1
        elif validation_count:
            validation_count -= 1
    result = {path: "train" for path in ranked}
    for path in ranked[:test_count]:
        result[path] = "test"
    for path in ranked[test_count : test_count + validation_count]:
        result[path] = "validation"
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: DatasetManifest | Path | str) -> list[str]:
    """Return a list of integrity errors; an empty list means the dataset is valid."""
    if isinstance(manifest, (str, Path)):
        manifest = DatasetManifest.load(manifest)
    errors = []
    ids = set()
    for record in manifest.records:
        if record.id in ids:
            errors.append(f"duplicate id: {record.id}")
        ids.add(record.id)
        if record.split not in {"train", "validation", "test"}:
            errors.append(f"{record.id}: invalid split {record.split}")
        for key in ("audio_path", "f0_path"):
            path = (manifest.root / getattr(record, key)).resolve()
            if manifest.root not in path.parents:
                errors.append(f"{record.id}: {key} escapes dataset root")
            elif not path.is_file():
                errors.append(f"{record.id}: missing {key} ({path})")
        if record.duration_s <= 0:
            errors.append(f"{record.id}: non-positive duration")
    return errors


def summarize(records: Iterable[DatasetRecord], skipped: Optional[dict] = None) -> dict:
    """Aggregate record counts, duration, splits, and quality warnings.

    ``skipped`` (see ``prepare_dataset``'s ``metadata["skipped"]``) adds a
    ``skipped`` key with per-reason counts when provided.
    """
    records = list(records)
    by_split = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
    }
    result = {
        "files": len(records),
        "duration_s": sum(record.duration_s for record in records),
        "splits": by_split,
        "sample_rates": sorted({record.sample_rate for record in records}),
        "clipped_files": sum(record.clipped_fraction > 0 for record in records),
        "unvoiced_files": sum(record.voiced_fraction == 0 for record in records),
    }
    if skipped is not None:
        result["skipped"] = {
            reason: (len(entries) if isinstance(entries, list) else entries)
            for reason, entries in skipped.items()
        }
    return result
