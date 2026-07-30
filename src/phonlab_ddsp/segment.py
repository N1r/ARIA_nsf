"""Deterministic audio segmentation with an auditable CSV manifest."""

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
from typing import Optional

import numpy as np

from .audio import SUPPORTED_SUFFIXES, read_audio, resample, write_wav
from .manifest import discover_audio


@dataclass(frozen=True)
class SegmentRecord:
    id: str
    source_path: str
    source_sha256: str
    source_f0_sha256: str
    segment_path: str
    f0_path: str
    start_s: float
    end_s: float
    duration_s: float
    sample_rate: int
    samples: int
    mode: str


@dataclass
class SplitResult:
    root: Path
    records: list[SegmentRecord]
    metadata: dict

    def save(self) -> None:
        fields = list(SegmentRecord.__dataclass_fields__)
        with (self.root / "segments.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in self.records)
        (self.root / "split.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def split_audio(
    source: Path,
    output: Path,
    *,
    mode: str = "silence",
    segment_seconds: float = 2.0,
    overlap_seconds: float = 0.0,
    silence_threshold_db: float = -40.0,
    min_silence_seconds: float = 0.30,
    padding_seconds: float = 0.05,
    min_duration_seconds: float = 0.25,
    max_duration_seconds: float = 15.0,
    sample_rate: Optional[int] = None,
    keep_tail: bool = True,
    split_f0_sidecars: bool = False,
    f0_hop_seconds: float = 0.005,
) -> SplitResult:
    """Split one file or a directory tree into mono PCM WAV files.

    The output is built in a staging directory and renamed only after all source
    files succeed. Existing output directories are never overwritten.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio source does not exist: {source}")
    if source.is_file() and source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported audio file: {source}")
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"Audio source is neither a file nor directory: {source}")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another output directory")
    if mode not in {"fixed", "silence"}:
        raise ValueError("mode must be 'fixed' or 'silence'")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= segment_seconds:
        raise ValueError("overlap_seconds must satisfy 0 <= overlap < segment_seconds")
    if min_silence_seconds <= 0:
        raise ValueError("min_silence_seconds must be positive")
    if padding_seconds < 0:
        raise ValueError("padding_seconds must not be negative")
    if min_duration_seconds <= 0:
        raise ValueError("min_duration_seconds must be positive")
    if max_duration_seconds < min_duration_seconds:
        raise ValueError("max_duration_seconds must be >= min_duration_seconds")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if f0_hop_seconds <= 0:
        raise ValueError("f0_hop_seconds must be positive")

    if source.is_file():
        source_root = source.parent
        files = [source]
    else:
        source_root = source
        files = discover_audio(source)
    if not files:
        raise ValueError(f"No supported audio files found below {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.splitting-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "audio").mkdir(parents=True)
    records: list[SegmentRecord] = []
    skipped = []
    try:
        for path in files:
            audio, original_rate = read_audio(path)
            target_rate = sample_rate or original_rate
            audio = resample(audio, original_rate, target_rate)
            if not np.isfinite(audio).all():
                raise ValueError(f"Invalid samples in {path}")
            sidecar_f0 = None
            sidecar_hash = ""
            if split_f0_sidecars:
                sidecar = path.with_suffix(".pv")
                if not sidecar.is_file():
                    raise FileNotFoundError(f"F0 sidecar is missing: {sidecar}")
                sidecar_f0 = np.loadtxt(sidecar, dtype=np.float32, ndmin=1)
                if not np.isfinite(sidecar_f0).all() or np.any(sidecar_f0 < 0):
                    raise ValueError(f"Invalid F0 values in {sidecar}")
                audio_seconds = len(audio) / target_rate
                sidecar_seconds = len(sidecar_f0) * f0_hop_seconds
                if abs(audio_seconds - sidecar_seconds) > max(0.1, 2 * f0_hop_seconds):
                    raise ValueError(
                        f"F0 sidecar duration differs from audio by more than 100 ms: {sidecar}"
                    )
                sidecar_hash = _sha256(sidecar)
            if mode == "fixed":
                boundaries = _fixed_boundaries(
                    len(audio),
                    target_rate,
                    segment_seconds,
                    overlap_seconds,
                    min_duration_seconds,
                    keep_tail,
                )
            else:
                boundaries = _silence_boundaries(
                    audio,
                    target_rate,
                    silence_threshold_db,
                    min_silence_seconds,
                    padding_seconds,
                    min_duration_seconds,
                    max_duration_seconds,
                )
            if not boundaries:
                skipped.append(path.relative_to(source_root).as_posix())
                continue

            relative = path.relative_to(source_root)
            source_hash = _sha256(path)
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-") or "audio"
            path_key = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:8]
            source_id = f"{safe_stem}-{path_key}"
            for index, (start, end) in enumerate(boundaries, start=1):
                start_ms = round(start * 1000 / target_rate)
                end_ms = round(end * 1000 / target_rate)
                item_id = f"{source_id}__{index:04d}__{start_ms:09d}-{end_ms:09d}"
                relative_output = Path("audio") / f"{item_id}.wav"
                write_wav(staging / relative_output, audio[start:end], target_rate)
                relative_f0 = ""
                if sidecar_f0 is not None:
                    first_frame = max(0, int(np.floor(start / target_rate / f0_hop_seconds)))
                    last_frame = min(
                        len(sidecar_f0),
                        int(np.ceil(end / target_rate / f0_hop_seconds)),
                    )
                    relative_f0_path = relative_output.with_suffix(".pv")
                    np.savetxt(
                        staging / relative_f0_path,
                        sidecar_f0[first_frame:last_frame],
                        fmt="%.5f",
                    )
                    relative_f0 = relative_f0_path.as_posix()
                records.append(
                    SegmentRecord(
                        id=item_id,
                        source_path=relative.as_posix(),
                        source_sha256=source_hash,
                        source_f0_sha256=sidecar_hash,
                        segment_path=relative_output.as_posix(),
                        f0_path=relative_f0,
                        start_s=start / target_rate,
                        end_s=end / target_rate,
                        duration_s=(end - start) / target_rate,
                        sample_rate=target_rate,
                        samples=end - start,
                        mode=mode,
                    )
                )
        if not records:
            raise ValueError("No segments met the duration and activity criteria")
        result = SplitResult(
            root=staging,
            records=records,
            metadata={
                "schema_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "mode": mode,
                "segment_seconds": segment_seconds,
                "overlap_seconds": overlap_seconds,
                "silence_threshold_dbfs": silence_threshold_db,
                "min_silence_seconds": min_silence_seconds,
                "padding_seconds": padding_seconds,
                "min_duration_seconds": min_duration_seconds,
                "max_duration_seconds": max_duration_seconds,
                "target_sample_rate": sample_rate,
                "keep_tail": keep_tail,
                "split_f0_sidecars": split_f0_sidecars,
                "f0_hop_seconds": f0_hop_seconds,
                "source_files": len(files),
                "segments": len(records),
                "skipped_files": skipped,
                "tool": "phonlab-ddsp",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        )
        result.save()
        staging.rename(output)
        result.root = output
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def split_summary(result: SplitResult) -> dict:
    durations = [record.duration_s for record in result.records]
    return {
        "source_files": result.metadata["source_files"],
        "segments": len(result.records),
        "duration_s": sum(durations),
        "minimum_s": min(durations),
        "maximum_s": max(durations),
        "sample_rates": sorted({record.sample_rate for record in result.records}),
        "skipped_files": len(result.metadata["skipped_files"]),
        "output": str(result.root),
    }


def _fixed_boundaries(
    samples: int,
    sample_rate: int,
    segment_seconds: float,
    overlap_seconds: float,
    min_duration_seconds: float,
    keep_tail: bool,
) -> list[tuple[int, int]]:
    window = max(1, round(segment_seconds * sample_rate))
    step = max(1, round((segment_seconds - overlap_seconds) * sample_rate))
    minimum = max(1, round(min_duration_seconds * sample_rate))
    boundaries = []
    for start in range(0, samples, step):
        end = min(samples, start + window)
        if end - start < window and not keep_tail:
            break
        if end - start >= minimum:
            boundaries.append((start, end))
        if end == samples:
            break
    return boundaries


def _silence_boundaries(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    min_silence_seconds: float,
    padding_seconds: float,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> list[tuple[int, int]]:
    frame = max(1, round(0.020 * sample_rate))
    hop = max(1, round(0.010 * sample_rate))
    rms_db = []
    for start in range(0, len(audio), hop):
        window = audio[start : min(len(audio), start + frame)]
        rms = float(np.sqrt(np.mean(np.asarray(window, dtype=np.float64) ** 2)))
        rms_db.append(20 * np.log10(max(rms, 1e-12)))
    active = np.flatnonzero(np.asarray(rms_db) >= threshold_db)
    if not active.size:
        return []

    minimum_gap = max(1, round(min_silence_seconds * sample_rate / hop))
    groups = []
    first = previous = int(active[0])
    for index in active[1:]:
        index = int(index)
        if index - previous >= minimum_gap:
            groups.append((first, previous))
            first = index
        previous = index
    groups.append((first, previous))

    padding = round(padding_seconds * sample_rate)
    minimum = max(1, round(min_duration_seconds * sample_rate))
    maximum = max(minimum, round(max_duration_seconds * sample_rate))
    boundaries = []
    for first_frame, last_frame in groups:
        start = max(0, first_frame * hop - padding)
        end = min(len(audio), last_frame * hop + frame + padding)
        if end - start < minimum:
            continue
        boundaries.extend(_limit_duration(start, end, minimum, maximum))
    return boundaries


def _limit_duration(start: int, end: int, minimum: int, maximum: int) -> list[tuple[int, int]]:
    boundaries = []
    cursor = start
    while end - cursor > maximum:
        boundaries.append((cursor, cursor + maximum))
        cursor += maximum
    if end - cursor >= minimum:
        boundaries.append((cursor, end))
    elif boundaries:
        previous_start, _ = boundaries[-1]
        boundaries[-1] = (previous_start, end)
    return boundaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
