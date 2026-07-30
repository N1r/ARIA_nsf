"""Reproducible acquisition of a duration-capped CMU ARCTIC subset.

The module intentionally uses only the Python standard library.  Its public
functions are suitable for a CLI or GUI, but contain no presentation or shell
logic themselves.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import urllib.request
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Tuple

CMU_ARCTIC_URL = (
    "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2"
)
CMU_ARCTIC_SHA256 = "9fddec16fbfbfb7d4989dff0fe77ccbe31f80b07b57be49d09994aa7a67d6dba"
CMU_ARCTIC_SIZE = 119_914_432
CMU_ARCTIC_ARCHIVE = "cmu_us_slt_arctic-0.95-release.tar.bz2"

# Short aliases are convenient for callers which already have an ARCTIC-specific
# context.
ARCTIC_URL = CMU_ARCTIC_URL
ARCTIC_SHA256 = CMU_ARCTIC_SHA256
ARCTIC_SIZE = CMU_ARCTIC_SIZE

DEFAULT_TARGET_DURATION_S = 1800.0
DEFAULT_MAX_DURATION_S = 3600.0
DEFAULT_SILENCE_GAP_S = 0.35
CORPUS_METADATA = "corpus.json"
SCHEMA_VERSION = "1.0"

_ORDER_LINE = re.compile(r"^\s*\(\s*([A-Za-z0-9_.-]+)(?:\s|\))")
_LICENSE_NAMES = {
    "copying",
    "copying.txt",
    "license",
    "license.txt",
    "licence",
    "licence.txt",
}


@dataclass(frozen=True)
class Utterance:
    """One WAV in corpus order."""

    id: str
    path: Path
    frames: int
    sample_rate: int
    channels: int
    sample_width: int
    duration_s: float


@dataclass(frozen=True)
class SubsetSelection:
    """A deterministic prefix of corpus utterances."""

    utterances: Tuple[Utterance, ...]
    duration_s: float
    target_duration_s: float
    max_duration_s: float
    reached_target: bool


@dataclass(frozen=True)
class CorpusResult:
    """Paths and provenance returned by :func:`acquire_cmu_arctic`."""

    archive: Path
    extracted: Path
    selected_dir: Path
    continuous_wav: Optional[Path]
    metadata: dict


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA256 digest of *path*."""

    path = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(
    path: Path,
    expected_size: int = CMU_ARCTIC_SIZE,
    expected_sha256: str = CMU_ARCTIC_SHA256,
) -> str:
    """Verify an archive's exact byte size and SHA256, raising on mismatch."""

    path = Path(path)
    expected_sha256 = _normalise_sha256(expected_sha256)
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError("expected_size must be a non-negative integer")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Archive is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Archive size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Archive SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    opener: Optional[Callable[..., Any]] = None,
) -> Path:
    """Download *url* to ``destination + '.part'`` and atomically finish it.

    An existing verified destination is reused.  An incomplete ``.part`` file
    is appended to only after the server confirms the requested byte range.
    Invalid files are retained for inspection; this function never deletes or
    overwrites them.
    """

    destination = Path(destination)
    expected_sha256 = _normalise_sha256(expected_sha256)
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError("expected_size must be a non-negative integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if opener is None:
        opener = urllib.request.urlopen
    if destination.exists() or destination.is_symlink():
        verify_archive(destination, expected_size, expected_sha256)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise FileExistsError(f"Partial download is not a regular file: {partial}")

    partial_exists = partial.exists()
    offset = partial.stat().st_size if partial_exists else 0
    if offset > expected_size:
        raise ValueError(
            f"Partial download is larger than expected: {offset} > {expected_size} bytes"
        )
    if partial_exists and offset == expected_size:
        verify_archive(partial, expected_size, expected_sha256)
        _publish_no_replace(partial, destination)
        return destination

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "phonlab-ddsp/0.1 corpus acquisition",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)

    with opener(request, timeout=timeout) as response:
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "getcode"):
            status = response.getcode()
        if offset:
            content_range = response.headers.get("Content-Range", "")
            range_match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            valid_range = (
                range_match is not None
                and int(range_match.group(1)) == offset
                and int(range_match.group(2)) == expected_size - 1
                and int(range_match.group(3)) == expected_size
            )
            if status != 206 or not valid_range:
                raise RuntimeError(
                    f"Server did not honour resume request for byte {offset}; "
                    f"the partial file was left unchanged at {partial}"
                )
        elif status not in {None, 200, 206}:
            raise RuntimeError(f"Unexpected HTTP status {status} while downloading {url}")

        mode = "ab" if partial_exists else "xb"
        remaining = expected_size - offset
        with partial.open(mode) as stream:
            while True:
                block = response.read(min(chunk_size, remaining + 1))
                if not block:
                    break
                if len(block) > remaining:
                    raise ValueError(f"Download exceeded expected size of {expected_size} bytes")
                stream.write(block)
                remaining -= len(block)
            stream.flush()
            os.fsync(stream.fileno())

    verify_archive(partial, expected_size, expected_sha256)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Download destination appeared during transfer: {destination}")
    _publish_no_replace(partial, destination)
    return destination


def download_archive(
    destination: Path,
    *,
    url: str = CMU_ARCTIC_URL,
    expected_size: int = CMU_ARCTIC_SIZE,
    expected_sha256: str = CMU_ARCTIC_SHA256,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    opener: Optional[Callable[..., Any]] = None,
) -> Path:
    """Download and verify an archive, resuming through its ``.part`` file."""

    return download_with_resume(
        url,
        destination,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        timeout=timeout,
        chunk_size=chunk_size,
        opener=opener,
    )


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    """Extract regular files/directories without traversal or special entries.

    All members are validated before writing starts.  Extraction is built in a
    uniquely named sibling staging directory and atomically renamed into place.
    Existing destinations, symlinks, devices, FIFOs, hard links, and symbolic
    links are rejected.  An interrupted staging directory is retained and a
    later call safely starts a new staging attempt without touching it.
    """

    archive = Path(archive)
    destination = Path(destination)
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(f"Tar archive is not a regular file: {archive}")
    _reject_symlink_ancestors(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Extraction destination already exists: {destination}")

    with tarfile.open(archive, mode="r:*") as bundle:
        planned: list[tuple[tarfile.TarInfo, Path]] = []
        kinds: dict[Path, str] = {}
        for member in bundle.getmembers():
            relative = _safe_tar_member_path(member)
            if relative is None:
                continue
            kind = "directory" if member.isdir() else "file"
            if relative in kinds:
                raise ValueError(f"Duplicate tar member path: {member.name!r}")
            kinds[relative] = kind
            planned.append((member, relative))

        for relative in kinds:
            for parent in relative.parents:
                if parent == Path("."):
                    break
                if kinds.get(parent) == "file":
                    raise ValueError(
                        f"Tar member {relative.as_posix()!r} is below file {parent.as_posix()!r}"
                    )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = _new_staging_directory(destination, "extracting")
        for member, relative in planned:
            target = staging / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read regular tar member: {member.name!r}")
            written = 0
            with source, target.open("xb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    written += len(block)
            if written != member.size:
                raise ValueError(
                    f"Tar member size mismatch for {member.name!r}: "
                    f"expected {member.size}, wrote {written}"
                )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Extraction destination appeared during extraction: {destination}")
    _publish_no_replace(staging, destination)
    return destination


def locate_arctic_corpus(extracted: Path) -> Path:
    """Locate the unique ARCTIC corpus root below an extraction directory."""

    extracted = Path(extracted)
    if not extracted.is_dir():
        raise FileNotFoundError(f"Extraction directory does not exist: {extracted}")
    candidates = []
    for candidate in (extracted, *sorted(path for path in extracted.rglob("*") if path.is_dir())):
        if (candidate / "wav").is_dir() and (candidate / "etc" / "txt.done.data").is_file():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No directory containing wav/ and etc/txt.done.data was found below {extracted}"
        )
    shallowest_depth = min(len(path.relative_to(extracted).parts) for path in candidates)
    shallowest = [
        path for path in candidates if len(path.relative_to(extracted).parts) == shallowest_depth
    ]
    if len(shallowest) != 1:
        listed = ", ".join(str(path) for path in shallowest)
        raise ValueError(f"Multiple CMU ARCTIC corpus roots found: {listed}")
    return shallowest[0]


def corpus_wavs_in_order(corpus_root: Path) -> list[Path]:
    """Return utterance WAV paths in ``etc/txt.done.data`` corpus order."""

    corpus_root = Path(corpus_root)
    order_path = corpus_root / "etc" / "txt.done.data"
    wav_dir = corpus_root / "wav"
    if not order_path.is_file():
        raise FileNotFoundError(f"CMU ARCTIC order file is missing: {order_path}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"CMU ARCTIC WAV directory is missing: {wav_dir}")

    result = []
    seen = set()
    with order_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            match = _ORDER_LINE.match(line)
            if not match:
                continue
            utterance_id = match.group(1)
            if utterance_id in seen:
                raise ValueError(
                    f"Duplicate utterance {utterance_id!r} in {order_path}:{line_number}"
                )
            seen.add(utterance_id)
            wav_path = wav_dir / f"{utterance_id}.wav"
            if not wav_path.is_file() or wav_path.is_symlink():
                raise FileNotFoundError(
                    f"WAV listed in corpus order is missing or unsafe: {wav_path}"
                )
            result.append(wav_path)
    if not result:
        raise ValueError(f"No utterance identifiers were found in {order_path}")
    return result


def inspect_wav(path: Path) -> Utterance:
    """Read deterministic PCM metadata without decoding samples."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"WAV is not a regular file: {path}")
    with wave.open(str(path), "rb") as stream:
        if stream.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is unsupported: {path}")
        frames = stream.getnframes()
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
    if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise ValueError(f"Invalid WAV format metadata: {path}")
    return Utterance(
        id=path.stem,
        path=path,
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_s=frames / sample_rate,
    )


def select_duration_subset(
    wav_paths: Iterable[Path],
    target_duration_s: float = DEFAULT_TARGET_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> SubsetSelection:
    """Select a corpus-order prefix until the target, without crossing the cap."""

    target, maximum = _duration_limits(target_duration_s, max_duration_s)
    selected = []
    total = Fraction(0)
    identifiers = set()
    for wav_path in wav_paths:
        if total >= target:
            break
        utterance = inspect_wav(Path(wav_path))
        if utterance.id in identifiers:
            raise ValueError(f"Duplicate utterance id in WAV sequence: {utterance.id}")
        identifiers.add(utterance.id)
        duration = Fraction(utterance.frames, utterance.sample_rate)
        if total + duration > maximum:
            break
        selected.append(utterance)
        total += duration
    if not selected:
        raise ValueError("No corpus utterance fits within max_duration_s")
    return SubsetSelection(
        utterances=tuple(selected),
        duration_s=float(total),
        target_duration_s=float(target),
        max_duration_s=float(maximum),
        reached_target=total >= target,
    )


def select_utterances(
    corpus_root: Path,
    target_duration_s: float = DEFAULT_TARGET_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> SubsetSelection:
    """Select CMU ARCTIC utterances deterministically in official corpus order."""

    return select_duration_subset(
        corpus_wavs_in_order(corpus_root),
        target_duration_s=target_duration_s,
        max_duration_s=max_duration_s,
    )


def create_continuous_wav(
    wav_paths: Iterable[Path],
    output: Path,
    silence_gap_s: float = DEFAULT_SILENCE_GAP_S,
) -> Path:
    """Concatenate compatible PCM WAVs with silence between utterances."""

    if not math.isfinite(silence_gap_s) or silence_gap_s < 0:
        raise ValueError("silence_gap_s must be a finite non-negative number")
    paths = [Path(path) for path in wav_paths]
    if not paths:
        raise ValueError("At least one WAV is required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Continuous WAV already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    inspected = [inspect_wav(path) for path in paths]
    reference = inspected[0]
    expected_format = (
        reference.channels,
        reference.sample_width,
        reference.sample_rate,
    )
    for item in inspected[1:]:
        item_format = (item.channels, item.sample_width, item.sample_rate)
        if item_format != expected_format:
            raise ValueError(
                f"Incompatible WAV format for {item.path}: {item_format}, "
                f"expected {expected_format}"
            )
    for item in inspected:
        with wave.open(str(item.path), "rb") as source:
            samples = source.readframes(item.frames)
        expected_bytes = item.frames * item.channels * item.sample_width
        if len(samples) != expected_bytes:
            raise ValueError(f"Truncated WAV data in {item.path}")
    gap_frames = round(silence_gap_s * reference.sample_rate)
    silence_byte = b"\x80" if reference.sample_width == 1 else b"\x00"
    silence = silence_byte * gap_frames * reference.channels * reference.sample_width

    with output.open("xb") as raw_output:
        with wave.open(raw_output, "wb") as destination:
            destination.setnchannels(reference.channels)
            destination.setsampwidth(reference.sample_width)
            destination.setframerate(reference.sample_rate)
            for index, item in enumerate(inspected):
                with wave.open(str(item.path), "rb") as source:
                    samples = source.readframes(item.frames)
                expected_bytes = item.frames * item.channels * item.sample_width
                if len(samples) != expected_bytes:
                    raise ValueError(f"Truncated WAV data in {item.path}")
                destination.writeframesraw(samples)
                if index + 1 < len(paths) and gap_frames:
                    destination.writeframesraw(silence)
            destination.writeframes(b"")
    return output


def load_corpus_result(output: Path) -> CorpusResult:
    """Load a completed result and verify all provenance-critical artifacts."""

    output = Path(output).absolute()
    _reject_symlink_ancestors(output)
    output = output.resolve()
    metadata_path = output / CORPUS_METADATA
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError(f"Completed corpus metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid corpus metadata: {metadata_path}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("complete") is not True:
        raise ValueError(f"Corpus metadata is not a complete {SCHEMA_VERSION} result")

    source = _mapping(metadata, "source")
    archive = _output_path(output, source.get("archive"), "source.archive")
    expected_size = _metadata_int(source, "size_bytes")
    expected_sha256 = _metadata_sha256(source, "sha256")
    verify_archive(archive, expected_size, expected_sha256)

    outputs = _mapping(metadata, "outputs")
    extracted = _output_path(output, outputs.get("extracted"), "outputs.extracted")
    selected_dir = _output_path(output, outputs.get("selected_dir"), "outputs.selected_dir")
    if not extracted.is_dir() or extracted.is_symlink():
        raise ValueError(f"Recorded extraction directory is missing or unsafe: {extracted}")
    if not selected_dir.is_dir() or selected_dir.is_symlink():
        raise ValueError(f"Recorded subset directory is missing or unsafe: {selected_dir}")

    license_metadata = _mapping(metadata, "license")
    license_path = _output_path(output, license_metadata.get("file"), "license.file")
    _verify_recorded_file(
        license_path,
        _metadata_int(license_metadata, "size_bytes"),
        _metadata_sha256(license_metadata, "sha256"),
    )

    provenance = _mapping(metadata, "provenance")
    order_path = _output_path(output, provenance.get("order_file"), "provenance.order_file")
    _verify_recorded_file(
        order_path,
        _metadata_int(provenance, "order_file_size_bytes"),
        _metadata_sha256(provenance, "order_file_sha256"),
    )
    if order_path != extracted / "etc" / "txt.done.data":
        raise ValueError("Recorded order file is not the extracted corpus order file")

    selection = _mapping(metadata, "selection")
    records = selection.get("utterances")
    if not isinstance(records, list) or not records:
        raise ValueError("selection.utterances must be a non-empty list")
    target = _fraction_from_number(
        selection.get("target_duration_s"), "selection.target_duration_s"
    )
    maximum = _fraction_from_number(selection.get("max_duration_s"), "selection.max_duration_s")
    expected_selection = select_utterances(extracted, float(target), float(maximum))
    if len(records) != len(expected_selection.utterances):
        raise ValueError("Recorded utterance count does not match deterministic selection")
    if selection.get("utterance_count") != len(records):
        raise ValueError("selection.utterance_count is inconsistent")
    if selection.get("reached_target") is not expected_selection.reached_target:
        raise ValueError("selection.reached_target is inconsistent")

    expected_selected = set()
    selected_paths = []
    total = Fraction(0)
    for index, (raw_record, expected_utterance) in enumerate(
        zip(records, expected_selection.utterances)
    ):
        if not isinstance(raw_record, dict):
            raise ValueError(f"selection.utterances[{index}] must be an object")
        if raw_record.get("id") != expected_utterance.id:
            raise ValueError("Recorded utterances are not in deterministic corpus order")
        selected_path = _output_path(
            output, raw_record.get("selected_path"), f"selection.utterances[{index}].selected_path"
        )
        source_path = _output_path(
            output, raw_record.get("source_path"), f"selection.utterances[{index}].source_path"
        )
        if selected_dir != selected_path.parent:
            raise ValueError(f"Selected WAV is outside selected_dir: {selected_path}")
        if source_path != expected_utterance.path:
            raise ValueError("Recorded source WAVs are not the deterministic corpus prefix")
        if selected_path != selected_dir / f"{expected_utterance.id}.wav":
            raise ValueError(f"Unexpected selected WAV path: {selected_path}")
        size = _metadata_int(raw_record, "size_bytes")
        digest = _metadata_sha256(raw_record, "sha256")
        _verify_recorded_file(selected_path, size, digest)
        _verify_recorded_file(source_path, size, digest)
        frames = _metadata_int(raw_record, "frames")
        sample_rate = _metadata_int(raw_record, "sample_rate")
        if sample_rate <= 0 or frames < 0:
            raise ValueError(f"Invalid WAV duration metadata for {selected_path}")
        inspected = inspect_wav(selected_path)
        channels = _metadata_int(raw_record, "channels")
        sample_width = _metadata_int(raw_record, "sample_width_bytes")
        recorded_duration = _fraction_from_number(
            raw_record.get("duration_s"), f"selection.utterances[{index}].duration_s"
        )
        if (
            inspected.frames != frames
            or inspected.sample_rate != sample_rate
            or inspected.channels != channels
            or inspected.sample_width != sample_width
            or abs(float(recorded_duration) - inspected.duration_s) > 1e-9
        ):
            raise ValueError(f"WAV metadata changed for {selected_path}")
        total += Fraction(frames, sample_rate)
        expected_selected.add(selected_path)
        selected_paths.append(selected_path)
    actual_selected = set(selected_dir.iterdir())
    if actual_selected != expected_selected:
        raise ValueError("Selected directory contents do not match corpus metadata")

    recorded_total = _fraction_from_number(
        selection.get("selected_duration_s"), "selection.selected_duration_s"
    )
    if (
        total > maximum
        or abs(float(total) - float(recorded_total)) > 1e-9
        or abs(float(total) - expected_selection.duration_s) > 1e-9
    ):
        raise ValueError("Recorded subset duration is inconsistent")

    continuous_value = outputs.get("continuous_wav")
    continuous_size = outputs.get("continuous_size_bytes")
    continuous_digest = outputs.get("continuous_sha256")
    continuous_fields_present = [
        value is not None for value in (continuous_value, continuous_size, continuous_digest)
    ]
    if any(continuous_fields_present) and not all(continuous_fields_present):
        raise ValueError("Continuous WAV path, size, and SHA256 must be all present or all null")
    silence_gap_value = outputs.get("silence_gap_s")
    if all(continuous_fields_present):
        continuous_wav = _output_path(output, continuous_value, "outputs.continuous_wav")
        if continuous_wav != output / "continuous.wav":
            raise ValueError("Continuous WAV is not at the canonical output path")
        _verify_recorded_file(
            continuous_wav,
            _metadata_int(outputs, "continuous_size_bytes"),
            _metadata_sha256(outputs, "continuous_sha256"),
        )
        silence_gap = _fraction_from_number(silence_gap_value, "outputs.silence_gap_s")
        if silence_gap < 0:
            raise ValueError("outputs.silence_gap_s must be non-negative")
        _verify_continuous_contents(continuous_wav, selected_paths, float(silence_gap))
    else:
        continuous_wav = None
        if silence_gap_value is not None:
            raise ValueError("A silence gap was recorded without a continuous WAV")

    return CorpusResult(
        archive=archive,
        extracted=extracted,
        selected_dir=selected_dir,
        continuous_wav=continuous_wav,
        metadata=metadata,
    )


def acquire_cmu_arctic(
    output: Path,
    target_duration_s: float = DEFAULT_TARGET_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    silence_gap_s: Optional[float] = DEFAULT_SILENCE_GAP_S,
    *,
    url: str = CMU_ARCTIC_URL,
    expected_sha256: str = CMU_ARCTIC_SHA256,
    expected_size: int = CMU_ARCTIC_SIZE,
    archive_source: Optional[Path] = None,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    opener: Optional[Callable[..., Any]] = None,
) -> CorpusResult:
    """Acquire CMU ARCTIC, select a capped prefix, and write ``corpus.json``.

    Passing ``silence_gap_s=None`` disables the optional continuous WAV.
    ``archive_source`` can seed the output from an already-downloaded archive;
    it is verified, copied under *output*, and verified again.  A complete
    matching result is verified and reused.  Apart from a recognised archive
    or its resumable ``.part`` file, a non-empty incomplete output directory is
    refused without modification.
    """

    target, maximum = _duration_limits(target_duration_s, max_duration_s)
    expected_sha256 = _normalise_sha256(expected_sha256)
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError("expected_size must be a non-negative integer")
    if silence_gap_s is not None and (not math.isfinite(silence_gap_s) or silence_gap_s < 0):
        raise ValueError("silence_gap_s must be None or a finite non-negative number")

    output = Path(output).absolute()
    _reject_symlink_ancestors(output)
    output = output.resolve()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise FileExistsError(f"Corpus output is not a regular directory: {output}")
    metadata_path = output / CORPUS_METADATA
    if metadata_path.exists() or metadata_path.is_symlink():
        try:
            result = load_corpus_result(output)
        except Exception as error:
            raise FileExistsError(
                f"Existing corpus result failed verification and was left unchanged: {output}"
            ) from error
        if not _request_matches(
            result.metadata,
            url=url,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            target=target,
            maximum=maximum,
            silence_gap_s=silence_gap_s,
        ):
            raise FileExistsError(
                f"Existing verified corpus uses different acquisition settings: {output}"
            )
        return result

    output.mkdir(parents=True, exist_ok=True)
    archive = output / CMU_ARCTIC_ARCHIVE
    partial = archive.with_name(archive.name + ".part")
    extraction_dir = output / "extracted"
    selected_dir = output / "selected"
    continuous_path = output / "continuous.wav"
    known_paths = {
        archive,
        partial,
        extraction_dir,
        selected_dir,
        continuous_path,
    }
    unknown = {
        path
        for path in output.iterdir()
        if path not in known_paths and not _is_known_acquisition_staging(path)
    }
    if unknown:
        listed = ", ".join(sorted(path.name for path in unknown))
        raise FileExistsError(
            "Refusing non-empty incomplete corpus output with unknown paths "
            f"({listed}); existing paths were left unchanged: {output}"
        )
    if not archive.exists() and (
        extraction_dir.exists() or selected_dir.exists() or continuous_path.exists()
    ):
        raise FileExistsError("Incomplete corpus has derived outputs but no verified archive")
    if selected_dir.exists() and not extraction_dir.exists():
        raise FileExistsError("Incomplete corpus has a selected subset but no extraction")
    if continuous_path.exists() and not selected_dir.exists():
        raise FileExistsError("Incomplete corpus has a continuous WAV but no selected subset")

    if archive_source is not None and not archive.exists():
        archive_source = Path(archive_source).resolve()
        _seed_archive_from_source(
            archive_source,
            archive,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    else:
        download_archive(
            archive,
            url=url,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            timeout=timeout,
            chunk_size=chunk_size,
            opener=opener,
        )

    if extraction_dir.exists():
        if extraction_dir.is_symlink() or not extraction_dir.is_dir():
            raise FileExistsError(f"Unsafe interrupted extraction output: {extraction_dir}")
    else:
        safe_extract_tar(archive, extraction_dir)
    corpus_root = locate_arctic_corpus(extraction_dir)
    selection = select_utterances(corpus_root, float(target), float(maximum))

    utterance_records = _prepare_selected_subset(
        output,
        selection,
        selected_dir,
    )

    continuous_wav = None
    continuous_sha256 = None
    continuous_size = None
    if silence_gap_s is not None:
        continuous_wav = continuous_path
        selected_wavs = [selected_dir / f"{item.id}.wav" for item in selection.utterances]
        if continuous_wav.exists() or continuous_wav.is_symlink():
            _verify_continuous_contents(continuous_wav, selected_wavs, silence_gap_s)
        else:
            continuous_staging = _new_staging_file(continuous_wav, "preparing")
            create_continuous_wav(selected_wavs, continuous_staging, silence_gap_s)
            _verify_continuous_contents(continuous_staging, selected_wavs, silence_gap_s)
            if continuous_wav.exists() or continuous_wav.is_symlink():
                raise FileExistsError(
                    f"Continuous WAV appeared during preparation: {continuous_wav}"
                )
            _publish_no_replace(continuous_staging, continuous_wav)
        continuous_sha256 = sha256_file(continuous_wav)
        continuous_size = continuous_wav.stat().st_size
    elif continuous_path.exists() or continuous_path.is_symlink():
        raise FileExistsError("Incomplete corpus contains continuous.wav but silence_gap_s=None")

    license_path = _find_license_file(extraction_dir)
    order_path = corpus_root / "etc" / "txt.done.data"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "corpus": {
            "name": "CMU ARCTIC",
            "release": "0.95",
            "speaker": "slt",
        },
        "source": {
            "url": url,
            "archive": archive.relative_to(output).as_posix(),
            "size_bytes": expected_size,
            "sha256": expected_sha256,
        },
        "license": {
            "name": "CMU ARCTIC distribution terms",
            "identifier": "LicenseRef-CMU-ARCTIC",
            "file": license_path.relative_to(output).as_posix(),
            "size_bytes": license_path.stat().st_size,
            "sha256": sha256_file(license_path),
        },
        "provenance": {
            "tool": "phonlab-ddsp",
            "module": "phonlab_ddsp.corpus",
            "selection_policy": (
                "prefix in etc/txt.done.data order; stop after reaching target or "
                "before exceeding max_duration_s"
            ),
            "order_file": order_path.relative_to(output).as_posix(),
            "order_file_size_bytes": order_path.stat().st_size,
            "order_file_sha256": sha256_file(order_path),
        },
        "selection": {
            "target_duration_s": float(target),
            "max_duration_s": float(maximum),
            "selected_duration_s": selection.duration_s,
            "reached_target": selection.reached_target,
            "utterance_count": len(selection.utterances),
            "utterances": utterance_records,
        },
        "outputs": {
            "extracted": corpus_root.relative_to(output).as_posix(),
            "selected_dir": selected_dir.relative_to(output).as_posix(),
            "continuous_wav": (
                continuous_wav.relative_to(output).as_posix()
                if continuous_wav is not None
                else None
            ),
            "continuous_size_bytes": continuous_size,
            "continuous_sha256": continuous_sha256,
            "silence_gap_s": silence_gap_s,
        },
    }
    metadata_staging = _new_staging_file(metadata_path, "preparing")
    _write_json_exclusive(metadata_staging, metadata)
    if metadata_path.exists() or metadata_path.is_symlink():
        raise FileExistsError(f"Corpus metadata appeared during preparation: {metadata_path}")
    _publish_no_replace(metadata_staging, metadata_path)
    return load_corpus_result(output)


def _normalise_sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _duration_limits(target_duration_s: float, max_duration_s: float) -> tuple[Fraction, Fraction]:
    target = _fraction_from_number(target_duration_s, "target_duration_s")
    maximum = _fraction_from_number(max_duration_s, "max_duration_s")
    if target <= 0:
        raise ValueError("target_duration_s must be positive")
    if maximum <= 0:
        raise ValueError("max_duration_s must be positive")
    if target > maximum:
        raise ValueError("target_duration_s must not exceed max_duration_s")
    return target, maximum


def _fraction_from_number(value: Any, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return Fraction(str(value))


def _safe_tar_member_path(member: tarfile.TarInfo) -> Optional[Path]:
    name = member.name
    if member.issym() or member.islnk():
        raise ValueError(f"Tar links are forbidden: {name!r}")
    if not member.isdir() and not member.isreg():
        raise ValueError(f"Tar special entry is forbidden: {name!r}")
    if "\x00" in name or "\\" in name:
        raise ValueError(f"Unsafe tar member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"Tar path traversal is forbidden: {name!r}")
    parts = tuple(part for part in pure.parts if part not in {"", "."})
    if not parts:
        if member.isdir():
            return None
        raise ValueError(f"Unsafe empty tar member path: {name!r}")
    if ":" in parts[0]:
        raise ValueError(f"Drive-like tar member path is forbidden: {name!r}")
    return Path(*parts)


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(path).absolute()
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            raise ValueError(f"Symbolic-link path component is forbidden: {candidate}")


def _staging_name(destination: Path, label: str, index: int) -> Path:
    suffix = "" if index == 0 else f"-{index}"
    return destination.parent / f".{destination.name}.{label}{suffix}"


def _new_staging_directory(destination: Path, label: str) -> Path:
    index = 0
    while True:
        candidate = _staging_name(destination, label, index)
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
            continue
        return candidate


def _new_staging_file(destination: Path, label: str) -> Path:
    index = 0
    while True:
        candidate = _staging_name(destination, label, index)
        if not os.path.lexists(candidate):
            return candidate
        index += 1


def _publish_no_replace(staging: Path, destination: Path) -> None:
    """Atomically publish a sibling staging path without replacing a target."""

    if os.name == "posix":
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(staging),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(error_number, os.strerror(error_number), destination)
            if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise OSError(error_number, os.strerror(error_number), destination)

    if staging.is_file():
        os.link(staging, destination)
        staging.unlink()
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Publication destination already exists: {destination}")
    staging.rename(destination)


def _is_known_acquisition_staging(path: Path) -> bool:
    return bool(
        re.fullmatch(
            r"\.(?:extracted\.extracting|selected\.preparing|"
            r"continuous\.wav\.preparing|corpus\.json\.preparing)(?:-\d+)?",
            path.name,
        )
    )


def _seed_archive_from_source(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    verify_archive(source, expected_size, expected_sha256)
    if destination.exists() or destination.is_symlink():
        verify_archive(destination, expected_size, expected_sha256)
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise FileExistsError(f"Partial archive copy is not a regular file: {partial}")
    partial_exists = partial.exists()
    offset = partial.stat().st_size if partial_exists else 0
    if offset > expected_size:
        raise ValueError(
            f"Partial archive copy is larger than expected: {offset} > {expected_size}"
        )

    with source.open("rb") as source_stream:
        if partial_exists:
            with partial.open("rb") as partial_stream:
                remaining = offset
                while remaining:
                    length = min(1024 * 1024, remaining)
                    if partial_stream.read(length) != source_stream.read(length):
                        raise ValueError(
                            f"Partial archive does not match archive_source: {partial}"
                        )
                    remaining -= length
        source_stream.seek(offset)
        with partial.open("ab" if partial_exists else "xb") as output_stream:
            shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    verify_archive(partial, expected_size, expected_sha256)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Archive destination appeared during copy: {destination}")
    _publish_no_replace(partial, destination)
    return destination


def _prepare_selected_subset(
    output: Path,
    selection: SubsetSelection,
    selected_dir: Path,
) -> list[dict]:
    expected_names = {f"{item.id}.wav" for item in selection.utterances}
    if selected_dir.is_symlink() or (selected_dir.exists() and not selected_dir.is_dir()):
        raise FileExistsError(f"Unsafe selected subset directory: {selected_dir}")
    if not selected_dir.exists():
        staging = _new_staging_directory(selected_dir, "preparing")
        for utterance in selection.utterances:
            _copy_file_exclusive(utterance.path, staging / f"{utterance.id}.wav")
        if selected_dir.exists() or selected_dir.is_symlink():
            raise FileExistsError(f"Selected subset appeared during preparation: {selected_dir}")
        _publish_no_replace(staging, selected_dir)

    actual_names = {path.name for path in selected_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("Interrupted selected subset does not match requested corpus prefix")
    records = []
    for utterance in selection.utterances:
        selected_path = selected_dir / f"{utterance.id}.wav"
        if not selected_path.is_file() or selected_path.is_symlink():
            raise ValueError(f"Selected WAV is missing or unsafe: {selected_path}")
        source_digest = sha256_file(utterance.path)
        if (
            selected_path.stat().st_size != utterance.path.stat().st_size
            or sha256_file(selected_path) != source_digest
        ):
            raise ValueError(f"Selected WAV differs from extracted source: {selected_path}")
        records.append(
            {
                "id": utterance.id,
                "source_path": utterance.path.relative_to(output).as_posix(),
                "selected_path": selected_path.relative_to(output).as_posix(),
                "size_bytes": selected_path.stat().st_size,
                "sha256": source_digest,
                "frames": utterance.frames,
                "sample_rate": utterance.sample_rate,
                "channels": utterance.channels,
                "sample_width_bytes": utterance.sample_width,
                "duration_s": utterance.duration_s,
            }
        )
    return records


def _verify_continuous_contents(
    continuous: Path,
    wav_paths: Iterable[Path],
    silence_gap_s: float,
) -> None:
    paths = [Path(path) for path in wav_paths]
    if not continuous.is_file() or continuous.is_symlink():
        raise ValueError(f"Continuous WAV is missing or unsafe: {continuous}")
    items = [inspect_wav(path) for path in paths]
    if not items:
        raise ValueError("Cannot verify a continuous WAV without utterances")
    reference = items[0]
    expected_format = (
        reference.channels,
        reference.sample_width,
        reference.sample_rate,
    )
    gap_frames = round(silence_gap_s * reference.sample_rate)
    silence_byte = b"\x80" if reference.sample_width == 1 else b"\x00"
    silence = silence_byte * gap_frames * reference.channels * reference.sample_width
    expected_frames = sum(item.frames for item in items) + gap_frames * (len(items) - 1)

    with wave.open(str(continuous), "rb") as combined:
        actual_format = (
            combined.getnchannels(),
            combined.getsampwidth(),
            combined.getframerate(),
        )
        if (
            combined.getcomptype() != "NONE"
            or actual_format != expected_format
            or combined.getnframes() != expected_frames
        ):
            raise ValueError(f"Continuous WAV format or duration is inconsistent: {continuous}")
        for index, item in enumerate(items):
            item_format = (item.channels, item.sample_width, item.sample_rate)
            if item_format != expected_format:
                raise ValueError(f"Incompatible selected WAV format: {item.path}")
            with wave.open(str(item.path), "rb") as source:
                expected_samples = source.readframes(item.frames)
            if combined.readframes(item.frames) != expected_samples:
                raise ValueError(f"Continuous WAV audio differs at utterance {item.id}")
            if index + 1 < len(items) and combined.readframes(gap_frames) != silence:
                raise ValueError(f"Continuous WAV has an invalid silence gap after {item.id}")
        if combined.readframes(1):
            raise ValueError(f"Continuous WAV contains unexpected trailing frames: {continuous}")


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Copy source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _find_license_file(extracted: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in extracted.rglob("*")
            if path.is_file() and path.name.lower() in _LICENSE_NAMES
        ),
        key=lambda path: (len(path.relative_to(extracted).parts), path.as_posix()),
    )
    if not candidates:
        raise FileNotFoundError(f"No COPYING or LICENSE file was found below {extracted}")
    return candidates[0]


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object in corpus metadata")
    return value


def _metadata_int(parent: dict, key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer in corpus metadata")
    return value


def _metadata_sha256(parent: dict, key: str) -> str:
    try:
        return _normalise_sha256(parent.get(key))
    except ValueError as error:
        raise ValueError(f"{key} is not a SHA256 in corpus metadata") from error


def _output_path(output: Path, raw_value: Any, field: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(raw_value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in raw_value
        or (relative.parts and ":" in relative.parts[0])
    ):
        raise ValueError(f"{field} is not a safe relative path")
    lexical = output / relative
    current = output
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{field} contains a symbolic link")
    resolved = lexical.resolve()
    if resolved != output and output not in resolved.parents:
        raise ValueError(f"{field} escapes the corpus output")
    return lexical


def _verify_recorded_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Recorded file is missing or unsafe: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"Recorded file size changed: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Recorded file SHA256 changed: {path}")


def _request_matches(
    metadata: dict,
    *,
    url: str,
    expected_sha256: str,
    expected_size: int,
    target: Fraction,
    maximum: Fraction,
    silence_gap_s: Optional[float],
) -> bool:
    try:
        source = _mapping(metadata, "source")
        selection = _mapping(metadata, "selection")
        outputs = _mapping(metadata, "outputs")
        return (
            source.get("url") == url
            and source.get("sha256") == expected_sha256
            and source.get("size_bytes") == expected_size
            and _fraction_from_number(
                selection.get("target_duration_s"), "selection.target_duration_s"
            )
            == target
            and _fraction_from_number(selection.get("max_duration_s"), "selection.max_duration_s")
            == maximum
            and outputs.get("silence_gap_s") == silence_gap_s
        )
    except (TypeError, ValueError):
        return False
