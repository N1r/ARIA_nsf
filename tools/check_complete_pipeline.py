#!/usr/bin/env python3
"""Mechanical acceptance predicate for the CMU ARCTIC SLT demonstration.

The checker intentionally depends only on the Python standard library so it can
run as the final command of a Slurm job without importing the training stack.
Every returned issue represents one independently actionable acceptance gap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

SUCCESS_TOKEN = "PHONLAB_COMPLETE_PIPELINE_OK"
CMU_ARCTIC_SHA256 = "9fddec16fbfbfb7d4989dff0fe77ccbe31f80b07b57be49d09994aa7a67d6dba"
CMU_ARCTIC_SIZE = 119_914_432
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^version_(\d+)$")
MIN_SPEECH_SECONDS = 1800.0
MAX_SPEECH_SECONDS = 3600.0
MIN_DATASET_RECORDS = 500
MIN_TRAIN_STEP = 100
MIN_REPORT_BYTES = 500


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_issues(root: Path) -> List[str]:
    """Return every acceptance failure found below *root*."""

    root = Path(root).expanduser().resolve()
    issues: List[str] = []
    corpus_records = _check_corpus(root, issues)
    _check_segments(root, issues)
    dataset_records, dataset_fingerprint = _check_dataset(root, issues)
    checkpoint = _check_experiment_and_training(root, issues, dataset_fingerprint)
    test_ids = {
        row.get("id", "")
        for row in dataset_records
        if row.get("split", "").strip().lower() == "test" and row.get("id", "")
    }
    _check_outputs(root, issues, test_ids, checkpoint)

    # This catches a corrupt "complete" corpus whose metadata lists no usable
    # utterances without adding a second error for an entirely missing JSON.
    if corpus_records is not None and not corpus_records:
        issues.append("corpus: selection.utterances contains no usable records")
    return issues


def missing_units(root: Path) -> Tuple[int, List[str]]:
    """Compatibility wrapper returning ``(count, details)``."""

    issues = collect_issues(root)
    return len(issues), issues


def _check_corpus(root: Path, issues: List[str]) -> Optional[List[Mapping[str, Any]]]:
    corpus_root = root / "corpus"
    metadata_path = corpus_root / "corpus.json"
    metadata = _load_json(metadata_path, "corpus", issues)
    if metadata is None:
        return None

    if metadata.get("schema_version") != "1.0":
        issues.append("corpus: corpus.json schema_version must be '1.0'")
    if metadata.get("complete") is not True:
        issues.append("corpus: corpus.json is not marked complete")

    identity = _mapping(metadata.get("corpus"))
    if identity is None:
        issues.append("corpus: corpus.json has no corpus object")
    else:
        if str(identity.get("name", "")).strip().upper() != "CMU ARCTIC":
            issues.append("corpus: corpus name is not CMU ARCTIC")
        if str(identity.get("speaker", "")).strip().lower() != "slt":
            issues.append("corpus: corpus speaker is not SLT")

    source = _mapping(metadata.get("source"))
    if source is None:
        issues.append("corpus: corpus.json has no source object")
    else:
        source_sha = str(source.get("sha256", "")).strip().lower()
        if source_sha != CMU_ARCTIC_SHA256:
            issues.append("corpus: source SHA-256 is not the fixed CMU ARCTIC SLT release digest")
        if source.get("size_bytes") != CMU_ARCTIC_SIZE:
            issues.append("corpus: source size is not the fixed CMU ARCTIC SLT release size")
        archive = _resolve_relative(
            corpus_root, source.get("archive"), "corpus source archive", issues
        )
        if archive is not None and _required_file(archive, "corpus source archive", issues):
            if archive.stat().st_size != CMU_ARCTIC_SIZE:
                issues.append("corpus: downloaded archive does not have the official byte size")
            else:
                _check_hash(
                    archive,
                    CMU_ARCTIC_SHA256,
                    "corpus source archive",
                    issues,
                )

    license_info = _mapping(metadata.get("license"))
    if license_info is None:
        issues.append("corpus: corpus.json has no license provenance")
    else:
        _check_recorded_file(
            corpus_root,
            license_info,
            "file",
            "license",
            issues,
        )

    provenance = _mapping(metadata.get("provenance"))
    if provenance is None:
        issues.append("corpus: corpus.json has no ordering provenance")
    else:
        order = _resolve_relative(
            corpus_root,
            provenance.get("order_file"),
            "corpus order file",
            issues,
        )
        if order is not None and _required_file(order, "corpus order file", issues):
            expected_size = _positive_int(provenance.get("order_file_size_bytes"))
            if expected_size is None or order.stat().st_size != expected_size:
                issues.append("corpus: order file size does not match corpus.json")
            expected_sha = _sha_value(provenance.get("order_file_sha256"))
            if expected_sha is None:
                issues.append("corpus: order file has no valid recorded SHA-256")
            else:
                _check_hash(order, expected_sha, "corpus order file", issues)

    selection = _mapping(metadata.get("selection"))
    if selection is None:
        issues.append("corpus: corpus.json has no selection object")
        return None
    records_value = selection.get("utterances")
    if not isinstance(records_value, list):
        issues.append("corpus: selection.utterances must be a list")
        return None
    records = [row for row in records_value if isinstance(row, dict)]
    if len(records) != len(records_value):
        issues.append("corpus: selection.utterances contains a non-object entry")
    if selection.get("utterance_count") != len(records_value):
        issues.append("corpus: utterance_count does not match selection.utterances")

    selected_paths: Set[Path] = set()
    speech_seconds = 0.0
    for index, record in enumerate(records):
        label = "corpus utterance[{0}]".format(index)
        selected = _resolve_relative(
            corpus_root, record.get("selected_path"), label + " WAV", issues
        )
        if selected is None:
            continue
        selected_paths.add(selected)
        info = _inspect_wav(selected, label + " WAV", issues)
        if info is None:
            continue
        frames, sample_rate, channels, sample_width = info
        speech_seconds += frames / sample_rate
        if record.get("frames") != frames or record.get("sample_rate") != sample_rate:
            issues.append(label + ": WAV duration metadata does not match the file")
        if record.get("channels") != channels:
            issues.append(label + ": WAV channel metadata does not match the file")
        if record.get("sample_width_bytes") != sample_width:
            issues.append(label + ": WAV sample-width metadata does not match the file")
        expected_size = _positive_int(record.get("size_bytes"))
        if expected_size is None or selected.stat().st_size != expected_size:
            issues.append(label + ": WAV size does not match corpus.json")
        expected_sha = _sha_value(record.get("sha256"))
        if expected_sha is None:
            issues.append(label + ": no valid recorded SHA-256")
        else:
            _check_hash(selected, expected_sha, label + " WAV", issues)

    selected_dir = None
    outputs = _mapping(metadata.get("outputs"))
    if outputs is None:
        issues.append("corpus: corpus.json has no outputs object")
    else:
        selected_dir = _resolve_relative(
            corpus_root, outputs.get("selected_dir"), "corpus selected directory", issues
        )
        if selected_dir is not None:
            if not selected_dir.is_dir() or selected_dir.is_symlink():
                issues.append("corpus: selected directory is missing or unsafe")
            else:
                actual_selected = {
                    path.resolve()
                    for path in selected_dir.iterdir()
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".wav"
                }
                if actual_selected != selected_paths:
                    issues.append("corpus: selected directory does not exactly match corpus.json")
        continuous = outputs.get("continuous_wav")
        if continuous is None:
            issues.append("corpus: no continuous WAV is recorded")
        else:
            continuous_path = _resolve_relative(
                corpus_root, continuous, "corpus continuous WAV", issues
            )
            if continuous_path is not None:
                _inspect_wav(continuous_path, "corpus continuous WAV", issues)
                expected_size = _positive_int(outputs.get("continuous_size_bytes"))
                if _required_file(continuous_path, "corpus continuous WAV", issues) and (
                    expected_size is None or continuous_path.stat().st_size != expected_size
                ):
                    issues.append("corpus: continuous WAV size does not match corpus.json")
                expected_sha = _sha_value(outputs.get("continuous_sha256"))
                if expected_sha is None:
                    issues.append("corpus: continuous WAV has no valid recorded SHA-256")
                elif continuous_path.is_file():
                    _check_hash(
                        continuous_path,
                        expected_sha,
                        "corpus continuous WAV",
                        issues,
                    )

    recorded_duration = _finite_number(selection.get("selected_duration_s"))
    if recorded_duration is None:
        issues.append("corpus: selected_duration_s is not finite")
    else:
        if not MIN_SPEECH_SECONDS <= recorded_duration <= MAX_SPEECH_SECONDS:
            issues.append("corpus: recorded pure-speech duration is outside 1800-3600 s")
        if records and abs(recorded_duration - speech_seconds) > 0.05:
            issues.append("corpus: recorded pure-speech duration does not match selected WAVs")
    if records and not MIN_SPEECH_SECONDS <= speech_seconds <= MAX_SPEECH_SECONDS:
        issues.append("corpus: measured pure-speech duration is outside 1800-3600 s")
    if selection.get("reached_target") is not True:
        issues.append("corpus: duration selection did not reach its target")
    return records


def _check_segments(root: Path, issues: List[str]) -> None:
    segment_root = root / "segments"
    rows = _read_csv(segment_root / "segments.csv", "segments", issues)
    if rows is None:
        return
    if not rows:
        issues.append("segments: segments.csv has no records")
        return
    required = {"id", "segment_path", "duration_s", "sample_rate", "samples"}
    _check_csv_fields(rows, required, "segments", issues)
    seen: Set[str] = set()
    for index, row in enumerate(rows):
        item_id = row.get("id", "").strip()
        label = "segments row[{0}]".format(index)
        if not item_id:
            issues.append(label + ": id is empty")
        elif item_id in seen:
            issues.append(label + ": duplicate id " + item_id)
        seen.add(item_id)
        path = _resolve_relative(segment_root, row.get("segment_path"), label + " WAV", issues)
        if path is None:
            continue
        info = _inspect_wav(path, label + " WAV", issues)
        if info is None:
            continue
        frames, sample_rate, _, _ = info
        declared_samples = _integer(row.get("samples"))
        declared_rate = _integer(row.get("sample_rate"))
        declared_duration = _finite_number(row.get("duration_s"))
        if declared_samples != frames or declared_rate != sample_rate:
            issues.append(label + ": WAV frame/rate metadata does not match")
        if declared_duration is None or abs(declared_duration - frames / sample_rate) > 0.01:
            issues.append(label + ": WAV duration metadata does not match")


def _check_dataset(root: Path, issues: List[str]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    dataset = root / "dataset"
    rows = _read_csv(dataset / "manifest.csv", "dataset manifest", issues)
    metadata = _load_json(dataset / "dataset.json", "dataset", issues)
    _required_file(
        dataset / "report.html",
        "dataset report.html",
        issues,
        minimum_size=MIN_REPORT_BYTES,
    )
    if rows is None:
        return [], None
    required = {
        "id",
        "split",
        "audio_path",
        "f0_path",
        "source_sha256",
        "sample_rate",
        "samples",
    }
    _check_csv_fields(rows, required, "dataset manifest", issues)
    if len(rows) < MIN_DATASET_RECORDS:
        issues.append(
            "dataset: manifest has {0} records; at least {1} are required".format(
                len(rows), MIN_DATASET_RECORDS
            )
        )

    seen: Set[str] = set()
    split_counts: Dict[str, int] = {}
    usable_for_fingerprint: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        item_id = row.get("id", "").strip()
        label = "dataset row[{0}]".format(index)
        split = row.get("split", "").strip().lower()
        if not item_id:
            issues.append(label + ": id is empty")
        elif item_id in seen:
            issues.append(label + ": duplicate id " + item_id)
        seen.add(item_id)
        if split not in {"train", "validation", "test"}:
            issues.append(label + ": invalid split " + repr(split))
        else:
            split_counts[split] = split_counts.get(split, 0) + 1
        source_sha = _sha_value(row.get("source_sha256"))
        if source_sha is None:
            issues.append(label + ": source_sha256 is invalid")
        if item_id and source_sha is not None and split:
            usable_for_fingerprint.append(row)

        audio = _resolve_relative(dataset, row.get("audio_path"), label + " audio", issues)
        if audio is not None:
            info = _inspect_wav(audio, label + " audio", issues)
            if info is not None:
                frames, sample_rate, _, _ = info
                if sample_rate != 16000:
                    issues.append(label + ": dataset WAV is not 16 kHz")
                if (
                    _integer(row.get("samples")) != frames
                    or _integer(row.get("sample_rate")) != sample_rate
                ):
                    issues.append(label + ": audio frame/rate metadata does not match")
        f0 = _resolve_relative(dataset, row.get("f0_path"), label + " F0", issues)
        if f0 is not None:
            _check_f0_file(f0, label + " F0", issues)

    for split in ("train", "validation", "test"):
        if not split_counts.get(split):
            issues.append("dataset: no {0} records".format(split))

    fingerprint = _dataset_fingerprint(usable_for_fingerprint)
    if not SHA256_PATTERN.fullmatch(fingerprint):
        issues.append("dataset: computed fingerprint is invalid")
        fingerprint = None
    recorded = metadata.get("dataset_fingerprint") if metadata is not None else None
    if not isinstance(recorded, str) or not SHA256_PATTERN.fullmatch(recorded.lower()):
        issues.append("dataset: dataset.json has no valid dataset_fingerprint")
    elif fingerprint is not None and recorded.lower() != fingerprint:
        issues.append("dataset: dataset fingerprint does not match manifest.csv")
    if metadata is not None and metadata.get("sample_rate") != 16000:
        issues.append("dataset: dataset.json sample_rate is not 16000")
    return rows, fingerprint


def _check_experiment_and_training(
    root: Path, issues: List[str], dataset_fingerprint: Optional[str]
) -> Optional[Path]:
    experiment = root / "experiment"
    metadata_path = experiment / "experiment.json"
    config = experiment / "config.yaml"
    decoder = experiment / "decoder.yaml"
    metadata = _load_json(metadata_path, "experiment", issues)
    config_ok = _required_file(config, "experiment config.yaml", issues)
    decoder_ok = _required_file(decoder, "experiment decoder.yaml", issues)

    if metadata is not None:
        recorded_fingerprint = metadata.get("dataset_fingerprint")
        if not isinstance(recorded_fingerprint, str) or not SHA256_PATTERN.fullmatch(
            recorded_fingerprint.lower()
        ):
            issues.append("experiment: dataset_fingerprint is invalid")
        elif (
            dataset_fingerprint is not None and recorded_fingerprint.lower() != dataset_fingerprint
        ):
            issues.append("experiment: dataset fingerprint does not match dataset")
        if config_ok:
            expected = _sha_value(metadata.get("config_sha256"))
            if expected is None:
                issues.append("experiment: config_sha256 is invalid")
            else:
                _check_hash(config, expected, "experiment config.yaml", issues)
        if decoder_ok:
            expected = _sha_value(metadata.get("decoder_sha256"))
            if expected is None:
                issues.append("experiment: decoder_sha256 is invalid")
            else:
                _check_hash(decoder, expected, "experiment decoder.yaml", issues)
        if metadata.get("decoder_config") != "decoder.yaml":
            issues.append("experiment: decoder_config does not name decoder.yaml")
        if not str(metadata.get("model", "")).strip():
            issues.append("experiment: model is empty")

    _check_metrics(experiment, issues)
    _required_file(
        root / "metrics.html",
        "training metrics.html",
        issues,
        minimum_size=MIN_REPORT_BYTES,
    )

    checkpoints = sorted(
        path
        for path in (experiment / "runs" / "checkpoints").glob("*.ckpt")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    )
    checkpoint = None
    if not checkpoints:
        issues.append("training: no non-empty checkpoint")
    else:
        last = experiment / "runs" / "checkpoints" / "last.ckpt"
        checkpoint = last if last in checkpoints else checkpoints[-1]

    job = _load_json(root / "training_job.json", "training job", issues)
    if job is not None:
        for key in ("job_id", "node", "gpu"):
            value = job.get(key)
            if value is None or not str(value).strip():
                issues.append("training job: {0} is missing or empty".format(key))
        state = str(job.get("state", "")).strip().upper()
        stage = str(job.get("stage", "")).strip().lower().replace("_", "-")
        completion_flag = job.get("pipeline_complete") is True
        if (
            state != "COMPLETED"
            and stage
            not in {
                "pipeline-complete",
                "postprocess-complete",
            }
            and not completion_flag
        ):
            issues.append(
                "training job: neither COMPLETED state nor pipeline-complete marker is present"
            )
        if checkpoint is not None and "checkpoint_sha256" in job:
            expected = _sha_value(job.get("checkpoint_sha256"))
            if expected is None:
                issues.append("training job: checkpoint_sha256 is invalid")
            else:
                _check_hash(checkpoint, expected, "training checkpoint", issues)
    return checkpoint


def _check_metrics(experiment: Path, issues: List[str]) -> None:
    candidates: Dict[int, List[Path]] = {}
    for path in experiment.rglob("metrics.csv") if experiment.is_dir() else []:
        match = VERSION_PATTERN.fullmatch(path.parent.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.setdefault(int(match.group(1)), []).append(path)
    if not candidates:
        issues.append("training metrics: no version_N/metrics.csv")
        return
    latest = max(candidates)
    paths = sorted(candidates[latest])
    if len(paths) != 1:
        issues.append("training metrics: latest version_{0} is ambiguous".format(latest))
        return
    path = paths[0]
    rows = _read_csv(path, "training metrics", issues)
    if rows is None or not rows:
        if rows == []:
            issues.append("training metrics: latest metrics.csv has no rows")
        return
    fieldnames = set(rows[0])
    train_names = sorted(name for name in fieldnames if name.lower() == "train_loss")
    validation_names = sorted(name for name in fieldnames if name.lower() == "val_loss")
    lr_names = sorted(name for name in fieldnames if _is_lr_metric(name))
    if not train_names:
        issues.append("training metrics: train_loss series is missing")
    if not validation_names:
        issues.append("training metrics: val_loss series is missing")
    if not lr_names:
        issues.append("training metrics: learning-rate series is missing")

    max_step: Optional[int] = None
    for row in rows:
        step = _integer(row.get("step"))
        if step is not None:
            max_step = step if max_step is None else max(max_step, step)
    if max_step is None or max_step < MIN_TRAIN_STEP:
        issues.append("training metrics: maximum step is below {0}".format(MIN_TRAIN_STEP))
    for label, names in (
        ("train_loss", train_names),
        ("val_loss", validation_names),
        ("learning-rate", lr_names),
    ):
        if names and not any(
            _finite_number(row.get(name)) is not None for row in rows for name in names
        ):
            issues.append("training metrics: {0} has no finite values".format(label))


def _check_outputs(
    root: Path,
    issues: List[str],
    test_ids: Set[str],
    checkpoint: Optional[Path],
) -> None:
    reconstruction = root / "reconstruction"
    if not test_ids:
        issues.append("reconstruction: dataset supplies no test IDs")
    for item_id in sorted(test_ids):
        _inspect_wav(
            reconstruction / (item_id + ".wav"),
            "reconstruction " + item_id,
            issues,
        )
    _required_file(
        root / "comparison.html",
        "comparison.html",
        issues,
        minimum_size=MIN_REPORT_BYTES,
    )

    manipulation_root = root / "manipulations"
    metadata = _load_json(manipulation_root / "manipulation.json", "manipulation", issues)
    if metadata is not None:
        outputs_value = metadata.get("outputs")
        if not isinstance(outputs_value, list):
            issues.append("manipulation: outputs must be a list")
        else:
            checked_outputs: List[Tuple[str, Path]] = []
            seen_shifts: Set[float] = set()
            for index, raw in enumerate(outputs_value):
                label = "manipulation output[{0}]".format(index)
                if not isinstance(raw, dict):
                    issues.append(label + ": entry is not an object")
                    continue
                directory = _resolve_relative(
                    manipulation_root,
                    raw.get("directory"),
                    label + " directory",
                    issues,
                )
                if "semitones" in raw:
                    shift = _finite_number(raw.get("semitones"))
                    if shift is None or shift == 0.0:
                        issues.append(label + ": semitone shift must be finite and non-zero")
                        continue
                    if shift in seen_shifts:
                        issues.append(label + ": duplicate semitone shift")
                    seen_shifts.add(shift)
                    output_label = "manipulation {0:+g} st".format(shift)
                else:
                    controls = raw.get("controls")
                    if not isinstance(controls, dict) or not controls:
                        issues.append(
                            label + ": non-pitch output must declare a non-empty controls object"
                        )
                        continue
                    output_label = "manipulation " + str(
                        raw.get("name", raw.get("directory", index))
                    )
                if directory is not None:
                    checked_outputs.append((output_label, directory))
            if len(seen_shifts) < 2:
                issues.append("manipulation: at least two distinct non-zero shifts are required")
            for label, directory in checked_outputs:
                if not directory.is_dir() or directory.is_symlink():
                    issues.append(label + ": output directory is missing or unsafe")
                    continue
                wavs = [
                    path
                    for path in directory.rglob("*.wav")
                    if path.is_file() and not path.is_symlink()
                ]
                if not wavs:
                    issues.append(label + ": contains no WAV")
                for item_id in sorted(test_ids):
                    _inspect_wav(
                        directory / (item_id + ".wav"),
                        label + " " + item_id,
                        issues,
                    )
        recorded_checkpoint = _sha_value(metadata.get("checkpoint_sha256"))
        if recorded_checkpoint is None:
            issues.append("manipulation: checkpoint_sha256 is invalid")
        elif checkpoint is not None:
            _check_hash(
                checkpoint,
                recorded_checkpoint,
                "manipulation checkpoint",
                issues,
            )
    _required_file(
        root / "manipulation.html",
        "manipulation.html",
        issues,
        minimum_size=MIN_REPORT_BYTES,
    )


def _load_json(path: Path, label: str, issues: List[str]) -> Optional[Dict[str, Any]]:
    if not _required_file(path, label + " JSON", issues):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(label + ": invalid JSON ({0})".format(error))
        return None
    if not isinstance(payload, dict):
        issues.append(label + ": JSON top level is not an object")
        return None
    return payload


def _read_csv(path: Path, label: str, issues: List[str]) -> Optional[List[Dict[str, str]]]:
    if not _required_file(path, label + " CSV", issues):
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames is None:
                issues.append(label + ": CSV has no header")
                return None
            fields = [field.strip() for field in reader.fieldnames]
            if any(not field for field in fields) or len(fields) != len(set(fields)):
                issues.append(label + ": CSV header is empty or duplicated")
                return None
            reader.fieldnames = fields
            rows: List[Dict[str, str]] = []
            for row in reader:
                if None in row:
                    issues.append(label + ": CSV row has more values than columns")
                    return None
                rows.append(
                    {
                        key: (value.strip() if value is not None else "")
                        for key, value in row.items()
                    }
                )
            return rows
    except (OSError, UnicodeError, csv.Error) as error:
        issues.append(label + ": unreadable CSV ({0})".format(error))
        return None


def _check_csv_fields(
    rows: Sequence[Mapping[str, str]],
    required: Set[str],
    label: str,
    issues: List[str],
) -> None:
    if not rows:
        return
    missing = sorted(required - set(rows[0]))
    if missing:
        issues.append(label + ": missing columns " + ", ".join(missing))


def _required_file(
    path: Path,
    label: str,
    issues: List[str],
    minimum_size: int = 1,
) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < minimum_size:
        issues.append(
            "{0}: missing, unsafe, or smaller than {1} byte(s): {2}".format(
                label, minimum_size, path
            )
        )
        return False
    return True


def _resolve_relative(base: Path, value: Any, label: str, issues: List[str]) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        issues.append(label + ": relative path is missing")
        return None
    relative = Path(value)
    if relative.is_absolute():
        issues.append(label + ": path must be relative")
        return None
    base = base.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        issues.append(label + ": path escapes its artifact directory")
        return None
    return candidate


def _inspect_wav(path: Path, label: str, issues: List[str]) -> Optional[Tuple[int, int, int, int]]:
    if not _required_file(path, label, issues):
        return None
    try:
        with wave.open(str(path), "rb") as stream:
            frames = stream.getnframes()
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
    except (OSError, EOFError, wave.Error) as error:
        issues.append(label + ": unreadable WAV ({0})".format(error))
        return None
    if frames <= 0 or sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        issues.append(label + ": WAV has invalid audio dimensions")
        return None
    return frames, sample_rate, channels, sample_width


def _check_f0_file(path: Path, label: str, issues: List[str]) -> None:
    if not _required_file(path, label, issues):
        return
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                for token in line.split():
                    count += 1
                    try:
                        value = float(token)
                    except ValueError:
                        issues.append(
                            "{0}: non-numeric value on line {1}".format(label, line_number)
                        )
                        return
                    if not math.isfinite(value) or value < 0:
                        issues.append(
                            "{0}: F0 value is negative or non-finite on line {1}".format(
                                label, line_number
                            )
                        )
                        return
    except (OSError, UnicodeError) as error:
        issues.append(label + ": unreadable F0 file ({0})".format(error))
        return
    if count == 0:
        issues.append(label + ": F0 file has no values")


def _check_recorded_file(
    base: Path,
    metadata: Mapping[str, Any],
    path_key: str,
    label: str,
    issues: List[str],
) -> None:
    path = _resolve_relative(base, metadata.get(path_key), "corpus " + label, issues)
    if path is None or not _required_file(path, "corpus " + label, issues):
        return
    expected_size = _positive_int(metadata.get("size_bytes"))
    if expected_size is None or path.stat().st_size != expected_size:
        issues.append("corpus: {0} size does not match corpus.json".format(label))
    expected_sha = _sha_value(metadata.get("sha256"))
    if expected_sha is None:
        issues.append("corpus: {0} has no valid recorded SHA-256".format(label))
    else:
        _check_hash(path, expected_sha, "corpus " + label, issues)


def _check_hash(path: Path, expected: str, label: str, issues: List[str]) -> None:
    try:
        actual = sha256_file(path)
    except OSError as error:
        issues.append(label + ": could not compute SHA-256 ({0})".format(error))
        return
    if actual.lower() != expected.lower():
        issues.append(label + ": SHA-256 mismatch")


def _dataset_fingerprint(rows: Iterable[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.get("id", "")):
        digest.update(row.get("id", "").encode("utf-8"))
        digest.update(row.get("source_sha256", "").encode("utf-8"))
        digest.update(row.get("split", "").encode("utf-8"))
    return digest.hexdigest()


def _mapping(value: Any) -> Optional[Mapping[str, Any]]:
    return value if isinstance(value, dict) else None


def _sha_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if SHA256_PATTERN.fullmatch(value) else None


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _positive_int(value: Any) -> Optional[int]:
    number = _integer(value)
    return number if number is not None and number > 0 else None


def _is_lr_metric(name: str) -> bool:
    lowered = name.strip().lower()
    return (
        lowered == "lr"
        or lowered.startswith("lr-")
        or lowered.startswith("lr_")
        or "learning_rate" in lowered
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the complete CMU ARCTIC SLT PhonLab-DDSP pipeline"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--count",
        action="store_true",
        help="print only the number of acceptance failures",
    )
    args = parser.parse_args(argv)
    issues = collect_issues(args.root)
    if args.count:
        print(len(issues))
        return 0
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    print(SUCCESS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
