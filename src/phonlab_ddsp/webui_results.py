"""Safe, dependency-free browsing and export of postprocess audio results.

This module is deliberately independent of a particular Web UI framework.  A
caller first loads a validated :class:`ResultCatalog`, then uses only catalog
item identifiers and condition names for export.  Every export reloads the
catalog before opening source files, so a result tree changed after discovery
is rejected rather than silently copied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

MAX_CONDITIONS = 128
MAX_FILES_PER_CONDITION = 10_000
MAX_EXPORT_FILES = 10_000
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_WAV_BYTES = 512 * 1024 * 1024
MAX_EXPORT_BYTES = 2 * 1024 * 1024 * 1024

_REPORT_NAMES = ("reconstruction.html", "manipulation.html", "metrics.html")
_PROVENANCE_FIELDS = (
    "schema_version",
    "created_at",
    "operation",
    "unvoiced_policy",
    "model",
    "dataset_fingerprint",
    "checkpoint_sha256",
)


@dataclass(frozen=True)
class ClippingSummary:
    """Aggregate clipping information from one ``_render.json`` file."""

    clipped_samples: int
    samples: int
    clipped_fraction: float
    files_with_clipping: int

    def to_dict(self) -> dict[str, object]:
        return {
            "clipped_samples": self.clipped_samples,
            "samples": self.samples,
            "clipped_fraction": self.clipped_fraction,
            "files_with_clipping": self.files_with_clipping,
        }


@dataclass(frozen=True)
class ResultCondition:
    """One baseline or manipulation condition in a validated result tree."""

    name: str
    label: str
    controls: Mapping[str, float]
    directory: str
    render_metadata: str
    files: tuple[str, ...]
    file_audits: Mapping[str, tuple[int, int]]
    clipping: ClippingSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "controls": dict(self.controls),
            "directory": self.directory,
            "render_metadata": self.render_metadata,
            "file_count": len(self.files),
            "clipping": self.clipping.to_dict(),
        }


@dataclass(frozen=True)
class ResultAudio:
    """One directly playable catalog audio entry."""

    item_id: str
    condition: str
    label: str
    controls: Mapping[str, float]
    path: str
    clipped_samples: int
    samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "condition": self.condition,
            "label": self.label,
            "controls": dict(self.controls),
            "path": self.path,
            "clipped_samples": self.clipped_samples,
            "samples": self.samples,
            "clipped_fraction": self.clipped_samples / self.samples if self.samples else 0.0,
        }


@dataclass(frozen=True)
class ResultItem:
    """One utterance with baseline and variants in a uniform audio sequence."""

    item_id: str
    audio: tuple[ResultAudio, ...]

    @property
    def baseline_path(self) -> str:
        return self.audio[0].path

    @property
    def variant_paths(self) -> dict[str, str]:
        return {entry.condition: entry.path for entry in self.audio[1:]}

    def to_dict(self) -> dict[str, object]:
        entries = [entry.to_dict() for entry in self.audio]
        return {
            "id": self.item_id,
            "item_id": self.item_id,
            "audio": entries,
            "baseline": entries[0],
            "variants": entries[1:],
        }


@dataclass(frozen=True)
class ResultCatalog:
    """Validated, JSON-serializable view of one postprocess directory."""

    workspace_root: Path
    result_root: Path
    provenance: Mapping[str, object]
    reports: Mapping[str, str]
    baseline: ResultCondition
    conditions: tuple[ResultCondition, ...]
    items: tuple[ResultItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "workspace_root": str(self.workspace_root),
            "result_root": str(self.result_root),
            "result_relative_path": self.result_root.relative_to(self.workspace_root).as_posix(),
            "provenance": dict(self.provenance),
            "reports": dict(self.reports),
            "baseline": self.baseline.to_dict(),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "items": [item.to_dict() for item in self.items],
        }

    def as_dict(self) -> dict[str, object]:
        """Compatibility-friendly alias for :meth:`to_dict`."""

        return self.to_dict()


@dataclass(frozen=True)
class ExportReceipt:
    """Description of a completed directory export."""

    destination: Path
    provenance_path: Path
    files: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": str(self.destination),
            "provenance_path": str(self.provenance_path),
            "file_count": len(self.files),
            "files": [str(path) for path in self.files],
        }


def load_result_catalog(
    workspace_root: PathLike,
    result_root: PathLike,
    *,
    max_files_per_condition: int = MAX_FILES_PER_CONDITION,
) -> ResultCatalog:
    """Validate a postprocess directory and return its read-only catalog.

    ``result_root`` may be absolute or relative to ``workspace_root``.  It must
    remain beneath that workspace.  Metadata paths, source WAV files, render
    metadata, reports, and every existing path component used here must be
    regular non-symbolic-link entries.
    """

    file_limit = _bounded_positive_int(
        max_files_per_condition,
        "max_files_per_condition",
        MAX_FILES_PER_CONDITION,
    )
    workspace = _existing_workspace(workspace_root)
    root = _existing_result_root(workspace, result_root)

    baseline_directory = _safe_existing_directory(root, "reconstruction")
    manipulations_directory = _safe_existing_directory(root, "manipulations")
    baseline = _load_render_condition(
        result_root=root,
        directory=baseline_directory,
        name="baseline",
        label="Reconstruction",
        controls={},
        directory_relative="reconstruction",
        render_relative="reconstruction/_render.json",
        file_limit=file_limit,
    )

    metadata_path = _safe_existing_file(root, "manipulations/manipulation.json")
    metadata = _read_json_object(metadata_path)
    raw_outputs = metadata.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ValueError("manipulation.json outputs must be a non-empty array")
    if len(raw_outputs) > MAX_CONDITIONS:
        raise ValueError(f"Too many manipulation conditions (maximum {MAX_CONDITIONS})")

    conditions = []
    names = set()
    directories = set()
    baseline_files = set(baseline.files)
    for index, raw_condition in enumerate(raw_outputs):
        if not isinstance(raw_condition, dict):
            raise ValueError(f"manipulation output {index} must be an object")
        name = _safe_name(raw_condition.get("name"), f"outputs[{index}].name")
        if name == "baseline" or name in names:
            raise ValueError(f"Duplicate or reserved manipulation name: {name!r}")
        names.add(name)
        raw_directory = raw_condition.get("directory")
        relative_directory = _safe_relative_path(
            raw_directory, f"outputs[{index}].directory", single_component=True
        )
        directory_string = relative_directory.as_posix()
        if directory_string in directories:
            raise ValueError(f"Duplicate manipulation directory: {directory_string!r}")
        directories.add(directory_string)
        condition_directory = _safe_existing_directory(manipulations_directory, directory_string)
        controls = _controls(raw_condition.get("controls"), f"outputs[{index}].controls")
        label_value = raw_condition.get("label", name)
        if not isinstance(label_value, str) or not label_value.strip() or len(label_value) > 512:
            raise ValueError(f"outputs[{index}].label must be a short non-empty string")

        expected_render = f"{directory_string}/_render.json"
        render_audit = raw_condition.get("render_audit")
        if not isinstance(render_audit, dict):
            raise ValueError(f"outputs[{index}].render_audit must be an object")
        supplied_render = _safe_relative_path(
            render_audit.get("metadata"),
            f"outputs[{index}].render_audit.metadata",
        ).as_posix()
        if supplied_render != expected_render:
            raise ValueError(f"outputs[{index}].render_audit.metadata must be {expected_render!r}")

        condition = _load_render_condition(
            result_root=root,
            directory=condition_directory,
            name=name,
            label=label_value.strip(),
            controls=controls,
            directory_relative=f"manipulations/{directory_string}",
            render_relative=f"manipulations/{expected_render}",
            file_limit=file_limit,
        )
        condition_files = set(condition.files)
        if condition_files != baseline_files:
            missing = len(baseline_files - condition_files)
            unexpected = len(condition_files - baseline_files)
            raise ValueError(
                f"Condition {name!r} does not match baseline WAV set "
                f"(missing={missing}, unexpected={unexpected})"
            )
        conditions.append(condition)

    reports = {}
    for report_name in _REPORT_NAMES:
        raw_path = root / report_name
        if os.path.lexists(raw_path):
            report = _safe_existing_file(root, report_name)
            reports[report.stem] = report.relative_to(root).as_posix()

    condition_by_name = {condition.name: condition for condition in conditions}
    items = []
    for item_id in baseline.files:
        audio = [
            _result_audio(
                baseline,
                item_id,
                f"{baseline.directory}/{item_id}",
            )
        ]
        audio.extend(
            _result_audio(
                condition_by_name[condition.name],
                item_id,
                f"{condition.directory}/{item_id}",
            )
            for condition in conditions
        )
        items.append(
            ResultItem(
                item_id=item_id,
                audio=tuple(audio),
            )
        )

    provenance = {field: metadata[field] for field in _PROVENANCE_FIELDS if field in metadata}
    return ResultCatalog(
        workspace_root=workspace,
        result_root=root,
        provenance=provenance,
        reports=reports,
        baseline=baseline,
        conditions=tuple(conditions),
        items=tuple(items),
    )


def discover_result_catalog(
    workspace_root: PathLike,
    result_root: PathLike,
    *,
    max_files_per_condition: int = MAX_FILES_PER_CONDITION,
) -> ResultCatalog:
    """Descriptive alias for :func:`load_result_catalog`."""

    return load_result_catalog(
        workspace_root,
        result_root,
        max_files_per_condition=max_files_per_condition,
    )


def export_wav(
    catalog: ResultCatalog,
    condition: str,
    item_id: str,
    destination: PathLike,
) -> ExportReceipt:
    """Copy one catalog WAV plus provenance into a new destination directory."""

    fresh = _reload_catalog(catalog)
    selected_condition = _select_condition(fresh, condition)
    selected_item = _select_item(fresh, item_id)
    source_relative = _item_path(selected_item, selected_condition.name)
    return _export_directory(
        fresh,
        selected_condition,
        (selected_item.item_id,),
        (source_relative,),
        destination,
        selection_kind="wav",
    )


def export_condition(
    catalog: ResultCatalog,
    condition: str,
    destination: PathLike,
    *,
    max_files: int = MAX_EXPORT_FILES,
) -> ExportReceipt:
    """Copy every WAV from one condition plus provenance to a new directory."""

    limit = _bounded_positive_int(max_files, "max_files", MAX_EXPORT_FILES)
    fresh = _reload_catalog(catalog)
    selected_condition = _select_condition(fresh, condition)
    if len(fresh.items) > limit:
        raise ValueError(f"Condition has {len(fresh.items)} files; export limit is {limit}")
    item_ids = tuple(item.item_id for item in fresh.items)
    paths = tuple(_item_path(item, selected_condition.name) for item in fresh.items)
    return _export_directory(
        fresh,
        selected_condition,
        item_ids,
        paths,
        destination,
        selection_kind="condition",
    )


def create_export_zip(
    catalog: ResultCatalog,
    condition: str,
    *,
    item_id: Optional[str] = None,
    cache_root: Optional[PathLike] = None,
    max_files: int = MAX_EXPORT_FILES,
) -> Path:
    """Atomically create a ZIP for one WAV or a complete condition.

    The default cache is ``<workspace>/.cache/webui_exports``.  A caller-
    supplied cache must also remain inside the catalog workspace and outside
    the result tree.  The returned archive name is unique and is never reused.
    """

    limit = _bounded_positive_int(max_files, "max_files", MAX_EXPORT_FILES)
    fresh = _reload_catalog(catalog)
    selected_condition = _select_condition(fresh, condition)
    if item_id is None:
        if len(fresh.items) > limit:
            raise ValueError(f"Condition has {len(fresh.items)} files; ZIP limit is {limit}")
        selected_items = fresh.items
        kind = "condition"
    else:
        selected_items = (_select_item(fresh, item_id),)
        kind = "wav"
    item_ids = tuple(item.item_id for item in selected_items)
    source_paths = tuple(_item_path(item, selected_condition.name) for item in selected_items)

    raw_cache = cache_root
    if raw_cache is None:
        raw_cache = fresh.workspace_root / ".cache" / "webui_exports"
    cache = _safe_output_directory(fresh, raw_cache)
    safe_label = selected_condition.name
    final = cache / f"phonlab-{safe_label}-{uuid.uuid4().hex}.zip"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".webui-export-", suffix=".tmp", dir=cache)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        file_records = []
        total_bytes = 0
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=False,
        ) as archive:
            for item, source_relative in zip(selected_items, source_paths):
                source = _safe_existing_file(fresh.result_root, source_relative)
                archive_relative = _archive_audio_path(selected_condition, item.item_id)
                digest, size = _copy_into_zip(archive, source, archive_relative)
                total_bytes = _checked_total_size(total_bytes, size)
                file_records.append(
                    {
                        "item_id": item.item_id,
                        "source": source_relative,
                        "exported": archive_relative,
                        "bytes": size,
                        "sha256": digest,
                    }
                )
            provenance = _export_provenance(
                fresh,
                selected_condition,
                kind,
                item_ids,
                file_records,
            )
            archive.writestr(
                "provenance.json",
                json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            )
        os.replace(temporary, final)
    except Exception:
        if os.path.lexists(temporary):
            temporary.unlink()
        raise
    return final


def _load_render_condition(
    *,
    result_root: Path,
    directory: Path,
    name: str,
    label: str,
    controls: Mapping[str, float],
    directory_relative: str,
    render_relative: str,
    file_limit: int,
) -> ResultCondition:
    render_path = _safe_existing_file(result_root, render_relative)
    render = _read_json_object(render_path)
    render_controls = _controls(render.get("controls"), f"{render_relative}.controls")
    expected_runtime_controls = {
        control_name: value
        for control_name, value in controls.items()
        if control_name != "pitch_semitones"
    }
    if render_controls != expected_runtime_controls:
        raise ValueError(
            f"{render_relative} controls do not match manipulation metadata "
            f"({render_controls!r} != {expected_runtime_controls!r})"
        )

    raw_files = render.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"{render_relative}.files must be a non-empty array")
    if len(raw_files) > file_limit:
        raise ValueError(f"{render_relative} exceeds file limit {file_limit}")
    files = []
    file_audits = {}
    seen = set()
    clipped_sum = 0
    sample_sum = 0
    clipped_files = 0
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ValueError(f"{render_relative}.files[{index}] must be an object")
        relative = _safe_relative_path(
            raw_file.get("path"), f"{render_relative}.files[{index}].path"
        )
        if relative.suffix.lower() != ".wav":
            raise ValueError(f"Render file must have a .wav suffix: {relative.as_posix()!r}")
        relative_string = relative.as_posix()
        if relative_string in seen:
            raise ValueError(f"Duplicate render file path: {relative_string!r}")
        seen.add(relative_string)
        source_relative = f"{directory_relative}/{relative_string}"
        source = _safe_existing_file(result_root, source_relative)
        size = source.stat().st_size
        if size > MAX_WAV_BYTES:
            raise ValueError(f"WAV is too large for Web UI export: {source_relative}")
        clipped = _nonnegative_int(
            raw_file.get("clipped_samples"),
            f"{render_relative}.files[{index}].clipped_samples",
        )
        samples = _nonnegative_int(
            raw_file.get("samples"), f"{render_relative}.files[{index}].samples"
        )
        if clipped > samples:
            raise ValueError(f"Clipped samples exceed sample count in {relative_string!r}")
        clipped_sum += clipped
        sample_sum += samples
        clipped_files += int(clipped > 0)
        files.append(relative_string)
        file_audits[relative_string] = (clipped, samples)

    files_written = _nonnegative_int(
        render.get("files_written"), f"{render_relative}.files_written"
    )
    aggregate_clipped = _nonnegative_int(
        render.get("clipped_samples"), f"{render_relative}.clipped_samples"
    )
    aggregate_samples = _nonnegative_int(render.get("samples"), f"{render_relative}.samples")
    fraction = render.get("clipped_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError(f"{render_relative}.clipped_fraction must be a finite number")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{render_relative}.clipped_fraction must be between zero and one")
    expected_fraction = clipped_sum / sample_sum if sample_sum else 0.0
    if (
        files_written != len(files)
        or aggregate_clipped != clipped_sum
        or aggregate_samples != sample_sum
        or not math.isclose(fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError(f"{render_relative} aggregate render audit is inconsistent")

    return ResultCondition(
        name=name,
        label=label,
        controls=dict(controls),
        directory=directory_relative,
        render_metadata=render_relative,
        files=tuple(files),
        file_audits=file_audits,
        clipping=ClippingSummary(
            clipped_samples=clipped_sum,
            samples=sample_sum,
            clipped_fraction=expected_fraction,
            files_with_clipping=clipped_files,
        ),
    )


def _export_directory(
    catalog: ResultCatalog,
    condition: ResultCondition,
    item_ids: Sequence[str],
    source_paths: Sequence[str],
    destination: PathLike,
    *,
    selection_kind: str,
) -> ExportReceipt:
    if len(source_paths) > MAX_EXPORT_FILES:
        raise ValueError(f"Export exceeds maximum of {MAX_EXPORT_FILES} files")
    target = _new_destination(catalog, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(catalog.workspace_root, target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.export-", dir=target.parent))
    exported_paths = []
    try:
        file_records = []
        total_bytes = 0
        for item_id, source_relative in zip(item_ids, source_paths):
            source = _safe_existing_file(catalog.result_root, source_relative)
            exported_relative = _archive_audio_path(condition, item_id)
            output = staging.joinpath(*PurePosixPath(exported_relative).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest, size = _copy_regular_file(source, output)
            total_bytes = _checked_total_size(total_bytes, size)
            file_records.append(
                {
                    "item_id": item_id,
                    "source": source_relative,
                    "exported": exported_relative,
                    "bytes": size,
                    "sha256": digest,
                }
            )
            exported_paths.append(target / exported_relative)
        provenance = _export_provenance(
            catalog,
            condition,
            selection_kind,
            item_ids,
            file_records,
        )
        provenance_path = staging / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Export destination already exists: {target}")
        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExportReceipt(
        destination=target,
        provenance_path=target / "provenance.json",
        files=tuple(exported_paths),
    )


def _export_provenance(
    catalog: ResultCatalog,
    condition: ResultCondition,
    kind: str,
    item_ids: Sequence[str],
    files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "result_relative_path": catalog.result_root.relative_to(
                catalog.workspace_root
            ).as_posix(),
            "provenance": dict(catalog.provenance),
            "condition": condition.to_dict(),
        },
        "selection": {"kind": kind, "item_ids": list(item_ids)},
        "files": list(files),
    }


def _archive_audio_path(condition: ResultCondition, item_id: str) -> str:
    relative = _safe_relative_path(item_id, "item_id")
    directory = "baseline" if condition.name == "baseline" else condition.name
    return PurePosixPath("audio", directory, *relative.parts).as_posix()


def _reload_catalog(catalog: ResultCatalog) -> ResultCatalog:
    if not isinstance(catalog, ResultCatalog):
        raise TypeError("catalog must be a ResultCatalog returned by load_result_catalog")
    return load_result_catalog(catalog.workspace_root, catalog.result_root)


def _select_condition(catalog: ResultCatalog, name: str) -> ResultCondition:
    if not isinstance(name, str):
        raise ValueError("condition must be a catalog condition name")
    if name == catalog.baseline.name:
        return catalog.baseline
    for condition in catalog.conditions:
        if condition.name == name:
            return condition
    raise ValueError(f"Unknown catalog condition: {name!r}")


def _select_item(catalog: ResultCatalog, item_id: str) -> ResultItem:
    if not isinstance(item_id, str):
        raise ValueError("item_id must be a catalog item identifier")
    for item in catalog.items:
        if item.item_id == item_id:
            return item
    raise ValueError(f"Unknown catalog item: {item_id!r}")


def _item_path(item: ResultItem, condition: str) -> str:
    for audio in item.audio:
        if audio.condition == condition:
            return audio.path
    raise ValueError(f"Item {item.item_id!r} has no condition {condition!r}")


def _result_audio(condition: ResultCondition, item_id: str, path: str) -> ResultAudio:
    clipped_samples, samples = condition.file_audits[item_id]
    return ResultAudio(
        item_id=item_id,
        condition=condition.name,
        label=condition.label,
        controls=dict(condition.controls),
        path=path,
        clipped_samples=clipped_samples,
        samples=samples,
    )


def _existing_workspace(raw_root: PathLike) -> Path:
    lexical = _lexical_absolute(Path(raw_root).expanduser())
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"Workspace root must be a non-symlink directory: {lexical}")
    return lexical.resolve(strict=True)


def _existing_result_root(workspace: Path, raw_root: PathLike) -> Path:
    supplied = Path(raw_root).expanduser()
    candidate = supplied if supplied.is_absolute() else workspace / supplied
    lexical = _lexical_absolute(candidate)
    _require_beneath(lexical, workspace, "Result root")
    _reject_symlink_components(workspace, lexical)
    if not lexical.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {lexical}")
    resolved = lexical.resolve(strict=True)
    _require_beneath(resolved, workspace, "Result root")
    return resolved


def _safe_existing_directory(root: Path, raw_relative: object) -> Path:
    relative = _safe_relative_path(raw_relative, "directory")
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_components(root, candidate)
    if not candidate.is_dir() or candidate.is_symlink():
        raise ValueError(f"Required directory is missing or unsafe: {relative.as_posix()}")
    resolved = candidate.resolve(strict=True)
    _require_beneath(resolved, root, "Directory")
    return resolved


def _safe_existing_file(root: Path, raw_relative: object) -> Path:
    relative = _safe_relative_path(raw_relative, "file")
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_components(root, candidate)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"Required file is missing or unsafe: {relative.as_posix()}")
    resolved = candidate.resolve(strict=True)
    _require_beneath(resolved, root, "File")
    return resolved


def _safe_relative_path(
    raw_path: object,
    field: str,
    *,
    single_component: bool = False,
) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path or len(raw_path) > 4096:
        raise ValueError(f"{field} must be a non-empty relative path")
    if "\x00" in raw_path or "\\" in raw_path:
        raise ValueError(f"{field} contains forbidden path characters")
    raw_parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{field} contains path traversal")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains path traversal")
    if not path.parts or ":" in path.parts[0]:
        raise ValueError(f"{field} is not a portable relative path")
    if single_component and len(path.parts) != 1:
        raise ValueError(f"{field} must be a single directory name")
    return path


def _safe_name(raw_name: object, field: str) -> str:
    path = _safe_relative_path(raw_name, field, single_component=True)
    name = path.as_posix()
    if len(name) > 128:
        raise ValueError(f"{field} is too long")
    return name


def _controls(raw_controls: object, field: str) -> dict[str, float]:
    if not isinstance(raw_controls, dict):
        raise ValueError(f"{field} must be an object")
    controls = {}
    for raw_name, raw_value in raw_controls.items():
        if not isinstance(raw_name, str) or not raw_name or len(raw_name) > 128:
            raise ValueError(f"{field} contains an invalid control name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{field}.{raw_name} must be a finite number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{field}.{raw_name} must be a finite number")
        controls[raw_name] = value
    return controls


def _read_json_object(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"JSON metadata exceeds {MAX_JSON_BYTES} bytes: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON metadata must be an object: {path}")
    return value


def _new_destination(catalog: ResultCatalog, raw_destination: PathLike) -> Path:
    supplied = Path(raw_destination).expanduser()
    candidate = supplied if supplied.is_absolute() else catalog.workspace_root / supplied
    target = _lexical_absolute(candidate)
    _require_beneath(target, catalog.workspace_root, "Export destination")
    if target == catalog.result_root or catalog.result_root in target.parents:
        raise ValueError("Export destination must be outside the source result tree")
    _reject_symlink_components(catalog.workspace_root, target.parent)
    if os.path.lexists(target):
        raise FileExistsError(f"Export destination already exists: {target}")
    return target


def _safe_output_directory(catalog: ResultCatalog, raw_directory: PathLike) -> Path:
    supplied = Path(raw_directory).expanduser()
    candidate = supplied if supplied.is_absolute() else catalog.workspace_root / supplied
    directory = _lexical_absolute(candidate)
    _require_beneath(directory, catalog.workspace_root, "Export cache")
    if directory == catalog.result_root or catalog.result_root in directory.parents:
        raise ValueError("Export cache must be outside the source result tree")
    _reject_symlink_components(catalog.workspace_root, directory.parent)
    directory.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(catalog.workspace_root, directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Export cache is not a safe directory: {directory}")
    return directory


def _copy_regular_file(source: Path, destination: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Source is not a regular file: {source}")
        with os.fdopen(descriptor, "rb") as source_stream, destination.open("xb") as output:
            descriptor = -1
            while True:
                block = source_stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_WAV_BYTES:
                    raise ValueError(f"WAV exceeds export size limit: {source}")
                digest.update(block)
                output.write(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), size


def _copy_into_zip(
    archive: zipfile.ZipFile,
    source: Path,
    archive_relative: str,
) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Source is not a regular file: {source}")
        with os.fdopen(descriptor, "rb") as source_stream:
            descriptor = -1
            with archive.open(archive_relative, "w") as output:
                while True:
                    block = source_stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_WAV_BYTES:
                        raise ValueError(f"WAV exceeds export size limit: {source}")
                    digest.update(block)
                    output.write(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), size


def _checked_total_size(previous: int, addition: int) -> int:
    total = previous + addition
    if total > MAX_EXPORT_BYTES:
        raise ValueError(f"Export exceeds byte limit {MAX_EXPORT_BYTES}")
    return total


def _nonnegative_int(raw_value: object, field: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return raw_value


def _bounded_positive_int(raw_value: object, field: str, maximum: int) -> int:
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, int)
        or not 1 <= raw_value <= maximum
    ):
        raise ValueError(f"{field} must be an integer between 1 and {maximum}")
    return raw_value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_beneath(candidate: Path, root: Path, label: str) -> None:
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"{label} must be below workspace/result root: {candidate}")


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path must remain below its root: {candidate}")
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"Symbolic-link path component is forbidden: {current}")


__all__ = [
    "ClippingSummary",
    "ExportReceipt",
    "ResultCatalog",
    "ResultAudio",
    "ResultCondition",
    "ResultItem",
    "create_export_zip",
    "discover_result_catalog",
    "export_condition",
    "export_wav",
    "load_result_catalog",
]
