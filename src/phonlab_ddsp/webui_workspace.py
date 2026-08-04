"""Bounded, symlink-safe discovery of WebUI workspace artifacts.

The WebUI needs a short catalog of paths, not a general-purpose file browser.
This module therefore recognizes only workflow sentinel files, never reads
their contents, and deliberately avoids payload trees such as audio, caches,
and the imported research engine.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Union

PathLike = Union[str, os.PathLike]

DEFAULT_MAX_ENTRIES = 1_000
MAX_ENTRIES = 10_000
MAX_DEPTH = 8
MAX_SCANNED_ENTRIES = 50_000

_FROZEN_ENGINE_DIRECTORIES = frozenset(
    {"cfg", "configs", "datasets", "loss", "ltng", "models", "scripts"}
)
_PAYLOAD_DIRECTORIES = frozenset(
    {
        "audio",
        "build",
        "corpora",
        "corpus",
        "dist",
        "download",
        "downloads",
        "extracted",
        "f0",
        "jobs",
        "metrics",
        "node_modules",
        "raw",
        "raw-audio",
        "raw-data",
        "raw_audio",
        "raw_data",
        "segments",
        "selected",
        "source",
        "sources",
        "swanlab",
        "wandb",
        "wav",
        "wavs",
    }
)


@dataclass(frozen=True)
class _Entry:
    name: str
    is_directory: bool
    is_file: bool


@dataclass
class _ScanState:
    root: Path
    max_entries: int
    datasets: Dict[str, dict] = field(default_factory=dict)
    experiments: Dict[str, dict] = field(default_factory=dict)
    checkpoints: Dict[str, dict] = field(default_factory=dict)
    results: Dict[str, dict] = field(default_factory=dict)
    scanned_entries: int = 0
    truncated: bool = False
    stop: bool = False

    @property
    def discovered_entries(self) -> int:
        return (
            len(self.datasets) + len(self.experiments) + len(self.checkpoints) + len(self.results)
        )

    def base_item(self, relative: PurePosixPath) -> dict:
        relative_text = relative.as_posix()
        path = self.root if relative_text == "." else self.root.joinpath(*relative.parts)
        return {"path": str(path), "relative_path": relative_text}

    def add(self, collection: Dict[str, dict], relative: PurePosixPath, item: dict) -> bool:
        key = relative.as_posix()
        if key in collection:
            collection[key].update(item)
            return True
        if self.discovered_entries >= self.max_entries:
            self.truncated = True
            self.stop = True
            return False
        collection[key] = item
        return True

    def add_dataset(self, relative: PurePosixPath) -> None:
        item = self.base_item(relative)
        item["manifest"] = _relative_child(relative, "manifest.csv")
        item["metadata"] = _relative_child(relative, "dataset.json")
        self.add(self.datasets, relative, item)

    def add_experiment(self, relative: PurePosixPath) -> None:
        item = self.base_item(relative)
        item["metadata"] = _relative_child(relative, "experiment.json")
        self.add(self.experiments, relative, item)

    def add_checkpoint(self, relative: PurePosixPath, experiment: PurePosixPath) -> None:
        item = self.base_item(relative)
        experiment_item = self.base_item(experiment)
        item["experiment"] = experiment_item["path"]
        item["experiment_relative_path"] = experiment_item["relative_path"]
        self.add(self.checkpoints, relative, item)

    def add_result(
        self,
        relative: PurePosixPath,
        *,
        manipulation: bool = False,
        reconstruction: bool = False,
    ) -> None:
        key = relative.as_posix()
        if key in self.results:
            item = self.results[key]
            item["has_manipulation"] = bool(item["has_manipulation"] or manipulation)
            item["has_reconstruction"] = bool(item["has_reconstruction"] or reconstruction)
            return
        item = self.base_item(relative)
        item["has_manipulation"] = manipulation
        item["has_reconstruction"] = reconstruction
        self.add(self.results, relative, item)


def scan_workspace(workspace_root: PathLike, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> dict:
    """Return a deterministic catalog of workflow artifacts below ``workspace_root``.

    ``max_entries`` bounds the total number of returned datasets, experiments,
    checkpoints, and result bundles.  Traversal is additionally bounded by
    :data:`MAX_DEPTH` and :data:`MAX_SCANNED_ENTRIES`.  ``scanned_entries`` in
    the result counts directory entries observed, including skipped entries;
    it is not the number of returned artifacts.  Every catalog item has an
    absolute ``path`` confined to the workspace and a ``relative_path`` for
    display or persistence.

    Symbolic links are never followed or returned.  On platforms supporting
    ``dir_fd`` and ``O_NOFOLLOW`` (including the supported Linux cluster),
    child directories are opened relative to their already-open parent so a
    concurrent symlink replacement cannot redirect traversal outside the
    workspace.
    """

    limit = _entry_limit(max_entries)
    root, root_metadata = _workspace_root(workspace_root)
    state = _ScanState(root=root, max_entries=limit)

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_descriptor = os.open(root, flags)
    except OSError as error:
        raise ValueError(f"Cannot safely open workspace root: {root}") from error
    try:
        _verify_workspace_identity(root, root_descriptor, root_metadata)
        _walk(root_descriptor, PurePosixPath(), 0, state, flags)
        _verify_workspace_identity(root, root_descriptor, root_metadata)
    finally:
        os.close(root_descriptor)

    return {
        "schema_version": "1.0",
        "workspace": str(root),
        "datasets": _sorted_values(state.datasets),
        "experiments": _sorted_values(state.experiments),
        "checkpoints": _sorted_values(state.checkpoints),
        "results": _sorted_values(state.results),
        "entry_count": state.discovered_entries,
        "scanned_entries": state.scanned_entries,
        "max_entries": limit,
        "max_depth": MAX_DEPTH,
        "truncated": state.truncated,
    }


def _walk(
    directory_descriptor: int,
    relative: PurePosixPath,
    depth: int,
    state: _ScanState,
    directory_flags: int,
) -> None:
    if state.stop:
        return
    entries = _directory_entries(directory_descriptor, state)
    regular_files = {entry.name for entry in entries if entry.is_file}
    directories = {entry.name for entry in entries if entry.is_directory}

    is_dataset = {"dataset.json", "manifest.csv"}.issubset(regular_files)
    if is_dataset:
        state.add_dataset(relative)
    if "experiment.json" in regular_files and not state.stop:
        state.add_experiment(relative)
    if "reconstruction" in directories and not state.stop:
        state.add_result(relative, reconstruction=True)
    is_manipulation_directory = (
        relative.name == "manipulations" and "manipulation.json" in regular_files
    )
    if is_manipulation_directory and not state.stop:
        state.add_result(relative.parent, manipulation=True)

    is_checkpoint_directory = relative.parts[-2:] == ("runs", "checkpoints")
    if is_checkpoint_directory and not state.stop:
        experiment = PurePosixPath(*relative.parts[:-2])
        for name in sorted(regular_files):
            if not name.endswith(".ckpt"):
                continue
            state.add_checkpoint(relative / name, experiment)
            if state.stop:
                break

    if state.stop or is_dataset or is_manipulation_directory or is_checkpoint_directory:
        return

    children = [entry for entry in entries if entry.is_directory]
    for entry in children:
        if state.stop:
            return
        if _skip_directory(entry.name) or entry.name == "reconstruction":
            continue
        if depth >= MAX_DEPTH:
            state.truncated = True
            continue
        try:
            child_descriptor = os.open(
                entry.name,
                directory_flags,
                dir_fd=directory_descriptor,
            )
        except OSError:
            # The entry may have disappeared or become a symlink after scandir.
            # Skipping is safe; marking the result incomplete is honest.
            state.truncated = True
            continue
        try:
            _walk(
                child_descriptor,
                relative / entry.name,
                depth + 1,
                state,
                directory_flags,
            )
        finally:
            os.close(child_descriptor)


def _directory_entries(directory_descriptor: int, state: _ScanState) -> List[_Entry]:
    entries: List[_Entry] = []
    try:
        with os.scandir(directory_descriptor) as iterator:
            for raw_entry in iterator:
                if state.scanned_entries >= MAX_SCANNED_ENTRIES:
                    state.truncated = True
                    state.stop = True
                    break
                state.scanned_entries += 1
                try:
                    if raw_entry.is_symlink():
                        continue
                    is_directory = raw_entry.is_dir(follow_symlinks=False)
                    is_file = raw_entry.is_file(follow_symlinks=False)
                except OSError:
                    state.truncated = True
                    continue
                if is_directory or is_file:
                    entries.append(
                        _Entry(
                            name=raw_entry.name,
                            is_directory=is_directory,
                            is_file=is_file,
                        )
                    )
    except OSError:
        state.truncated = True
        return []
    return sorted(entries, key=lambda item: item.name)


def _workspace_root(raw_root: PathLike) -> tuple[Path, os.stat_result]:
    supplied = Path(raw_root).expanduser()
    lexical = Path(os.path.abspath(os.fspath(supplied)))
    if lexical.parent == lexical:
        raise ValueError("Workspace root may not be a filesystem root")
    if not os.path.lexists(lexical):
        raise FileNotFoundError(f"Workspace root does not exist: {lexical}")
    _reject_symlink_components(lexical)
    try:
        metadata = os.stat(lexical, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Cannot inspect workspace root: {lexical}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Workspace root must be a directory: {lexical}")
    return lexical, metadata


def _verify_workspace_identity(root: Path, descriptor: int, expected: os.stat_result) -> None:
    """Reject a root path replaced between validation, traversal, and return."""

    _reject_symlink_components(root)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Workspace root changed during discovery: {root}") from error
    expected_identity = (expected.st_dev, expected.st_ino)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or expected_identity != (opened.st_dev, opened.st_ino)
        or expected_identity != (current.st_dev, current.st_ino)
    ):
        raise ValueError(f"Workspace root changed during discovery: {root}")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Workspace root contains a symbolic link: {current}")


def _entry_limit(raw_limit: int) -> int:
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or not 1 <= raw_limit <= MAX_ENTRIES
    ):
        raise ValueError(f"max_entries must be an integer between 1 and {MAX_ENTRIES}")
    return raw_limit


def _skip_directory(name: str) -> bool:
    return (
        name.startswith(".")
        or name.endswith(".egg-info")
        or name in _FROZEN_ENGINE_DIRECTORIES
        or name in _PAYLOAD_DIRECTORIES
    )


def _relative_child(relative: PurePosixPath, name: str) -> str:
    return (relative / name).as_posix()


def _sorted_values(items: Dict[str, dict]) -> List[dict]:
    return [items[key] for key in sorted(items)]


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "MAX_DEPTH",
    "MAX_ENTRIES",
    "MAX_SCANNED_ENTRIES",
    "scan_workspace",
]
