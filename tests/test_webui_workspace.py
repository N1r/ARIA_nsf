import json
from pathlib import Path

import pytest

from phonlab_ddsp.webui_workspace import (
    MAX_DEPTH,
    MAX_ENTRIES,
    MAX_SCANNED_ENTRIES,
    scan_workspace,
)

KINDS = ("datasets", "experiments", "checkpoints", "results")


def _touch(path: Path, content: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _dataset(path: Path) -> None:
    # Discovery deliberately does not parse metadata; malformed or old metadata
    # can still be listed and validated by the workflow action that opens it.
    _touch(path / "dataset.json", b"\xff")
    _touch(path / "manifest.csv", b"id,split\n")


def _experiment(path: Path, checkpoints=()) -> None:
    _touch(path / "experiment.json")
    for name in checkpoints:
        _touch(path / "runs" / "checkpoints" / name, b"checkpoint")


def _relative_paths(catalog: dict, kind: str) -> list[str]:
    return [item["relative_path"] for item in catalog[kind]]


def test_discovers_workflow_sentinels_and_control_postprocess_layout(tmp_path):
    workspace = tmp_path / "workspace"
    dataset = workspace / "artifacts" / "demo" / "dataset"
    experiment = workspace / "artifacts" / "demo" / "experiment"
    result = workspace / "artifacts" / "demo" / "control_postprocess_v2"
    reconstruction_only = workspace / "artifacts" / "demo" / "baseline-output"
    _dataset(dataset)
    _experiment(experiment, ("last.ckpt", "epoch=1.ckpt", "notes.txt"))
    _touch(result / "manipulations" / "manipulation.json")
    (result / "reconstruction").mkdir(parents=True)
    (reconstruction_only / "reconstruction").mkdir(parents=True)

    catalog = scan_workspace(workspace)

    assert _relative_paths(catalog, "datasets") == ["artifacts/demo/dataset"]
    assert _relative_paths(catalog, "experiments") == ["artifacts/demo/experiment"]
    assert _relative_paths(catalog, "checkpoints") == [
        "artifacts/demo/experiment/runs/checkpoints/epoch=1.ckpt",
        "artifacts/demo/experiment/runs/checkpoints/last.ckpt",
    ]
    assert _relative_paths(catalog, "results") == [
        "artifacts/demo/baseline-output",
        "artifacts/demo/control_postprocess_v2",
    ]
    control_result = catalog["results"][1]
    assert control_result["has_manipulation"] is True
    assert control_result["has_reconstruction"] is True
    assert Path(control_result["path"]) == result.resolve()
    assert catalog["entry_count"] == 6
    assert catalog["truncated"] is False
    json.dumps(catalog, sort_keys=True)


def test_never_follows_symlinks_or_scans_pruned_trees(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    _experiment(outside / "external-experiment", ("secret.ckpt",))
    _touch(outside / "dataset.json")
    _touch(outside / "secret.ckpt", b"secret")
    workspace.mkdir()

    try:
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
        fake_dataset = workspace / "fake-dataset"
        fake_dataset.mkdir()
        _touch(fake_dataset / "manifest.csv")
        (fake_dataset / "dataset.json").symlink_to(outside / "dataset.json")
        checkpoint_directory = workspace / "safe" / "runs" / "checkpoints"
        checkpoint_directory.mkdir(parents=True)
        (checkpoint_directory / "outside.ckpt").symlink_to(outside / "secret.ckpt")
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    for directory in (
        ".git",
        ".venv",
        ".cache",
        "audio",
        "cfg",
        "configs",
        "datasets",
        "loss",
        "ltng",
        "models",
        "raw",
        "raw-data",
        "scripts",
    ):
        _experiment(workspace / directory / "hidden-experiment")
    _experiment(workspace / "visible-experiment")

    catalog = scan_workspace(workspace)
    serialized = json.dumps(catalog, sort_keys=True)

    assert _relative_paths(catalog, "datasets") == []
    assert _relative_paths(catalog, "experiments") == ["visible-experiment"]
    assert _relative_paths(catalog, "checkpoints") == []
    assert str(outside) not in serialized
    root = workspace.resolve()
    for kind in KINDS:
        for item in catalog[kind]:
            discovered = Path(item["path"])
            assert discovered == root or root in discovered.parents
            assert not discovered.is_symlink()

    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        scan_workspace(workspace_link)

    real_parent = tmp_path / "real-parent"
    nested_workspace = real_parent / "nested"
    nested_workspace.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        scan_workspace(linked_parent / "nested")


def test_catalog_is_deterministic_and_sorted(tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("zulu", "alpha", "middle"):
        base = workspace / "artifacts" / name
        _dataset(base / "dataset")
        _experiment(base / "experiment", ("last.ckpt", "epoch=2.ckpt"))
        _touch(base / "postprocess" / "manipulations" / "manipulation.json")

    first = scan_workspace(workspace)
    second = scan_workspace(workspace)

    assert first == second
    for kind in KINDS:
        paths = _relative_paths(first, kind)
        assert paths == sorted(paths)


def test_entry_and_depth_limits_are_enforced(tmp_path):
    workspace = tmp_path / "limited-workspace"
    for index in reversed(range(8)):
        _dataset(workspace / "artifacts" / f"item-{index:02d}" / "dataset")

    catalog = scan_workspace(workspace, max_entries=3)

    assert catalog["entry_count"] == 3
    assert sum(len(catalog[kind]) for kind in KINDS) == 3
    assert _relative_paths(catalog, "datasets") == [
        "artifacts/item-00/dataset",
        "artifacts/item-01/dataset",
        "artifacts/item-02/dataset",
    ]
    assert catalog["truncated"] is True
    assert catalog["scanned_entries"] <= MAX_SCANNED_ENTRIES

    depth_workspace = tmp_path / "depth-workspace"
    too_deep = depth_workspace
    for index in range(MAX_DEPTH + 1):
        too_deep /= f"level-{index}"
    _experiment(too_deep)

    depth_catalog = scan_workspace(depth_workspace)

    assert depth_catalog["experiments"] == []
    assert depth_catalog["truncated"] is True


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2", MAX_ENTRIES + 1])
def test_rejects_invalid_entry_limits(tmp_path, invalid):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="max_entries"):
        scan_workspace(workspace, max_entries=invalid)
