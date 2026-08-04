from pathlib import Path

from tools import engine_checksums


def _write_manifest(root: Path, manifest: Path) -> None:
    lines = [
        f"{engine_checksums.digest(path)}  {path.relative_to(root).as_posix()}"
        for path in engine_checksums.engine_files(root)
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_rejects_untracked_files_in_frozen_roots(tmp_path, capsys):
    source = tmp_path / "models" / "model.py"
    source.parent.mkdir()
    source.write_text("baseline\n", encoding="utf-8")
    manifest = tmp_path / "ENGINE_FILES.sha256"
    _write_manifest(tmp_path, manifest)

    assert engine_checksums.verify(tmp_path, manifest) == 0
    (tmp_path / "models" / "unregistered.py").write_text("new\n", encoding="utf-8")

    assert engine_checksums.verify(tmp_path, manifest) == 1
    assert "untracked: models/unregistered.py" in capsys.readouterr().err


def test_verify_rejects_changed_and_malformed_entries(tmp_path, capsys):
    source = tmp_path / "autoencode.py"
    source.write_text("baseline\n", encoding="utf-8")
    manifest = tmp_path / "ENGINE_FILES.sha256"
    _write_manifest(tmp_path, manifest)
    source.write_text("changed\n", encoding="utf-8")

    assert engine_checksums.verify(tmp_path, manifest) == 1
    assert "changed: autoencode.py" in capsys.readouterr().err

    manifest.write_text("not-a-checksum\n", encoding="utf-8")
    assert engine_checksums.verify(tmp_path, manifest) == 1
    assert "malformed manifest line" in capsys.readouterr().err
