from pathlib import Path

from tools import repo_audit


def _write_minimal_repository(root: Path) -> None:
    required_files = (
        "README.md",
        "README_EN.md",
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "pyproject.toml",
        "src/phonlab_ddsp/__init__.py",
        "docs/ARCHITECTURE.md",
        "docs/QUICKSTART_ZH.md",
        "docs/DATA_AND_REPRODUCIBILITY.md",
        "docs/MANIPULATION_ZH.md",
        "docs/WEBUI_ZH.md",
        "docs/REPOSITORY_MAP_ZH.md",
        "docs/GITHUB_RELEASE_ZH.md",
        ".github/workflows/test.yml",
    )
    for relative in required_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")


def test_clean_repository_passes_all_categories(tmp_path, capsys):
    _write_minimal_repository(tmp_path)

    report = repo_audit.audit_repository(tmp_path)

    assert report.ok
    assert {check.category for check in report.checks} == {
        "navigation",
        "package",
        "docs",
        "ci",
        "generated",
    }
    assert repo_audit.main([str(tmp_path), "--strict"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("REPO_AUDIT_OK")


def test_missing_navigation_and_ci_are_reported(tmp_path):
    _write_minimal_repository(tmp_path)
    (tmp_path / "README.md").unlink()
    (tmp_path / ".github/workflows/test.yml").unlink()

    report = repo_audit.audit_repository(tmp_path)

    failures = {(check.category, check.target) for check in report.failures}
    assert ("navigation", "README.md") in failures
    assert ("ci", ".github/workflows/test.yml") in failures
    assert repo_audit.main([str(tmp_path)]) == 0
    assert repo_audit.main([str(tmp_path), "--strict"]) == 1


def test_forbidden_generated_top_level_entries_are_physical_checks(tmp_path):
    _write_minimal_repository(tmp_path)
    for directory in ("build", "dist", "__pycache__", "phonlab_ddsp.egg-info"):
        (tmp_path / directory).mkdir()
    for filename in ("checkpoint.ckpt", "weights.pt"):
        (tmp_path / filename).touch()

    found = {
        path.relative_to(tmp_path).as_posix()
        for path in repo_audit.find_forbidden_top_level(tmp_path)
    }

    assert found == {
        "__pycache__",
        "build",
        "checkpoint.ckpt",
        "dist",
        "phonlab_ddsp.egg-info",
        "weights.pt",
    }
    assert not repo_audit.audit_repository(tmp_path).ok


def test_intentional_local_trees_and_nested_outputs_are_allowed(tmp_path):
    _write_minimal_repository(tmp_path)
    for directory in (".cache", ".venv", "artifacts"):
        (tmp_path / directory).mkdir()
    nested = tmp_path / "artifacts" / "dist"
    nested.mkdir()
    (nested / "model.ckpt").touch()

    report = repo_audit.audit_repository(tmp_path)

    assert report.ok


def test_json_report_is_machine_readable(tmp_path, capsys):
    _write_minimal_repository(tmp_path)

    assert repo_audit.main([str(tmp_path), "--strict", "--json"]) == 0

    output = capsys.readouterr().out
    assert '"failure_count": 0' in output
    assert '"ok": true' in output
