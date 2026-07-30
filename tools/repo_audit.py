#!/usr/bin/env python3
"""Report whether the repository satisfies its lightweight governance contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

NAVIGATION_FILES = (
    ("README.md", "project overview"),
    ("LICENSE", "license"),
    ("pyproject.toml", "Python project metadata"),
)
PACKAGE_FILES = (("src/phonlab_ddsp/__init__.py", "public Python package"),)
DOCUMENTATION_FILES = (
    ("docs/ARCHITECTURE.md", "architecture guide"),
    ("docs/QUICKSTART_ZH.md", "quick-start guide"),
    ("docs/DATA_AND_REPRODUCIBILITY.md", "data and reproducibility guide"),
)
CI_FILES = ((".github/workflows/test.yml", "continuous-integration workflow"),)

FORBIDDEN_DIRECTORY_NAMES = frozenset({"__pycache__", "build", "dist"})
FORBIDDEN_DIRECTORY_SUFFIXES = (".egg-info",)
FORBIDDEN_FILE_SUFFIXES = (".ckpt", ".pt")


@dataclass(frozen=True)
class AuditCheck:
    """One mechanically verifiable repository requirement."""

    category: str
    target: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """All audit checks for one repository root."""

    root: Path
    checks: tuple[AuditCheck, ...]

    @property
    def failures(self) -> tuple[AuditCheck, ...]:
        return tuple(check for check in self.checks if not check.ok)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
            "failure_count": len(self.failures),
        }


def _required_file_checks(
    root: Path,
    category: str,
    requirements: Sequence[tuple[str, str]],
) -> list[AuditCheck]:
    checks = []
    for relative, description in requirements:
        exists = (root / relative).is_file()
        detail = description if exists else f"missing {description}"
        checks.append(AuditCheck(category, relative, exists, detail))
    return checks


def find_forbidden_top_level(root: Path) -> tuple[Path, ...]:
    """Return forbidden generated entries physically present below ``root``."""

    forbidden = []
    if not root.is_dir():
        return ()

    for path in root.iterdir():
        name = path.name
        lowered = name.lower()
        if path.is_dir() and (
            name in FORBIDDEN_DIRECTORY_NAMES or lowered.endswith(FORBIDDEN_DIRECTORY_SUFFIXES)
        ):
            forbidden.append(path)
        elif path.is_file() and lowered.endswith(FORBIDDEN_FILE_SUFFIXES):
            forbidden.append(path)
    return tuple(sorted(forbidden, key=lambda path: path.name.casefold()))


def audit_repository(root: Path) -> AuditReport:
    """Audit the navigation, package, docs, CI, and generated-file policy."""

    root = root.expanduser().resolve()
    checks = []
    checks.extend(_required_file_checks(root, "navigation", NAVIGATION_FILES))
    checks.extend(_required_file_checks(root, "package", PACKAGE_FILES))
    checks.extend(_required_file_checks(root, "docs", DOCUMENTATION_FILES))
    checks.extend(_required_file_checks(root, "ci", CI_FILES))

    forbidden = find_forbidden_top_level(root)
    detail = (
        "no forbidden generated top-level entries"
        if not forbidden
        else "forbidden generated entries: "
        + ", ".join(path.relative_to(root).as_posix() for path in forbidden)
    )
    checks.append(AuditCheck("generated", "top-level", not forbidden, detail))
    return AuditReport(root, tuple(checks))


def render_text(report: AuditReport) -> str:
    """Render a stable, human-readable audit report."""

    lines = [f"Repository audit: {report.root}"]
    for check in report.checks:
        status = "OK" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.category}: {check.target} - {check.detail}")
    passed = len(report.checks) - len(report.failures)
    lines.append(f"Summary: {passed} passed, {len(report.failures)} failed")
    lines.append("REPO_AUDIT_OK" if report.ok else "REPO_AUDIT_FAILED")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report repository navigation and generated-file policy checks."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any check fails",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_repository(args.root)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 1 if args.strict and not report.ok else 0


if __name__ == "__main__":
    sys.exit(main())
