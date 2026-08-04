#!/usr/bin/env python3
"""Write or verify checksums for the imported frozen research engine."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ENGINE_ROOTS = ("models", "ltng", "loss", "datasets", "cfg", "configs", "scripts")
ENGINE_FILES = ("autoencode.py", "main.py", "biquads.py", "harm_and_noise.py")


def engine_files(root: Path):
    files = [root / name for name in ENGINE_FILES]
    for directory in ENGINE_ROOTS:
        files.extend(path for path in (root / directory).rglob("*") if path.is_file())
    return sorted(path for path in files if path.is_file() and "__pycache__" not in path.parts)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(root: Path, manifest: Path):
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in engine_files(root)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {manifest}")


def verify(root: Path, manifest: Path) -> int:
    failures = []
    checked = 0
    recorded = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed manifest line: {line!r}")
            continue
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not relative
        ):
            failures.append(f"malformed manifest line: {line!r}")
            continue
        if relative in recorded:
            failures.append(f"duplicate: {relative}")
            continue
        recorded.add(relative)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"changed: {relative}")
        checked += 1
    current = {path.relative_to(root).as_posix() for path in engine_files(root)}
    failures.extend(f"untracked: {relative}" for relative in sorted(current - recorded))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"OK: {checked} imported engine files match {manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="replace the checksum manifest")
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("provenance/ENGINE_FILES.sha256"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest.resolve()
    if args.write:
        write(root, manifest)
        return 0
    return verify(root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
