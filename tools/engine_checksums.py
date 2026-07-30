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
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"changed: {relative}")
        checked += 1
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
