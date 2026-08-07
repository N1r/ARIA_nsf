"""Environment diagnostics with actionable remediation."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass


@dataclass
class Check:
    """Result of one dependency check."""

    name: str
    ok: bool
    detail: str
    required_for: str


def checks() -> list[Check]:
    """Probe optional dependencies and report what each one enables."""
    result = [
        Check("Python", sys.version_info >= (3, 9), platform.python_version(), "all commands"),
        _module("numpy", "data preparation"),
        _module("soundfile", "non-WAV input and training"),
        _module("scipy", "measurement-grade resampling"),
        _module("pyworld", "recommended F0 extraction"),
        _module("torch", "training"),
        _module("lightning", "training"),
    ]
    ffmpeg = shutil.which("ffmpeg")
    result.append(
        Check("ffmpeg", bool(ffmpeg), ffmpeg or "not found (optional)", "audio conversion")
    )
    return result


def _module(name: str, purpose: str) -> Check:
    found = importlib.util.find_spec(name) is not None
    detail = "installed" if found else "missing"
    if name in {"soundfile", "scipy", "pyworld"} and not found:
        extra = "world" if name == "pyworld" else "audio"
        detail += f"; install with: uv sync --extra {extra} (or: pip install 'aris[{extra}]')"
    if name in {"torch", "lightning"} and not found:
        detail += "; install with: uv sync --extra train (or: pip install 'aris[train]')"
    return Check(name, found, detail, purpose)


def as_json() -> str:
    """Return all checks as a JSON string."""
    return json.dumps([asdict(item) for item in checks()], indent=2)
