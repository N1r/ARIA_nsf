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
    from .cuda_env import auto_configure_cuda, get_cuda_info

    auto_configure_cuda()
    cuda_info = get_cuda_info()

    result = [
        Check("Python", sys.version_info >= (3, 9), platform.python_version(), "all commands"),
        _module("numpy", "data preparation"),
        _module("soundfile", "non-WAV input and training"),
        _module("scipy", "measurement-grade resampling"),
        _module("pyworld", "recommended F0 extraction"),
        _module("torch", "training and inference"),
        _module("lightning", "training"),
        _module("gradio", "interactive Studio web workspace"),
    ]

    if cuda_info["available"]:
        gpu_detail = f"{cuda_info['device_name']} ({cuda_info['memory_total_gb']} GB VRAM, sm_{cuda_info['compute_capability']}, CUDA {cuda_info['cuda_version']})"
        result.append(Check("CUDA GPU", True, gpu_detail, "fast neural vocoder training and synthesis"))
    else:
        result.append(
            Check(
                "CUDA GPU",
                False,
                "No CUDA GPU detected (CPU mode available for synthesis/manipulation; training will use CPU)",
                "fast GPU acceleration",
            )
        )

    ffmpeg = shutil.which("ffmpeg")
    result.append(
        Check("ffmpeg", bool(ffmpeg), ffmpeg or "not found (optional)", "audio conversion")
    )
    return result


def _module(name: str, purpose: str) -> Check:
    found = importlib.util.find_spec(name) is not None
    detail = "installed" if found else "missing"
    if not found:
        if name in {"soundfile", "scipy", "pyworld"}:
            extra = "world" if name == "pyworld" else "audio"
            detail += f"; install with: pip install 'aris[{extra}]' (or: pip install 'aris[all]')"
        elif name in {"torch", "lightning"}:
            detail += "; install with: pip install 'aris[train]' (or: pip install 'aris[all]')"
        elif name == "gradio":
            detail += "; install with: pip install 'aris[studio]' (or: pip install 'aris[all]')"
        else:
            detail += "; install with: pip install 'aris[all]'"
    return Check(name, found, detail, purpose)


def as_json() -> str:
    """Return all checks as a JSON string."""
    return json.dumps([asdict(item) for item in checks()], indent=2)
