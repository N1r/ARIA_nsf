"""Automatic CUDA environment discovery and configuration.

Ensures that Numba CUDA JIT, TorchLPC, and PyTorch find the necessary
CUDA libraries (libcudart.so, libnvvm.so, and libdevice.10.bc) without
manual intervention, whether running locally, on SLURM clusters, or on Google Colab.
"""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_cuda_home() -> Optional[Path]:
    """Search for the most suitable CUDA Toolkit installation directory."""
    # 1. Respect existing environment variables
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env_var)
        if val:
            path = Path(val).resolve()
            if path.exists():
                return path

    # 2. Check nvcc binary in PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        if nvcc_path.name == "nvcc" and nvcc_path.parent.name == "bin":
            candidate = nvcc_path.parents[1]
            if candidate.exists():
                return candidate

    # 3. Known standard install locations (ordered by preference)
    well_known_locations: List[Path] = [
        Path("/usr/local/cuda"),  # Google Colab & standard Linux symlink
    ]

    # Check versioned installs in /usr/local
    usr_local = Path("/usr/local")
    if usr_local.exists():
        for item in sorted(usr_local.glob("cuda-12*"), reverse=True):
            if item.is_dir():
                well_known_locations.append(item)
        for item in sorted(usr_local.glob("cuda-11*"), reverse=True):
            if item.is_dir():
                well_known_locations.append(item)

    # Check EasyBuild / cluster modules
    easybuild_cuda = Path("/easybuild/software/CUDA")
    if easybuild_cuda.exists():
        for item in sorted(easybuild_cuda.glob("12*"), reverse=True):
            if item.is_dir():
                well_known_locations.append(item)
        for item in sorted(easybuild_cuda.glob("11*"), reverse=True):
            if item.is_dir():
                well_known_locations.append(item)

    well_known_locations.append(Path("/opt/cuda"))

    for loc in well_known_locations:
        if loc.exists() and (loc / "lib64").exists():
            return loc.resolve()

    return None


def auto_configure_cuda() -> Dict[str, Any]:
    """Configure environment variables and preload libraries for Numba/TorchLPC.

    Returns a status dict detailing what was configured.
    """
    cuda_dir = find_cuda_home()
    status: Dict[str, Any] = {
        "cuda_home": str(cuda_dir) if cuda_dir else None,
        "libdevice": None,
        "libs_preloaded": [],
    }

    if not cuda_dir:
        return status

    # Set CUDA_HOME if not set
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = str(cuda_dir)

    # Locate libdevice
    libdevice_dir = cuda_dir / "nvvm" / "libdevice"
    if libdevice_dir.exists():
        status["libdevice"] = str(libdevice_dir)
        if "NUMBA_CUDA_LIBDEVICE" not in os.environ:
            os.environ["NUMBA_CUDA_LIBDEVICE"] = str(libdevice_dir)

    # Prepend lib paths to LD_LIBRARY_PATH
    new_lib_paths: List[str] = []
    for sub in ("lib64", "nvvm/lib64", "lib"):
        p = cuda_dir / sub
        if p.exists():
            new_lib_paths.append(str(p))

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [x for x in current_ld.split(":") if x]
    to_add = [p for p in new_lib_paths if p not in existing]
    if to_add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(to_add + existing)

    # Preload libcudart and libnvvm so numba's ctypes.CDLL resolution succeeds immediately
    for sub in ("lib64", "lib"):
        p = cuda_dir / sub / "libcudart.so"
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                status["libs_preloaded"].append(str(p))
                break
            except Exception:
                pass

    for sub in ("nvvm/lib64", "lib64", "lib"):
        p = cuda_dir / sub / "libnvvm.so"
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                status["libs_preloaded"].append(str(p))
                break
            except Exception:
                pass

    return status


def get_cuda_info() -> Dict[str, Any]:
    """Inspect and return current CUDA device & driver information."""
    info: Dict[str, Any] = {
        "available": False,
        "device_name": None,
        "device_count": 0,
        "compute_capability": None,
        "memory_total_gb": None,
        "cuda_version": None,
        "cuda_home": os.environ.get("CUDA_HOME"),
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
            props = torch.cuda.get_device_properties(0)
            info["memory_total_gb"] = round(props.total_memory / (1024**3), 1)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    return info
