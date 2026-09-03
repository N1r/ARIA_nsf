"""Test top-level ARIS Python API exports and CUDA environment auto-discovery."""

import aris
from aris.cuda_env import get_cuda_info


def test_top_level_api_exports():
    """Verify that high-level functions are directly accessible from the aris module."""
    assert callable(aris.doctor)
    assert callable(aris.split)
    assert callable(aris.prepare)
    assert callable(aris.validate)
    assert callable(aris.init_experiment)
    assert callable(aris.train)
    assert callable(aris.synthesize)
    assert callable(aris.manipulate)
    assert callable(aris.launch_studio)
    assert callable(aris.auto_configure_cuda)
    assert callable(aris.get_cuda_info)


def test_doctor_check():
    """Verify that aris.doctor returns structured check results."""
    findings = aris.doctor()
    assert isinstance(findings, list)
    assert any(c.name == "Python" and c.ok for c in findings)
    assert any(c.name == "CUDA GPU" for c in findings)


def test_cuda_env_helpers():
    """Verify get_cuda_info returns expected keys."""
    info = get_cuda_info()
    assert "available" in info
    assert "device_name" in info
    assert "device_count" in info
    assert "compute_capability" in info
