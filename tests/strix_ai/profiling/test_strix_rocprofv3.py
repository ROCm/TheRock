"""
ROCProfiler v3 Profiling Tests for Strix

Tests for ROCProfiler v3 on Strix GPUs.
Market segments: All
"""

import pytest
import torch
import os
import subprocess


@pytest.fixture(scope="module")
def check_strix_gpu():
    """Verify Strix GPU is available"""
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    if "gfx115" not in amdgpu_family:
        pytest.skip(f"Strix GPU not detected. AMDGPU_FAMILIES={amdgpu_family}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    
    return torch.cuda.get_device_name(0)


@pytest.fixture(scope="module")
def check_rocprofv3():
    """Check if rocprofv3 is available"""
    try:
        result = subprocess.run(
            ["rocprofv3", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    pytest.skip("rocprofv3 not available. Install ROCProfiler SDK to enable profiling tests.")


@pytest.mark.profiling
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.healthcare
@pytest.mark.p1
def test_rocprofv3_basic_profiling(check_strix_gpu, check_rocprofv3):
    """Test: Basic ROCProfiler v3 profiling on Strix"""
    pytest.skip("ROCProfiler v3 profiling workflow not yet configured for Strix. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.automotive
@pytest.mark.p1
def test_rocprofv3_kernel_tracing(check_strix_gpu, check_rocprofv3):
    """Test: Kernel tracing with ROCProfiler v3"""
    pytest.skip("ROCProfiler v3 kernel tracing not yet configured for Strix. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.automotive
@pytest.mark.p1
def test_rocprofv3_perfetto_export(check_strix_gpu, check_rocprofv3):
    """Test: Export Perfetto traces with ROCProfiler v3"""
    pytest.skip("Perfetto trace export not yet configured for Strix. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.quick
@pytest.mark.p1
def test_rocprofv3_quick_smoke(check_strix_gpu):
    """Quick smoke test for ROCProfiler v3"""
    pytest.skip("ROCProfiler v3 not yet configured for Strix testing. "
                "Test will be enabled when profiling tools are available.")
