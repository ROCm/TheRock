"""
Profile Existing Tests for Strix

Tests for profiling existing test suite with ROCProfiler v3.
Market segments: All
"""

import pytest
import torch


@pytest.fixture(scope="module")
def check_strix_gpu():
    """Verify Strix GPU is available"""
    import os
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    if "gfx115" not in amdgpu_family:
        pytest.skip(f"Strix GPU not detected. AMDGPU_FAMILIES={amdgpu_family}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    
    return torch.cuda.get_device_name(0)


@pytest.mark.profiling
@pytest.mark.vlm
@pytest.mark.p1
def test_profile_existing_vlm_tests(check_strix_gpu):
    """Test: Profile existing VLM tests"""
    pytest.skip("VLM test profiling not yet implemented. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.vla
@pytest.mark.p1
def test_profile_existing_vla_tests(check_strix_gpu):
    """Test: Profile existing VLA tests"""
    pytest.skip("VLA test profiling not yet implemented. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.instruct
@pytest.mark.p1
def test_profile_existing_instruct_tests(check_strix_gpu):
    """Test: Profile existing Instruct tests"""
    pytest.skip("Instruct test profiling not yet implemented. "
                "Test will be enabled when profiling infrastructure is ready.")


@pytest.mark.profiling
@pytest.mark.quick
@pytest.mark.p1
def test_profile_existing_tests_quick_smoke(check_strix_gpu):
    """Quick smoke test for profiling existing tests"""
    pytest.skip("Test profiling infrastructure not yet configured. "
                "Test will be enabled when profiling tools are available.")
