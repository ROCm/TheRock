"""
Pi0 Vision-Language-Action Model Tests for Strix

Tests for Pi0 0.5B models.
Market segment: Robotics
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


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.functional
def test_pi0_05b_load(check_strix_gpu):
    """Test: Load Pi0 0.5B model"""
    pytest.skip("Pi0-0.5B model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.quick
@pytest.mark.p0
def test_pi0_quick_smoke(check_strix_gpu):
    """Quick smoke test for Pi0"""
    pytest.skip("Pi0 model not yet configured for Strix. "
                "Test will be enabled when model is available.")
