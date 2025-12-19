"""
OpenVLA Vision-Language-Action Model Tests for Strix

Tests for OpenVLA 7B models with GPTQ/AWQ quantization.
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
@pytest.mark.gptq
def test_openvla_7b_gptq_load(check_strix_gpu):
    """Test: Load OpenVLA 7B with GPTQ quantization"""
    pytest.skip("OpenVLA-7B-GPTQ model not yet configured for Strix. "
                "Test will be enabled when model is optimized and published.")


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_openvla_7b_awq_load(check_strix_gpu):
    """Test: Load OpenVLA 7B with AWQ quantization"""
    pytest.skip("OpenVLA-7B-AWQ model not yet configured for Strix. "
                "Test will be enabled when model is optimized and published.")


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.quick
@pytest.mark.p0
def test_openvla_quick_smoke(check_strix_gpu):
    """Quick smoke test for OpenVLA"""
    pytest.skip("OpenVLA models not yet configured for Strix. "
                "Test will be enabled when models are available.")
