"""
GPTQ Quantization Tests for Strix

Tests for GPTQ quantization on Strix GPUs.
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


@pytest.mark.optimization
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.healthcare
@pytest.mark.p0
@pytest.mark.gptq
def test_gptq_quantization_basic(check_strix_gpu):
    """Test: Basic GPTQ quantization functionality"""
    pytest.skip("GPTQ quantization workflow not yet implemented for Strix. "
                "Test will be enabled when quantization tools are configured.")


@pytest.mark.optimization
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.gptq
@pytest.mark.performance
def test_gptq_quantization_performance(check_strix_gpu):
    """Test: GPTQ quantization performance gains"""
    pytest.skip("GPTQ quantization workflow not yet implemented for Strix. "
                "Test will be enabled when quantization tools are configured.")


@pytest.mark.optimization
@pytest.mark.quick
@pytest.mark.p0
def test_gptq_quick_smoke(check_strix_gpu):
    """Quick smoke test for GPTQ quantization"""
    pytest.skip("GPTQ quantization not yet configured for Strix. "
                "Test will be enabled when tools are available.")
