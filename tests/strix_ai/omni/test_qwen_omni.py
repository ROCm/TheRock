"""
Qwen Omni Multimodal Model Tests for Strix

Tests for Qwen2.5-Omni and Qwen3-Omni 7B models with AWQ quantization.
Market segment: Automotive
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


@pytest.mark.omni
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen25_omni_7b_awq_load(check_strix_gpu):
    """Test: Load Qwen2.5-Omni 7B with AWQ quantization"""
    pytest.skip("Qwen2.5-Omni-7B-AWQ model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.omni
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen3_omni_7b_awq_load(check_strix_gpu):
    """Test: Load Qwen3-Omni 7B with AWQ quantization"""
    pytest.skip("Qwen3-Omni-7B-AWQ model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.omni
@pytest.mark.automotive
@pytest.mark.quick
@pytest.mark.p0
def test_qwen_omni_quick_smoke(check_strix_gpu):
    """Quick smoke test for Qwen Omni models"""
    pytest.skip("Qwen Omni models not yet available. "
                "Test will be enabled when models are published.")
