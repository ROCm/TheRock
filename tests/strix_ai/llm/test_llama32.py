"""
Llama 3.2 LLM Tests for Strix

Tests for Llama 3.2 3B models with AWQ quantization.
Market segments: Industrial, Robotics, Healthcare
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


@pytest.mark.llm
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.healthcare
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_llama32_3b_awq_load(check_strix_gpu):
    """Test: Load Llama 3.2 3B with AWQ quantization"""
    pytest.skip("Llama-3.2-3B-AWQ model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.llm
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_llama32_3b_awq_inference(check_strix_gpu):
    """Test: Run inference with Llama 3.2 3B AWQ"""
    pytest.skip("Llama-3.2-3B-AWQ model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.llm
@pytest.mark.quick
@pytest.mark.p0
def test_llama32_quick_smoke(check_strix_gpu):
    """Quick smoke test for Llama 3.2"""
    pytest.skip("Llama 3.2 model not yet available. "
                "Test will be enabled when model is published.")
