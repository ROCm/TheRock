"""
Crossformer ASR Model Tests for Strix

Tests for crossformer <0.3B models.
Market segments: Automotive, Industrial
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


@pytest.mark.asr
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
def test_crossformer_load(check_strix_gpu):
    """Test: Load crossformer ASR model"""
    pytest.skip("crossformer model not yet configured for Strix. "
                "Test will be enabled when model is available via icefall.")


@pytest.mark.asr
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
def test_crossformer_inference(check_strix_gpu):
    """Test: Run ASR inference with crossformer"""
    pytest.skip("crossformer model not yet configured for Strix. "
                "Test will be enabled when model is available via icefall.")


@pytest.mark.asr
@pytest.mark.quick
@pytest.mark.p0
def test_crossformer_quick_smoke(check_strix_gpu):
    """Quick smoke test for crossformer"""
    pytest.skip("crossformer model not yet configured for Strix. "
                "Test will be enabled when model is available.")
