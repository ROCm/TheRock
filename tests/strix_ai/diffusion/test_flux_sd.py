"""
Flux/Stable Diffusion Model Tests for Strix

Tests for Flux 4B and Stable Diffusion 2B models with AWQ/FP16.
Market segments: Industrial, Healthcare
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


@pytest.mark.diffusion
@pytest.mark.industrial
@pytest.mark.healthcare
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_flux_4b_awq_load(check_strix_gpu):
    """Test: Load Flux 4B with AWQ quantization"""
    pytest.skip("Flux-4B-AWQ model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.diffusion
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.fp16
def test_sd_2b_fp16_load(check_strix_gpu):
    """Test: Load Stable Diffusion 2B with FP16"""
    pytest.skip("Stable-Diffusion-2B-FP16 model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.diffusion
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
def test_flux_4b_inference(check_strix_gpu):
    """Test: Run image generation with Flux 4B"""
    pytest.skip("Flux-4B model not yet available. "
                "Test will be enabled when model is published.")


@pytest.mark.diffusion
@pytest.mark.quick
@pytest.mark.p0
def test_diffusion_quick_smoke(check_strix_gpu):
    """Quick smoke test for diffusion models"""
    pytest.skip("Diffusion models not yet configured for Strix. "
                "Test will be enabled when models are available.")
