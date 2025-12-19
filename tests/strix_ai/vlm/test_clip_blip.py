"""
CLIP/BLIP Vision-Language Model Tests for Strix

Tests for CLIP and BLIP 0.5B models with AWQ quantization.
Market segments: Automotive, Industrial
"""

import pytest
import torch
from PIL import Image


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


@pytest.fixture(scope="module")
def sample_image():
    """Create a sample test image"""
    img = Image.new('RGB', (224, 224), color='white')
    return img


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_clip_05b_awq_load(check_strix_gpu):
    """Test: Load CLIP 0.5B with AWQ quantization"""
    pytest.skip("CLIP-0.5B-AWQ model not yet available. "
                "Test will be enabled when quantized model is published.")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
def test_blip_05b_awq_load(check_strix_gpu):
    """Test: Load BLIP 0.5B with AWQ quantization"""
    pytest.skip("BLIP-0.5B-AWQ model not yet available. "
                "Test will be enabled when quantized model is published.")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.quick
@pytest.mark.p0
def test_clip_blip_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for CLIP/BLIP models"""
    pytest.skip("CLIP/BLIP quantized models not yet available. "
                "Test will be enabled when models are published.")
