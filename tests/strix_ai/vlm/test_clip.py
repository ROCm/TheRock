"""
CLIP Model Tests for Strix

Tests for CLIP models on Strix GPUs.
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
def test_clip_load(check_strix_gpu):
    """Test: Load CLIP model"""
    pytest.skip("CLIP model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
def test_clip_inference(check_strix_gpu, sample_image):
    """Test: Run CLIP inference"""
    pytest.skip("CLIP model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.vlm
@pytest.mark.quick
@pytest.mark.p0
def test_clip_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for CLIP"""
    pytest.skip("CLIP model not yet configured for Strix. "
                "Test will be enabled when model is available.")
