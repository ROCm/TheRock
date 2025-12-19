"""
YOLO Model Tests for Strix

Tests for YOLO models on Strix GPUs.
Market segments: Automotive, Industrial, Robotics
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
    img = Image.new('RGB', (640, 640), color='white')
    return img


@pytest.mark.cv
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.functional
def test_yolo_load(check_strix_gpu):
    """Test: Load YOLO model"""
    pytest.skip("YOLO model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.cv
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
def test_yolo_inference(check_strix_gpu, sample_image):
    """Test: Run YOLO inference"""
    pytest.skip("YOLO model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.cv
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
def test_yolo_latency(check_strix_gpu, sample_image):
    """Test: Measure YOLO latency"""
    pytest.skip("YOLO model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.cv
@pytest.mark.quick
@pytest.mark.p0
def test_yolo_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for YOLO"""
    pytest.skip("YOLO model not yet configured for Strix. "
                "Test will be enabled when model is available.")
