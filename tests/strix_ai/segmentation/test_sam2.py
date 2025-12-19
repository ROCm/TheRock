"""
SAM2 Segmentation Model Tests for Strix

Tests for SAM2 0.2B models.
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


@pytest.mark.segmentation
@pytest.mark.industrial
@pytest.mark.healthcare
@pytest.mark.p0
@pytest.mark.functional
def test_sam2_02b_load(check_strix_gpu):
    """Test: Load SAM2 0.2B model"""
    pytest.skip("SAM2-0.2B model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.segmentation
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
def test_sam2_02b_inference(check_strix_gpu):
    """Test: Run inference with SAM2 0.2B"""
    pytest.skip("SAM2-0.2B model not yet configured for Strix. "
                "Test will be enabled when model is available.")


@pytest.mark.segmentation
@pytest.mark.quick
@pytest.mark.p0
def test_sam2_quick_smoke(check_strix_gpu):
    """Quick smoke test for SAM2"""
    pytest.skip("SAM2 model not yet configured for Strix. "
                "Test will be enabled when model is available.")
