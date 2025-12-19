"""
Action Prediction Tests for Strix

Tests for action prediction models on Strix GPUs.
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
def test_action_prediction_basic(check_strix_gpu):
    """Test: Basic action prediction"""
    pytest.skip("Action prediction models not yet configured for Strix. "
                "Test will be enabled when models are available.")


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.performance
def test_action_prediction_latency(check_strix_gpu):
    """Test: Action prediction latency"""
    pytest.skip("Action prediction models not yet configured for Strix. "
                "Test will be enabled when models are available.")


@pytest.mark.vla
@pytest.mark.robotics
@pytest.mark.quick
@pytest.mark.p0
def test_action_prediction_quick_smoke(check_strix_gpu):
    """Quick smoke test for action prediction"""
    pytest.skip("Action prediction models not yet configured for Strix. "
                "Test will be enabled when models are available.")
