"""
Shared pytest fixtures for Strix AI tests
"""

import pytest
import os

# Optional imports - skip if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


@pytest.fixture(scope="session")
def strix_device():
    """Get Strix GPU device"""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {device_name}")
    
    # Check if it's a Strix device (gfx115x)
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    if amdgpu_family not in ["gfx1150", "gfx1151"]:
        pytest.skip(f"Not a Strix device. AMDGPU_FAMILIES={amdgpu_family}")
    
    return torch.device("cuda")


@pytest.fixture(scope="session")
def test_image_224():
    """Create standard 224x224 test image"""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    return Image.new('RGB', (224, 224), color='blue')


@pytest.fixture(scope="session")
def test_image_512():
    """Create standard 512x512 test image"""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    return Image.new('RGB', (512, 512), color='green')


@pytest.fixture(scope="function")
def cleanup_gpu():
    """Cleanup GPU memory after each test"""
    yield
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="session")
def amdgpu_family():
    """Get AMDGPU family from environment"""
    return os.getenv("AMDGPU_FAMILIES", "")


@pytest.fixture(scope="session")
def is_strix(amdgpu_family):
    """Check if running on Strix platform"""
    return amdgpu_family in ["gfx1150", "gfx1151"]


@pytest.fixture(scope="session")
def is_strix_halo(amdgpu_family):
    """Check if running on Strix Halo (gfx1151)"""
    return amdgpu_family == "gfx1151"


@pytest.fixture(scope="session")
def is_strix_point(amdgpu_family):
    """Check if running on Strix Point (gfx1150)"""
    return amdgpu_family == "gfx1150"


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "strix: Tests specific to Strix platforms")
    config.addinivalue_line("markers", "vlm: Vision Language Model tests")
    config.addinivalue_line("markers", "vla: Vision Language Action tests")
    config.addinivalue_line("markers", "vit: Vision Transformer tests")
    config.addinivalue_line("markers", "cv: Computer Vision tests")
    config.addinivalue_line("markers", "slow: Tests that take > 30 seconds")
    config.addinivalue_line("markers", "quick: Quick smoke tests")
    config.addinivalue_line("markers", "windows: Windows-specific tests")
    config.addinivalue_line("markers", "p0: Priority 0 (Critical)")
    config.addinivalue_line("markers", "p1: Priority 1 (High)")
    config.addinivalue_line("markers", "p2: Priority 2 (Medium)")

