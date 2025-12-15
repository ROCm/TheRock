"""
Shared pytest fixtures and configuration for Strix AI tests
"""

import pytest
import os
import subprocess
import shutil
import sys

@pytest.fixture(scope="session")
def strix_device():
    """
    Get Strix device for testing
    
    Returns torch.device('cuda') if Strix GPU available, skips test otherwise
    """
    import torch
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    
    # Check for Strix device (gfx115x)
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    
    if amdgpu_family not in ["gfx1150", "gfx1151"]:
        pytest.skip(f"Not running on Strix GPU (AMDGPU_FAMILIES={amdgpu_family})")
    
    device = torch.device("cuda")
    
    # Verify device
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {device_name}")
    
    return device


@pytest.fixture
def cleanup_gpu():
    """Clean up GPU memory after test"""
    yield
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@pytest.fixture
def test_image_224():
    """Create a 224x224 test image"""
    from PIL import Image
    return Image.new('RGB', (224, 224), color='blue')


@pytest.fixture
def test_image_512():
    """Create a 512x512 test image"""
    from PIL import Image
    return Image.new('RGB', (512, 512), color='green')


@pytest.fixture
def record_property(request):
    """
    JUnit XML property recording fixture
    
    Usage in tests:
        def test_example(record_property):
            record_property("metric_fps", "45.2")
            record_property("gpu_family", "gfx1151")
    """
    def _record_property(name: str, value):
        if hasattr(request.config, '_xml'):
            request.config._xml.add_property(name, value)
    
    return _record_property


# Profiling fixtures
@pytest.fixture(scope="session")
def rocprof_available():
    """Check if rocprof CLI tool is available"""
    return shutil.which("rocprof") is not None


@pytest.fixture(scope="session")
def rocprofiler_sdk_available():
    """Check if rocprofv3 (rocprofiler-sdk) CLI tool is available"""
    return shutil.which("rocprofv3") is not None


@pytest.fixture(scope="function")
def profiler_context(tmp_path_factory):
    """Create a temporary directory for profiler outputs and clean up"""
    profiling_dir = tmp_path_factory.mktemp("profiling_output")
    print(f"\nCreated profiling output directory: {profiling_dir}")
    yield str(profiling_dir)
    print(f"Cleaning up profiling output directory: {profiling_dir}")
    # shutil.rmtree(profiling_dir) # Keep for inspection if needed


@pytest.fixture(scope="function")
def enable_profiling(request):
    """
    Enable profiling for a test if --profile flag is passed
    
    Usage:
        @pytest.mark.profile
        def test_clip(strix_device, enable_profiling):
            if enable_profiling:
                print("Profiling enabled!")
            # ... test code ...
    
    Run with:
        pytest tests/strix_ai/vlm/ -v -s --profile
    """
    return request.config.getoption("--profile", False)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling mode for tests marked with @pytest.mark.profile"
    )


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "strix: Strix-specific tests")
    config.addinivalue_line("markers", "vlm: Vision Language Model tests")
    config.addinivalue_line("markers", "vla: Vision Language Action tests")
    config.addinivalue_line("markers", "vit: Vision Transformer tests")
    config.addinivalue_line("markers", "cv: Computer Vision tests")
    config.addinivalue_line("markers", "optimization: Model optimization tests")
    config.addinivalue_line("markers", "profiling: ROCProfiler integration tests")
    config.addinivalue_line("markers", "quick: Quick smoke tests")
    config.addinivalue_line("markers", "slow: Slow comprehensive tests")
    config.addinivalue_line("markers", "p0: Priority 0 (critical) tests")
    config.addinivalue_line("markers", "p1: Priority 1 (important) tests")
    config.addinivalue_line("markers", "p2: Priority 2 (nice to have) tests")
    config.addinivalue_line("markers", "profile: Test supports profiling mode with --profile flag")


@pytest.fixture(scope="session")
def is_ci():
    """Check if running in CI environment"""
    return bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))


@pytest.fixture(scope="session")
def is_strix_halo():
    """Check if running on Strix Halo (gfx1151)"""
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    return amdgpu_family == "gfx1151"


@pytest.fixture(scope="session")
def is_strix_point():
    """Check if running on Strix Point (gfx1150)"""
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    return amdgpu_family == "gfx1150"


@pytest.fixture(scope="session")
def is_strix():
    """Check if running on any Strix GPU"""
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    return amdgpu_family in ["gfx1150", "gfx1151"]
