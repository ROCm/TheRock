"""
Pytest configuration for Strix AI/ML tests

Defines markers, fixtures, and configuration for the Strix test suite.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers"""
    
    # Model category markers
    config.addinivalue_line("markers", "vlm: Vision-Language Model tests")
    config.addinivalue_line("markers", "vla: Vision-Language-Action tests")
    config.addinivalue_line("markers", "omni: Multimodal Omni model tests")
    config.addinivalue_line("markers", "instruct: Instruction following model tests")
    config.addinivalue_line("markers", "segmentation: Segmentation model tests (SAM2)")
    config.addinivalue_line("markers", "asr: Automatic Speech Recognition tests")
    config.addinivalue_line("markers", "diffusion: Diffusion/Generative AI tests")
    config.addinivalue_line("markers", "llm: Large Language Model tests")
    
    # Market segment markers
    config.addinivalue_line("markers", "automotive: Automotive market segment tests")
    config.addinivalue_line("markers", "industrial: Industrial market segment tests")
    config.addinivalue_line("markers", "robotics: Robotics market segment tests")
    config.addinivalue_line("markers", "healthcare: Healthcare market segment tests")
    
    # Test type markers
    config.addinivalue_line("markers", "functional: Functional correctness tests")
    config.addinivalue_line("markers", "performance: Performance measurement tests")
    config.addinivalue_line("markers", "profiling: ROCProfiler profiling tests")
    config.addinivalue_line("markers", "benchmark: Market segment benchmark tests")
    config.addinivalue_line("markers", "quick: Quick smoke tests (< 10 seconds)")
    config.addinivalue_line("markers", "slow: Slow tests (> 30 seconds)")
    config.addinivalue_line("markers", "optimization: Quantization/optimization tests")
    
    # Priority markers
    config.addinivalue_line("markers", "p0: Priority 0 - Critical (market launch)")
    config.addinivalue_line("markers", "p1: Priority 1 - High priority")
    config.addinivalue_line("markers", "p2: Priority 2 - Nice to have")
    
    # Quantization markers
    config.addinivalue_line("markers", "awq: AWQ quantized model tests")
    config.addinivalue_line("markers", "gptq: GPTQ quantized model tests")
    config.addinivalue_line("markers", "fp16: FP16 baseline tests")
    
    # ROCm version markers
    config.addinivalue_line("markers", "rocm644: ROCm 6.4.4 specific tests")
    config.addinivalue_line("markers", "rocm702: ROCm 7.0.2 specific tests")
    
    # GPU family markers
    config.addinivalue_line("markers", "gfx1150: Strix Point (gfx1150) specific")
    config.addinivalue_line("markers", "gfx1151: Strix Halo (gfx1151) specific")


@pytest.fixture(scope="session")
def strix_gpu_info():
    """Get Strix GPU information"""
    import os
    import torch
    
    gpu_info = {
        "amdgpu_family": os.getenv("AMDGPU_FAMILIES", ""),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }
    
    return gpu_info


@pytest.fixture(scope="session")
def rocm_info():
    """Get ROCm installation information"""
    import os
    import subprocess
    
    rocm_info = {
        "rocm_home": os.getenv("ROCM_HOME", ""),
        "therock_bin_dir": os.getenv("THEROCK_BIN_DIR", ""),
    }
    
    # Try to get ROCm version
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            rocm_info["rocminfo_available"] = True
        else:
            rocm_info["rocminfo_available"] = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        rocm_info["rocminfo_available"] = False
    
    return rocm_info


@pytest.fixture(scope="session")
def rocprofv3_available():
    """Check if rocprofv3 is available"""
    import subprocess
    
    try:
        result = subprocess.run(
            ["rocprofv3", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Cleanup GPU memory after each test"""
    yield
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic skips"""
    import os
    
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    
    # Skip tests if not on Strix GPU
    if "gfx115" not in amdgpu_family:
        skip_marker = pytest.mark.skip(reason="Not running on Strix GPU (gfx1150/gfx1151)")
        for item in items:
            # Skip all Strix tests if not on Strix hardware
            if "strix" in str(item.fspath).lower():
                item.add_marker(skip_marker)


def pytest_report_header(config):
    """Add custom header to pytest report"""
    import os
    import torch
    
    header = [
        "Strix AI/ML Test Suite",
        f"AMDGPU_FAMILIES: {os.getenv('AMDGPU_FAMILIES', 'Not set')}",
        f"ROCM_HOME: {os.getenv('ROCM_HOME', 'Not set')}",
        f"PyTorch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    
    if torch.cuda.is_available():
        header.append(f"GPU: {torch.cuda.get_device_name(0)}")
        header.append(f"GPU count: {torch.cuda.device_count()}")
    
    return header
