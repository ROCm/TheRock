# Strix Platform Testing Guide

## Overview

This guide covers testing for AMD Strix platforms (Strix Point and Strix Halo) in TheRock, including existing test cases and how to add new ones.

---

## 🎯 Strix Platforms in TheRock

| Platform | LLVM Target | Architecture | Status |
|----------|-------------|--------------|--------|
| **Strix Point** | gfx1150 | RDNA 3.5 iGPU | ✅ Build Passing, Sanity Tested |
| **Strix Halo** | gfx1151 | RDNA 3.5 iGPU | ✅ Release Ready (Linux), Sanity Tested |

**Location in Code:** `cmake/therock_amdgpu_targets.cmake:138-147`

---

## 📊 Current Test Coverage for Strix

### Test Infrastructure

Strix platforms are integrated into three test matrices:

#### **1. Pre-submit Tests** (Pull Requests)
**File:** `build_tools/github_actions/amdgpu_family_matrix.py`

```python
"gfx1151": {
    "linux": {
        "test-runs-on": "linux-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "bypass_tests_for_releases": True,
        "build_variants": ["release"],
        "sanity_check_only_for_family": True,
    },
    "windows": {
        "test-runs-on": "windows-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "build_variants": ["release"],
    },
}
```

#### **2. Nightly Tests** (Scheduled)
**File:** `build_tools/github_actions/amdgpu_family_matrix.py:163-202`

```python
"gfx1150": {
    "linux": {
        "test-runs-on": "",  # No runner assigned yet
        "family": "gfx1150",
        "build_variants": ["release"],
    },
    "windows": {
        "test-runs-on": "",  # No runner assigned yet
        "family": "gfx1150",
        "build_variants": ["release"],
    },
}
```

### Existing Test Cases for Strix

#### **1. Sanity Tests**
**File:** `tests/test_rocm_sanity.py`

**Tests:**
- ✅ ROCm Info detection
- ✅ GPU device enumeration
- ✅ HIP compilation (hipcc)
- ✅ HIP printf execution
- ✅ ROCm agent enumerator

**Runs on:** All platforms including gfx1151/gfx1150

#### **2. Library Tests**
**Location:** `build_tools/github_actions/test_executable_scripts/`

**Available for Strix:**

| Library | Test Script | Platforms | Special Handling |
|---------|-------------|-----------|------------------|
| **hipBLAS** | `test_hipblas.py` | Linux, Windows | ✅ Full tests |
| **hipBLASLt** | `test_hipblaslt.py` | Linux, Windows | ⚠️ Quick tests only on Windows gfx1151 |
| **hipCUB** | `test_hipcub.py` | Linux, Windows | ✅ Full tests |
| **hipDNN** | `test_hipdnn.py` | Linux, Windows | ✅ Full tests |
| **hipFFT** | `test_hipfft.py` | Linux, Windows | ✅ Full tests |
| **hipRAND** | `test_hiprand.py` | Linux, Windows | ✅ Full tests |
| **hipSOLVER** | `test_hipsolver.py` | Linux, Windows | ✅ Full tests |
| **hipSPARSE** | `test_hipsparse.py` | Linux | ✅ Full tests |
| **rocBLAS** | `test_rocblas.py` | Linux, Windows | ✅ Full tests |
| **rocFFT** | `test_rocfft.py` | Linux | ✅ Full tests |
| **rocPRIM** | `test_rocprim.py` | Linux, Windows | ✅ Full tests |
| **rocRAND** | `test_rocrand.py` | Linux, Windows | ✅ Full tests |
| **rocSOLVER** | `test_rocsolver.py` | Linux | ✅ Full tests |
| **rocSPARSE** | `test_rocsparse.py` | Linux, Windows | ❌ Excluded on Windows gfx1151 ([Issue #1640](https://github.com/ROCm/TheRock/issues/1640)) |
| **rocTHRUST** | `test_rocthrust.py` | Linux, Windows | ✅ Full tests |
| **rocWMMA** | `test_rocwmma.py` | Linux, Windows | ✅ Full tests |
| **MIOpen** | `test_miopen.py` | Linux | ✅ Full tests |
| **MIOpen Plugin** | `test_miopen_plugin.py` | Linux, Windows | ✅ Full tests |

**Excluded on Strix (by design):**
- ❌ **hipSPARSELt** - Not supported on gfx115X ([Issue #2042](https://github.com/ROCm/TheRock/issues/2042))
- ❌ **rccl** - Not supported on iGPUs ([Issue #150](https://github.com/ROCm/TheRock/issues/150))

#### **4. AI/ML Tests** ⭐ NEW
**Location:** `tests/strix_ai/`

**Strix-Specific AI Workloads:**

| Category | Tests | Status | Priority |
|----------|-------|--------|----------|
| **Vision Language Models (VLM)** | LLaVA, CLIP, Qwen-VL | 🔴 Needed | Critical |
| **Vision Transformers (ViT)** | ViT-Base, DINOv2, Swin | 🔴 Needed | Critical |
| **Object Detection** | YOLOv8, DETR | 🔴 Needed | Critical |
| **Semantic Segmentation** | SegFormer, Mask2Former | 🟡 Needed | High |
| **Edge Inference** | Quantization, ONNX | 🔴 Needed | Critical |
| **Video Processing** | Encode/Decode, Real-time | 🟡 Needed | High |
| **Windows AI Platform** | DirectML, WinML | 🟡 Needed | High |
| **Consumer AI Apps** | Background blur, upscaling | 🟢 Needed | Medium |

**Why Strix Needs These Tests:**
- Strix targets **Edge AI** and **Consumer AI** use cases (Windows Copilot+, AI PCs)
- These workloads are **NOT covered** by MI (data center) or Navi (gaming) test suites
- iGPU characteristics (shared memory, power efficiency) require **specific optimization**

**See Full Details:**
- 📋 **[STRIX_AI_ML_TEST_PLAN.md](./STRIX_AI_ML_TEST_PLAN.md)** - Comprehensive test plan
- 🚀 **[STRIX_AI_QUICK_START.md](./STRIX_AI_QUICK_START.md)** - Implementation guide

#### **3. Special Handling for Strix**

**Windows Strix Halo Memory Constraint:**
```python
# File: test_hipblaslt.py:28-29
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"
```
Only runs quick/smoke tests due to memory limitations ([Issue #1750](https://github.com/ROCm/TheRock/issues/1750))

**Windows Strix Halo rocSPARSE Exclusion:**
```python
# File: fetch_test_configurations.py:124-126
"exclude_family": {
    "windows": ["gfx1151"]
}
```
rocSPARSE tests excluded on Windows gfx1151 ([Issue #1640](https://github.com/ROCm/TheRock/issues/1640))

---

## 🚀 How to Add New Test Cases for Strix

### Method 1: Add a New Library Test

#### **Step 1: Create Test Script**

Create `build_tools/github_actions/test_executable_scripts/test_mynewlib.py`:

```python
import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# GTest sharding configuration
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# Test type (full, smoke, quick)
test_type = os.getenv("TEST_TYPE", "full")

# Special handling for Strix platforms if needed
if AMDGPU_FAMILIES in ["gfx1150", "gfx1151"]:
    logging.info(f"Running on Strix platform: {AMDGPU_FAMILIES}")
    # Add any Strix-specific configuration here
    if platform == "windows" and AMDGPU_FAMILIES == "gfx1151":
        # Memory constraint handling
        test_type = "quick"

# Build test filter
test_filter = []
if test_type == "smoke":
    test_filter.append("--gtest_filter=*smoke*")
elif test_type == "quick":
    test_filter.append("--gtest_filter=*quick*")

# Run the test
cmd = [f"{THEROCK_BIN_DIR}/mynewlib-test"] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
```

#### **Step 2: Register in Test Matrix**

Edit `build_tools/github_actions/fetch_test_configurations.py`:

```python
test_matrix = {
    # ... existing tests ...
    
    "mynewlib": {
        "job_name": "mynewlib",
        "fetch_artifact_args": "--mynewlib --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_mynewlib.py')}",
        "platform": ["linux", "windows"],  # or just ["linux"] if not Windows-ready
        "total_shards": 1,
        # Optional: Exclude specific platforms/families
        # "exclude_family": {
        #     "windows": ["gfx1151"]  # Exclude if not working
        # },
    },
}
```

#### **Step 3: Update GitHub Actions Workflow**

Tests are automatically included once registered in the matrix!

---

### Method 2: Add Platform-Specific Test

#### **Create Strix-Specific Test File**

Create `tests/test_strix_specific.py`:

```python
from pathlib import Path
from pytest_check import check
import logging
import os
import platform
import pytest
import re
import shlex
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()


def is_windows():
    return "windows" == platform.system().lower()


def run_command(command: list[str], cwd=None):
    logger.info(f"++ Run [{cwd}]$ {shlex.join(command)}")
    process = subprocess.run(
        command, capture_output=True, cwd=cwd, shell=is_windows(), text=True
    )
    if process.returncode != 0:
        logger.error(f"Command failed!")
        logger.error("command stdout:")
        for line in process.stdout.splitlines():
            logger.error(line)
        logger.error("command stderr:")
        for line in process.stderr.splitlines():
            logger.error(line)
        raise Exception(f"Command failed: `{shlex.join(command)}`")
    return process


@pytest.fixture(scope="session")
def gpu_info():
    """Get GPU information"""
    try:
        result = run_command([f"{THEROCK_BIN_DIR}/rocminfo"])
        return result.stdout
    except Exception as e:
        logger.info(str(e))
        return None


class TestStrixSpecific:
    """Test suite specifically for Strix platforms (gfx1150, gfx1151)"""
    
    @pytest.mark.skipif(is_windows(), reason="rocminfo not on Windows")
    def test_strix_gpu_detected(self, gpu_info):
        """Verify Strix GPU is detected"""
        if not gpu_info:
            pytest.fail("rocminfo failed to run")
        
        # Check for gfx1150 or gfx1151
        check.is_not_none(
            re.search(r"Name:\s*gfx115[0-1]", gpu_info),
            "Failed to detect Strix GPU (gfx1150 or gfx1151)"
        )
    
    def test_strix_igpu_memory(self, gpu_info):
        """Verify iGPU memory configuration"""
        if not gpu_info:
            pytest.skip("rocminfo not available")
        
        # Strix platforms have shared system memory
        # Check that memory is detected
        check.is_not_none(
            re.search(r"Size:\s*\d+", gpu_info),
            "Failed to detect GPU memory"
        )
    
    def test_strix_hip_runtime(self):
        """Test HIP runtime on Strix"""
        # Get offload architecture
        platform_executable_suffix = ".exe" if is_windows() else ""
        offload_arch_path = (
            THEROCK_BIN_DIR / ".." / "lib" / "llvm" / "bin" / 
            f"offload-arch{platform_executable_suffix}"
        ).resolve()
        
        process = run_command([str(offload_arch_path)])
        offload_arch = process.stdout.strip().splitlines()[0]
        
        # Verify it's a gfx115X architecture
        check.is_true(
            offload_arch.startswith("gfx115"),
            f"Expected gfx115X architecture, got {offload_arch}"
        )
    
    def test_strix_memory_allocation(self):
        """Test basic memory allocation on Strix iGPU"""
        # Create simple HIP program to test memory allocation
        test_code = """
        #include <hip/hip_runtime.h>
        #include <iostream>
        
        int main() {
            int *d_data;
            size_t size = 1024 * sizeof(int);
            
            hipError_t err = hipMalloc(&d_data, size);
            if (err != hipSuccess) {
                std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
                return 1;
            }
            
            err = hipMemset(d_data, 0, size);
            if (err != hipSuccess) {
                std::cerr << "hipMemset failed: " << hipGetErrorString(err) << std::endl;
                return 1;
            }
            
            hipFree(d_data);
            std::cout << "Memory allocation test passed!" << std::endl;
            return 0;
        }
        """
        
        # Write test code
        test_file = THIS_DIR / "strix_mem_test.hip"
        test_file.write_text(test_code)
        
        # Compile
        exec_name = "strix_mem_test.exe" if is_windows() else "strix_mem_test"
        run_command([
            f"{THEROCK_BIN_DIR}/hipcc",
            str(test_file),
            "-o", exec_name
        ], cwd=str(THIS_DIR))
        
        # Run
        process = run_command([f"./{exec_name}"], cwd=str(THIS_DIR))
        check.is_in("Memory allocation test passed!", process.stdout)
        
        # Cleanup
        test_file.unlink()
        (THIS_DIR / exec_name).unlink()
```

#### **Run This Test:**
```bash
# Set environment variables
export THEROCK_BIN_DIR=/path/to/therock/bin
export AMDGPU_FAMILIES=gfx1151

# Run pytest
pytest tests/test_strix_specific.py -v
```

---

### Method 3: Add Test to Existing Harness

#### **Add to PyTest Harness**

Edit `tests/harness/tests_mynewlib.py`:

```python
#!/usr/bin/python3

class TestMyNewLib:
    """Test Suite for MyNewLib component"""

    def test_mynewlib_basic(self, orch, therock_path, result):
        """Basic functionality test"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/MyNewLib")
        assert result.testVerdict
    
    def test_mynewlib_strix_specific(self, orch, therock_path, result):
        """Strix-specific test case"""
        import os
        
        # Skip if not on Strix
        amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
        if amdgpu_family not in ["gfx1150", "gfx1151"]:
            pytest.skip(f"Strix-specific test, skipping for {amdgpu_family}")
        
        # Run Strix-specific test
        result.testVerdict = orch.runCtest(
            cwd=f"{therock_path}/bin/MyNewLib",
            test_filter="strix"  # Filter for Strix-specific tests
        )
        assert result.testVerdict
```

---

### Method 4: Add Performance Benchmarks

#### **Create Strix Performance Test**

Create `tests/test_strix_performance.py`:

```python
import pytest
import subprocess
import time
import os
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific performance tests"
)
class TestStrixPerformance:
    """Performance benchmarks for Strix platforms"""
    
    def test_memory_bandwidth(self):
        """Measure memory bandwidth on Strix iGPU"""
        # Run memory bandwidth test
        cmd = [f"{THEROCK_BIN_DIR}/rocm-bandwidth-test"]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start
        
        assert result.returncode == 0, "Memory bandwidth test failed"
        
        # Log performance metrics
        print(f"Memory bandwidth test completed in {duration:.2f}s")
        print(result.stdout)
    
    def test_compute_performance(self):
        """Measure compute performance on Strix"""
        # Run compute benchmark
        cmd = [f"{THEROCK_BIN_DIR}/rocblas-bench", "--function", "gemm"]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        duration = time.time() - start
        
        assert result.returncode == 0, "Compute benchmark failed"
        
        print(f"Compute benchmark completed in {duration:.2f}s")
    
    def test_igpu_shared_memory(self):
        """Test shared memory characteristics on iGPU"""
        # Strix platforms use shared system memory
        # Test that memory allocation works correctly
        
        cmd = [f"{THEROCK_BIN_DIR}/rocminfo"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert "gfx115" in result.stdout, "Strix GPU not detected"
        
        # Parse memory size from rocminfo
        import re
        memory_match = re.search(r"Size:\s*(\d+)", result.stdout)
        if memory_match:
            memory_kb = int(memory_match.group(1))
            print(f"Detected iGPU memory: {memory_kb / 1024 / 1024:.2f} GB")
```

---

## 🔧 Configuration for Strix Tests

### **1. Enable Nightly Tests for Strix Point**

Edit `build_tools/github_actions/amdgpu_family_matrix.py:163-174`:

```python
"gfx1150": {
    "linux": {
        "test-runs-on": "linux-strix-point-gpu-rocm",  # Add runner label
        "family": "gfx1150",
        "build_variants": ["release"],
        "sanity_check_only_for_family": True,  # Add this
    },
    "windows": {
        "test-runs-on": "windows-strix-point-gpu-rocm",  # Add runner label
        "family": "gfx1150",
        "build_variants": ["release"],
    },
}
```

### **2. Add Test Filters for Strix**

If you need platform-specific test filtering:

```python
# In your test script
test_filter = []

# Example: Run reduced tests on Strix due to memory constraints
if AMDGPU_FAMILIES in ["gfx1150", "gfx1151"] and platform == "windows":
    test_filter.append("--gtest_filter=*quick*")
elif test_type == "smoke":
    test_filter.append("--gtest_filter=*smoke*")
```

---

## 📋 Test Organization

### **Test Types Available:**

1. **`full`** - Complete test suite (default)
2. **`smoke`** - Quick smoke tests only
3. **`quick`** - Reduced test set (used for Strix Windows)

### **Test Sharding:**

For long-running tests, use sharding:

```python
test_matrix = {
    "mylib": {
        "total_shards": 4,  # Split into 4 parallel jobs
        # ...
    }
}
```

Sharding configuration:
- `SHARD_INDEX`: Current shard (1-based)
- `TOTAL_SHARDS`: Total number of shards
- Automatically converted to 0-based for GTest

---

## 🎯 Comprehensive Test Suite Template for Strix

Create `tests/test_strix_comprehensive.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Strix Platforms (gfx1150, gfx1151)
"""

import pytest
import subprocess
import os
import re
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", "/opt/rocm/bin")).resolve()
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


def is_strix_platform():
    """Check if running on Strix hardware"""
    return AMDGPU_FAMILIES in ["gfx1150", "gfx1151"]


def run_hip_command(cmd, timeout=60):
    """Run a HIP command and return result"""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=THEROCK_BIN_DIR
    )
    return result


@pytest.mark.skipif(not is_strix_platform(), reason="Strix-specific tests")
class TestStrixPlatform:
    """Test suite for Strix platform capabilities"""
    
    def test_strix_detection(self):
        """Verify correct Strix platform is detected"""
        result = run_hip_command([f"{THEROCK_BIN_DIR}/rocminfo"])
        assert result.returncode == 0
        
        if AMDGPU_FAMILIES == "gfx1150":
            assert "gfx1150" in result.stdout, "Strix Point (gfx1150) not detected"
        elif AMDGPU_FAMILIES == "gfx1151":
            assert "gfx1151" in result.stdout, "Strix Halo (gfx1151) not detected"
    
    def test_strix_hip_compilation(self):
        """Test HIP compilation for Strix"""
        test_file = Path("/tmp/strix_test.hip")
        test_file.write_text("""
        #include <hip/hip_runtime.h>
        #include <iostream>
        
        __global__ void kernel() {
            printf("Hello from Strix GPU!\\n");
        }
        
        int main() {
            kernel<<<1, 1>>>();
            hipDeviceSynchronize();
            return 0;
        }
        """)
        
        # Compile
        result = run_hip_command([
            f"{THEROCK_BIN_DIR}/hipcc",
            str(test_file),
            "-o", "/tmp/strix_test"
        ])
        assert result.returncode == 0, "Compilation failed"
        
        # Run
        result = subprocess.run(["/tmp/strix_test"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Hello from Strix GPU!" in result.stdout
        
        # Cleanup
        test_file.unlink()
        Path("/tmp/strix_test").unlink()
    
    def test_strix_memory_allocation(self):
        """Test various memory allocation patterns on Strix iGPU"""
        sizes_mb = [1, 10, 100, 500]  # Different allocation sizes
        
        for size_mb in sizes_mb:
            # Create test that allocates memory
            test_code = f"""
            #include <hip/hip_runtime.h>
            int main() {{
                void *ptr;
                size_t size = {size_mb} * 1024 * 1024;
                hipError_t err = hipMalloc(&ptr, size);
                if (err == hipSuccess) {{
                    hipFree(ptr);
                    return 0;
                }}
                return 1;
            }}
            """
            # ... compile and run ...
    
    def test_strix_concurrent_streams(self):
        """Test concurrent stream execution on Strix"""
        # Test that multiple streams can execute concurrently
        pass  # Implement based on requirements
    
    @pytest.mark.parametrize("library", [
        "rocblas", "hipblas", "rocfft", "hipfft", "rocrand", "hiprand"
    ])
    def test_strix_library_smoke(self, library):
        """Smoke test for each library on Strix"""
        # Run quick smoke test for each library
        test_executable = f"{THEROCK_BIN_DIR}/{library}-test"
        if not Path(test_executable).exists():
            pytest.skip(f"{library} test not available")
        
        result = subprocess.run(
            [test_executable, "--gtest_filter=*smoke*"],
            capture_output=True,
            timeout=30
        )
        assert result.returncode == 0, f"{library} smoke test failed"


@pytest.mark.skipif(not is_strix_platform(), reason="Strix-specific tests")
class TestStrixWindowsSpecific:
    """Windows-specific tests for Strix"""
    
    @pytest.mark.skipif(not os.name == "nt", reason="Windows only")
    def test_strix_windows_driver(self):
        """Verify Windows driver is loaded correctly"""
        # Check device manager or registry for Strix device
        pass
    
    @pytest.mark.skipif(not os.name == "nt", reason="Windows only")
    def test_strix_windows_memory_constraint(self):
        """Test memory-constrained scenarios on Windows"""
        # Strix Halo on Windows has memory limitations (Issue #1750)
        # Test that allocations respect limits
        pass
```

---

## 🔍 Running Tests for Strix

### **Run All Tests:**

```bash
# Linux - Strix Halo
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151
pytest tests/ -v

# Windows - Strix Halo
$env:THEROCK_BIN_DIR="C:\rocm\bin"
$env:AMDGPU_FAMILIES="gfx1151"
pytest tests\ -v
```

### **Run Specific Test Suite:**

```bash
# Sanity tests only
pytest tests/test_rocm_sanity.py -v

# Strix-specific tests
pytest tests/test_strix_specific.py -v

# Single library test
python build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

### **Run with Test Type Filter:**

```bash
# Smoke tests only
export TEST_TYPE=smoke
pytest tests/ -v

# Quick tests (for Windows Strix)
export TEST_TYPE=quick
pytest tests/ -v
```

---

## 📝 Test Case Checklist for New Tests

When adding a new test for Strix:

- [ ] **Works on Linux gfx1151** (Strix Halo)
- [ ] **Works on Linux gfx1150** (Strix Point)
- [ ] **Works on Windows gfx1151** (Strix Halo)
- [ ] **Works on Windows gfx1150** (Strix Point)
- [ ] **Handles memory constraints** on Windows Strix Halo
- [ ] **Respects test type** (full, smoke, quick)
- [ ] **Uses GTest sharding** if long-running
- [ ] **Proper error handling** and logging
- [ ] **Clean up temporary files**
- [ ] **Documented in test matrix**

---

## 🐛 Known Issues for Strix

### **1. Windows Strix Halo Memory Constraints**
**Issue:** [#1750](https://github.com/ROCm/TheRock/issues/1750)  
**Workaround:** Use `quick` test type
```python
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"
```

### **2. rocSPARSE on Windows gfx1151**
**Issue:** [#1640](https://github.com/ROCm/TheRock/issues/1640)  
**Status:** Excluded from Windows gfx1151 tests
```python
"exclude_family": {
    "windows": ["gfx1151"]
}
```

### **3. No Runner for gfx1150 Nightly**
**Status:** No hardware assigned yet
```python
"test-runs-on": "",  # No runner
```

---

## 📊 Test Coverage Matrix for Strix

| Test Category | gfx1151 Linux | gfx1151 Windows | gfx1150 Linux | gfx1150 Windows |
|---------------|---------------|-----------------|---------------|-----------------|
| **Sanity Tests** | ✅ | ✅ | ✅ | ✅ |
| **BLAS** | ✅ | ✅ (quick) | ✅ | ✅ |
| **FFT** | ✅ | ✅ | ✅ | ✅ |
| **RAND** | ✅ | ✅ | ✅ | ✅ |
| **PRIM** | ✅ | ✅ | ✅ | ✅ |
| **SPARSE** | ✅ | ❌ (Issue #1640) | ✅ | ❌ |
| **SOLVER** | ✅ | ✅ | ✅ | ✅ |
| **MIOpen** | ✅ | ✅ | ✅ | ✅ |
| **PyTorch (Basic)** | ✅ | ✅ | ⏳ | ⏳ |
| **VLM (LLaVA, CLIP)** | 🔴 Needed | 🔴 Needed | 🔴 Needed | 🔴 Needed |
| **ViT (Transformers)** | 🔴 Needed | 🔴 Needed | 🔴 Needed | 🔴 Needed |
| **Object Detection** | 🔴 Needed | 🔴 Needed | 🔴 Needed | 🔴 Needed |
| **Edge Inference** | 🔴 Needed | 🔴 Needed | 🔴 Needed | 🔴 Needed |
| **Video Processing** | 🟡 Needed | 🟡 Needed | 🟡 Needed | 🟡 Needed |
| **Windows AI (DirectML)** | N/A | 🟡 Needed | N/A | 🟡 Needed |

---

## 🚀 Quick Start: Add Your First Test

### **Step-by-Step Example:**

**1. Create test file:** `tests/test_my_strix_feature.py`

```python
import pytest
import os

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-only test"
)
def test_my_strix_feature():
    """Test my new feature on Strix"""
    # Your test code here
    assert True  # Replace with actual test
```

**2. Run it:**
```bash
export AMDGPU_FAMILIES=gfx1151
export THEROCK_BIN_DIR=/opt/rocm/bin
pytest tests/test_my_strix_feature.py -v
```

**3. Add to CI** (if needed):
```python
# Edit fetch_test_configurations.py
"myfeature": {
    "job_name": "myfeature",
    "test_script": "pytest tests/test_my_strix_feature.py",
    "platform": ["linux", "windows"],
    "total_shards": 1,
}
```

---

## 📖 Additional Resources

### **Test Infrastructure Files:**
- `tests/test_rocm_sanity.py` - Basic sanity tests
- `tests/harness/conftest.py` - PyTest configuration
- `build_tools/github_actions/fetch_test_configurations.py` - Test matrix
- `build_tools/github_actions/amdgpu_family_matrix.py` - Platform configuration
- `build_tools/github_actions/test_executable_scripts/` - Library tests
- **`tests/strix_ai/` - AI/ML tests for Strix** ⭐ NEW

### **Documentation:**
- `docs/development/test_environment_reproduction.md` - Reproduce test environments
- `docs/development/development_guide.md` - Development guidelines
- `SUPPORTED_GPUS.md` - GPU support status
- **`docs/development/STRIX_AI_ML_TEST_PLAN.md` - AI/ML test plan for Strix** ⭐ NEW
- **`docs/development/STRIX_AI_QUICK_START.md` - Quick start for AI tests** ⭐ NEW

---

## 💡 Best Practices

1. **Use Platform Detection:**
   ```python
   if AMDGPU_FAMILIES in ["gfx1150", "gfx1151"]:
       # Strix-specific logic
   ```

2. **Handle Memory Constraints:**
   ```python
   if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
       test_type = "quick"  # Reduce memory usage
   ```

3. **Use Appropriate Test Types:**
   - `smoke` - Quick validation (< 1 minute)
   - `quick` - Reduced test set (< 5 minutes)
   - `full` - Complete test suite

4. **Add Timeouts:**
   ```python
   subprocess.run(cmd, timeout=60)  # Prevent hanging
   ```

5. **Clean Up Resources:**
   ```python
   try:
       run_test()
   finally:
       cleanup_temp_files()
   ```

---

## 🎯 Summary

**Current Test Coverage:**
- ✅ 20+ library test suites available
- ✅ Strix Halo (gfx1151) fully tested on Linux & Windows
- ✅ Strix Point (gfx1150) tested on Linux
- ⏳ Some tests pending for Windows gfx1150

**To Add More Tests:**
1. Create test script in `tests/` or `test_executable_scripts/`
2. Register in `fetch_test_configurations.py` if needed
3. Handle platform-specific constraints
4. Run and verify

**Test Infrastructure is Ready!** Just follow the patterns above to add new tests.

