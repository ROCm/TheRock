# Strix Testing Quick Reference

## 🎯 Strix Platforms

- **gfx1150** (Strix Point) - RDNA 3.5 iGPU
- **gfx1151** (Strix Halo) - RDNA 3.5 iGPU

---

## ✅ Current Test Coverage Summary

### Sanity Tests (tests/test_rocm_sanity.py)
- ✅ ROCm Info
- ✅ GPU Detection
- ✅ HIP Compilation
- ✅ ROCm Agent Enumerator

### Library Tests (20+ libraries)
| Library | Linux | Windows | Notes |
|---------|-------|---------|-------|
| hipBLAS, hipBLASLt | ✅ | ✅ | Quick tests on Win gfx1151 |
| rocBLAS | ✅ | ✅ | Full support |
| hipFFT, rocFFT | ✅ | ✅ | rocFFT Linux only |
| hipRAND, rocRAND | ✅ | ✅ | Full support |
| rocSPARSE | ✅ | ❌ | Win gfx1151 excluded ([#1640](https://github.com/ROCm/TheRock/issues/1640)) |
| MIOpen | ✅ | ✅ | Full support |
| rocPRIM, hipCUB | ✅ | ✅ | Full support |
| rocWMMA | ✅ | ✅ | Full support |

---

## 🚀 Add New Test in 3 Steps

### 1. Create Test Script

**File:** `build_tools/github_actions/test_executable_scripts/test_mynewlib.py`

```python
import logging, os, subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()

logging.basicConfig(level=logging.INFO)

# Special handling for Strix Windows
test_type = os.getenv("TEST_TYPE", "full")
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"

# Run test
test_filter = ["--gtest_filter=*quick*"] if test_type == "quick" else []
cmd = [f"{THEROCK_BIN_DIR}/mynewlib-test"] + test_filter

subprocess.run(cmd, check=True)
```

### 2. Register in Test Matrix

**File:** `build_tools/github_actions/fetch_test_configurations.py`

```python
test_matrix = {
    "mynewlib": {
        "job_name": "mynewlib",
        "fetch_artifact_args": "--mynewlib --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_mynewlib.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
}
```

### 3. Done! Test Runs Automatically in CI

---

## 🧪 Run Tests Locally

### Linux
```bash
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151
pytest tests/test_rocm_sanity.py -v
```

### Windows
```powershell
$env:THEROCK_BIN_DIR="C:\rocm\bin"
$env:AMDGPU_FAMILIES="gfx1151"
pytest tests\test_rocm_sanity.py -v
```

### Run Specific Test
```bash
python build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

---

## 📊 Test Types

| Type | Usage | Duration |
|------|-------|----------|
| `full` | Complete suite (default) | Varies |
| `smoke` | Quick validation | < 1 min |
| `quick` | Reduced set (Windows gfx1151) | < 5 min |

Set with: `export TEST_TYPE=smoke`

---

## 🐛 Known Issues

### 1. Windows gfx1151 Memory Constraints ([#1750](https://github.com/ROCm/TheRock/issues/1750))
**Fix:** Use `quick` tests
```python
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"
```

### 2. rocSPARSE Windows gfx1151 ([#1640](https://github.com/ROCm/TheRock/issues/1640))
**Fix:** Exclude from Windows gfx1151
```python
"exclude_family": {"windows": ["gfx1151"]}
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `tests/test_rocm_sanity.py` | Basic sanity tests |
| `build_tools/github_actions/amdgpu_family_matrix.py` | Platform configuration |
| `build_tools/github_actions/fetch_test_configurations.py` | Test matrix |
| `build_tools/github_actions/test_executable_scripts/` | Library test scripts |

---

## 🎨 Test Template

```python
import pytest
import os

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-only test"
)
def test_strix_feature():
    """Test Strix-specific feature"""
    # Your test code
    assert True
```

---

## 💡 Best Practices

1. ✅ **Platform Detection:** Check `AMDGPU_FAMILIES` for Strix
2. ✅ **Memory Aware:** Use `quick` tests on Windows gfx1151
3. ✅ **Timeouts:** Add `timeout` to subprocess calls
4. ✅ **Cleanup:** Remove temporary files in `finally` block
5. ✅ **Sharding:** Use for tests > 15 minutes

---

## 📖 Full Guide

See [`STRIX_TESTING_GUIDE.md`](./STRIX_TESTING_GUIDE.md) for comprehensive documentation, examples, and advanced usage.

---

## 🚦 Test Status

| Platform | Pre-submit | Nightly |
|----------|-----------|---------|
| Linux gfx1151 | ✅ Sanity only | ✅ Enabled |
| Windows gfx1151 | ✅ Full | ✅ Enabled |
| Linux gfx1150 | ✅ Sanity only | ⏳ No runner |
| Windows gfx1150 | ✅ Full | ⏳ No runner |

---

**Need Help?** Check the full guide or contact the ROCm testing team!

