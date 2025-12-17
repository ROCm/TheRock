# Strix Tests - Quick Reference

## 📋 Available Tests for Strix (gfx1150/gfx1151)

### ✅ Fully Supported Tests

| Test | Linux | Windows | Notes |
|------|-------|---------|-------|
| rocblas | ✅ | ✅ | |
| hipblas | ✅ | ✅ | |
| hipblaslt | ✅ | ✅ | Windows: quick mode only |
| rocprim | ✅ | ✅ | |
| hipcub | ✅ | ✅ | |
| rocthrust | ✅ | ✅ | |
| rocrand | ✅ | ✅ | |
| hiprand | ✅ | ✅ | |
| hipfft | ✅ | ✅ | |
| hipsolver | ✅ | ✅ | |
| hipdnn | ✅ | ✅ | |
| miopen_plugin | ✅ | ✅ | |
| rocwmma | ✅ | ✅ | |

### ⚠️ Platform-Specific Tests

| Test | Linux | Windows | Reason |
|------|-------|---------|--------|
| rocroller | ✅ | ❌ | Linux only |
| rocsolver | ✅ | ❌ | Windows support pending |
| hipsparse | ✅ | ❌ | Linux only |
| rocsparse | ✅ | ❌ | Windows gfx1151 excluded (Issue #1640) |
| hipsparselt | ✅ | ❌ | Linux only |
| rocfft | ✅ | ❌ | Windows support pending |
| miopen | ✅ | ❌ | Linux only |

### ❌ Not Applicable for Strix

| Test | Reason |
|------|--------|
| rccl | Multi-GPU only (Strix is single iGPU) |

---

## 🚀 Quick Start: Add a New Test

### 1. Create Test Script

**File:** `build_tools/github_actions/test_executable_scripts/test_mylib.py`

```python
import logging
import os
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()

logging.basicConfig(level=logging.INFO)

# Strix-specific handling
if AMDGPU_FAMILIES in ["gfx1150", "gfx1151"]:
    logging.info(f"Running on Strix: {AMDGPU_FAMILIES}")

test_type = os.getenv("TEST_TYPE", "full")
test_filter = []
if test_type == "smoke":
    test_filter = ["--gtest_filter=*smoke*"]

cmd = [f"{THEROCK_BIN_DIR}/mylib-test"] + test_filter
subprocess.run(cmd, check=True)
```

### 2. Register in Test Matrix

**File:** `build_tools/github_actions/fetch_test_configurations.py`

```python
"mylib": {
    "job_name": "mylib",
    "fetch_artifact_args": "--mylib --tests",
    "timeout_minutes": 30,
    "test_script": f"python {_get_script_path('test_mylib.py')}",
    "platform": ["linux", "windows"],
    "total_shards": 1,
},
```

### 3. Test Locally

```bash
export THEROCK_BIN_DIR=/path/to/build/bin
export AMDGPU_FAMILIES=gfx1151
python build_tools/github_actions/test_executable_scripts/test_mylib.py
```

---

## 🔧 Common Strix Configurations

### Exclude Strix from a Test

```python
"mytest": {
    # ...
    "exclude_family": {
        "windows": ["gfx1151"],  # Exclude Windows Strix Halo
        "linux": ["gfx1150"],    # Exclude Linux Strix Point
    },
}
```

### Add Strix-Specific Logic

```python
# Memory constraint for Windows Strix Halo
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"
    logging.info("Using reduced test set for Windows Strix Halo")

# Custom environment for all Strix
if AMDGPU_FAMILIES in ["gfx1150", "gfx1151"]:
    environ_vars["STRIX_MODE"] = "1"
```

### Enable Test Sharding

```python
"mytest": {
    # ...
    "total_shards": 4,  # Split into 4 parallel jobs
}

# In test script:
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)
```

---

## 📍 Strix Runner Configuration

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

---

## 🐛 Known Strix Issues

### Issue #1750: Windows Strix Halo Memory Constraint

```python
# Workaround:
if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
    test_type = "quick"
```

### Issue #1640: rocSPARSE on Windows gfx1151

```python
# Excluded:
"rocsparse": {
    # ...
    "exclude_family": {
        "windows": ["gfx1151"]
    },
}
```

---

## 🧪 Test Types

| Type | Filter | Use Case |
|------|--------|----------|
| `full` | None | Complete test suite (default) |
| `smoke` | `--gtest_filter=*smoke*` | Quick validation |
| `quick` | `--gtest_filter=*quick*` | Reduced set (memory-constrained) |

---

## 📊 Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `THEROCK_BIN_DIR` | `/opt/rocm/bin` | Test binaries location |
| `AMDGPU_FAMILIES` | `gfx1151` | GPU family |
| `RUNNER_OS` | `Linux` or `Windows` | Operating system |
| `TEST_TYPE` | `smoke` | Test filter type |
| `SHARD_INDEX` | `1` | Current shard (1-based) |
| `TOTAL_SHARDS` | `4` | Total number of shards |

---

## 📖 Full Documentation

See: [`STRIX_ADD_TESTS_GUIDE.md`](./STRIX_ADD_TESTS_GUIDE.md) for complete guide

---

## ✅ Quick Checklist

Adding a new test:
- [ ] Create test script in `test_executable_scripts/`
- [ ] Register in `fetch_test_configurations.py`
- [ ] Add Strix-specific logic if needed
- [ ] Test locally with Strix environment variables
- [ ] Commit and test in CI

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `build_tools/github_actions/fetch_test_configurations.py` | Test registry |
| `build_tools/github_actions/amdgpu_family_matrix.py` | GPU family config |
| `build_tools/github_actions/test_executable_scripts/` | Test scripts |
| `.github/workflows/test_artifacts.yml` | Test workflow |

---

**Need more details?** See full guide: [`STRIX_ADD_TESTS_GUIDE.md`](./STRIX_ADD_TESTS_GUIDE.md)

