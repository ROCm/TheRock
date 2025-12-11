# Strix AI Testing CI/CD Workflow

## 🎯 Overview

This document explains how tests are executed on Strix platforms (gfx1150, gfx1151) in GitHub Actions CI/CD pipelines.

---

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Pull Request / Push to main / Nightly Schedule             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  .github/workflows/ci.yml                                    │
│  - Determines trigger type (presubmit/postsubmit/nightly)   │
│  - Builds TheRock for target platforms                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  .github/workflows/test_artifacts.yml                        │
│  - Calls test workflows for each GPU family                 │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │ MI325  │  │ Strix  │  │ Others │
    │ gfx94x │  │gfx1151 │  │  ...   │
    └────────┘  └────┬───┘  └────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  configure_test_matrix                                       │
│  - build_tools/github_actions/fetch_test_configurations.py  │
│  - Reads amdgpu_family_matrix.py                           │
│  - Generates test matrix for Strix                          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│test_sanity   │ │test_component│ │test_component│
│  - rocminfo  │ │  - rocBLAS   │ │  - hipBLAS   │
│  - HIP test  │ │  - MIOpen    │ │  - YOLOv8    │
└──────────────┘ └──────────────┘ └──────────────┘
                                   │
                                   ▼  (Future)
                            ┌──────────────┐
                            │  Strix AI    │
                            │  - VLM/VLA   │
                            │  - ViT       │
                            │  - CV        │
                            └──────────────┘
```

---

## 🔧 Current Strix Configuration

### **File:** `build_tools/github_actions/amdgpu_family_matrix.py`

#### **Presubmit Matrix** (Pull Requests)

```python
"gfx1151": {  # Strix Halo
    "linux": {
        "test-runs-on": "linux-strix-halo-gpu-rocm",     # ← Runner label
        "family": "gfx1151",
        "bypass_tests_for_releases": True,
        "build_variants": ["release"],
        "sanity_check_only_for_family": True,            # ← Only sanity tests!
    },
    "windows": {
        "test-runs-on": "windows-strix-halo-gpu-rocm",   # ← Runner label  
        "family": "gfx1151",
        "build_variants": ["release"],
        # No sanity_check_only! Windows runs full test suite
    },
}
```

#### **Nightly Matrix** (Scheduled)

```python
"gfx1150": {  # Strix Point
    "linux": {
        "test-runs-on": "",                              # ← NO RUNNER!
        "family": "gfx1150",
        "build_variants": ["release"],
    },
    "windows": {
        "test-runs-on": "",                              # ← NO RUNNER!
        "family": "gfx1150",
        "build_variants": ["release"],
    },
},
"gfx1152": {  # Strix (variant)
    "linux": {
        "test-runs-on": "",                              # ← NO RUNNER!
        "family": "gfx1152",
        "expect_failure": True,
        "build_variants": ["release"],
    },
    "windows": {
        "test-runs-on": "",                              # ← NO RUNNER!
        "family": "gfx1152",
        "expect_failure": True,
        "build_variants": ["release"],
    },
}
```

---

## 🎯 Test Execution Flow

### **Step 1: Configure Test Matrix**

**Workflow:** `.github/workflows/test_artifacts.yml`

```yaml
jobs:
  configure_test_matrix:
    runs-on: ${{ inputs.test_runs_on }}  # Runs on Strix runner
    steps:
      - name: "Configuring CI options"
        env:
          AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}  # gfx1151
          TEST_TYPE: ${{ inputs.test_type }}               # full/smoke/quick
        run: python ./build_tools/github_actions/fetch_test_configurations.py
```

**Script:** `build_tools/github_actions/fetch_test_configurations.py`

This script:
1. Reads `amdgpu_families` (e.g., "gfx1151")
2. Looks up test configuration from `test_matrix` dict
3. Generates JSON matrix of tests to run
4. Returns list of components to test

### **Step 2: Run Sanity Tests**

**Workflow:** `.github/workflows/test_sanity_check.yml`

```yaml
test_sanity_check:
  runs-on: ${{ inputs.test_runs_on }}  # linux-strix-halo-gpu-rocm
  container:
    image: 'ghcr.io/rocm/no_rocm_image_ubuntu24_04'
    options: --device /dev/kfd --device /dev/dri
  env:
    THEROCK_BIN_DIR: ${{ github.workspace }}/build/bin
    AMDGPU_FAMILIES: gfx1151
  steps:
    - name: Test
      run: pytest tests/test_rocm_sanity.py -v
```

**Tests Run:**
- `test_rocm_output` - rocminfo GPU detection
- `test_hip_printf` - HIP compilation & execution  
- `test_rocm_agent_enumerator` - GPU enumeration (Linux only)

### **Step 3: Run Component Tests** (If not sanity_check_only)

**Workflow:** `.github/workflows/test_component.yml`

```yaml
test_component:
  runs-on: ${{ inputs.test_runs_on }}  # Strix runner
  strategy:
    matrix:
      shard: ${{ fromJSON(inputs.component).shard_arr }}  # [1, 2, 3, 4]
  env:
    THEROCK_BIN_DIR: "./build/bin"
    AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}
    SHARD_INDEX: ${{ matrix.shard }}
    TOTAL_SHARDS: ${{ fromJSON(inputs.component).total_shards }}
    TEST_TYPE: ${{ fromJSON(inputs.component).test_type }}
  steps:
    - name: Test
      timeout-minutes: ${{ fromJSON(inputs.component).timeout_minutes }}
      run: |
        ${{ fromJSON(inputs.component).test_script }}
```

**Example test_script:**
```bash
python build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

---

## 📋 Current Test Matrix for Strix

**File:** `build_tools/github_actions/fetch_test_configurations.py`

```python
test_matrix = {
    "rocblas": {
        "job_name": "rocblas",
        "test_script": "python build_tools/.../test_rocblas.py",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "hipblas": {
        "job_name": "hipblas",
        "test_script": "python build_tools/.../test_hipblas.py",
        "platform": ["linux", "windows"],
        "total_shards": 4,
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "test_script": "python build_tools/.../test_hipblaslt.py",
        "platform": ["linux", "windows"],
        "total_shards": 6,
        # Special handling for Windows gfx1151 - quick tests only
    },
    # ... 20+ more libraries
}
```

---

## ➕ Adding Strix AI Tests to CI/CD

### **Step 1: Add to Test Matrix**

Edit `build_tools/github_actions/fetch_test_configurations.py`:

```python
test_matrix = {
    # ... existing tests ...
    
    # NEW: Strix AI Tests
    "strix_vlm": {
        "job_name": "strix_vlm",
        "fetch_artifact_args": "--pytorch --tests",
        "timeout_minutes": 30,
        "test_script": "pytest tests/strix_ai/vlm/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "strix_vla": {
        "job_name": "strix_vla",
        "fetch_artifact_args": "--pytorch --tests",
        "timeout_minutes": 30,
        "test_script": "pytest tests/strix_ai/vla/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "strix_vit": {
        "job_name": "strix_vit",
        "fetch_artifact_args": "--pytorch --tests",
        "timeout_minutes": 30,
        "test_script": "pytest tests/strix_ai/vit/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "strix_cv": {
        "job_name": "strix_cv",
        "fetch_artifact_args": "--pytorch --tests",
        "timeout_minutes": 45,
        "test_script": "pytest tests/strix_ai/cv/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
}
```

### **Step 2: Enable Full Test Suite for Strix**

Edit `build_tools/github_actions/amdgpu_family_matrix.py`:

```python
"gfx1151": {
    "linux": {
        "test-runs-on": "linux-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "bypass_tests_for_releases": True,
        "build_variants": ["release"],
        "sanity_check_only_for_family": False,  # ← Change to False!
    },
    "windows": {
        "test-runs-on": "windows-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "build_variants": ["release"],
    },
}
```

### **Step 3: Handle PyTorch Dependencies**

The test runners need PyTorch and transformers installed. Two options:

**Option A: Install in test setup action**

Edit `.github/actions/setup_test_environment/action.yml`:

```yaml
- name: Install AI dependencies for Strix
  if: ${{ contains(env.AMDGPU_FAMILIES, 'gfx115') }}
  run: |
    pip install transformers accelerate ultralytics opencv-python pillow torch torchvision timm einops
```

**Option B: Use conda environment**

Create `.github/workflows/strix_ai_tests.yml`:

```yaml
name: Strix AI Tests

on:
  workflow_call:
    inputs:
      amdgpu_families:
        type: string
      test_runs_on:
        type: string

jobs:
  strix_ai_tests:
    runs-on: ${{ inputs.test_runs_on }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Conda
        uses: conda-incubator/setup-miniconda@v3
        with:
          auto-activate-base: true
      
      - name: Install dependencies
        run: |
          conda install pytorch torchvision -c pytorch
          pip install transformers ultralytics opencv-python
      
      - name: Run Strix AI Tests
        env:
          THEROCK_BIN_DIR: ./build/bin
          AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}
        run: |
          pytest tests/strix_ai/ -v --junit-xml=results.xml
      
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: strix-ai-test-results
          path: results.xml
```

---

## 🔍 Debugging Test Execution

### **View Workflow Runs**

```
GitHub Repository → Actions tab → Select workflow run
```

### **Check Test Matrix**

Look for "Configure test matrix" job output:
```json
{
  "components": [
    {"job_name": "rocblas", "test_script": "python ...", ...},
    {"job_name": "hipblas", "test_script": "python ...", ...}
  ],
  "platform": "linux"
}
```

### **Check Individual Test Logs**

```
Workflow run → Test component (job) → Test (step) → View logs
```

Look for:
- `AMDGPU_FAMILIES=gfx1151` (environment variable)
- `THEROCK_BIN_DIR=./build/bin`
- `pytest` or test executable output

---

## 📊 Current Test Coverage on Strix

| Test Category | Linux gfx1151 | Windows gfx1151 | Status |
|---------------|---------------|-----------------|--------|
| **Sanity Tests** | ✅ Presubmit | ✅ Presubmit | Running |
| **Library Tests** | ❌ Disabled (sanity_check_only) | ✅ Enabled | Partial |
| **AI/ML Tests (VLM)** | ⏳ Not in CI | ⏳ Not in CI | Ready to add |
| **AI/ML Tests (VLA)** | ⏳ Not in CI | ⏳ Not in CI | Ready to add |
| **AI/ML Tests (ViT)** | ⏳ Not in CI | ⏳ Not in CI | Ready to add |
| **AI/ML Tests (CV)** | ⏳ Not in CI | ⏳ Not in CI | Ready to add |

---

## 🎯 Recommended CI/CD Integration

### **Phase 1: Add to Nightly (Low Risk)**

```python
# In amdgpu_family_matrix.py - nightly section
"gfx1151": {
    "linux": {
        "test-runs-on": "linux-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "build_variants": ["release"],
        "test_labels": ["strix_vlm", "strix_vit"],  # Run specific AI tests
    },
}
```

### **Phase 2: Add to Presubmit (After validation)**

```python
# In amdgpu_family_matrix.py - presubmit section
"gfx1151": {
    "linux": {
        "test-runs-on": "linux-strix-halo-gpu-rocm",
        "family": "gfx1151",
        "build_variants": ["release"],
        "sanity_check_only_for_family": False,  # Enable full tests
        "test_labels": ["strix_vlm", "strix_vla", "strix_vit"],
    },
}
```

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI entry point |
| `.github/workflows/test_artifacts.yml` | Test orchestration |
| `.github/workflows/test_component.yml` | Individual test execution |
| `build_tools/github_actions/amdgpu_family_matrix.py` | Platform configuration |
| `build_tools/github_actions/fetch_test_configurations.py` | Test matrix generation |
| `tests/strix_ai/` | Strix AI test suite |

---

## 📝 Manual Workflow Dispatch

You can manually trigger tests on Strix:

```bash
# Via GitHub UI
Repository → Actions → Test Artifacts → Run workflow
  - artifact_group: linux_release_gfx1151
  - amdgpu_families: gfx1151
  - test_runs_on: linux-strix-halo-gpu-rocm
  - test_type: full

# Via GitHub CLI
gh workflow run test_artifacts.yml \
  -f artifact_group=linux_release_gfx1151 \
  -f amdgpu_families=gfx1151 \
  -f test_runs_on=linux-strix-halo-gpu-rocm \
  -f test_type=full
```

---

## ✅ Summary

**Current State:**
- ✅ Strix runners available (`linux-strix-halo-gpu-rocm`, `windows-strix-halo-gpu-rocm`)
- ✅ Sanity tests running on presubmit (Linux)
- ✅ Full test suite running on Windows
- ❌ AI/ML tests not yet in CI (tests ready, just need integration)

**To Add AI Tests:**
1. Add test entries to `fetch_test_configurations.py`
2. Set `sanity_check_only_for_family: False` for Linux gfx1151
3. Install PyTorch dependencies in test setup
4. Run in nightly first, then promote to presubmit

**Ready to integrate when:**
- Strix hardware is available/stable
- PyTorch installation fixed on Windows runners
- Baseline performance established

