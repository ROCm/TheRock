# TheRock CI Workflow - Complete Documentation

## Overview

This document provides a comprehensive, end-to-end explanation of the TheRock Continuous Integration (CI) workflow, including all downstream YAML files and Python scripts that are invoked.

## Table of Contents

- [CI Workflow Entry Point](#ci-workflow-entry-point)
- [Complete Workflow Diagram](#complete-workflow-diagram)
- [Workflow Jobs Detailed](#workflow-jobs-detailed)
- [Key Python Scripts Reference](#key-python-scripts-reference)
- [Data Flow](#data-flow)
- [Execution Flow Summary](#execution-flow-summary)

---

## CI Workflow Entry Point

**File**: `.github/workflows/ci.yml`

### Triggers

The workflow runs when:

1. **Push to main branch** - Runs full build and test for all AMD GPU families
2. **Pull requests** - Runs default builds for:
   - Linux: gfx94X, gfx110X
   - Windows: gfx110X
   - Can add labels like `gfx120X-linux` to trigger additional GPU families
3. **Manual dispatch** - Can be manually triggered with custom options for:
   - Specific GPU families
   - Test label filtering
   - Using pre-built artifacts

### Key Features

- **Concurrency control**: Cancels in-progress runs when new commits are pushed
- **Fail-fast disabled**: Continues testing all GPU families even if one fails
- **Flexible testing**: Can run full builds or just tests on existing artifacts
- **Label-based filtering**: Can reduce test scope using labels like `test:rocprim`, `test:hipcub`

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  ci.yml                                       │
│  Triggered by: Push to main | Pull Request | Manual Dispatch                 │
│  Concurrency: Cancel in-progress runs                                        │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  JOB 1: setup (setup.yml)                                                  │
│  ├─ Python Scripts:                                                        │
│  │  ├─ configure_ci.py ⭐ [Main CI Configuration]                         │
│  │  │  • Reads: amdgpu_family_matrix.py                                   │
│  │  │  • Parses PR labels, input parameters                               │
│  │  │  • Determines which GPU families to build/test                      │
│  │  │  • Outputs: linux_variants, windows_variants, test_labels           │
│  │  │                                                                      │
│  │  └─ compute_rocm_package_version.py                                    │
│  │     • Generates version string for packages                            │
│  │                                                                         │
│  └─ Outputs:                                                               │
│     • enable_build_jobs                                                    │
│     • linux_variants (JSON matrix)                                         │
│     • windows_variants (JSON matrix)                                       │
│     • linux_test_labels, windows_test_labels                              │
│     • rocm_package_version                                                 │
│     • test_type (smoke | full)                                            │
└───────────┬───────────────────────────────────────────────────────────────┘
            │
            ├──────────────────────────────┬───────────────────────────────┐
            ▼                              ▼                               ▼
┌───────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────┐
│ JOB 2: linux_build_and_   │  │ JOB 3: windows_build_    │  │                        │
│        test (ci_linux.yml) │  │        and_test          │  │  (Matrix Strategy:     │
│                            │  │        (ci_windows.yml)  │  │   One job per GPU      │
│ Strategy: Matrix           │  │                          │  │   family variant)      │
│ • gfx94X, gfx110X,        │  │ Strategy: Matrix         │  │                        │
│   gfx1201X, etc.          │  │ • gfx1151, etc.          │  │                        │
└─────┬─────────────────────┘  └────┬─────────────────────┘  └────────────────────────┘
      │                             │
      ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LINUX BUILD PATH                                                           │
│  ═══════════════════                                                        │
│                                                                             │
│  SUB-JOB 2.1: build_portable_linux_artifacts                               │
│               (build_portable_linux_artifacts.yml)                         │
│  ├─ Runs on: azure-linux-scale-rocm (or heavy runners for ASAN)           │
│  ├─ Container: ghcr.io/rocm/therock_build_manylinux_x86_64                │
│  ├─ Python Scripts (in order):                                             │
│  │  ├─ setup_ccache.py                                                     │
│  │  │  • Configures compiler cache for faster builds                       │
│  │  │                                                                       │
│  │  ├─ health_status.py ⭐                                                 │
│  │  │  • Checks runner system health (disk, memory, GPU)                   │
│  │  │                                                                       │
│  │  ├─ fetch_sources.py ⭐                                                 │
│  │  │  • Downloads all ROCm component sources (git repos)                  │
│  │  │  • Uses DVC for dependency management                                │
│  │  │                                                                       │
│  │  ├─ build_configure.py ⭐ [Build Configuration]                        │
│  │  │  • Configures CMake with appropriate presets                         │
│  │  │  • Sets up GPU family targets                                        │
│  │  │  • Generates build/CMakeLists.txt                                    │
│  │  │                                                                       │
│  │  ├─ [CMAKE BUILD] therock-archives & therock-dist targets              │
│  │  │  • Builds all ROCm libraries and tools                               │
│  │  │  • Creates .tar.xz archives                                          │
│  │  │                                                                       │
│  │  ├─ [CTEST] Packaging tests                                             │
│  │  │                                                                       │
│  │  ├─ analyze_build_times.py                                              │
│  │  │  • Parses ninja build logs                                           │
│  │  │  • Generates per-component timing reports                            │
│  │  │                                                                       │
│  │  └─ post_build_upload.py ⭐                                             │
│  │     • Uploads artifacts to S3 bucket                                    │
│  │     • Stores build logs, ccache stats                                   │
│  │                                                                          │
│  └─ Artifacts Generated:                                                   │
│     • build/artifacts/*.tar.xz (ROCm packages)                             │
│     • build/dist/rocm (Full SDK distribution)                              │
│     • build/logs/ (Build logs and metrics)                                 │
└────────┬────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────┬──────────────────────────────────────┐
         ▼                             ▼                                      ▼
┌────────────────────────┐  ┌──────────────────────────────────────────────────┐
│  SUB-JOB 2.2:          │  │  SUB-JOB 2.3: build_portable_linux_python_       │
│  test_linux_artifacts  │  │               packages.yml                       │
│  (test_artifacts.yml)  │  │  ├─ Python Scripts:                              │
│                        │  │  │  ├─ fetch_artifacts.py ⭐                     │
│  ⚡Needs: Build job    │  │  │  │  • Downloads artifacts from S3 or GitHub   │
│  ⚡Matrix: GPU family  │  │  │  │                                            │
│                        │  │  │  └─ linux_portable_build.py                   │
│                        │  │  │     • Builds Python wheels                     │
│                        │  │  │     • Creates pip packages (rocm[devel], etc.) │
│                        │  │  │                                               │
└───┬────────────────────┘  └──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SUB-SUB-JOB 2.2.1: configure_test_matrix                                 │
│  ├─ Runs on: GPU test machine (${{ inputs.test_runs_on }})               │
│  ├─ Python Scripts:                                                       │
│  │  └─ fetch_test_configurations.py ⭐ [Test Matrix Generator]           │
│  │     • Reads test metadata from components                             │
│  │     • Generates test matrix with sharding                             │
│  │     • Filters by test_labels if specified                             │
│  │     • Outputs: components JSON, platform, shard_arr                   │
│  │                                                                        │
│  └─ Outputs:                                                              │
│     • components: List of test jobs to run                               │
│     • platform: linux | windows                                          │
│     • shard_arr: Test sharding configuration                             │
└───┬──────────────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────┬─────────────────────────────────────────┐
    ▼                             ▼                                         ▼
┌────────────────────────┐  ┌────────────────────────────────────────────────┐
│  SUB-SUB-JOB 2.2.2:    │  │  SUB-SUB-JOB 2.2.3: test_components           │
│  test_sanity_check     │  │                     (test_component.yml)       │
│  (test_sanity_check.   │  │                                                │
│   yml)                 │  │  ⚡Needs: test_sanity_check                   │
│                        │  │  ⚡Strategy: Matrix by component & shard       │
│  ⚡Container: Ubuntu   │  │  ⚡Container: Ubuntu (Linux) or Native (Win)   │
│   24.04 (no ROCm)      │  │                                                │
│  ⚡GPU devices mounted │  │  ├─ Composite Action: setup_test_environment  │
│                        │  │  │  ├─ setup_venv.py                           │
│  ├─ Composite Action:  │  │  │  │  • Creates Python virtual env            │
│  │  setup_test_env..  │  │  │  │                                          │
│  │  ├─ setup_venv.py  │  │  │  └─ install_rocm_from_artifacts.py ⭐      │
│  │  ├─ install_rocm.. │  │  │     • Downloads and unpacks artifacts       │
│  │  │   _from_artifa..│  │  │     • Installs ROCm into test environment   │
│  │  │                 │  │  │                                             │
│  ├─ print_driver_gpu..│  │  ├─ health_status.py                           │
│  │  _info.py          │  │  │  • System health check                       │
│  │  • GPU driver      │  │  │                                             │
│  │    validation      │  │  ├─ print_driver_gpu_info.py ⭐                │
│  │                    │  │  │  • Validates GPU driver and devices         │
│  └─ pytest tests/     │  │  │  • Prints GPU capabilities                  │
│     • Basic smoke     │  │  │                                             │
│       tests           │  │  └─ Test Execution (Component-Specific):       │
│     • Runtime checks  │  │     • test_rocprim.py                          │
│                       │  │     • test_hipcub.py                           │
└───────────────────────┘  │     • test_rocthrust.py                        │
                           │     • test_rocblas.py                          │
                           │     • test_hipblas.py                          │
                           │     • test_rocsparse.py                        │
                           │     • test_rocrand.py                          │
                           │     • test_rocfft.py                           │
                           │     • test_rccl.py                             │
                           │     • test_hipfft.py                           │
                           │     • test_hipsolver.py                        │
                           │     • test_rocwmma.py                          │
                           │     • test_miopen.py                           │
                           │     • ... and more (25+ components)            │
                           │                                                │
                           │  Each test script:                             │
                           │  • Uses ctest or pytest                         │
                           │  • Supports smoke/full test modes              │
                           │  • Respects SHARD_INDEX/TOTAL_SHARDS          │
                           │  • Runs with GPU isolation                     │
                           │                                                │
                           └────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  WINDOWS BUILD PATH                                                         │
│  ═══════════════════                                                        │
│                                                                             │
│  SUB-JOB 3.1: build_windows_artifacts                                      │
│               (build_windows_artifacts.yml)                                │
│  ├─ Runs on: azure-windows-scale-rocm                                     │
│  ├─ Python Scripts (in order):                                             │
│  │  ├─ health_status.py                                                    │
│  │  ├─ fetch_sources.py                                                    │
│  │  ├─ build_configure.py                                                  │
│  │  ├─ [CMAKE BUILD] therock-archives & therock-dist                      │
│  │  └─ post_build_upload.py                                                │
│  │                                                                          │
│  └─ Similar to Linux but:                                                  │
│     • Uses MSVC compiler                                                   │
│     • Uses chocolatey for dependencies                                     │
│     • Different cache strategy                                             │
└────────┬────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────┬──────────────────────────────────────┐
         ▼                             ▼                                      ▼
┌────────────────────────┐  ┌──────────────────────────────────────────────────┐
│  SUB-JOB 3.2:          │  │  SUB-JOB 3.3: build_windows_python_packages.yml │
│  test_windows_..       │  │  ├─ Python Scripts:                              │
│  artifacts             │  │  │  ├─ fetch_artifacts.py                        │
│  (test_artifacts.yml)  │  │  │  └─ build_python_packages.py ⭐               │
│                        │  │  │     • Builds Windows Python wheels            │
│  [Same structure as    │  │  │     • Runs sanity checks                      │
│   Linux testing but    │  │  │                                               │
│   on Windows runners]  │  └──────────────────────────────────────────────────┘
│                        │
└────────────────────────┘

            ┌───────────────────────────────────────┐
            ▼                                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  JOB 4: ci_summary                                                         │
│  ├─ Runs: always() (even if other jobs fail)                              │
│  ├─ Depends on: setup, linux_build_and_test, windows_build_and_test      │
│  └─ Logic:                                                                 │
│     • Collects all job results                                            │
│     • Filters out continue-on-error jobs                                  │
│     • Fails if any critical job failed                                    │
│     • Success if all required jobs passed                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Jobs Detailed

### Job 1: Setup (`setup.yml`)

**Purpose**: Configures the CI run by determining which GPU families and tests to execute.

**Key Steps**:
1. Checkout repository with fetch-depth of 2 (for diffing)
2. Set PR labels if this is a pull request
3. Run `configure_ci.py` to determine build matrix
4. Run `compute_rocm_package_version.py` to generate version string

**Outputs**:
- `enable_build_jobs`: Boolean to enable/disable builds
- `linux_variants`: JSON array of Linux build configurations
- `windows_variants`: JSON array of Windows build configurations
- `linux_test_labels`, `windows_test_labels`: Test filtering labels
- `rocm_package_version`: Package version string
- `test_type`: Either "smoke" or "full"

### Job 2: Linux Build and Test (`ci_linux.yml`)

**Matrix Strategy**: Creates one job per GPU family variant (gfx94X, gfx110X, gfx1201X, etc.)

#### Sub-Job 2.1: Build Artifacts (`build_portable_linux_artifacts.yml`)

**Container**: `ghcr.io/rocm/therock_build_manylinux_x86_64`
**Timeout**: 12 hours

**Build Steps**:
1. Install Python dependencies from `requirements.txt`
2. Setup ccache for faster compilation
3. Check runner health status
4. Test build_tools with pytest
5. Fetch ROCm component sources
6. Configure CMake projects with GPU family targets
7. Build `therock-archives` and `therock-dist` targets
8. Run packaging tests with CTest
9. Analyze build times
10. Upload artifacts to S3

**Artifacts Produced**:
- `build/artifacts/*.tar.xz` - Compressed ROCm libraries
- `build/dist/rocm` - Full SDK distribution
- `build/logs/` - Build logs and metrics

#### Sub-Job 2.2: Test Artifacts (`test_artifacts.yml`)

Depends on build completion (or uses pre-built artifacts).

##### Step 2.2.1: Configure Test Matrix

**Script**: `fetch_test_configurations.py`

Generates a test matrix including:
- Which components to test (rocprim, hipcub, rocblas, etc.)
- Shard configuration for parallel testing
- Platform detection (linux/windows)
- Test label filtering

##### Step 2.2.2: Sanity Check (`test_sanity_check.yml`)

**Purpose**: Quick validation that ROCm installation and GPU drivers work.

**Container**: Ubuntu 24.04 (no ROCm pre-installed)
**GPU Access**: Devices mounted into container

**Steps**:
1. Setup test environment (download and install artifacts)
2. Print GPU driver and device information
3. Run basic pytest smoke tests from `tests/` directory

##### Step 2.2.3: Component Tests (`test_component.yml`)

**Matrix Strategy**: One job per component × shard combination

**Container**: Ubuntu 24.04 with GPU access
**Timeout**: 210 minutes per component

**Steps**:
1. Setup test environment
2. Health status check
3. GPU driver validation
4. Execute component-specific test script
5. Post-job cleanup

**Test Scripts** (25+ total):
- ROCm Primitives: `test_rocprim.py`
- HIP/ROC Libraries: `test_hipcub.py`, `test_rocthrust.py`
- Linear Algebra: `test_rocblas.py`, `test_hipblas.py`, `test_rocsolver.py`, `test_hipsolver.py`
- Sparse Operations: `test_rocsparse.py`, `test_hipsparse.py`, `test_hipsparselt.py`
- FFT: `test_rocfft.py`, `test_hipfft.py`
- Random Numbers: `test_rocrand.py`, `test_hiprand.py`
- Communication: `test_rccl.py`
- Neural Networks: `test_miopen.py`, `test_hipdnn.py`
- Other: `test_rocwmma.py`, `test_rocroller.py`, `test_hipblaslt.py`

Each test script:
- Uses CTest or Pytest framework
- Supports smoke test mode (subset of tests) or full test mode
- Implements test sharding for parallel execution
- Has configurable timeouts and retry logic

#### Sub-Job 2.3: Build Python Packages (`build_portable_linux_python_packages.yml`)

**Purpose**: Build Python wheels for ROCm SDK

**Steps**:
1. Fetch artifacts from S3/GitHub
2. Build Python packages in manylinux container
3. Inspect generated wheels
4. (TODO: Sanity check and upload)

### Job 3: Windows Build and Test (`ci_windows.yml`)

Similar structure to Linux jobs but with Windows-specific configurations:

#### Sub-Job 3.1: Build Windows Artifacts (`build_windows_artifacts.yml`)

**Runner**: `azure-windows-scale-rocm`
**Build Directory**: `B:\build`

**Key Differences from Linux**:
- Uses MSVC compiler toolchain
- Installs dependencies via Chocolatey (ccache, ninja, strawberryperl, awscli)
- Uses Windows cache/restore actions
- PowerShell scripts for disk space reporting

#### Sub-Job 3.2: Test Windows Artifacts

Same testing structure as Linux but runs on Windows GPU test machines.

#### Sub-Job 3.3: Build Windows Python Packages (`build_windows_python_packages.yml`)

Uses `build_python_packages.py` instead of Docker container approach.

### Job 4: CI Summary

**Always runs**, even if previous jobs fail.

**Purpose**: Aggregate results and determine overall CI status.

**Logic**:
1. Collect all job results as JSON
2. Filter out jobs marked with `continue-on-error`
3. If any critical job failed → exit 1 (fail the CI)
4. If all required jobs succeeded → exit 0 (pass the CI)

---

## Key Python Scripts Reference

### Configuration & Setup Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **`configure_ci.py`** | `build_tools/github_actions/` | **Main CI orchestrator** - Determines which GPU families and tests to run based on triggers (PR labels, workflow dispatch inputs, push to main). Reads GPU family matrix and generates build variants for Linux and Windows. |
| **`amdgpu_family_matrix.py`** | `build_tools/github_actions/` | Defines all AMD GPU family configurations (gfx94X, gfx110X, gfx1201X, etc.) and maps them to test runner labels. |
| **`compute_rocm_package_version.py`** | `build_tools/` | Generates semantic version strings for ROCm packages based on release type (dev, rc, final). |
| **`setup_ccache.py`** | `build_tools/` | Configures compiler cache (ccache) with appropriate settings for faster incremental builds. |
| **`setup_venv.py`** | `build_tools/` | Creates and configures Python virtual environments for testing. |

### Build Phase Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **`fetch_sources.py`** | `build_tools/` | **Downloads all ROCm component sources** from git repositories. Uses DVC for dependency version management. Supports parallel fetching with `--jobs` flag. |
| **`build_configure.py`** | `build_tools/github_actions/` | **Configures CMake build** with GPU family targets and build variant presets. Sets up appropriate CMake options for manylinux or native builds. |
| **`analyze_build_times.py`** | `build_tools/` | Parses Ninja build logs (`.ninja_log`) to generate per-component build timing reports. |
| **`post_build_upload.py`** | `build_tools/github_actions/` | **Uploads build artifacts to S3 bucket** for later use in testing phase. Handles both successful and failed builds. |
| **`health_status.py`** | `build_tools/` | **System health checks** - Validates disk space, memory, CPU info, and GPU status before builds/tests. |

### Test Phase Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **`fetch_test_configurations.py`** | `build_tools/github_actions/` | **Generates test matrix** with component list, sharding configuration, and test label filtering. Reads test metadata from ROCm components. |
| **`fetch_artifacts.py`** | `build_tools/` | **Downloads build artifacts** from S3 or GitHub Actions artifacts. Supports filtering by artifact group and pattern matching. |
| **`install_rocm_from_artifacts.py`** | `build_tools/` | **Installs ROCm from downloaded artifacts** into test environment. Unpacks tar.xz files and sets up directory structure. |
| **`print_driver_gpu_info.py`** | `build_tools/` | **GPU driver validation** - Detects AMD GPUs, prints driver version, device capabilities, and memory information. |

### Component Test Scripts

**Location**: `build_tools/github_actions/test_executable_scripts/`

All test scripts follow similar patterns:
- Read environment variables: `THEROCK_BIN_DIR`, `TEST_TYPE`, `SHARD_INDEX`, `TOTAL_SHARDS`
- Support smoke test mode (subset) or full test suite
- Use CTest or Pytest frameworks
- Include test filtering and sharding logic

| Script | Component | Framework | Notes |
|--------|-----------|-----------|-------|
| `test_rocprim.py` | ROCm Primitives | CTest | Parallel primitives library |
| `test_hipcub.py` | HIP CUB | CTest | CUB-like primitives for HIP |
| `test_rocthrust.py` | ROCm Thrust | CTest | Thrust-like parallel algorithms |
| `test_rocblas.py` | ROCm BLAS | CTest | Basic Linear Algebra Subprograms |
| `test_hipblas.py` | HIP BLAS | CTest | HIP wrapper for BLAS |
| `test_rocsparse.py` | ROCm Sparse | CTest | Sparse linear algebra |
| `test_hipsparse.py` | HIP Sparse | CTest | HIP wrapper for sparse ops |
| `test_hipsparselt.py` | HIP SparseLT | CTest | Sparse matrix ops (lightweight) |
| `test_rocfft.py` | ROCm FFT | CTest | Fast Fourier Transform |
| `test_hipfft.py` | HIP FFT | CTest | HIP wrapper for FFT |
| `test_rocrand.py` | ROCm Random | CTest | Random number generation |
| `test_hiprand.py` | HIP Random | CTest | HIP wrapper for random numbers |
| `test_rocsolver.py` | ROCm Solver | CTest | Linear system solvers |
| `test_hipsolver.py` | HIP Solver | CTest | HIP wrapper for solvers |
| `test_rccl.py` | RCCL | Pytest | Collective communication library |
| `test_miopen.py` | MIOpen | CTest | Deep neural network library |
| `test_hipdnn.py` | HIP DNN | CTest | HIP wrapper for DNN ops |
| `test_rocwmma.py` | ROCm WMMA | CTest | Wave Matrix Multiply-Accumulate |
| `test_rocroller.py` | ROCm Roller | CTest | Tensor operations |
| `test_hipblas_lt.py` | HIP BLAS LT | CTest | BLAS lightweight operations |
| `test_miopen_plugin.py` | MIOpen Plugin | CTest | MIOpen plugin system |

### Package Building Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **`linux_portable_build.py`** | `build_tools/` | Builds Linux Python packages (ROCm wheels) in manylinux container. Creates packages like `rocm[devel]`, `rocm[libraries]`. |
| **`build_python_packages.py`** | `build_tools/` | Builds Windows Python packages. Generates pip-installable wheels for ROCm SDK. |

### Utility Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `github_actions_utils.py` | `build_tools/github_actions/` | Shared utilities for GitHub Actions workflows |
| `determine_version.py` | `build_tools/github_actions/` | Version determination logic |
| `fetch_package_targets.py` | `build_tools/github_actions/` | Package target fetching utilities |
| `configure_target_run.py` | `build_tools/github_actions/` | Target run configuration |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUTS                                                           │
│ • PR Labels (gfx94X-linux, test:rocprim, etc.)                  │
│ • Workflow Dispatch Inputs (GPU families, test labels, etc.)    │
│ • Git Trigger (push to main, PR, manual)                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ CONFIGURATION PHASE                                              │
│ configure_ci.py                                                  │
│ • Reads: amdgpu_family_matrix.py                                │
│ • Outputs: Build Matrix JSON (linux_variants, windows_variants) │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ SOURCE ACQUISITION                                               │
│ fetch_sources.py                                                 │
│ • Clones ROCm component repositories                            │
│ • Checks out correct versions via DVC                           │
│ • Outputs: Local source tree                                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ BUILD CONFIGURATION                                              │
│ build_configure.py                                               │
│ • Reads: CMakePresets.json, GPU family targets                  │
│ • Runs: CMake configuration                                     │
│ • Outputs: Build system (Ninja files)                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPILATION                                                      │
│ CMake + Ninja Build                                              │
│ • Builds all ROCm libraries and tools                           │
│ • Uses ccache for incremental builds                            │
│ • Outputs: Compiled binaries                                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ARTIFACT PACKAGING                                               │
│ CMake targets: therock-archives, therock-dist                   │
│ • Creates .tar.xz archives                                      │
│ • Organizes files by component                                  │
│ • Outputs: build/artifacts/*.tar.xz                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ARTIFACT UPLOAD                                                  │
│ post_build_upload.py                                             │
│ • Uploads to S3: s3://therock-ci-artifacts/                     │
│ • Organized by: run_id/artifact_group/                          │
│ • Includes: artifacts, logs, build metrics                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ TEST CONFIGURATION                                               │
│ fetch_test_configurations.py                                     │
│ • Generates component test matrix                               │
│ • Applies test label filtering                                  │
│ • Configures test sharding                                      │
│ • Outputs: components JSON array                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ARTIFACT DOWNLOAD                                                │
│ install_rocm_from_artifacts.py                                   │
│ • Downloads from S3 or GitHub Actions                           │
│ • Unpacks .tar.xz files                                         │
│ • Sets up test environment directory structure                  │
│ • Outputs: build/ directory with ROCm installation              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ TEST EXECUTION (Parallel Matrix)                                │
│ Component Test Scripts (test_*.py)                              │
│ • Sanity check first (basic GPU/runtime validation)            │
│ • Then component tests in parallel                              │
│ • Each script runs CTest or Pytest                              │
│ • Supports sharding for large test suites                       │
│ • Outputs: Test results (pass/fail)                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PYTHON PACKAGE BUILDING (Parallel)                              │
│ • Linux: linux_portable_build.py (in manylinux container)       │
│ • Windows: build_python_packages.py                             │
│ • Outputs: Python wheels (.whl files)                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ CI SUMMARY                                                       │
│ ci_summary job (ci.yml)                                          │
│ • Aggregates all job results                                    │
│ • Filters continue-on-error jobs                                │
│ • Reports: Overall pass/fail status                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Execution Flow Summary

### 1. Trigger Phase

A CI run is initiated by:
- Push to `main` branch → Full build for all GPU families
- Pull request → Builds for default families (gfx94X, gfx110X)
- Manual dispatch → Custom GPU families and test filters

### 2. Setup Phase (Job 1)

**Script**: `configure_ci.py`

- Determines trigger type (PR, push, manual)
- Reads PR labels if applicable
- Consults `amdgpu_family_matrix.py` for GPU configurations
- Generates build matrix JSON for Linux and Windows
- Determines test type (smoke vs full)
- Filters test labels if specified

**Output**: JSON matrices defining what to build and test

### 3. Build Phase (Jobs 2 & 3 - Parallel per GPU family)

#### For Each GPU Family:

**Step 1**: Health check (`health_status.py`)
- Validate runner has sufficient disk space
- Check memory and CPU
- Verify GPU is available (for GPU runners)

**Step 2**: Fetch sources (`fetch_sources.py`)
- Clone ROCm component repositories
- Checkout appropriate versions using DVC
- Parallel fetching with 12 workers

**Step 3**: Configure build (`build_configure.py`)
- Setup CMake with correct preset (release, asan, etc.)
- Configure for target GPU family
- Setup ccache for faster incremental builds

**Step 4**: Build (`cmake --build`)
- Compile all ROCm libraries and tools
- Create distribution archives
- Generates `.tar.xz` files

**Step 5**: Package tests (`ctest`)
- Run CTest packaging validation

**Step 6**: Analyze (`analyze_build_times.py`)
- Parse Ninja logs
- Generate timing reports

**Step 7**: Upload (`post_build_upload.py`)
- Upload artifacts to S3
- Store logs and metrics

### 4. Test Phase (Depends on Build - Parallel per Component)

#### Configure Test Matrix

**Script**: `fetch_test_configurations.py`

- Reads component metadata
- Determines which tests to run
- Applies test label filters
- Configures sharding for large test suites

#### Sanity Check (Sequential - Must Pass)

**Workflow**: `test_sanity_check.yml`

- Downloads base artifacts
- Installs ROCm SDK
- Validates GPU driver
- Runs basic smoke tests
- Must pass before component tests start

#### Component Tests (Parallel Matrix)

**Workflow**: `test_component.yml`

For each component (rocprim, rocblas, hipcub, etc.):

1. **Setup Environment**:
   - `setup_venv.py` - Create Python venv
   - `install_rocm_from_artifacts.py` - Download and install ROCm
   
2. **Pre-test Validation**:
   - `health_status.py` - System check
   - `print_driver_gpu_info.py` - GPU validation
   
3. **Execute Tests**:
   - Component-specific test script (e.g., `test_rocblas.py`)
   - Runs CTest or Pytest
   - Respects sharding if configured
   - Smoke or full test mode based on `TEST_TYPE`

### 5. Python Package Phase (Parallel with Testing)

**Linux**: `linux_portable_build.py`
- Builds in manylinux container
- Creates pip-installable wheels
- Packages: `rocm[devel]`, `rocm[libraries]`, etc.

**Windows**: `build_python_packages.py`
- Native Windows build
- Creates Windows wheels
- Runs sanity checks

### 6. Summary Phase (Job 4 - Always Runs)

**Workflow**: ci_summary job in `ci.yml`

- Collects all job results as JSON
- Filters out jobs marked `continue-on-error`
- Reports failed jobs
- Exits with:
  - `exit 0` if all required jobs passed
  - `exit 1` if any critical job failed

---

## Matrix Strategies

### GPU Family Matrix

Different GPU architectures require different builds:

- **gfx94X**: RDNA 2 architecture (e.g., RX 6000 series)
- **gfx110X**: RDNA 3 architecture (e.g., RX 7000 series)
- **gfx1201X**: RDNA 3.5 architecture
- **gfx1151**: Integrated GPUs
- **And more**: Defined in `amdgpu_family_matrix.py`

### Test Sharding

Large test suites are split into shards for parallel execution:

```python
# Example from test script
SHARD_INDEX = os.getenv("SHARD_INDEX")  # e.g., "1"
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS")  # e.g., "4"

# CTest sharding
cmd = [
    "ctest",
    "--parallel", "8",
    "--shard-index", SHARD_INDEX,
    "--shard-count", TOTAL_SHARDS
]
```

This allows a single component test to run across multiple machines simultaneously.

### Build Variants

Different build configurations for different purposes:

- **release**: Standard optimized build (default)
- **asan**: Address sanitizer build for memory debugging
- **debug**: Debug symbols and assertions
- Custom presets defined in `CMakePresets.json`

---

## Environment Variables

### Build Phase

| Variable | Purpose | Example |
|----------|---------|---------|
| `AMDGPU_FAMILIES` | Target GPU families | `"gfx94X,gfx110X"` |
| `BUILD_DIR` | Build directory path | `"build"` or `"B:\build"` |
| `CACHE_DIR` | Cache directory for ccache | `".container-cache"` |
| `CCACHE_CONFIGPATH` | ccache config file | `".ccache/ccache.conf"` |
| `TEATIME_FORCE_INTERACTIVE` | Disable interactive prompts | `0` |

### Test Phase

| Variable | Purpose | Example |
|----------|---------|---------|
| `THEROCK_BIN_DIR` | ROCm binaries location | `"./build/bin"` |
| `OUTPUT_ARTIFACTS_DIR` | Downloaded artifacts dir | `"./build"` |
| `ARTIFACT_RUN_ID` | GitHub run ID for artifacts | `"12345678"` |
| `TEST_TYPE` | Test mode | `"smoke"` or `"full"` |
| `SHARD_INDEX` | Current shard (0-indexed) | `"0"` |
| `TOTAL_SHARDS` | Total number of shards | `"4"` |
| `AMD_LOG_LEVEL` | HIP logging verbosity | `"4"` |

---

## Containers Used

### Build Containers

**Linux**: `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:583d473f...`
- Based on manylinux for portable binary builds
- Includes: CMake, Ninja, GCC, Python, ROCm build tools
- Used for building portable Linux artifacts

### Test Containers

**Linux**: `ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:405945a...`
- Ubuntu 24.04 base (no ROCm pre-installed)
- ROCm installed from test artifacts
- GPU devices mounted: `/dev/kfd`, `/dev/dri`
- IPC host mode for GPU communication

**Windows**: Native (no container)
- Runs directly on Windows Server runners
- Uses MSVC toolchain

---

## Runners Used

### Build Runners

- **Linux**: `azure-linux-scale-rocm` (standard), `azure-linux-u2404-hx176-cpu-rocm` (heavy, for ASAN)
- **Windows**: `azure-windows-scale-rocm`

### Test Runners

Dynamically determined by `configure_ci.py` based on:
- GPU family availability
- Test runner labels from `amdgpu_family_matrix.py`
- Stored in `ROCM_THEROCK_TEST_RUNNERS` organization variable

Example runner labels:
- `linux-gfx94X-gpu`
- `windows-gfx1151-gpu`

---

## Artifact Storage

### Build Artifacts

**Location**: S3 bucket `s3://therock-ci-artifacts/`

**Structure**:
```
s3://therock-ci-artifacts/
  ├── {run_id}/
  │   ├── {artifact_group}/
  │   │   ├── artifacts/
  │   │   │   ├── therock-{component}-{version}-{arch}.tar.xz
  │   │   │   └── ...
  │   │   └── logs/
  │   │       ├── build.log
  │   │       ├── ccache_stats.log
  │   │       └── build_times.json
```

**Artifact Groups**: `gfx94X-dcgpu`, `gfx110X-all`, `gfx1151`, etc.

### Artifact Patterns

Different artifact types indicated by naming:
- `*_dev_*`: Development headers and CMake files
- `*_lib_*`: Runtime libraries
- `*_run_*`: Runtime executables and tools
- `*_tests_*`: Test binaries

Test jobs fetch only what they need:
```bash
# Example: Only fetch lib and run artifacts for testing
fetch_artifacts.py --artifact-group=gfx94X-dcgpu _lib_ _run_
```

---

## Test Label Filtering

Tests can be filtered by component labels:

**Label Format**: `test:{component_name}`

**Examples**:
- `test:rocprim` - Run only rocprim tests
- `test:rocblas` - Run only rocblas tests
- `test:hipcub` - Run only hipcub tests

**Usage**:
1. Add label to PR → Filters tests automatically
2. Workflow dispatch input → Specify test labels manually

This allows focused testing when changes only affect specific components.

---

## Special Features

### Concurrency Control

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event.number || github.sha }}
  cancel-in-progress: true
```

Cancels in-progress CI runs when new commits are pushed to the same PR/branch.

### Continue-on-Error

Some builds are marked as expected to fail:

```yaml
continue-on-error: ${{ inputs.expect_failure }}
```

These failures don't block the overall CI but are tracked separately.

### Conditional Execution

Jobs use complex conditionals to determine if they should run:

```yaml
if: >-
  ${{
    !failure() &&
    !cancelled() &&
    (
      inputs.use_prebuilt_artifacts == 'false' ||
      inputs.use_prebuilt_artifacts == 'true'
    ) &&
    inputs.expect_failure == false
  }}
```

### GPU Isolation

Test containers use pod info for GPU isolation:

```yaml
--env-file /etc/podinfo/gha-gpu-isolation-settings
```

Prevents test jobs from interfering with each other on shared GPU runners.

---

## Debugging Tips

### View Specific Logs

Build logs are uploaded to S3:
```bash
aws s3 cp s3://therock-ci-artifacts/{run_id}/{artifact_group}/logs/build.log .
```

### Re-run Tests Only

Use workflow dispatch with `use_prebuilt_artifacts: true`:
1. Go to Actions tab
2. Click "CI" workflow
3. Click "Run workflow"
4. Check "use_prebuilt_artifacts"
5. Provide `artifact_run_id` from a previous successful build

### Local Testing

Download artifacts locally:
```bash
python build_tools/install_rocm_from_artifacts.py \
  --run-id=12345678 \
  --artifact-group=gfx94X-dcgpu \
  --output-dir=./build
```

Then run tests:
```bash
export THEROCK_BIN_DIR=./build/bin
pytest build_tools/github_actions/test_executable_scripts/test_rocprim.py
```

---

## Summary

The TheRock CI is a sophisticated, multi-stage pipeline that:

1. **Configures** dynamically based on triggers and labels
2. **Builds** ROCm SDK for multiple AMD GPU families in parallel
3. **Tests** 25+ ROCm components with GPU isolation and sharding
4. **Packages** Python wheels for easy distribution
5. **Summarizes** results and reports overall status

The system handles:
- Multiple GPU architectures (RDNA 2, RDNA 3, etc.)
- Both Linux and Windows platforms
- Smoke and full test modes
- Build variants (release, asan, debug)
- Artifact caching and reuse
- Parallel execution with over 100 concurrent jobs

This allows the ROCm team to efficiently validate changes across a complex, heterogeneous ecosystem while maintaining fast feedback loops for developers.

---

## Workflow Files Index

### Primary Workflows

- `ci.yml` - Main CI entry point
- `setup.yml` - Configuration and setup
- `ci_linux.yml` - Linux platform orchestration
- `ci_windows.yml` - Windows platform orchestration

### Build Workflows

- `build_portable_linux_artifacts.yml` - Linux builds
- `build_windows_artifacts.yml` - Windows builds
- `build_portable_linux_python_packages.yml` - Linux Python packages
- `build_windows_python_packages.yml` - Windows Python packages

### Test Workflows

- `test_artifacts.yml` - Test orchestration
- `test_sanity_check.yml` - Sanity checks
- `test_component.yml` - Component testing

### Composite Actions

- `.github/actions/setup_test_environment/action.yml` - Test environment setup

### Other Workflows

- `ci_asan.yml` - AddressSanitizer builds
- `ci_nightly.yml` - Nightly builds
- `ci_weekly.yml` - Weekly builds
- `multi_arch_ci.yml` - Multi-architecture builds
- `build_linux_jax_wheels.yml` - JAX wheel building
- `build_portable_linux_pytorch_wheels.yml` - PyTorch wheel building
- `build_windows_pytorch_wheels.yml` - Windows PyTorch wheels
- Various release and publish workflows

---

**Document Author**: Automated CI Documentation Generator  
**Last Updated**: 2025-12-13  
**Repository**: [ROCm/TheRock](https://github.com/ROCm/TheRock)  
**Branch**: users/asudhanw/my-work

