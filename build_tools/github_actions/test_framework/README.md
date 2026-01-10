# TheRock Test Framework

Unified testing framework for TheRock ROCm distribution, supporting benchmark and functional testing with automated execution, system detection, and results management.

## Table of Contents

- [Overview](#overview)
- [Test Types](#test-types)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD Integration](#cicd-integration)
- [Architecture](#architecture)
- [Adding Tests](#adding-tests)

## Overview

The test framework provides a unified infrastructure for multiple test types:

### Features

- **Multi-Type Testing** - Benchmark and functional tests
- **Shared Infrastructure** - Common utilities, configuration, and results management
- **System Auto-Detection** - Hardware, OS, GPU, and ROCm version detection
- **Results Management** - Local storage (JSON) and API upload with retry logic
- **Comprehensive Logging** - File rotation and configurable log levels
- **Error Handling** - Custom exceptions with clear, actionable messages
- **Modular Architecture** - Extensible design for adding new test types
- **CI/CD Integration** - Parallel execution in nightly CI

## Test Types

### 1. Benchmark Tests (`benchmark/`)

**Purpose:** Detect performance regressions by comparing against Last Known Good (LKG) baselines.

- **Result:** PASS/FAIL/UNKNOWN
- **Comparison:** Current vs baseline (with tolerance)
- **Frequency:** Every nightly CI
- **Use Case:** Automated CI gates to prevent regressions
- **Example:** "GEMM performance is 5% slower than baseline" → FAIL

**See:** [benchmark/README.md](benchmark/README.md)

### 2. Functional Tests (`functional/`)

**Purpose:** Validate correctness, API contracts, and expected behavior.

- **Result:** PASS/FAIL with detailed error messages
- **Comparison:** Output vs expected results
- **Frequency:** Every nightly CI
- **Use Case:** Correctness validation, API testing
- **Example:** "Matrix multiplication result matches expected output" → PASS

**See:** [functional/README.md](functional/README.md)

## Quick Start

### Running Tests

```bash
# Set environment variables
export THEROCK_BIN_DIR=/path/to/therock/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

# Run benchmark test
python test_framework/benchmark/scripts/test_hipblaslt_benchmark.py

# Run functional test
python test_framework/functional/scripts/test_miopendriver_conv.py
```

### Available Tests

**Benchmark Tests:**

- `test_hipblaslt_benchmark.py` - hipBLASLt GEMM benchmarks
- `test_rocfft_benchmark.py` - rocFFT transform benchmarks
- `test_rocrand_benchmark.py` - rocRAND generation benchmarks
- `test_rocsolver_benchmark.py` - ROCsolver linear algebra benchmarks

**Functional Tests:**

- `test_miopendriver_conv.py` - MIOpen driver convolution functional tests

## Project Structure

```
test_framework/
├── __init__.py
├── README.md                       # This file
│
├── configs/                        # SHARED configuration
│   └── config.yml                 # Framework config (logging, API, execution)
│
├── benchmark/                      # Benchmark tests (LKG comparison)
│   ├── scripts/                   # Test implementations
│   │   ├── benchmark_base.py      # Base class with LKG logic
│   │   └── test_*_benchmark.py    # Individual benchmark tests
│   ├── configs/                   # Test-specific configurations
│   │   ├── hipblaslt.json
│   │   └── rocfft.json
│   ├── benchmark_matrix.py        # CI test matrix
│   └── README.md                  # Benchmark-specific docs
│
├── functional/                     # Functional/correctness tests
│   ├── scripts/                   # Test implementations
│   │   ├── functional_base.py     # Base class for functional tests
│   │   └── test_miopendriver_conv.py  # MIOpen convolution tests
│   ├── configs/                   # Test-specific configurations
│   │   └── miopen_driver_conv.json
│   ├── functional_matrix.py       # CI test matrix
│   └── README.md                  # (to be created)
│
└── utils/                          # SHARED utilities for all test types
    ├── exceptions.py              # Custom exception classes
    │   ├── BenchmarkExecutionError   # Execution/parsing failures
    │   ├── BenchmarkResultError      # Result validation failures
    │   └── FrameworkException        # Base exception
    │
    ├── logger.py                  # Logging utilities
    ├── test_client.py             # Test execution client
    ├── constants.py               # Global constants
    │
    ├── config/                    # Configuration parsers
    │   ├── config_parser.py
    │   ├── config_validator.py
    │   └── config_helper.py
    │
    ├── results/                   # Results handling & LKG
    │   ├── results_api.py        # API for storing/retrieving results
    │   └── results_handler.py    # Process and format results
    │
    └── system/                    # Hardware & ROCm detection
        ├── hardware.py           # GPU detection and capabilities
        ├── platform.py           # Platform-specific utilities
        └── rocm_detector.py      # ROCm version detection
```

## CI/CD Integration

### Test Execution Schedule

Benchmark tests run **only on nightly CI builds** to save time and resources on pull request validation:

| Workflow Trigger           | Benchmark/Functional Tests     | Regular Tests          |
| -------------------------- | ------------------------------ | ---------------------- |
| **Pull Request (PR)**      | Skipped                        | Run (smoke: 1 shard)   |
| **Nightly CI (scheduled)** | Run (in parallel, always full) | Run (full: all shards) |
| **Push to main**           | Skipped                        | Run (smoke: 1 shard)   |
| **Manual workflow**        | Optional                       | Optional               |

**Note:** Benchmarks always run with `total_shards=1` and do not use `test_type` or `test_labels` filtering.

### Parallel Execution Architecture

Benchmarks run **in parallel** with regular tests for faster CI execution:

```
ci_nightly.yml → ci_linux.yml
                   │
                   ├─ build_artifacts (30 min)
                   │
                   ├─ test_artifacts (45 min) ────┐
                   │   └─ Regular tests            │  Run in
                   │      (rocblas, hipblas, ...)  │  PARALLEL
                   │                                │
                   └─ test_benchmarks (60 min) ────┘
                        └─ Benchmark tests
                           (hipblaslt_bench, rocfft_bench, ...)

```

### Workflow Integration Details

```
.github/workflows/ci_nightly.yml
  └─ calls → ci_linux.yml
              ├─ job: build_artifacts
              │   └─ Builds TheRock binaries
              │
              ├─ job: test_artifacts (parallel)
              │   └─ calls → test_artifacts.yml
              │       └─ Functional tests matrix
              │
              └─ job: test_benchmarks (parallel)
                  └─ calls → test_benchmarks.yml
                      ├─ configure_benchmark_matrix
                      │   └─ fetch_test_configurations.py
                      │      (IS_BENCHMARK_WORKFLOW=true)
                      └─ run_benchmarks
                          └─ test_component.yml (matrix)
```

## Architecture

### Execution Flow (Common Pattern)

```
┌─────────────────────────────────────────┐
│ 1. Initialize Test Runner               │
│    - Auto-detect system (GPU, ROCm)     │
│    - Load configuration                 │
│    - Setup logging                      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Execute Tests                        │
│    - Run test binaries/scripts          │
│    - Capture output to log files        │
│    - Handle errors gracefully           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. Parse Results                        │
│    - Extract metrics from logs          │
│    - Structure data according to schema │
│    - Validate results                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. Process Results (Type-Specific)      │
│    Benchmark: Compare with LKG baseline │
│    Functional: Validate correctness     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. Report Results                       │
│    - Display formatted output           │
│    - Upload to results API              │
│    - Append to GitHub Actions summary   │
│    - Return exit code                   │
└─────────────────────────────────────────┘
```

## Adding Tests

### Adding a Benchmark Test

See detailed guide in [benchmark/README.md](benchmark/README.md#adding-a-new-benchmark)

### Adding a Functional Test

See detailed guide in [functional/README.md](functional/README.md#adding-a-new-benchmark)

### Getting Help

- **Framework issues:** See this README
- **Benchmark-specific:** See [benchmark/README.md](benchmark/README.md)
- **Functional tests:** See [functional/README.md](functional/README.md)
- **Utilities:** See [utils/README.md](utils/README.md)

## Related Files

- `configure_ci.py` - CI workflow orchestration
- `fetch_test_configurations.py` - Test matrix builder
- `github_actions_utils.py` - GitHub Actions utilities
- `.github/workflows/test_benchmarks.yml` - Benchmark execution workflow
- `.github/workflows/ci_nightly.yml` - Nightly CI orchestration
