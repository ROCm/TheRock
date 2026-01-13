# TheRock Test Framework

Unified testing framework for TheRock ROCm distribution, supporting benchmark and functional testing with automated execution, system detection, and results management.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD Integration](#cicd-integration)
- [Architecture](#architecture)

## Overview

The test framework provides infrastructure for two test types:

| Test Type                     | Purpose                          | Result Types         | When to Use                                  |
| ----------------------------- | -------------------------------- | -------------------- | -------------------------------------------- |
| **[Benchmark](benchmark/)**   | Performance regression detection | PASS/FAIL/UNKNOWN    | Prevent performance degradation (nightly CI) |
| **[Functional](functional/)** | Correctness validation           | PASS/FAIL/ERROR/SKIP | Verify expected behavior (nightly CI)        |

### Key Features

- **Shared Infrastructure** - Common utilities, configuration, and results management
- **System Auto-Detection** - Hardware, OS, GPU, and ROCm version detection
- **Results Management** - Local storage (JSON) and API upload with retry logic
- **Comprehensive Logging** - File rotation and configurable log levels
- **Error Handling** - Custom exceptions with clear, actionable messages
- **Modular Architecture** - Extensible design for adding new test types
- **CI/CD Integration** - Parallel execution in nightly CI

## Quick Start

### Environment Setup

All tests require these environment variables:

```bash
export THEROCK_BIN_DIR=/path/to/therock/build/bin  # Path to TheRock binaries
export ARTIFACT_RUN_ID=local-test                  # Unique identifier for this test run
export AMDGPU_FAMILIES=gfx950-dcgpu                # Target GPU family
```

### Running Tests

See test-specific READMEs for detailed instructions and examples:

- **[Benchmark Tests](benchmark/README.md)** - Performance regression testing
- **[Functional Tests](functional/README.md)** - Correctness validation testing

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
│   └── README.md                  # Functional-specific docs
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

| Workflow Trigger           | Benchmark Tests | Functional Tests |
| -------------------------- | --------------- | ---------------- |
| **Pull Request (PR)**      | Skipped         | Optional         |
| **Nightly CI (scheduled)** | Run (parallel)  | Run (parallel)   |
| **Push to main**           | Skipped         | Optional         |

### Parallel Execution Architecture

Tests run in **parallel** for faster CI execution:

```
ci_nightly.yml
  └─ ci_linux.yml
      ├─ build_artifacts
      │
      ├─ test_artifacts ────────┐ Run in parallel
      │   └─ Functional tests   │ after build
      │                          │
      └─ test_benchmarks ────────┘
          └─ Benchmark tests
```

**Workflow Files:**

- `.github/workflows/ci_nightly.yml` - Nightly CI orchestration
- `.github/workflows/test_benchmarks.yml` - Benchmark execution
- `.github/workflows/test_artifacts.yml` - Functional test execution

## Architecture

### Common Test Execution Flow

All tests follow this pattern:

1. **Initialize** - Auto-detect system (GPU, ROCm), load configuration, setup logging
1. **Execute** - Run test binaries/scripts, capture output to log files
1. **Parse** - Extract metrics/results from logs, structure data
1. **Process** - Type-specific validation (LKG comparison or correctness check)
1. **Report** - Display results, upload to API, update GitHub Actions summary

### Implementation Details

See test-specific READMEs for detailed implementation guides:

- **[Benchmark Tests](benchmark/README.md)** - LKG comparison logic and adding new benchmarks
- **[Functional Tests](functional/README.md)** - Result validation and adding new tests
- **[Shared Utils](utils/README.md)** - Common utilities, exceptions, and helpers
