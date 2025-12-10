# Benchmark Testing Framework

Automated benchmark testing framework for ROCm libraries with system detection, results collection, and performance tracking.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD Integration](#cicd-integration)
- [Architecture](#architecture)
- [Adding New Benchmarks](#adding-new-benchmarks)

## Features

- **Automated Benchmark Execution** - ROCfft, ROCrand, ROCsolver, hipBLASLt
- **System Auto-Detection** - Hardware, OS, GPU, and ROCm version detection
- **Results Management** - Local storage (JSON) and API upload with retry logic
- **Performance Tracking** - LKG (Last Known Good) comparison
- **Comprehensive Logging** - File rotation and configurable log levels
- **Modular Architecture** - Extensible design for adding new benchmarks
- **CI/CD Integration** - Smart test scheduling (smoke vs full tests)

## Quick Start

### Available Benchmarks

- `test_hipblaslt_benchmark.py` - hipBLASLt benchmark suite
- `test_rocsolver_benchmark.py` - ROCsolver benchmark suite
- `test_rocrand_benchmark.py` - ROCrand benchmark suite
- `test_rocfft_benchmark.py` - ROCfft benchmark suite

## Project Structure

```
build_tools/github_actions/
├── configs/
│   ├── config.yml              # Main framework configuration
│   ├── README.md               # This file
│   └── benchmarks/             # Benchmark-specific configs
│       ├── hipblaslt.json
│       ├── rocsolver.json
│       ├── rocrand.json
│       └── rocfft.json
│
├── test_executable_scripts/    # Regular test scripts
│
├── benchmark_scripts/          # Benchmark test scripts
│   ├── test_hipblaslt_benchmark.py
│   ├── test_rocsolver_benchmark.py
│   ├── test_rocrand_benchmark.py
│   └── test_rocfft_benchmark.py
│
├── utils/                      # Framework utilities
│   ├── test_client.py          # Main client API
│   ├── logger.py               # Logging utilities
│   ├── config/                 # Configuration management
│   ├── system/                 # System detection
│   └── results/                # Results handling & schemas
│
├── fetch_test_configurations.py  # Regular test matrix generation
├── benchmark_test_matrix.py      # Benchmark test matrix definitions
├── configure_ci.py               # CI workflow configuration
└── github_actions_utils.py          # GitHub Actions utilities
```

## CI/CD Integration

### When Benchmark Tests Run

Benchmark tests are configured to run **only on nightly CI builds** to save time and resources on pull request validation:

| Workflow Trigger | Test Type | Benchmark Tests | Regular Tests |
|------------------|-----------|-----------------|---------------|
| **Pull Request (PR)** | `smoke` | Skipped | Run (1 shard) |
| **Nightly CI (scheduled)** | `full` | Run | Run (all shards) |
| **Push to main** | `smoke` | Skipped* | Run (1 shard) |
| **Manual workflow** | configurable | Depends on inputs | Depends on inputs |

*\*Push to main can trigger full tests if:*
- *Git submodules are modified, OR*
- *Test labels (e.g., `test:rocfft_bench`) are specified*

### Available Benchmark Tests in CI

The following benchmark tests are defined in `benchmark_test_matrix.py`:

| Test Name | Library | Platform | Timeout | Shards |
|-----------|---------|----------|---------|--------|
| `hipblaslt_bench` | hipBLASLt | Linux | 60 min | 1 |
| `rocsolver_bench` | ROCsolver | Linux | 60 min | 1 |
| `rocrand_bench` | ROCrand | Linux | 60 min | 1 |
| `rocfft_bench` | ROCfft | Linux | 60 min | 1 |

**Implementation:** During nightly CI runs, `configure_ci.py` adds benchmark test names to test labels, which are then processed by `fetch_test_configurations.py` to include benchmarks in the test execution matrix.

## Architecture

### Test Execution Flow

```
1. Initialize TestClient
   ↓ Auto-detect system (GPU, OS, ROCm version)
   ↓ Load configuration from config.yml
   
2. Run Benchmarks
   ↓ Execute benchmark binary
   ↓ Capture output to log file
   
3. Parse Results
   ↓ Extract metrics from log file
   ↓ Structure data according to schema
   
4. Upload Results
   ↓ Submit to API (with retry)
   ↓ Save JSON locally
   
5. Compare with LKG
   ↓ Fetch last known good results
   ↓ Calculate performance delta
   
6. Report Results
   ↓ Display formatted table
   ↓ Append to GitHub Actions step summary
   ↓ Return exit code (0=success, 1=failure)
```

## Adding New Benchmarks

To add a new benchmark test to the nightly CI:

### 1. Create Benchmark Script

Create `build_tools/github_actions/benchmark_scripts/test_your_benchmark.py`. Reference existing benchmarks like `test_rocfft_benchmark.py` as a template.

Key components:
- Import `TestClient` from `utils`
- Define `run_benchmarks()` - executes binary and logs output
- Define `parse_results()` - parses logs and returns structured data
- Call `client.upload_results()` to submit to API

### 2. Add to Benchmark Test Matrix

Edit `build_tools/github_actions/benchmark_test_matrix.py`:

```python
"your_benchmark": {
    "job_name": "your_benchmark",
    "fetch_artifact_args": "--your-lib --tests",
    "timeout_minutes": 60,
    "test_script": f"python {_get_benchmark_script_path('test_your_benchmark.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

The benchmark will automatically be included in nightly CI runs via test labels set by `configure_ci.py`.

### 3. Test Locally

```bash
# Set environment variables
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

# Run the benchmark
python3 build_tools/github_actions/benchmark_scripts/test_your_benchmark.py
```

## Related Documentation

- [Main TheRock Documentation](../../../README.md)
- [Utils Module Documentation](../utils/README.md)
- [CI Nightly Workflow](https://github.com/ROCm/TheRock/actions/workflows/ci_nightly.yml)
