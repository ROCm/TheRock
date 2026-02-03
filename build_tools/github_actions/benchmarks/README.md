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

- **Automated Benchmark Execution** - ROCfft, ROCrand, ROCsolver, hipBLASLt, rocBLAS, RCCL
- **System Auto-Detection** - Hardware, OS, GPU, and ROCm version detection
- **Distributed Testing** - Multi-GPU RCCL benchmarks (requires OpenMPI in Docker image)
- **Results Management** - Local storage (JSON) and API upload with retry logic
- **Performance Tracking** - LKG (Last Known Good) comparison
- **Comprehensive Logging** - File rotation and configurable log levels
- **Modular Architecture** - Extensible design for adding new benchmarks
- **CI/CD Integration** - Parallel execution with regular tests in nightly CI

## Quick Start

### Available Benchmarks

- `benchmarks/scripts/test_hipblaslt_benchmark.py` - hipBLASLt benchmark suite
- `benchmarks/scripts/test_rccl_benchmark.py` - RCCL collective communication benchmarks (requires OpenMPI)
- `benchmarks/scripts/test_rocblas_benchmark.py` - rocBLAS benchmark suite
- `benchmarks/scripts/test_rocfft_benchmark.py` - ROCfft benchmark suite
- `benchmarks/scripts/test_rocrand_benchmark.py` - ROCrand benchmark suite
- `benchmarks/scripts/test_rocsolver_benchmark.py` - ROCsolver benchmark suite

## Project Structure

```
build_tools/github_actions/
├── benchmarks/                 # All benchmark-related code
│   ├── scripts/                # Benchmark test implementations
│   │   ├── benchmark_base.py   # Base class for all benchmarks
│   │   ├── test_hipblaslt_benchmark.py
│   │   ├── test_rccl_benchmark.py
│   │   ├── test_rocblas_benchmark.py
│   │   ├── test_rocfft_benchmark.py
│   │   ├── test_rocrand_benchmark.py
│   │   └── test_rocsolver_benchmark.py
│   │
│   ├── configs/                # Benchmark configs
│   │   ├── config.yml          # Framework configuration
│   │   ├── hipblaslt.json      # hipBLASLt benchmark config
│   │   ├── rccl.json           # RCCL benchmark config
│   │   ├── rocblas.json        # rocBLAS benchmark config
│   │   └── rocfft.json         # ROCfft benchmark config
│   │
│   ├── utils/                  # Benchmark utilities
│   │   ├── benchmark_client.py # Main client API
│   │   ├── logger.py           # Logging utilities
│   │   ├── config/             # Configuration management
│   │   ├── system/             # System detection (GPU, ROCm, OS)
│   │   └── results/            # Results API client & schemas
│   │
│   ├── benchmark_test_matrix.py  # Benchmark matrix definitions
│   └── README.md               # This file
│
├── test_executable_scripts/    # Regular functional tests
│
├── configure_ci.py             # CI workflow orchestration
├── fetch_test_configurations.py  # Test matrix builder
└── github_actions_utils.py     # GitHub Actions utilities
```

## CI/CD Integration

### When Benchmark Tests Run

Benchmark tests run **only on nightly CI builds** to save time and resources on pull request validation:

| Workflow Trigger           | Benchmark Tests                | Regular Tests          |
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

### Available Benchmark Tests in CI

The following benchmark tests are defined in `benchmarks/benchmark_test_matrix.py`:

| Test Name         | Library   | Platform       | Timeout | Shards |
| ----------------- | --------- | -------------- | ------- | ------ |
| `hipblaslt_bench` | hipBLASLt | Linux, Windows | 60 min  | 1      |
| `rccl_bench`      | RCCL      | Linux          | 60 min  | 1      |
| `rocblas_bench`   | rocBLAS   | Linux          | 60 min  | 1      |
| `rocfft_bench`    | ROCfft    | Linux, Windows | 60 min  | 1      |
| `rocrand_bench`   | ROCrand   | Linux, Windows | 60 min  | 1      |
| `rocsolver_bench` | ROCsolver | Linux, Windows | 60 min  | 1      |

**GPU Family Support:**

| GPU Family | Platform | Architecture          | Benchmark Supported | Benchmark CI Status  |
| ---------- | -------- | --------------------- | ------------------- | -------------------- |
| `gfx94x`   | Linux    | MI300X/MI325X (CDNA3) | Yes                 | Enabled (nightly CI) |
| `gfx1151`  | Windows  | RDNA 3.5              | Yes                 | Enabled (nightly CI) |
| `gfx950`   | Linux    | MI355X (CDNA4)        | Yes                 | Not enabled          |
| `gfx110x`  | Windows  | RDNA 2                | Yes                 | Not enabled          |
| `gfx110x`  | Linux    | RDNA 2                | Yes                 | Not enabled          |
| `gfx120x`  | Linux    | RDNA 3                | Yes                 | Not enabled          |
| `gfx120x`  | Windows  | RDNA 3                | Yes                 | Not enabled          |
| `gfx90x`   | Linux    | MI200 (CDNA2)         | Yes                 | Not enabled          |
| `gfx1151`  | Linux    | RDNA 3.5              | Yes                 | Not enabled          |

> **Note:** All benchmarks are **architecture-agnostic** and support any ROCm-compatible GPU. The table above lists GPU families actively used in CI testing. To add support for additional GPU families, update [`amdgpu_family_matrix.py`](../amdgpu_family_matrix.py) with appropriate `benchmark-runs-on` runners.

### Implementation Details

1. **Nightly Trigger:** `configure_ci.py` adds benchmark test names to test labels
1. **Parallel Jobs:** `ci_linux.yml` spawns two parallel jobs:
   - `test_artifacts` → Regular tests via `test_artifacts.yml`
   - `test_benchmarks` → Benchmarks via `test_benchmarks.yml`
1. **Matrix Generation:** `fetch_test_configurations.py` uses `IS_BENCHMARK_WORKFLOW=true` flag to select only benchmarks from `benchmark_test_matrix.py`
1. **Dedicated Runners:** Benchmarks can use dedicated GPU runners specified by `benchmark-runs-on` in `amdgpu_family_matrix.py`

## Architecture

### Workflow Integration

```
.github/workflows/ci_nightly.yml
  └─ calls → ci_linux.yml
              ├─ job: build_artifacts
              ├─ job: test_artifacts (parallel)
              └─ job: test_benchmarks (parallel) ← NEW
                    └─ calls → test_benchmarks.yml
                                ├─ configure_benchmark_matrix
                                │   └─ fetch_test_configurations.py
                                │      (IS_BENCHMARK_WORKFLOW=true)
                                └─ run_benchmarks
                                    └─ test_component.yml (matrix)
```

### Benchmark Script Execution Flow

```
1. Initialize BenchmarkClient
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

Create `benchmarks/scripts/test_your_benchmark.py`. Reference existing benchmarks like `test_rocfft_benchmark.py` as a template.

Key components:

- Inherit from `BenchmarkBase` class
- Implement `run_benchmarks()` - executes binary and logs output
- Implement `parse_results()` - parses logs and returns structured data
- Results are automatically uploaded to API via base class

Example:

```python
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class YourBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(benchmark_name="your_lib", display_name="YourLib")
        self.log_file = self.script_dir / "your_lib_bench.log"

    def run_benchmarks(self) -> None:
        """Execute benchmark binary and log output."""
        # Load config if needed
        config_file = self.script_dir.parent / "configs" / "your_lib.json"

        # Your benchmark execution logic here
        pass

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse log file and return (test_results, table)."""
        # Your parsing logic here
        # Use self.create_test_result() to build result dictionaries
        pass


if __name__ == "__main__":
    run_benchmark_main(YourBenchmark())  # Handles sys.exit() internally
```

### 2. Add to Benchmark Test Matrix

Edit `benchmarks/benchmark_test_matrix.py`:

```python
"your_benchmark": {
    "job_name": "your_benchmark",
    "fetch_artifact_args": "--your-lib --tests",
    "timeout_minutes": 60,
    "test_script": f"python {_get_benchmark_script_path('test_your_benchmark.py')}",
    "platform": ["linux", "windows"],  # Supported platforms
    "total_shards": 1,
    # TODO: Remove xfail once dedicated performance servers are added
    "expect_failure": True,
},
```

The benchmark will automatically be included in nightly CI runs:

- `configure_ci.py` adds benchmark names to test labels
- `ci_linux.yml` spawns `test_benchmarks` job
- `test_benchmarks.yml` calls `fetch_test_configurations.py` with `IS_BENCHMARK_WORKFLOW=true`
- Only benchmarks from `benchmark_test_matrix.py` are executed

### 3. Test Locally

```bash
# Set environment variables
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

# Run the benchmark
python3 build_tools/github_actions/benchmarks/scripts/test_your_benchmark.py
```

## Related Documentation

- [Utils Module Documentation](utils/README.md) - Utility modules reference
- [CI Nightly Workflow](https://github.com/ROCm/TheRock/actions/workflows/ci_nightly.yml) - GitHub Actions
- [Test Benchmarks Workflow](../../.github/workflows/test_benchmarks.yml) - Benchmark execution workflow
