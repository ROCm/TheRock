# Benchmark Tests

Performance regression detection tests that compare current results against Last Known Good (LKG) baselines.

> **Prerequisites:** See [Extended Tests Framework Overview](../README.md) for environment setup and general architecture.

## Overview

Benchmark tests detect **performance regressions** by comparing against baselines:

- **Result Types:** PASS (within tolerance) / FAIL (regression) / UNKNOWN (no baseline)
- **Comparison:** Current performance vs. LKG baseline with configurable tolerance
- **CI Execution:** Nightly only (not on PRs to save resources)
- **Exit Code:** Non-zero if any test FAILS

## Available Tests

| Test Script                   | Library   | Description                             |
| ----------------------------- | --------- | --------------------------------------- |
| `test_hipblaslt_benchmark.py` | hipBLASLt | Matrix multiplication benchmarks (GEMM) |
| `test_rocfft_benchmark.py`    | rocFFT    | FFT benchmarks (1D, 2D, 3D transforms)  |
| `test_rocrand_benchmark.py`   | rocRAND   | Random number generation benchmarks     |
| `test_rocsolver_benchmark.py` | ROCsolver | Dense linear algebra benchmarks         |

## Quick Start

```bash
# Run a benchmark (environment variables from main README required)
cd build_tools/github_actions/extended_tests/benchmark/scripts
python test_hipblaslt_benchmark.py
```

## CI Test Matrix

Tests defined in `benchmark_test_matrix.py`:

| Test Name         | Library   | Platform       | Timeout | Artifacts Needed       | CI Status         |
| ----------------- | --------- | -------------- | ------- | ---------------------- | ----------------- |
| `hipblaslt_bench` | hipBLASLt | Linux, Windows | 60 min  | `--blas --tests`       | Enabled (nightly) |
| `rocfft_bench`    | rocFFT    | Linux, Windows | 60 min  | `--fft --rand --tests` | Enabled (nightly) |
| `rocrand_bench`   | rocRAND   | Linux, Windows | 60 min  | `--rand --tests`       | Enabled (nightly) |
| `rocsolver_bench` | ROCsolver | Linux, Windows | 60 min  | `--blas --tests`       | Enabled (nightly) |

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

## How Benchmark Tests Work

### LKG (Last Known Good) Comparison

```python
# Pseudocode
current_score = 42.5  # TFLOPS
lkg_baseline = 45.0  # TFLOPS from last successful run
tolerance = 0.05  # 5% tolerance

if current_score < lkg_baseline * (1 - tolerance):
    result = "FAIL"  # Performance regression!
elif lkg_baseline is None:
    result = "UNKNOWN"  # No baseline data yet
else:
    result = "PASS"  # Performance acceptable
```

### Result Statuses

| Status      | Meaning                                  | Action                          |
| ----------- | ---------------------------------------- | ------------------------------- |
| **PASS**    | Performance within tolerance of baseline | CI passes                       |
| **FAIL**    | Performance degraded beyond tolerance    | CI fails, blocks merge          |
| **UNKNOWN** | No baseline data available (new test)    | CI passes, baseline established |

## Adding a New Benchmark

### Step 1: Create Benchmark Script

Create `scripts/test_yourlib_benchmark.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base

from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class YourLibBenchmark(BenchmarkBase):
    """Benchmark tests for YourLib library."""

    def __init__(self):
        super().__init__(benchmark_name="yourlib", display_name="YourLib")
        self.benchmark_bin = "yourlib-bench"
        self.log_file = self.script_dir / "yourlib_bench.log"

    def run_benchmarks(self) -> None:
        """Execute benchmark binary and log output."""
        # Your benchmark execution logic
        # Use self.run_command() to execute binaries
        pass

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse log file and return (test_results, results_table)."""
        test_results = []
        table = PrettyTable()

        # Your parsing logic
        # Use self.create_test_result() to build result dictionaries

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(YourLibBenchmark())
```

### Step 2: Add to Benchmark Matrix

Edit `benchmark_test_matrix.py`:

```python
"yourlib_bench": {
    "job_name": "yourlib_bench",
    "fetch_artifact_args": "--yourlib --tests",
    "timeout_minutes": 60,
    "test_script": f"python {_get_benchmark_script_path('test_yourlib_benchmark.py')}",
    "platform": ["linux", "windows"],  # Supported platforms
    "total_shards": 1,
    # TODO: Remove xfail once dedicated performance servers are added
    "expect_failure": True,
},
```

### Step 3: Test Locally

```bash
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

python scripts/test_yourlib_benchmark.py
```

## Configuration

- **Test Matrix:** `benchmark_test_matrix.py` - CI test definitions
- **Test Parameters:** `configs/*.json` - Benchmark-specific parameters (sizes, precisions, etc.)
- **Performance Tolerance:** Default 5% degradation threshold (configurable per test)

## See Also

- [Extended Tests Framework Overview](../README.md) - Environment setup, CI/CD architecture
- [Functional Tests](../functional/README.md) - Correctness validation tests
