# Benchmark Tests

Regression detection tests that compare current performance against Last Known Good (LKG) baselines.

## Purpose

Benchmark tests detect **performance regressions** by comparing test results against established baselines:

- **Result:** PASS (within tolerance) / FAIL (regression detected) / UNKNOWN (no baseline)
- **Frequency:** Every nightly CI
- **Use Case:** Automated CI gates to prevent performance degradation

## Quick Start

### Available Benchmark Tests

| Test Script                   | Library   | Description                             |
| ----------------------------- | --------- | --------------------------------------- |
| `test_hipblaslt_benchmark.py` | hipBLASLt | Matrix multiplication benchmarks (GEMM) |
| `test_rocfft_benchmark.py`    | rocFFT    | FFT benchmarks (1D, 2D, 3D transforms)  |
| `test_rocrand_benchmark.py`   | rocRAND   | Random number generation benchmarks     |
| `test_rocsolver_benchmark.py` | ROCsolver | Dense linear algebra benchmarks         |

### Running Locally

```bash
# Set required environment variables
export THEROCK_BIN_DIR=/path/to/therock/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

# Run a benchmark
cd build_tools/github_actions/test_framework/benchmark/scripts
python test_hipblaslt_benchmark.py
```

## Directory Structure

```
benchmark/
├── scripts/                    # Benchmark implementations
│   ├── benchmark_base.py       # Base class (LKG comparison logic)
│   ├── test_hipblaslt_benchmark.py
│   ├── test_rocfft_benchmark.py
│   ├── test_rocrand_benchmark.py
│   └── test_rocsolver_benchmark.py
│
├── configs/                    # Benchmark-specific test configurations
│   ├── hipblaslt.json         # Test parameters (matrix sizes, precisions)
│   └── rocfft.json            # Test parameters (FFT sizes, dimensions)
│
├── benchmark_matrix.py         # CI test matrix definitions
└── README.md                   # This file
```

## Benchmark Test Matrix

Tests defined in `benchmark_matrix.py` for nightly CI:

| Test Name         | Library   | Platform | Timeout | Artifacts Needed       |
| ----------------- | --------- | -------- | ------- | ---------------------- |
| `hipblaslt_bench` | hipBLASLt | Linux    | 60 min  | `--blas --tests`       |
| `rocfft_bench`    | rocFFT    | Linux    | 60 min  | `--fft --rand --tests` |
| `rocrand_bench`   | rocRAND   | Linux    | 60 min  | `--rand --tests`       |
| `rocsolver_bench` | ROCsolver | Linux    | 60 min  | `--blas --tests`       |

## How Benchmark Tests Work

### 1. Execution Flow

```
┌─────────────────────────────────────────┐
│ 1. Initialize Benchmark                 │
│    - Auto-detect GPU, ROCm version      │
│    - Load configuration                 │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Run Benchmark Binary                 │
│    - Execute (e.g., rocblas-bench)      │
│    - Capture output to log file         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. Parse Results                        │
│    - Extract metrics from logs          │
│    - Structure as test results          │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. Compare with LKG Baseline            │
│    - Fetch last known good results      │
│    - Calculate performance delta        │
│    - Determine: PASS/FAIL/UNKNOWN       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. Report Results                       │
│    - Display formatted table            │
│    - Upload to results API              │
│    - Return exit code (0=pass, 1=fail) │
└─────────────────────────────────────────┘
```

### 2. LKG Comparison

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

### 3. Result Statuses

| Status      | Meaning                                  | Action                             |
| ----------- | ---------------------------------------- | ---------------------------------- |
| **PASS**    | Performance within tolerance of baseline | ✅ CI passes                       |
| **FAIL**    | Performance degraded beyond tolerance    | ❌ CI fails, blocks merge          |
| **UNKNOWN** | No baseline data available (new test)    | ⚠️ CI passes, baseline established |

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

Edit `benchmark_matrix.py`:

```python
"yourlib_bench": {
    "job_name": "yourlib_bench",
    "fetch_artifact_args": "--yourlib --tests",
    "timeout_minutes": 60,
    "test_script": f"python {_get_benchmark_script_path('test_yourlib_benchmark.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

### Step 3: Test Locally

```bash
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

python scripts/test_yourlib_benchmark.py
```

## CI Integration

Benchmark tests automatically run in nightly CI:

1. **Trigger:** Nightly scheduled run
1. **Execution:** Parallel with regular tests
1. **Matrix:** Generated from `benchmark_matrix.py`
1. **Runners:** Dedicated GPU runners (if configured)
1. **Results:** Uploaded to results API, compared with LKG

See [main test framework README](../README.md) for full CI architecture.

## Related Documentation

- [Test Framework README](../README.md) - Main framework documentation
- [Shared Utils](../utils/README.md) - Utility modules reference
- [Performance Tests](../performance/) - Performance characterization tests
- [Functional Tests](../functional/) - Correctness validation tests
