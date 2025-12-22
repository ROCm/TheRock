# Test Framework

Automated testing framework for ROCm libraries supporting both **performance** and **functional** tests with system detection, results collection, and performance tracking.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD Integration](#cicd-integration)
- [Architecture](#architecture)
- [Adding New Tests](#adding-new-tests)

## Features

- **Dual Test Types** - Performance (benchmarks) and Functional tests
- **Automated Test Execution** - ROCfft, ROCrand, ROCsolver, hipBLASLt
- **System Auto-Detection** - Hardware, OS, GPU, and ROCm version detection
- **Results Management** - Local storage (JSON) and API upload with retry logic
- **Performance Tracking** - LKG (Last Known Good) comparison
- **Comprehensive Logging** - File rotation and configurable log levels
- **Modular Architecture** - Extensible design for adding new tests
- **CI/CD Integration** - Parallel execution with regular tests in nightly CI

## Quick Start

### Available Performance Tests

- `scripts/performance/test_hipblaslt_perf.py` - hipBLASLt performance suite
- `scripts/performance/test_rocsolver_perf.py` - ROCsolver performance suite
- `scripts/performance/test_rocrand_perf.py` - ROCrand performance suite
- `scripts/performance/test_rocfft_perf.py` - ROCfft performance suite

### Available Functional Tests

Functional tests can be added in `scripts/functional/` directory.

## Project Structure

```
build_tools/github_actions/
├── test_framework/              # Test framework (was: benchmarks/)
│   ├── scripts/                 # Test implementations
│   │   ├── performance/                # Performance tests
│   │   │   ├── perf_base.py           # Performance test base class
│   │   │   ├── test_hipblaslt_perf.py
│   │   │   ├── test_rocsolver_perf.py
│   │   │   ├── test_rocrand_perf.py
│   │   │   └── test_rocfft_perf.py
│   │   │
│   │   └── functional/          # Functional tests
│   │       ├── functional_base.py     # Functional test base class
│   │       └── (your functional tests here)
│   │
│   ├── configs/                 # Test configurations
│   │   ├── config.yml           # Framework configuration
│   │   ├── performance/                # Performance test configs
│   │   │   ├── hipblaslt.json
│   │   │   └── rocfft.json
│   │   └── functional/          # Functional test configs
│   │
│   ├── utils/                   # Test utilities
│   │   ├── test_client.py       # Main client API (was: benchmark_client.py)
│   │   ├── logger.py            # Logging utilities
│   │   ├── config/              # Configuration management
│   │   ├── system/              # System detection (GPU, ROCm, OS)
│   │   └── results/             # Results API client & schemas
│   │
│   ├── performance_test_matrix.py      # Performance test matrix (was: benchmark_test_matrix.py)
│   ├── functional_test_matrix.py # Functional test matrix
│   └── README.md                # This file
│
├── test_executable_scripts/     # Regular functional tests
├── configure_ci.py               # CI workflow orchestration
├── fetch_test_configurations.py # Test matrix builder
└── github_actions_utils.py      # GitHub Actions utilities
```

## CI/CD Integration

### When Performance Tests Run

Performance tests run **only on nightly CI builds** to save time and resources on pull request validation:

| Workflow Trigger           | Performance Tests              | Regular Tests          |
| -------------------------- | ------------------------------ | ---------------------- |
| **Pull Request (PR)**      | Skipped                        | Run (smoke: 1 shard)   |
| **Nightly CI (scheduled)** | Run (in parallel, always full) | Run (full: all shards) |
| **Push to main**           | Skipped                        | Run (smoke: 1 shard)   |
| **Manual workflow**        | Optional                       | Optional               |

**Note:** Performance tests always run with `total_shards=1` and do not use `test_type` or `test_labels` filtering.

### Parallel Execution Architecture

Performance tests run **in parallel** with regular tests for faster CI execution:

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
                        └─ Performance tests
                           (hipblaslt_perf, rocfft_perf, ...)
```

### Available Performance Tests in CI

The following performance tests are defined in `performance_test_matrix.py`:

| Test Name         | Library   | Platform | Timeout | Shards |
| ----------------- | --------- | -------- | ------- | ------ |
| `hipblaslt_perf`  | hipBLASLt | Linux    | 60 min  | 1      |
| `rocsolver_perf`  | ROCsolver | Linux    | 60 min  | 1      |
| `rocrand_perf`    | ROCrand   | Linux    | 60 min  | 1      |
| `rocfft_perf`     | ROCfft    | Linux    | 60 min  | 1      |

### Implementation Details

1. **Nightly Trigger:** `configure_ci.py` adds performance test names to test labels
1. **Parallel Jobs:** `ci_linux.yml` spawns two parallel jobs:
   - `test_artifacts` → Regular tests via `test_artifacts.yml`
   - `test_benchmarks` → Performance tests via `test_benchmarks.yml`
1. **Matrix Generation:** `fetch_test_configurations.py` uses `IS_BENCHMARK_WORKFLOW=true` flag to select only performance tests from `performance_test_matrix.py`
1. **Dedicated Runners:** Performance tests can use dedicated GPU runners specified by `benchmark-runs-on` in `amdgpu_family_matrix.py`

## Architecture

### Workflow Integration

```
.github/workflows/ci_nightly.yml
  └─ calls → ci_linux.yml
              ├─ job: build_artifacts
              ├─ job: test_artifacts (parallel)
              └─ job: test_benchmarks (parallel)
                    └─ calls → test_benchmarks.yml
                                ├─ configure_benchmark_matrix
                                │   └─ fetch_test_configurations.py
                                │      (IS_BENCHMARK_WORKFLOW=true)
                                └─ run_benchmarks
                                    └─ test_component.yml (matrix)
```

### Test Execution Flow

```
1. Initialize TestClient
   ↓ Auto-detect system (GPU, OS, ROCm version)
   ↓ Load configuration from config.yml

2. Run Tests
   ↓ Execute test binary/script
   ↓ Capture output to log file

3. Parse Results
   ↓ Extract metrics from log file
   ↓ Structure data according to schema

4. Upload Results (Performance tests)
   ↓ Submit to API (with retry)
   ↓ Save JSON locally

5. Compare with LKG (Performance tests)
   ↓ Fetch last known good results
   ↓ Calculate performance delta

6. Report Results
   ↓ Display formatted table
   ↓ Append to GitHub Actions step summary
   ↓ Return exit code (0=success, 1=failure)
```

## Adding New Tests

### Adding a Performance Test

#### 1. Create Performance Test Script

Create `scripts/performance/test_your_perf.py`. Reference existing tests like `test_rocfft_perf.py` as a template.

Key components:

- Inherit from `PerfBase` class
- Implement `run_benchmarks()` - executes binary and logs output
- Implement `parse_results()` - parses logs and returns structured data
- Results are automatically uploaded to API via base class

Example:

```python
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # test_framework/
sys.path.insert(0, str(Path(__file__).parent))  # performance/
from perf_base import PerfBase, run_test_main
from utils.logger import log


class YourLibPerf(PerfBase):
    def __init__(self):
        super().__init__(test_name="your_lib", display_name="YourLib", test_type="performance")
        self.log_file = self.script_dir / "your_lib_perf.log"

    def run_benchmarks(self) -> None:
        """Execute performance test and log output."""
        # Load config if needed
        config_file = self.script_dir.parent.parent / "configs" / "perf" / "your_lib.json"
        
        # Your test execution logic here
        pass

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse log file and return (test_results, table)."""
        # Use self.create_test_result() to build result dictionaries
        pass


if __name__ == "__main__":
    run_test_main(YourLibPerf())
```

#### 2. Add to Performance Test Matrix

Edit `performance_test_matrix.py`:

```python
"your_lib_perf": {
    "job_name": "your_lib_perf",
    "fetch_artifact_args": "--your-lib --tests",
    "timeout_minutes": 60,
    "test_script": f"python {_get_performance_script_path('test_your_lib_perf.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

The performance test will automatically be included in nightly CI runs.

#### 3. Test Locally

```bash
# Set environment variables
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx950-dcgpu

# Run the test
python3 build_tools/github_actions/test_framework/scripts/performance/test_your_lib_perf.py
```

### Adding a Functional Test

#### 1. Create Functional Test Script

Create `scripts/functional/test_your_functional.py`:

```python
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For test_framework/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # For github_actions/
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))  # For perf_base
from functional_base import FunctionalBase
from perf_base import run_test_main
from utils.logger import log


class YourLibFunctional(FunctionalBase):
    def __init__(self):
        super().__init__(test_name="your_lib", display_name="YourLib")

    def run_benchmarks(self) -> None:
        """Execute functional tests."""
        # Your functional test logic
        pass

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse functional test results."""
        # Your result parsing logic
        pass


if __name__ == "__main__":
    run_test_main(YourLibFunctional())
```

#### 2. Add to Functional Test Matrix

Edit `functional_test_matrix.py`:

```python
"your_lib_functional": {
    "job_name": "your_lib_functional",
    "fetch_artifact_args": "--your-lib --tests",
    "timeout_minutes": 30,
    "test_script": f"python {_get_functional_script_path('test_your_lib_functional.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

## Migration Notes

### Key Changes from Old Structure

| Old Name | New Name | Type |
|----------|----------|------|
| `benchmarks/` | `test_framework/` | Directory |
| `benchmark_test_matrix.py` | `performance_test_matrix.py` | File |
| `BenchmarkBase` | `PerfBase` | Class |
| `BenchmarkClient` | `TestClient` | Class |
| `test_*_benchmark.py` | `test_*_perf.py` | Scripts |
| `benchmarks/scripts/` | `test_framework/scripts/performance/` | Directory |
| `benchmarks/configs/` | `test_framework/configs/performance/` | Directory |
| `benchmark_base.py` | `performance/perf_base.py` | File |
| `test_base.py` | `performance/perf_base.py` | File |
| `*_performance` (jobs) | `*_perf` (jobs) | Job names |
| `*Performance` (classes) | `*Perf` (classes) | Class names |

## Related Documentation

- [Utils Module Documentation](utils/README.md) - Utility modules reference
- [CI Nightly Workflow](https://github.com/ROCm/TheRock/actions/workflows/ci_nightly.yml) - GitHub Actions
- [Test Benchmarks Workflow](../../.github/workflows/test_benchmarks.yml) - Performance test execution workflow
