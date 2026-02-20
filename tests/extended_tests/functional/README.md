# Functional Testing Framework

Functional tests validate correctness and verify expected behavior without performance measurements.

> **Prerequisites:** See [Extended Tests Framework Overview](../README.md) for environment setup, CI/CD integration, and general architecture.

## Table of Contents

- [Available Tests](#available-tests)
- [CI Configuration](#ci-configuration)
- [Project Structure](#project-structure)
- [How Functional Tests Work](#how-functional-tests-work)
- [Adding New Functional Tests](#adding-new-functional-tests)

## Available Tests

| Test Script                 | Library | Platform | Timeout | Description                        |
| --------------------------- | ------- | -------- | ------- | ---------------------------------- |
| `test_miopendriver_conv.py` | MIOpen  | Linux    | 30 min  | Convolution forward/backward tests |

## CI Configuration

- **Test Matrix:** See [`functional_test_matrix.py`](functional_test_matrix.py) for complete test definitions (platforms, timeouts, artifacts needed)
- **Execution:** All functional tests run in nightly CI builds only
- **Architecture Support:** Tests are architecture-agnostic and support any ROCm-compatible GPU
- **Architecture Exclusions:** If a specific GPU architecture is not supported for a test, use the `exclude_family` field in the test matrix to skip that architecture/platform combination (see `fetch_test_configurations.py` for filtering logic)

## Project Structure

```
functional/
├── scripts/
│   ├── functional_base.py        # Base class for all functional tests
│   └── test_miopendriver_conv.py # MIOpen convolution test
│
├── configs/
│   └── miopendriver_conv.json    # Test configuration
│
├── functional_test_matrix.py     # CI test matrix definitions
└── README.md                     # This file
```

## How Functional Tests Work

### Result Tables

Functional tests generate two tables:

**Detailed Table:** One row per test case

```
+--------------+--------------------+--------+
| TestSuite    | TestCase           | Status |
+--------------+--------------------+--------+
| Forward_Conv | Forward_Conv_case1 | PASS   |
| Forward_Conv | Forward_Conv_case2 | PASS   |
+--------------+--------------------+--------+
```

**Summary Table:** Overall statistics

```
+-------------------+-------------------+--------+--------+---------+---------+-------------------+
| Total TestSuites  | Total TestCases   | Passed | Failed | Errored | Skipped |   Final Result    |
+-------------------+-------------------+--------+--------+---------+---------+-------------------+
|         2         |         18        |   18   |   0    |    0    |    0    |        PASS       |
+-------------------+-------------------+--------+--------+---------+---------+-------------------+
```

## Adding New Functional Tests

### 1. Create Test Script

Create `scripts/test_your_test.py`:

```python
"""YourTest Functional Test"""

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class YourTest(FunctionalBase):
    """YourTest functional test."""

    def __init__(self):
        super().__init__(test_name="your_test", display_name="YourTest Functional")
        self.results_json = self.script_dir / "your_test_results.json"

        # Load test configuration from JSON
        config = self.load_config("your_test.json")
        self.test_cases = config.get("test_cases", [])

    def run_tests(self) -> None:
        """Run functional tests and save results to JSON."""
        log.info(f"Running {self.display_name}")

        # Optional: Get GPU architecture for GPU-specific behavior
        gfx_id = self.get_gpu_architecture()

        all_results = []
        for test_case in self.test_cases:
            # Execute test, capture result
            status = "PASS"  # or "FAIL" based on return code
            all_results.append(
                {
                    "test_case": test_case["name"],
                    "status": status,
                }
            )

        # Save results to JSON
        with open(self.results_json, "w") as f:
            json.dump(all_results, f, indent=2)

    def parse_results(self) -> List[Dict[str, Any]]:
        """Parse results and return test_results list."""
        log.info("Parsing Results")

        test_results = []

        with open(self.results_json, "r") as f:
            json_results = json.load(f)

        for result in json_results:
            test_results.append(
                self.create_test_result(
                    test_name=self.test_name,
                    subtest_name=result["test_case"],
                    status=result["status"],
                    suite=result.get("test_suite", "default"),
                )
            )

        return test_results


if __name__ == "__main__":
    run_functional_main(YourTest())
```

**Required Methods:**

- `run_tests()` → Execute tests and save results to JSON
- `parse_results()` → Returns `List[Dict]` of test results
  - Must use `self.create_test_result(test_name, subtest_name, status, suite, **kwargs)`
  - Status must be: `"PASS"`, `"FAIL"`, `"ERROR"`, or `"SKIP"`
  - Base class generates detailed table and calculates num_suites automatically

**Available Helper Methods (from FunctionalBase):**

- `self.load_config(filename)` → Load JSON config from `configs/` directory
- `self.get_gpu_architecture()` → Get GPU gfx version (e.g., 'gfx942')
- `self.execute_command(cmd, cwd, env, log_file_handle)` → Execute command with streaming output
- `self.create_test_result(...)` → Create standardized result dictionary

### 2. Create Configuration File

Create `configs/your_test.json`:

```json
{
  "description": "YourTest Configuration",
  "test_cases": [
    {"name": "test_case_1", "command": "..."},
    {"name": "test_case_2", "command": "..."}
  ],
  "gpu_specific_flags": {
    "gfx906": {"flags": "-V 0"}
  }
}
```

### 3. Add to Test Matrix

Edit `functional_test_matrix.py`:

```python
"test_your_test": {
    "job_name": "test_your_test",
    "fetch_artifact_args": "--your-lib --tests",
    "timeout_minutes": 30,
    "test_script": f"python {_get_functional_script_path('test_your_test.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

### 4. Test Locally

> **Full Setup Guide:** See [Extended Tests Framework Overview](../README.md#quick-start) for complete environment setup, including how to obtain TheRock build artifacts.

```bash
# Required environment variables
export THEROCK_BIN_DIR=/path/to/therock/build/bin  # Path to TheRock build output
export AMDGPU_FAMILIES=gfx94X-dcgpu                # Your GPU family (gfx942, gfx1100, etc.)
export ARTIFACT_RUN_ID=local-test                  # Optional: test run identifier

# Run the test
python3 tests/extended_tests/functional/scripts/test_your_test.py
```
