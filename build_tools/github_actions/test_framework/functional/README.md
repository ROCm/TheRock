# Functional Tests

Correctness validation tests that verify expected behavior without performance measurements.

> **Prerequisites:** See [Test Framework Overview](../README.md) for environment setup and general architecture.

## Overview

Functional tests validate **correctness and behavior**:

- **Result Types:** PASS / FAIL / ERROR / SKIP
- **Validation:** Output correctness, API contracts, expected behavior
- **CI Execution:** Nightly CI and optionally on PRs
- **Exit Code:** Non-zero if any test FAILS or has ERRORs

## Available Tests

| Test Script                 | Library | Description                                    |
| --------------------------- | ------- | ---------------------------------------------- |
| `test_miopendriver_conv.py` | MIOpen  | Convolution forward/backward correctness tests |

## Quick Start

```bash
# Run a functional test (environment variables from main README required)
python build_tools/github_actions/test_framework/functional/scripts/test_miopendriver_conv.py
```

## CI Test Matrix

Tests defined in `functional_test_matrix.py`:

| Test Name            | Library | Platform | Timeout | Artifacts Needed   | CI Status         |
| -------------------- | ------- | -------- | ------- | ------------------ | ----------------- |
| `miopen_driver_conv` | MIOpen  | Linux    | 30 min  | `--miopen --tests` | Enabled (nightly) |

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

### Step 1: Create Test Script

Create `test_yourtest.py` in `scripts/`:

```python
"""YourTest Functional Test"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # test_framework/
sys.path.insert(0, str(Path(__file__).parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_test_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class YourTest(FunctionalBase):
    """YourTest functional test."""

    def __init__(self):
        super().__init__(
            test_name="yourtest_functional", display_name="YourTest Functional Test"
        )
        self.log_file = self.script_dir / "yourtest_functional.log"

        # Load test configuration from JSON
        config = self.load_config("yourtest_functional.json")
        self.test_cases = config.get("test_cases", [])

    def run_tests(self) -> None:
        """Run functional tests and save output to log file."""
        log.info(f"Running {self.display_name}")

        with open(self.log_file, "w+") as f:
            for test_case in self.test_cases:
                cmd = test_case["command"]
                # Execute command, capture output
                # Write to log file
                pass

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable, int]:
        """Parse log file and return (detailed_table, num_suites)."""
        log.info("Parsing Results")

        detailed_table = PrettyTable()
        detailed_table.field_names = ["TestCase", "Status"]

        test_results = []

        # Parse log file for each test case
        # Use self.create_test_result() to create result dictionaries

        return test_results, detailed_table, num_suites


if __name__ == "__main__":
    run_functional_test_main(YourTestFunctionalTest())
```

### Step 2: Add to Functional Matrix

Edit `functional_test_matrix.py`:

```python
"yourtest_name": {
    "job_name": "yourtest_name",
    "fetch_artifact_args": "--yourtest --tests",
    "timeout_minutes": 30,
    "test_script": f"python {_get_script_path('test_yourtest.py')}",
    "platform": ["linux"],
    "total_shards": 1,
},
```

### Step 3: Test Locally

```bash
export THEROCK_BIN_DIR=/path/to/build/bin
export ARTIFACT_RUN_ID=local-test
export AMDGPU_FAMILIES=gfx94x-dcgpu

python scripts/test_yourtest.py
```

## Configuration

- **Test Matrix:** `functional_test_matrix.py` - CI test definitions
- **Test Parameters:** `configs/*.json` - Test-specific parameters and configurations

## See Also

- [Test Framework Overview](../README.md) - Environment setup, CI/CD architecture
- [Benchmark Tests](../benchmark/README.md) - Performance regression tests
