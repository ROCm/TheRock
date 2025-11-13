# ROCm Test Kit - Usage Examples

This document provides real-world usage examples for the ROCm Component Test Kit.

## Basic Examples

### Example 1: Quick Health Check (5 minutes)
**Use Case**: You just installed ROCm and want to verify it's working.

```bash
cd rocm_test_kit
./run_tests.sh --quick
```

**What it does**: Tests 5 essential components (rocBLAS, hipBLAS, rocSOLVER, rocPRIM, MIOpen)

**Expected output**:
```
Configuration:
  Preset:     quick
  Test Type:  smoke
Running rocblas (smoke tests)...
✓ rocblas passed (25.3s)
...
Success Rate: 100.0%
```

---

### Example 2: Test Specific Components
**Use Case**: You're debugging an issue with BLAS libraries.

```bash
cd rocm_test_kit
python3 test_runner.py --components rocblas hipblas hipblaslt --verbose
```

**What it does**: Tests only the three BLAS libraries with detailed output

---

### Example 3: Test All Deep Learning Components
**Use Case**: You're setting up a machine learning workstation.

```bash
cd rocm_test_kit
python3 test_runner.py --category deep_learning --log-dir ./dl_test_logs
```

**What it does**: Tests MIOpen, MIOpen Plugin, and hipDNN, saving logs to `./dl_test_logs/`

---

### Example 4: Full Validation (2-4 hours)
**Use Case**: Production system validation before deployment.

```bash
cd rocm_test_kit
./run_tests.sh --full --parallel --log-dir ./prod_validation --with-report
```

**What it does**:
- Tests all 21 components
- Runs full test suites (not just smoke tests)
- Uses parallel execution for speed
- Saves detailed logs
- Generates HTML report

---

### Example 5: Hardware Check
**Use Case**: Verify you have MI300/MI350 hardware.

```bash
cd rocm_test_kit
./run_tests.sh --check-hardware
```

**Expected output**:
```
Hardware Detection Results
GPU Count: 8
MI300 Series: True
MI350 Series: False
Compatible: True
  GPU 0: AMD Instinct MI300X (gfx942)
  GPU 1: AMD Instinct MI300X (gfx942)
  ...
✓ Compatible hardware detected!
```

---

## Advanced Examples

### Example 6: CI/CD Integration
**Use Case**: Automated testing in GitHub Actions.

```yaml
# .github/workflows/rocm-tests.yml
name: ROCm Component Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: [self-hosted, mi300]
    steps:
      - uses: actions/checkout@v3

      - name: Run ROCm Tests
        run: |
          cd rocm_test_kit
          ./run_tests.sh --core --parallel --log-dir ./test_logs

      - name: Upload Logs
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-logs
          path: rocm_test_kit/test_logs/
```

---

### Example 7: Category Testing
**Use Case**: Test all libraries in a category.

```bash
# Test all BLAS libraries
python3 test_runner.py --category blas

# Test all sparse libraries
python3 test_runner.py --category sparse

# Test communication libraries
python3 test_runner.py --category communication
```

**Available categories**:
- `blas` - Basic Linear Algebra
- `solver` - LAPACK solvers
- `sparse` - Sparse matrix operations
- `primitives` - Parallel primitives
- `random` - Random number generation
- `fft` - Fast Fourier Transform
- `deep_learning` - Deep learning libraries
- `communication` - Multi-GPU communication

---

### Example 8: Parallel Testing with Custom Workers
**Use Case**: Optimize test speed for your hardware.

```bash
# Use all 8 GPUs
python3 test_runner.py --preset full --parallel --max-workers 8

# Conservative parallel (4 workers)
python3 test_runner.py --preset core --parallel --max-workers 4
```

---

### Example 9: Quick Pre-Commit Check
**Use Case**: Developers want to verify changes before committing.

```bash
#!/bin/bash
# scripts/pre-commit-rocm-check.sh

echo "Running ROCm quick tests..."
cd rocm_test_kit
./run_tests.sh --quick

if [ $? -eq 0 ]; then
    echo "✓ ROCm tests passed - ready to commit"
    exit 0
else
    echo "✗ ROCm tests failed - fix issues before committing"
    exit 1
fi
```

---

### Example 10: Generate Report for Documentation
**Use Case**: Create a test report to attach to a bug report or documentation.

```bash
# Run tests and generate comprehensive report
cd rocm_test_kit
python3 test_runner.py \
    --preset core \
    --test-type smoke \
    --log-dir ./report_$(date +%Y%m%d) \
    --verbose

# Then manually run report generator (if integrated)
python3 -c "
from report_generator import generate_html_report
from hardware_detector import detect_hardware
import json

# Load test results (example - replace with actual results)
# ... generate report
"
```

---

## Troubleshooting Examples

### Example 11: Debug a Failing Component
**Use Case**: rocBLAS tests are failing.

```bash
# Run only rocBLAS with verbose output and save logs
python3 test_runner.py \
    --components rocblas \
    --test-type full \
    --log-dir ./rocblas_debug \
    --verbose

# Check the log
cat ./rocblas_debug/rocblas_full.log
```

---

### Example 12: Test After ROCm Update
**Use Case**: You updated ROCm and want to verify everything still works.

```bash
# Before update - baseline test
./run_tests.sh --core --log-dir ./before_update

# After update - verification test
./run_tests.sh --core --log-dir ./after_update

# Compare results
diff ./before_update/ ./after_update/
```

---

## Scripting Examples

### Example 13: Automated Daily Testing

```bash
#!/bin/bash
# scripts/daily-rocm-test.sh

DATE=$(date +%Y%m%d)
LOG_DIR="./daily_tests/$DATE"

cd /path/to/TheRock/rocm_test_kit

echo "Starting daily ROCm test - $DATE"

./run_tests.sh \
    --core \
    --parallel \
    --log-dir "$LOG_DIR" \
    --with-report

# Email results (example)
if [ $? -eq 0 ]; then
    echo "All tests passed on $DATE" | mail -s "ROCm Tests: PASS" admin@example.com
else
    echo "Some tests failed on $DATE. Check logs at $LOG_DIR" | \
        mail -s "ROCm Tests: FAIL" admin@example.com
fi
```

---

### Example 14: Python Integration

```python
#!/usr/bin/env python3
"""Example: Programmatic test execution"""

import sys
sys.path.insert(0, '/path/to/TheRock/rocm_test_kit')

from test_runner import TestRunner
from hardware_detector import detect_hardware
from pathlib import Path

# Detect hardware
hw_info = detect_hardware()
print(f"Detected {hw_info.gpu_count} GPUs")

if not hw_info.compatible:
    print("Warning: No MI300/MI350 hardware detected")

# Initialize test runner
config_path = Path('/path/to/TheRock/rocm_test_kit/components.yaml')
runner = TestRunner(config_path)

# Run tests
summary = runner.run(
    preset='quick',
    test_type='smoke',
    parallel=True,
    log_dir=Path('./test_logs')
)

# Check results
if summary['failed'] > 0:
    print(f"⚠️  {summary['failed']} tests failed!")
    for result in summary['results']:
        if result['status'] == 'failed':
            print(f"  - {result['component']}: {result['error_message']}")
    sys.exit(1)
else:
    print(f"✓ All {summary['passed']} tests passed!")
    sys.exit(0)
```

---

## Performance Examples

### Example 15: Benchmark Test Execution Time

```bash
# Sequential execution (baseline)
time ./run_tests.sh --quick

# Parallel execution (optimized)
time ./run_tests.sh --quick --parallel

# Full parallel
time ./run_tests.sh --core --parallel --max-workers 8
```

---

## Summary

The ROCm Test Kit is designed to be:
- **Simple**: One command for most use cases
- **Flexible**: Advanced options for power users
- **Scriptable**: Easy integration into automation
- **Fast**: Parallel execution support
- **Clear**: Obvious output showing what works and what doesn't

Choose the example that matches your use case, or combine options to create your own workflow!
