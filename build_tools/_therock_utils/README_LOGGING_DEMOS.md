# TheRock Unified Logging Framework - Demo Applications

This directory contains sample applications that demonstrate TheRock's unified logging framework across different use cases.

## Demo Applications

### 1. Package Installer (`sample_package_installer.py`)
Demonstrates logging for package installation workflows:
- Package discovery and validation
- Dependency resolution
- Installation progress tracking
- Error handling

### 2. Build System (`sample_build_system.py`)
Demonstrates logging for build operations:
- Build configuration
- Compilation tracking
- Build artifact management
- Performance timing

### 3. GTest Demo (`sample_gtest_demo.py`)
Demonstrates logging for Google Test (GTest) execution:
- Test discovery and configuration
- Test sharding (GTEST_SHARD_INDEX, GTEST_TOTAL_SHARDS)
- Per-test execution tracking
- Result parsing and reporting (passed/failed tests)
- Exception handling for test failures

**Sample Tests Included:**
- MatrixMultiply.BasicFloat
- GEMM.SquareMatrix

### 4. CTest Demo (`sample_ctest_demo.py`)
Demonstrates logging for CMake Test (CTest) execution:
- CTest configuration and validation
- Parallel test execution tracking
- Test suite and subtest reporting
- Result aggregation
- Timeout handling

**Sample Tests Included:**
- rocfft_UnitTest (100 subtests)
- rocfft_accuracy_test_pow2 (150 subtests)

## Running the Demos

### Run All Demos
```bash
cd build_tools/_therock_utils
python run_logging_demos.py
```

This will run all four demo applications in sequence and display:
- Console output with colored logging (if supported)
- Log files in the `logs/` directory
- Summary of logging framework features

### Run Individual Demos
```bash
# Package installer demo
python sample_package_installer.py

# Build system demo
python sample_build_system.py

# GTest demo
python sample_gtest_demo.py

# CTest demo
python sample_ctest_demo.py
```

## Configuration Reference

See `logging_demo.yaml` for detailed configuration examples including:
- Global logging settings
- Component-specific configurations
- GTest and CTest integration patterns
- Real-world usage examples

## Key Features Demonstrated

### 1. Structured Logging
All demos show how to add structured data to logs:
```python
logger.info("Test completed", extra={
    "test_name": "GEMM.SquareMatrix",
    "duration_ms": 45.2,
    "status": "passed"
})
```

### 2. Timing Operations
Automatic timing with context managers:
```python
with logger.timed_operation("test_execution"):
    run_tests()
# Logs: "Completed operation: test_execution (1234.56ms)"
```

### 3. Exception Handling
Unified exception logging:
```python
try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed")
```

### 4. Test Result Parsing
- **GTest**: Extracts [PASSED], [FAILED] counts and test names
- **CTest**: Extracts test counts, durations, and failures

### 5. Multi-Level Logging
- DEBUG: Detailed execution information
- INFO: Standard operational messages
- WARNING: Non-critical issues
- ERROR: Test failures and exceptions

## Output Files

When you run the demos, log files are created in the `logs/` directory:
- `logs/package_installer_demo.log`
- `logs/build_system_demo.log`
- `logs/gtest_demo.log`
- `logs/ctest_demo.log`

## Real-World Examples

The logging framework is actively used in TheRock's test infrastructure:

### CTest Integration
- `build_tools/github_actions/test_executable_scripts/test_rocwmma.py`
- `build_tools/github_actions/test_executable_scripts/test_hipcub.py`

### GTest Integration
- `build_tools/github_actions/test_executable_scripts/test_rocroller.py`
- `build_tools/github_actions/test_executable_scripts/test_rocthrust.py`

## TestRunner Class

The `test_runner.py` module provides a unified `TestRunner` class that simplifies test execution with integrated logging:

```python
from test_runner import TestRunner

# GTest execution
runner = TestRunner(component="mycomponent", test_type="full")
runner.run_gtest(
    cmd=["./my_test_binary"],
    cwd=test_dir,
    env={"GTEST_SHARD_INDEX": "0", "GTEST_TOTAL_SHARDS": "4"}
)

# CTest execution
runner.run_ctest(
    test_dir=Path("/path/to/build/tests"),
    parallel=8,
    timeout="300"
)
```

## Customization

To create your own logging-enabled test script:

```python
from logging_config import get_logger, configure_root_logger
import logging

# Configure root logger
configure_root_logger(
    level=logging.INFO,
    log_file="logs/mytest.log",
    use_colors=True
)

# Get component logger
logger = get_logger(
    __name__,
    component="mycomponent",
    operation="test"
)

logger.info("Test starting")
```

## Benefits

✅ **Consistency**: Same logging format across all components  
✅ **Structured Data**: Easy to parse and analyze logs  
✅ **Performance Tracking**: Built-in timing for all operations  
✅ **CI/CD Integration**: Works seamlessly with GitHub Actions  
✅ **Cross-Platform**: Windows and Linux compatible  
✅ **Type Safety**: Full type hints for better IDE support  

---

For more details on the logging framework implementation, see `logging_config.py`.


