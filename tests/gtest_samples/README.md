# TheRock GTest Samples with Logging Framework

This directory contains sample GTest applications that demonstrate TheRock's logging framework integration with comprehensive test coverage and CTest support.

## Overview

The sample applications showcase:
- **GTest Integration**: Three sample test suites (math, string, integration)
- **Logging Framework**: C++ logging wrapper compatible with TheRock's Python logging
- **CTest Support**: Full CMake/CTest configuration
- **YAML Configuration**: Centralized test configuration and execution
- **Automated Runner**: Python script that orchestrates builds and test execution

## Directory Structure

```
tests/gtest_samples/
├── CMakeLists.txt                 # CMake configuration for tests
├── README.md                      # This file
├── test_logging.hpp               # C++ logging wrapper
├── sample_math_tests.cpp          # Math operations tests
├── sample_string_tests.cpp        # String operations tests
└── sample_integration_tests.cpp   # Integration tests
```

## Prerequisites

### Required
- CMake 3.14 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- Python 3.9+
- PyYAML (`pip install pyyaml`)

### Optional
- Ninja build system (recommended)
- Google Test (auto-downloaded if not found)

## Quick Start

### 1. Build and Run Tests

Using the automated runner:

```bash
# From TheRock root directory
python build_tools/run_logging_demo.py --config build_tools/logging_demo.yaml
```

### 2. Build Only (No Test Execution)

```bash
python build_tools/run_logging_demo.py --build-only
```

### 3. Dry Run (Show What Would Execute)

```bash
python build_tools/run_logging_demo.py --dry-run
```

## Manual Build and Test

### Build Tests with CMake

```bash
# Create build directory
mkdir -p build

# Configure
cmake -S tests/gtest_samples -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --parallel 8

# Run specific test
./build/tests/sample_math_tests

# Run all tests with CTest
cd build
ctest --output-on-failure --verbose
```

### Windows (PowerShell)

```powershell
# Create build directory
New-Item -ItemType Directory -Force -Path build

# Configure
cmake -S tests\gtest_samples -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --config Debug

# Run specific test
.\build\Debug\sample_math_tests.exe

# Run with CTest
cd build
ctest -C Debug --output-on-failure --verbose
```

## Test Suites

### 1. Math Tests (`sample_math_tests.cpp`)

Tests basic arithmetic and mathematical operations:
- Addition, subtraction, multiplication, division
- Square root and power functions
- Floating-point precision
- Edge cases (large numbers, negatives, division by zero)
- Performance benchmarks

**Run individually:**
```bash
./build/tests/sample_math_tests --gtest_output=xml:math_results.xml
```

### 2. String Tests (`sample_string_tests.cpp`)

Tests string manipulation operations:
- Concatenation and comparison
- Substring extraction and search
- Case conversion (upper/lower)
- String replace and split
- Empty string handling
- String reversal

**Run individually:**
```bash
./build/tests/sample_string_tests --gtest_output=xml:string_results.xml
```

### 3. Integration Tests (`sample_integration_tests.cpp`)

Tests complex integration scenarios:
- Data processing pipelines
- Multiple processing rounds
- Large dataset handling
- Container operations (vectors, maps)
- Memory management
- Exception handling
- Concurrency simulation

**Run individually:**
```bash
./build/tests/sample_integration_tests --gtest_output=xml:integration_results.xml
```

## Configuration File

The `logging_demo.yaml` configuration file controls:

### Logging Configuration
```yaml
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  use_colors: true        # Enable colored console output
  log_file:
    enabled: true
    path: "logs/logging_demo.log"
```

### Test Configuration
```yaml
tests:
  gtest:
    enabled: true
    test_executables:
      - name: "sample_math_tests"
        path: "tests/sample_math_tests"
        timeout: 60
        retry_on_failure: 2
```

### Build Configuration
```yaml
build:
  enabled: true
  cmake:
    generator: "Ninja"
    build_type: "Debug"
    options:
      - "-DTHEROCK_BUILD_TESTING=ON"
```

## Logging Framework

### C++ Logging Usage

```cpp
#include "test_logging.hpp"

// Create logger
TestLogger logger("MyTest");

// Basic logging
logger.info("Test started");
logger.debug("Debug information");
logger.warning("Warning message");
logger.error("Error occurred");

// Formatted logging
int value = 42;
logger.info("Processing value: {}", value);

// Multiple parameters
logger.debug("Testing: {} + {} = {}", a, b, result);

// Timed operations
{
    TIMED_OPERATION(logger, "slow_operation");
    // ... operation code ...
}  // Automatically logs duration on scope exit
```

### Python Logging Usage

The test runner uses TheRock's Python logging framework:

```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="tests", operation="run")
logger.info("Starting tests")
logger.error("Test failed", extra={"error_code": 1})

with logger.timed_operation("test_execution"):
    run_tests()
```

## Test Output

### Console Output
- Colored log messages (level-based colors)
- Test progress and results
- Execution timing for each test
- Summary statistics

### File Output
- **Log file**: `logs/logging_demo.log` (all log messages)
- **XML reports**: `test_results/*.xml` (GTest results)
- **JSON summary**: `test_results/summary.json` (aggregated results)

## GTest Features Demonstrated

### Test Fixtures
```cpp
class MathTest : public ::testing::Test {
protected:
    TestLogger logger;
    
    void SetUp() override {
        logger = TestLogger("MathTest");
    }
};

TEST_F(MathTest, Addition) {
    // Test implementation
}
```

### Assertions
- `EXPECT_EQ` / `ASSERT_EQ` - Equality
- `EXPECT_NE` - Inequality
- `EXPECT_TRUE` / `EXPECT_FALSE` - Boolean
- `EXPECT_DOUBLE_EQ` - Floating-point equality
- `EXPECT_GT` / `EXPECT_LT` - Comparisons

### Test Organization
- Test fixtures for setup/teardown
- Multiple test cases per fixture
- Clear test naming
- Edge case coverage

## CTest Integration

### Run All Tests
```bash
ctest --output-on-failure
```

### Run with Parallel Execution
```bash
ctest -j4
```

### Run Specific Label
```bash
ctest -L gtest
```

### Generate Test Report
```bash
ctest --output-junit test_results.xml
```

## Troubleshooting

### Google Test Not Found
If GTest is not found, CMake will automatically download it from GitHub. Ensure you have internet connectivity during the first build.

### Build Errors
```bash
# Clean build
rm -rf build
cmake -S tests/gtest_samples -B build -G Ninja
cmake --build build
```

### Test Failures
Check the detailed log file:
```bash
cat logs/logging_demo.log
```

Or run individual tests with verbose output:
```bash
./build/tests/sample_math_tests --gtest_output=xml:results.xml --gtest_verbose
```

### Windows Path Issues
Use forward slashes or escaped backslashes in paths:
```bash
cmake -S tests/gtest_samples -B build
# or
cmake -S tests\\gtest_samples -B build
```

## Advanced Usage

### Custom Configuration
Create your own YAML configuration:
```bash
cp build_tools/logging_demo.yaml my_config.yaml
# Edit my_config.yaml
python build_tools/run_logging_demo.py --config my_config.yaml
```

### Integration with CI/CD
```yaml
# .github/workflows/tests.yml
- name: Run Tests
  run: |
    python build_tools/run_logging_demo.py --config build_tools/logging_demo.yaml
```

### Custom Test Targets
Add to CMakeLists.txt:
```cmake
add_gtest_executable(my_custom_test my_test.cpp)
```

## Examples

### Running Math Tests Only
```bash
./build/tests/sample_math_tests --gtest_filter="MathTest.*"
```

### Running with Different Log Levels
Edit `logging_demo.yaml`:
```yaml
logging:
  level: DEBUG  # Show all debug messages
```

### Generate HTML Report (Future)
```yaml
reporting:
  html_report:
    enabled: true
    output_path: "test_results/report.html"
```

## Contributing

When adding new tests:
1. Create test file in `tests/gtest_samples/`
2. Include `test_logging.hpp` for logging
3. Add to `CMakeLists.txt` using `add_gtest_executable()`
4. Update `logging_demo.yaml` with test configuration
5. Run full test suite to verify

## License

Copyright Advanced Micro Devices, Inc.  
SPDX-License-Identifier: MIT

## Support

For issues or questions:
- Check TheRock documentation
- Review log files in `logs/`
- Run with `--dry-run` to debug configuration
- Check CMake cache: `build/CMakeCache.txt`


