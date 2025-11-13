# ROCm Component Test Kit

A comprehensive, easy-to-use testing framework for ROCm components on **MI300/MI350** hardware.

## 🚀 Quick Start

**Just click "run"!**

```bash
cd rocm_test_kit
./run_tests.sh
```

That's it! The test kit will:
- ✅ Detect your MI300/MI350 hardware
- ✅ Run component-level tests
- ✅ Show you which components pass/fail
- ✅ Complete in 5-10 minutes (quick mode)

## 📋 What Does It Test?

The test kit covers **21 ROCm components** across 8 categories:

### Core Libraries
- **BLAS**: rocBLAS, hipBLAS, hipBLASLt
- **Solvers**: rocSOLVER, hipSOLVER
- **Sparse**: rocSPARSE, hipSPARSE, hipSPARSELt
- **Primitives**: rocPRIM, hipCUB, rocThrust

### Specialized Libraries
- **FFT**: rocFFT, hipFFT
- **Random**: rocRAND, hipRAND
- **Deep Learning**: MIOpen, MIOpen Plugin, hipDNN
- **Communication**: RCCL
- **Other**: rocWMMA, rocRoller

## 🎯 Usage Examples

### Simple Usage (Recommended)

```bash
# Quick test (5-10 minutes) - DEFAULT
./run_tests.sh

# Core libraries test (15-20 minutes)
./run_tests.sh --core

# Full test suite (2-4 hours)
./run_tests.sh --full

# Parallel execution (faster)
./run_tests.sh --quick --parallel

# With HTML report
./run_tests.sh --quick --with-report

# Check hardware compatibility
./run_tests.sh --check-hardware

# List all components
./run_tests.sh --list
```

### Advanced Usage

```bash
# Test specific components
python3 test_runner.py --components rocblas hipblas miopen

# Test a category
python3 test_runner.py --category blas

# Full tests with detailed logs
python3 test_runner.py --preset full --test-type full --log-dir ./logs --verbose

# Parallel execution with custom workers
python3 test_runner.py --preset core --parallel --max-workers 8
```

## 🏗️ Architecture

```
rocm_test_kit/
├── run_tests.sh              # Simple "click to run" script
├── test_runner.py            # Main orchestration engine
├── hardware_detector.py      # MI300/MI350 detection
├── report_generator.py       # HTML/JSON report generation
├── components.yaml           # Component configuration
├── __init__.py              # Python package init
└── README.md                # This file
```

## 🔧 Requirements

- **Python 3.6+**
- **PyYAML** (auto-installed by run_tests.sh)
- **ROCm** installation with test binaries in `THEROCK_BIN_DIR`
- **MI300 or MI350** hardware (recommended, but will warn if not detected)

## 📊 Test Presets

### Quick (Default)
- **Components**: rocBLAS, hipBLAS, rocSOLVER, rocPRIM, MIOpen
- **Time**: 5-10 minutes
- **Purpose**: Fast sanity check of essential components

### Core
- **Components**: All core libraries (BLAS, solvers, FFT, MIOpen, RCCL)
- **Time**: 15-20 minutes
- **Purpose**: Comprehensive test of most-used libraries

### Full
- **Components**: All 21 components
- **Time**: 2-4 hours
- **Purpose**: Complete ROCm stack validation

## 🎨 Features

### ✨ Easy to Use
- Single command execution
- No complex configuration needed
- Clear, colored output

### 🔍 Hardware Detection
- Automatically detects MI300/MI350 GPUs
- Identifies GPU architecture (gfx940, gfx942, etc.)
- Warns if incompatible hardware detected

### 📈 Flexible Testing
- **Test Levels**: Smoke (fast) or Full (comprehensive)
- **Test Selection**: By component, category, or preset
- **Execution Mode**: Sequential or parallel

### 📝 Comprehensive Reporting
- Real-time progress updates
- Detailed test summaries
- Optional HTML reports
- Test logs for debugging

### 🚀 Performance
- Parallel execution support
- Smart test sharding
- Optimized for MI300/MI350

## 📖 Component Categories

### BLAS & Linear Algebra
Core numerical computation libraries for matrix operations.
```bash
python3 test_runner.py --category blas
```

### Deep Learning
Libraries for deep learning frameworks and operations.
```bash
python3 test_runner.py --category deep_learning
```

### Sparse Operations
Specialized libraries for sparse matrix computations.
```bash
python3 test_runner.py --category sparse
```

### Communication
Multi-GPU communication libraries (RCCL).
```bash
python3 test_runner.py --category communication
```

## 🐛 Troubleshooting

### "THEROCK_BIN_DIR not set"
```bash
export THEROCK_BIN_DIR=/path/to/rocm/bin
```

### "No MI300/MI350 hardware detected"
The test kit will still run but is optimized for MI300/MI350 hardware.

### Tests fail with "Test script not found"
Ensure you're running from the ROCm/TheRock repository with all test scripts available.

### Permission denied
```bash
chmod +x run_tests.sh
chmod +x test_runner.py
```

## 📚 Understanding Test Results

### Test Status
- **✓ Passed**: Component tests completed successfully
- **✗ Failed**: Component tests failed (check logs for details)
- **⊘ Skipped**: Component test was skipped (missing dependencies, etc.)

### Example Output
```
======================================================================
ROCm Component Test Kit
======================================================================
Test Type: smoke
Components: 5
Sequential execution
======================================================================
Running rocblas (smoke tests)...
✓ rocblas passed (25.3s)
Running hipblas (smoke tests)...
✓ hipblas passed (22.1s)
Running miopen (smoke tests)...
✗ miopen failed (45.2s)
...
======================================================================
Test Summary
======================================================================
Total Tests:    5
Passed:         3 ✓
Failed:         1 ✗
Skipped:        1
Duration:       125.5s
Success Rate:   75.0%
======================================================================
```

## 🔬 Advanced: Integration with CI/CD

The test kit can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run ROCm Component Tests
  run: |
    cd rocm_test_kit
    ./run_tests.sh --quick --with-report
```

## 🤝 Contributing

To add a new component:

1. Add component configuration to `components.yaml`
2. Ensure test script exists in `build_tools/github_actions/test_executable_scripts/`
3. Run `./run_tests.sh --list` to verify

## 📄 License

This test kit is part of the ROCm/TheRock project.

## 🆘 Support

For issues or questions:
- Check existing test scripts in `build_tools/github_actions/test_executable_scripts/`
- Review component configurations in `components.yaml`
- Run with `--verbose` flag for detailed output

## 🎯 Design Goals

This test kit was designed with these principles:

1. **Simplicity**: Just run one command
2. **Speed**: Quick mode completes in minutes
3. **Clarity**: Clear output showing what's broken
4. **Flexibility**: Test what you need, when you need it
5. **Hardware-Specific**: Optimized for MI300/MI350

---

**Made with ❤️ for the ROCm community**
