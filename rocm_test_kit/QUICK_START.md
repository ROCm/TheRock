# Quick Start Guide - ROCm Component Test Kit

## For Users Who Just Want to Test MI300/MI350 Hardware

### Step 1: Navigate to the test kit
```bash
cd /path/to/TheRock/rocm_test_kit
```

### Step 2: Run the test
```bash
./run_tests.sh
```

That's it! ✨

---

## Common Use Cases

### "I want to quickly check if my ROCm installation is working"
```bash
./run_tests.sh --quick
```
⏱️ Takes 5-10 minutes

### "I want to test everything thoroughly"
```bash
./run_tests.sh --full
```
⏱️ Takes 2-4 hours

### "I want to test faster using multiple GPUs"
```bash
./run_tests.sh --quick --parallel
```
⏱️ Faster completion

### "I want a nice HTML report"
```bash
./run_tests.sh --quick --with-report
```
📊 Generates visual report

### "I want to check if I have the right hardware"
```bash
./run_tests.sh --check-hardware
```
🔍 Shows GPU information

### "What components can I test?"
```bash
./run_tests.sh --list
```
📋 Lists all 21 components

---

## What the Output Means

### ✓ Green checkmarks
All tests passed for that component - it's working!

### ✗ Red X marks
Tests failed for that component - something is broken

### Final Summary
- **Passed**: Number of working components
- **Failed**: Number of broken components (these need attention!)
- **Success Rate**: Percentage of working components

---

## Example Output

```
========================================
  ROCm Component Test Kit
  MI300/MI350 Hardware Testing
========================================

Configuration:
  Preset:     quick
  Test Type:  smoke
  Parallel:   No

Starting tests...

Running rocblas (smoke tests)...
✓ rocblas passed (25.3s)
Running hipblas (smoke tests)...
✓ hipblas passed (22.1s)

Test Summary:
Total Tests:    5
Passed:         5 ✓
Failed:         0 ✗
Success Rate:   100.0%

Tests completed successfully! ✓
```

---

## If Something Goes Wrong

### Error: "python3 not found"
You need to install Python 3

### Error: "THEROCK_BIN_DIR not set"
```bash
export THEROCK_BIN_DIR=/path/to/rocm/bin
```

### Warning: "No MI300/MI350 hardware detected"
The test kit works best on MI300/MI350 but will still run on other hardware

### Tests fail
Look at the error messages - they'll tell you which component is broken

---

## More Options

For all options, run:
```bash
./run_tests.sh --help
```

For advanced usage, see [README.md](README.md)
