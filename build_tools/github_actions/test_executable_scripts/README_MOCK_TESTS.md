# Mock Test Scripts for Logging Demo

This directory contains mock/demo versions of GPU test scripts that can run on standard GitHub runners without requiring actual GPU hardware or compiled binaries.

## Purpose

These mock scripts demonstrate the unified logging framework integration with GTest and CTest without dependencies on:
- AMD GPU hardware
- Compiled test binaries
- ROCm installation
- TheRock build artifacts

## Mock Scripts

### 1. `demo_test_rocroller.py`
Mock version of `test_rocroller.py` that simulates GTest execution for rocROLLER library.

**Features:**
- Simulates 16 GTest cases
- Demonstrates test filtering (smoke/quick/full)
- Shows test sharding support
- Includes realistic timing and success/failure rates
- Full logging framework integration

**Environment Variables:**
- `TEST_TYPE`: Test type (smoke/quick/full) - default: full
- `SHARD_INDEX`: Shard number (1-based) - default: 1
- `TOTAL_SHARDS`: Total number of shards - default: 1
- `RUNNER_OS`: Platform name - default: Linux

**Example Usage:**
```bash
# Run smoke tests
TEST_TYPE=smoke python3 demo_test_rocroller.py

# Run with sharding
SHARD_INDEX=1 TOTAL_SHARDS=4 python3 demo_test_rocroller.py
```

### 2. `demo_test_rocwmma.py`
Mock version of `test_rocwmma.py` that simulates CTest execution for rocWMMA library.

**Features:**
- Simulates 25 CTest cases
- Demonstrates test filtering (smoke/regression/full)
- Shows CTest-specific output format
- Includes performance metrics
- Full logging framework integration

**Environment Variables:**
- `TEST_TYPE`: Test type (smoke/regression/full) - default: full
- `SHARD_INDEX`: Shard number (1-based) - default: 1
- `TOTAL_SHARDS`: Total number of shards - default: 1
- `AMDGPU_FAMILIES`: GPU architecture - default: gfx942
- `RUNNER_OS`: Platform name - default: Linux

**Example Usage:**
```bash
# Run regression tests
TEST_TYPE=regression python3 demo_test_rocwmma.py

# Run for specific GPU
AMDGPU_FAMILIES=gfx1201 python3 demo_test_rocwmma.py
```

## Integration in CI

These mock scripts are used in the `logging_demo.yml` workflow to demonstrate:
1. Unified logging across all test components
2. GTest and CTest integration patterns
3. Test filtering and sharding strategies
4. Performance metrics collection
5. Error handling and reporting

## Real vs Mock

| Feature | Real Scripts | Mock Scripts |
|---------|-------------|--------------|
| GPU Hardware | Required | Not required |
| Compiled Binaries | Required | Not required |
| ROCm Libraries | Required | Not required |
| Test Results | Real hardware tests | Simulated results |
| Logging Framework | ✅ Same | ✅ Same |
| Output Format | ✅ Same | ✅ Same |
| CI Integration | GPU runners | Standard runners |

## Differences from Real Scripts

1. **No Binary Execution**: Mock scripts don't execute actual test binaries
2. **Simulated Results**: Test outcomes are randomly generated with realistic failure rates
3. **Faster Execution**: Tests complete in seconds vs minutes/hours
4. **No GPU Dependencies**: Can run on any Python 3.9+ environment
5. **Same Logging**: Uses identical logging framework integration

## When to Use

**Use Mock Scripts For:**
- Demonstrating logging framework
- Testing CI workflow changes
- Documentation and examples
- Development without GPU access

**Use Real Scripts For:**
- Actual GPU testing
- Performance benchmarking
- Release validation
- Integration testing

## See Also

- Real scripts: `test_rocroller.py`, `test_rocwmma.py`
- Logging framework: `build_tools/_therock_utils/logging_config.py`
- Test runner: `build_tools/_therock_utils/test_runner.py`
- Demo workflow: `.github/workflows/logging_demo.yml`

