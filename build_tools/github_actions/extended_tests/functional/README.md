# Functional Testing Framework

Functional tests validate correctness, API contracts, and expected behavior of ROCm libraries.

## Overview

Functional tests are designed to verify that libraries work correctly and meet their API specifications. Unlike benchmark tests which measure performance, functional tests focus on correctness validation.

| Aspect | Functional Tests |
|--------|------------------|
| **Purpose** | Correctness validation and API testing |
| **Result Types** | PASS/FAIL/ERROR/SKIP |
| **When to Use** | Verify expected behavior (nightly CI) |
| **Frequency** | Nightly CI only |

## Status

**Note:** Functional tests are currently being developed. This directory structure is a placeholder for future functional test implementations.

## Planned Structure

When functional tests are implemented, the structure will be:

```
functional/
├── scripts/                   # Test implementations
│   ├── functional_base.py     # Base class for functional tests
│   └── test_*.py              # Individual functional tests
├── configs/                   # Test-specific configurations
│   └── *.json                 # Test configuration files
├── functional_test_matrix.py  # Functional test matrix
└── README.md                  # This file
```

## Adding Functional Tests

Functional tests will be added in follow-up PRs. When implemented, they will:

1. Inherit from `FunctionalBase` class (similar to `BenchmarkBase`)
2. Implement test execution and result validation logic
3. Use shared utilities from `extended_tests/utils/`
4. Be integrated into nightly CI workflows

## Related Documentation

- [Extended Tests Framework](../README.md) - Framework overview
- [Benchmark Tests](../benchmark/README.md) - Performance regression testing
- [Shared Utils](../utils/README.md) - Common utilities
