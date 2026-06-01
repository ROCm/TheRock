# Builder

This directory contains the experimental builder feature for composable_kernel.

* Status: In development (October - December 2025)

## Overview

The builder provides a high-level, semantically-clear interface for constructing composable kernel operations, with an initial focus on convolution kernels for MIOpen. It leverages modern C++20 features (such as POD structs as non-type template parameters, concepts, and designated initializers) to simplify kernel instantiation and improve developer experience.

This project is a prototype for a more general builder pattern for all of composable_kernel (CK) and CKTile, but is currently limited to formalizing the interface between MIOpen and CK.

## Design descriptions

- [CK Builder design description](include/ck_tile/builder/README.md)
- [CK Builder factory design](include/ck_tile/builder/factory/README.md)
- [CK Builder testing design](include/ck_tile/builder/testing/README.md)

## Directory Structure

- `include/ck_tile/builder/`
  Core builder headers and public API.
- `include/ck_tile/builder/reflect`
  Reflection mechanism.
- `include/ck_tile/builder/factory`
  Compile-time dispatch from builder descriptors to our exisitng specialized convolution kernel implementations.
- `test/`
  Unit tests and example usage of the builder pattern.
- `CMakeLists.txt`
  CMake configuration for building the experimental builder and its tests.

## CMake Configuration

To enable the experimental builder, configure your build with:

```bash
cmake                                                                                             \
  -D CMAKE_PREFIX_PATH=/opt/rocm                                                                  \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                       \
  -D CMAKE_BUILD_TYPE=Release                                                                     \
  -D GPU_TARGETS="gfx942"                                                                         \
  -D CK_EXPERIMENTAL_BUILDER=ON                                                                   \
  -D CMAKE_CXX_STANDARD=20                                                                        \
  -G Ninja                                                                                        \
  ..
```

## Building and Testing

The builder test suite is organized into two main categories:

### Smoke Tests (Fast Unit Tests)
Quick unit tests that verify the builder's internal logic without compiling GPU kernels. These complete in under 1 second total and are suitable for frequent execution during development.

```sh
ninja smoke-builder
```

### Regression Tests (Integration Tests)
Integration tests that compile actual GPU kernels to verify that the builder generates valid, compilable code. These are more expensive than smoke tests (can take minutes to compile) but cover more fuctionality.
)

```sh
ninja regression-builder
```

### Running All Tests
To build and run the complete test suite:

```sh
ninja check-builder
```

### Building Individual Tests
To build and run a specific test:

```sh
ninja test_ckb_conv_builder && bin/test_ckb_conv_builder
```

### Test Organization
- **Smoke tests**: Fast feedback during active development
- **Regression tests**: Thorough validation before submitting changes
- **Factory tests**: Expensive tests that build all MIOpen kernels (included in regression tests)

When adding new tests, please follow the convention where the CMake build target starts with a prefix `test_ckb`. This allows filtering of CK Builder tests from the full CK repository test suite.
