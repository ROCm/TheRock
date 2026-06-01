# Copilot Rules for MIOpen Provider Plugin

## Project Overview

MIOpen Provider is a hipDNN plugin that wraps AMD's MIOpen library to provide GPU-accelerated deep learning operations. This plugin implements the hipDNN Plugin SDK interface to expose MIOpen kernels as execution engines.

**hipDNN Context**: hipDNN is a graph-based deep learning library for AMD GPUs with a plugin-based architecture. Plugins like this one implement the Plugin SDK to provide engine implementations that execute operation graphs.

### Key Documentation
- **[Plugin README](../README.md)** - Build instructions and overview
- **[Operation Support](../docs/OperationSupport.md)** - Supported operations and data types
- **[hipDNN Plugin Development Guide](../../../projects/hipdnn/docs/PluginDevelopment.md)** - Plugin SDK reference
- **[Plugin SDK Headers](../../../projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/)** - API definitions

### Plugin Architecture

This plugin implements the hipDNN Plugin SDK interfaces:

| Component | Purpose |
|-----------|---------|
| **Plugin API** (`PluginApi.h`) | Plugin metadata (name, version, type), error reporting, logging |
| **Engine Plugin API** (`EnginePluginApi.h`) | Engine lifecycle (create, configure, execute graphs) |
| **IEngine** | Engine interface - checks graph applicability, creates execution plans |
| **IPlanBuilder** | Builds execution plans for specific operation types |
| **IPlan** | Executable plan that runs on GPU using MIOpen kernels |
| **Data SDK** | FlatBuffer schemas for serialization, logging utilities |

**Execution Flow**: Graph (FlatBuffer) → Engine Selection (isApplicable) → Plan Creation (IPlanBuilder) → Execution (IPlan::execute)

**Component Linkage (Do Not Modify)**:
- Plugin uses: Plugin SDK (header-only), Data SDK (header-only), MIOpen library
- Plugin SDK uses: Data SDK
- No direct dependencies on hipDNN Backend or Frontend

---

## Building & Testing

### Build Commands

**Building as part of hipDNN**:
```bash
# From hipDNN root directory
cd <workspace>/projects/hipdnn
mkdir -p build && cd build
cmake -GNinja ..
ninja  # Plugin built by default
```

**Building standalone** (requires hipDNN and MIOpen installed):
```bash
# From workspace root, navigate to plugin directory
cd <workspace>/dnn-providers/miopen-provider
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ ..
ninja
```

See [Plugin README](../README.md) for detailed build instructions.

### Test Binaries

Plugin-specific test binaries (in `build/bin/` when building from hipDNN root):

| Binary | Tests | Typical Use |
|--------|-------|-------------|
| `miopen_legacy_plugin_tests` | Plugin unit tests | Engine manager, plan builders, MIOpen integration |
| `miopen_plugin_integration_test` | GPU integration tests | End-to-end operation execution |

### Running Specific Tests with `--gtest_filter`

**Use `--gtest_filter` when iterating on specific functionality:**
```bash
# From hipDNN build directory
./bin/miopen_legacy_plugin_tests --gtest_filter="TestMiopenEngine.*"

# Run a specific test case
./bin/miopen_plugin_integration_test --gtest_filter="IntegrationGpuConvolutionFwdFp32.*"
```

### When Modifying Code

**Only build or run tests if explicitly requested in the user's prompt.** Do not proactively run `ninja`, test binaries, or build commands unless asked.

When requested to build/test:
1. Rebuild with `ninja` (from build directory)
2. Run relevant tests using `--gtest_filter` for fast iteration

---

## C++ Code Style

### Naming Conventions
- Use CamelCase for class and struct names (e.g., `MiopenEngine`, `ConvolutionPlanBuilder`)
- Use camelBack for functions, variables, and private members (e.g., `buildPlan()`, `engineId`)
- Prefix private members with underscore (e.g., `_miopenHandle`, `_planBuilders`)

### File Headers
- Always add copyright header to all source files:
  ```cpp
  // Copyright © Advanced Micro Devices, Inc., or its affiliates.
  // SPDX-License-Identifier:  MIT
  ```
- For header files (.h, .hpp), add `#pragma once` immediately after copyright

### Code Practices
- Use `auto` when initializing with a cast to avoid duplicating the type name
  ```cpp
  // Good
  auto tensor = static_cast<float*>(data);
  // Avoid
  float* tensor = static_cast<float*>(data);
  ```
- Use auto when initializing variables, unless the type is not obvious.
- **Avoid implicit casts** - use explicit `static_cast<>` for type conversions. The codebase compiles with `-Wconversion` and `-Wsign-conversion` which will error on implicit narrowing or sign conversions.
- Always use {}'s for if/for/while bodies, even if single line
  ```cpp
    // Good
    if(condition)
    {
        doSomething();
    }
    // Avoid
    if(condition) doSomething();
  ```

### Build System
- Always use CMake for managing C/C++ dependencies
- When discussing dependencies or build configuration, provide CMake-based solutions

### Serialization
- Use Flatbuffers for serialization needs (via Data SDK)
- Access FlatBuffer data via Data SDK wrapper classes (IGraph, GraphWrapper, etc.)

### Testing
- Use Google Test (gtest) framework for all C/C++ tests
- Never generate a main() function in test files - gtest provides its own
- Use TEST(), TEST_F(), or TEST_P() macros as appropriate

#### Test Naming Guidelines

Rules below apply ONLY to the TestSuite name (first parameter of `TEST` / `TEST_F` / `TEST_P`). The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.
TestCase should be PascalCase.

**Ordering & Composition (left → right):**

1. Optional `Integration` prefix for integration tests.
2. Optional `Gpu` (immediately after `Integration` if both apply) for GPU-required tests.
3. Core Feature / Subject under test (PascalCase, no underscores).
4. Optional Datatype token (`Bfp16`, `Fp16`, `Fp32`) at the end.

Omit any category that does not apply.

### Keywords (reserved positions)

- **Integration** (only for integration tests, always first if present).
- **Gpu** (always first unless preceded by Integration).
- **Datatypes**: Bfp16, Fp16, Fp32.

### Unit Tests

In most cases unit style tests should be named so they directly mirror the class under test. If the class is named `MyClass`, then the test suite should be named `TestMyClass`. In general these kinds of tests should try to avoid using anything that requires Gpu support. This is not always possible, in the cases where Gpu support is required, the test suite should be named `GpuTestMyClass`.

### Valid Examples

```cpp
IntegrationGpuConvolutionFwdFp32
GpuTestMiopenEngineFp32
TestMiopenEngineManager
TestConvolutionPlanBuilder
```

## Example Code Structure

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once  // For header files only

#include <memory>
#include <vector>

// Class names in PascalCase
class MiopenPlanBuilder
{
public:
    // Public methods in camelCase
    void buildPlan();
    size_t getWorkspaceSize() const;

private:
    // Private members prefixed with _
    miopenHandle_t _handle;
    std::vector<float> _data;
};

// Functions in camelCase
void executePlan(const MiopenPlanBuilder& builder);
```

## Test File Example

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

class TestMiopenEngine : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code
    }

private:
    miopenHandle_t _handle;
};

TEST_F(TestMiopenEngine, CreatesValidEngine)
{
    // Test implementation
}
```
