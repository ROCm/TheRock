---
author(s): Mitchell Ousdahl, Brian Harrison
created: 2025-01-21
modified: 2025-01-21
status: draft
discussion: https://github.com/ROCm/TheRock/discussions/1867
---

# hipDNN Integration into TheRock

This RFC proposes the integration of hipDNN, a graph-based deep learning library for AMD GPUs, into TheRock's build system and CI/CD infrastructure.

## Overview

[hipDNN](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn) is a graph-based deep learning library that enables multi-operation fusion for improved performance on AMD GPUs. It uses operation graphs as an intermediate representation to describe computations, allowing different backend engines to optimize and execute these graphs efficiently through a flexible plugin architecture.

hipDNN is currently in early development with limited operational support. This integration will establish hipDNN within TheRock's build infrastructure while the project continues to expand its capabilities.

## Core Design Principles

- **Graph-based API**: Operations are expressed as computational graphs rather than individual function calls, enabling optimization opportunities through multi-operation fusion
- **Plugin Architecture**: Functionality is provided through plugins, allowing extensibility without modifying the core library
- **Performance through Fusion**: Multiple operations can be fused into single kernels for better performance and reduced memory traffic
- **Engine Selection**: Selectable heuristics and benchmarking backends will provide intelligent engine selection for optimal performance
- **Industry Standard API**: Provides a familiar interface that matches established deep learning library conventions

Please refer to the [hipDNN design documentation](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Design.md) for more detailed information.

### Component Descriptions

#### Frontend

The Frontend is a header-only C++ library providing an industry-standard API for building and executing operation graphs. It wraps the lower-level Backend C API, offering:

- Intuitive graph construction interface
- Type-safe node and attribute management
- Simplified execution workflow
- No compiled dependencies for ease of integration

#### Backend

The Backend is a shared library providing the core C API for hipDNN. It is responsible for:

- Loading and managing plugins at runtime
- Orchestrating graph execution through plugin engines
- Providing descriptor-based API for language interoperability
- Abstracting plugin complexity from users

#### SDK

The SDK is a header-only C++ library providing shared utilities and interfaces for plugin development. It includes:

- Plugin API definitions
- FlatBuffers-based graph data structures
- Logging utilities
- Data-type utilities & definitions
- Utilities for verifying end-to-end graph execution
  - Reference graph executor
  - Output value validation utilities
  - Reference implementations for supported operations

#### Plugins

Plugins extend hipDNN's capabilities. Currently supported:

- **Engine Plugins**: Provide kernel implementations for operations
  - MIOpen Plugin: Integration with AMD's [MIOpen](https://github.com/ROCm/rocm-libraries/tree/develop/projects/miopen) library
  - IREE/Fusilli Plugin: JIT-compiled kernels (see [RFC0004](RFC0004-Fusilli-IREE-Kernel-Provider-hipDNN.md))

Future plugin types:

- **Heuristic Plugins**: Intelligent engine selection without sampling
- **Benchmarking Plugins**: Exhaustive tuning through performance sampling

## Dependencies

### Build-Time Dependencies

hipDNN's build requires:

- CMake 3.25.2+
- C++17 compatible compiler (AMD Clang)
- ROCm/HIP stack
- FlatBuffers
- GoogleTest
- spdlog
- nlohmann-json
- fmt

### Consumer Dependencies

Applications using hipDNN's Frontend or SDK also require:

- C++17 compatible compiler
- FlatBuffers
- spdlog
- fmt
- nlohmann-json

Note: The Frontend and SDK are header-only libraries, but they depend on these third-party headers being available at compile time.

### Runtime Dependencies

- HIP runtime
- ROCm libraries
- Plugin-specific dependencies (e.g., MIOpen for MIOpen plugin)

## Current Status and Roadmap

### Current Capabilities

- MIOpen plugin operational with limited support (please see [Operation Support](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/OperationSupport.md))

### Near-Term Roadmap

- Complete MIOpen plugin integration
- IREE/Fusilli plugin integration ([RFC0004](RFC0004-Fusilli-IREE-Kernel-Provider-hipDNN.md))

### Long-Term Roadmap

- Python frontend API
- Heuristic plugin system
- Benchmarking/tuning plugin system
- Performance benchmarking infrastructure
- Comprehensive operation coverage

## Documentation

hipDNN provides the following documentation:

- [README](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/README.md): Project overview and quick start
- [Building](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Building.md): Build instructions and configurations
- [Design](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Design.md): Architecture and component details
- [How-To](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/HowTo.md): Usage and extension guide
- [Plugin Development](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PluginDevelopment.md): Plugin creation guide
- [Testing Strategy](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/testing/TestingStrategy.md): Comprehensive testing approach
- [Roadmap](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Roadmap.md): Development priorities
- [Operation Support](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/OperationSupport.md): Current operation coverage

## Related RFCs

- [RFC0003](RFC0003-Build-Tree-Normalization.md): Build Tree Normalization - Defines directory structure for ML libraries
- [RFC0004](RFC0004-Fusilli-IREE-Kernel-Provider-hipDNN.md): Fusilli+IREE Kernel Provider - Proposes IREE/Fusilli plugin integration

## Revision History

- 2025-01-21: Initial draft (Mitchell Ousdahl, Brian Harrison)
