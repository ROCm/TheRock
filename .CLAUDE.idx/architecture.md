# Architecture Overview

## System Purpose

TheRock (The HIP Environment and ROCm Kit) is a CMake super-project that builds the complete ROCm software stack from source, including HIP, compilers (LLVM/Clang), runtime libraries, math libraries, ML frameworks, and profiling tools. It serves as a lightweight build platform for ROCm contributors, developers, and advanced users who need the latest capabilities without package-based installations.

## Project Type

- **Build System**: CMake 3.25+ super-project with Python orchestration
- **Primary Languages**: C++ (ROCm libraries/runtime), Python (build tooling)
- **Additional Languages**: CMake, TOML (topology), Bash (scripts)
- **Target Platforms**: Linux (Ubuntu, RHEL, SUSE), Windows 11

## High-Level Components

TheRock organizes 19 artifact groups across 9 major categories:

### 1. **Base Infrastructure** (`base/`)
Foundation components for ROCm build system
- **rocm-cmake**: CMake utilities and modules
- **half**: Half-precision floating-point library

### 2. **Compiler Toolchain** (`compiler/`)
LLVM-based compiler infrastructure
- **amd-llvm**: LLVM/Clang/LLD compiler with AMD GPU support
- **hipify**: CUDA-to-HIP source translation tool
- **spirv-llvm-translator**: SPIR-V ↔ LLVM translation

### 3. **Core Runtime** (`core/`)
Fundamental HIP and ROCm runtime components
- **hip-clr**: HIP runtime on CLR (Common Language Runtime)
- **rocr-runtime**: ROCr runtime and thunk
- **ocl/ocl-icd**: OpenCL implementation and ICD loader
- **amdsmi**: AMD System Management Interface
- **rocrtst**: ROCr runtime tests

### 4. **Math Libraries** (`math-libs/`)
High-performance mathematical operations for GPUs
- **BLAS family**: rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, hipBLASLt, hipSPARSELt
- **FFT**: rocFFT
- **Random**: rocRAND
- **Primitives**: rocPRIM (GPU parallel primitives)
- **Support**: rocWMMA (Wave Matrix Multiply-Accumulate), libhipcxx (HIP C++ standard library)

### 5. **ML Libraries** (`ml-libs/`)
Machine learning and deep neural network libraries
- **MIOpen**: AMD's deep learning primitives library
- **hipDNN**: DNN operations (samples and integration tests)
- **Composable Kernel**: GPU kernel library
- **Providers**: hipKernelProvider, hipBLASLtProvider, MIOpenProvider

### 6. **Media Libraries** (`media-libs/`)
Media processing and encoding/decoding
- **rocDecode**: Video decoding acceleration
- **rocJPEG**: JPEG encoding/decoding

### 7. **Communication Libraries** (`comm-libs/`)
Multi-GPU and distributed computing
- **RCCL**: ROCm Communication Collectives Library (like NCCL)
- **rocSHMEM**: OpenSHMEM implementation for ROCm

### 8. **Profiler Tools** (`profiler/`)
Performance analysis and debugging
- **rocprofiler-sdk**: Core profiling SDK
- **rocprofiler-compute**: Compute profiling
- **rocprofiler-systems**: System-level profiling
- **aqlprofile**: AQL profiling utilities

### 9. **Debug Tools** (`debug-tools/`)
Debugging infrastructure
- **rocgdb**: GDB with ROCm GPU debugging support
- **amd-dbgapi**: AMD Debug API
- **rocr-debug-agent**: ROCr debugging agent

### Additional Components

- **IREE Integration** (`iree-libs/`): IREE compiler and Fusilli kernel provider
- **Data Center Tools** (`dctools/`): RDC (ROCm Data Center Tool)
- **External Builds** (`external-builds/`): PyTorch, JAX, UCCL integrations
- **Third-Party Dependencies** (`third-party/`): 23 vendored libraries (googletest, grpc, fmt, LLVM, etc.)

## Directory Structure

```
TheRock/
├── base/                   # Base infrastructure (rocm-cmake, half)
├── compiler/               # LLVM/Clang, hipify, SPIR-V translator
├── core/                   # HIP runtime, ROCr, OpenCL, amdsmi
├── math-libs/              # rocBLAS, rocFFT, rocSPARSE, rocRAND, etc.
│   ├── BLAS/              # BLAS-family libraries
│   └── support/           # Support libraries (rocPRIM, rocWMMA, libhipcxx)
├── ml-libs/                # MIOpen, hipDNN, Composable Kernel
├── media-libs/             # rocDecode, rocJPEG
├── comm-libs/              # RCCL, rocSHMEM
├── profiler/               # rocprofiler tools
├── debug-tools/            # rocgdb, amd-dbgapi
├── iree-libs/              # IREE compiler, Fusilli
├── dctools/                # Data center tools
├── dnn-providers/          # DNN provider implementations
├── external-builds/        # PyTorch, JAX, UCCL
├── third-party/            # Vendored dependencies
│   ├── sysdeps/           # System dependencies (hwloc, libdrm, zlib)
│   └── [23 libraries]     # googletest, grpc, fmt, boost, eigen, etc.
├── build_tools/            # Python build automation
│   ├── *.py               # 38 build scripts
│   ├── _therock_utils/    # Python utility modules
│   ├── github_actions/    # CI/CD scripts
│   └── packaging/         # Packaging infrastructure
├── cmake/                  # CMake infrastructure
│   ├── *.cmake            # 21 core modules
│   ├── finders/           # Custom Find modules
│   ├── modules/           # Utility modules
│   └── toolchains/        # Toolchain files
├── docs/                   # Documentation
│   ├── development/       # Development guides
│   ├── design/            # Architecture docs
│   ├── rfcs/              # Design RFCs
│   └── packaging/         # Packaging docs
├── tests/                  # Test infrastructure
├── examples/               # Example code
├── patches/                # Git patch collections
├── projects/               # Component project files
├── dockerfiles/            # Docker build environments
├── BUILD_TOPOLOGY.toml     # **BUILD TOPOLOGY DEFINITION**
├── CMakeLists.txt          # Top-level CMake orchestration
├── FLAGS.cmake             # Compiler flags configuration
└── requirements.txt        # Python dependencies
```

## Build Topology System

TheRock uses **BUILD_TOPOLOGY.toml** as the single source of truth for build structure:

### Hierarchy

```
Source Sets (git submodule groupings)
    ↓
Build Stages (CI/CD pipeline jobs)
    ↓
Artifact Groups (19 groups with dependencies)
    ↓
Artifacts (individual packaging units: 50+ artifacts)
```

### Build Phases

Each component follows a standardized 4-phase build process:

1. **configure**: CMake configuration with subproject-specific options
2. **build**: Compile source code
3. **stage**: Install to component-specific directory (`build/component/stage/`)
4. **dist**: Merge with dependencies into unified distribution (`build/dist/rocm/`)

### Artifact Types

- **target-neutral**: Built once with all GPU architectures (headers, host code)
- **target-specific**: Built separately per GPU family (kernel libraries)
- **per-arch**: Full stack built per architecture for distribution

### Build Directory Layout

```
build/
├── component/
│   ├── build/             # CMake build tree
│   ├── stage/             # Component install (isolated)
│   ├── dist/              # Component + dependencies merged
│   └── stamp/             # Incremental build markers
├── dist/rocm/             # **Final unified ROCm installation**
└── cmake/
    └── therock_topology.cmake  # Auto-generated from BUILD_TOPOLOGY.toml
```

## Key Technologies

### Build Infrastructure
- **CMake**: 3.25+ with ExternalProject for subproject orchestration
- **Ninja**: High-performance build system (default generator)
- **Python**: 3.9+ for build orchestration and artifact management
- **Git**: Submodule-based component management (30+ submodules)
- **TOML**: Topology definition language

### Compiler Toolchain
- **AMD Clang**: LLVM/Clang with AMD GPU backend
- **hipcc**: HIP compiler driver
- **ROCm Device Libraries**: GPU math and runtime libraries

### Testing
- **CTest**: CMake test runner
- **GoogleTest**: C++ unit testing framework
- **pytest**: Python test framework

### Dependencies
- **System**: hwloc, libdrm, zlib, libegl, libnuma
- **Development**: googletest, grpc, fmt, Catch2, yaml-cpp, spdlog
- **Math/Science**: eigen, fftw3, SuiteSparse, boost, BLAS/LAPACK

### Packaging
- **Tarball**: `.tar.xz` archives with manifests
- **Python**: Wheel packages for Python bindings
- **Native**: DEB/RPM packages (Linux), MSI (Windows - planned)
- **S3**: Artifact storage backend for CI/CD

### CI/CD
- **GitHub Actions**: Primary CI/CD platform
- **Azure Pipelines**: Legacy support
- **DVC**: Data version control for large binary files (MIOpen kernels)
- **ccache**: Compiler cache for faster rebuilds

## Entry Points

### Build System Entry Points
- **`CMakeLists.txt`**: Top-level CMake entry, includes 15+ modules, generates topology
- **`BUILD_TOPOLOGY.toml`**: Defines artifact structure and dependencies (770 lines)
- **`build_tools/topology_to_cmake.py`**: Converts topology to CMake targets
- **`build_tools/fetch_sources.py`**: Clones submodules and applies patches

### Component Build Targets
Every component exposes standardized targets:
- `ninja component`: Full build (configure + build + stage + dist)
- `ninja component+build`: Rebuild after source changes
- `ninja component+dist`: Update artifacts without full rebuild
- `ninja component+expunge`: Clean slate rebuild

### Testing Entry Points
- **`ctest`**: Run all tests
- **`ninja build-tests`**: Build test binaries
- **Component tests**: `build/dist/rocm/bin/test_*`

### Artifact Management
- **`build_tools/build_tarballs.py`**: Package artifacts
- **`build_tools/fetch_artifacts.py`**: Download pre-built artifacts
- **`build_tools/artifact_manager.py`**: Artifact operations

## Dependency Flow

```
third-party-sysdeps → third-party-libs → base
    ↓
compiler (amd-llvm, hipify)
    ↓
core-runtime (hip-clr, rocr-runtime) ← core-amdsmi
    ↓
math-libs (rocBLAS, rocFFT, etc.)
    ↓
ml-libs (MIOpen, Composable Kernel) → comm-libs (RCCL)
    ↓
profiler-core → profiler-apps
    ↓
external-builds (PyTorch, JAX)
```

Orthogonal:
- **debug-tools** (rocgdb) depends on compiler + core-runtime
- **iree-compiler** + **fusilli-libs** depend on ml-libs
- **dctools** depends on core-amdsmi

## Configuration System

### GPU Target Selection

```cmake
-DTHEROCK_AMDGPU_FAMILIES=gfx1100  # Single architecture family
-DTHEROCK_AMDGPU_FAMILIES="gfx90a;gfx942"  # Multiple families
-DTHEROCK_AMDGPU_TARGETS=gfx1103  # Specific target
```

### Component Selection

```cmake
-DTHEROCK_ENABLE_ALL=OFF  # Disable all components
-DTHEROCK_ENABLE_ROCBLAS=ON  # Enable specific component
-DTHEROCK_ENABLE_MATH_LIBS=ON  # Enable entire artifact group
```

### Build Type Configuration

```cmake
-DCMAKE_BUILD_TYPE=Release  # Global build type
-Drocblas_BUILD_TYPE=RelWithDebInfo  # Per-component override
```

### Optimization Options

```cmake
-DCMAKE_C_COMPILER_LAUNCHER=ccache  # Enable ccache
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
-DTHEROCK_USE_SANITIZERS=address  # Enable sanitizers
```

---

See `components/` directory for detailed component documentation.
