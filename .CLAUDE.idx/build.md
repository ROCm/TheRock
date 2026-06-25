# Build and Test

## Build System

**Primary**: CMake 3.25+ super-project with Python 3.9+ orchestration  
**Generator**: Ninja (recommended) or Unix Makefiles  
**Parallelization**: Component-level and within-component  
**Topology**: Defined in `BUILD_TOPOLOGY.toml` (single source of truth)

## Quick Start

### Ubuntu 24.04

```bash
# Install dependencies
sudo apt update
sudo apt install gfortran git ninja-build cmake g++ pkg-config xxd \
    automake libtool python3-venv python3-dev libegl1-mesa-dev \
    texinfo bison flex

# Install patched patchelf
sudo apt install curl make
sudo env INSTALL_PREFIX=/usr/local ./dockerfiles/install_pinned_patchelf.sh

# Clone repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock

# Setup Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fetch sources
python3 ./build_tools/fetch_sources.py

# Configure (replace gfx1100 with your GPU)
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Build everything
ninja -C build

# Or build specific component
ninja -C build rocblas
```

### Windows 11 (VS 2022)

```cmd
:: Set UTF-8 encoding
chcp 65001

:: Clone repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock

:: Setup Python environment
python -m venv .venv
.venv\Scripts\Activate.bat
pip install --upgrade pip
pip install -r requirements.txt

:: Fetch sources
python ./build_tools/fetch_sources.py

:: Configure
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1100

:: Build
ninja -C build
```

## Build Commands

### Top-Level Targets

| Command | Purpose |
|---------|---------|
| `ninja -C build` | Build everything (all enabled components) |
| `ninja -C build dist` | Same as above (explicit dist target) |
| `ninja -C build artifacts` | Generate artifact directories and manifests |
| `ninja -C build archives` | Create `.tar.xz` distribution archives |
| `ninja -C build expunge` | Remove **all** build artifacts (clean slate) |

### Component Targets

Every component exposes these standardized targets:

| Target | Purpose | Example |
|--------|---------|---------|
| `ninja component` | Full build (configure+build+stage+dist) | `ninja rocblas` |
| `ninja component+build` | Rebuild after source changes | `ninja rocblas+build` |
| `ninja component+stage` | Re-run stage phase | `ninja rocblas+stage` |
| `ninja component+dist` | Re-run dist phase | `ninja rocblas+dist` |
| `ninja component+expunge` | Clean component and rebuild | `ninja rocblas+expunge` |

**Examples:**

```bash
# Build single component
ninja -C build rocblas

# Rebuild after editing rocBLAS source
ninja -C build rocblas+build

# Force complete rebuild of component
ninja -C build rocblas+expunge && ninja -C build rocblas

# Build multiple components
ninja -C build rocblas rocfft hip-clr
```

## Configuration

### Required Configuration

**Must specify GPU target:**

```cmake
# Option 1: By GPU family (recommended)
cmake -B build -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Multiple families
cmake -B build -DTHEROCK_AMDGPU_FAMILIES="gfx90a;gfx942;gfx1100"

# Option 2: Specific targets
cmake -B build -DTHEROCK_AMDGPU_TARGETS=gfx1103
```

**Discovering your GPU:**

```bash
# If you have ROCm/HIP installed:
rocm-smi  # or amd-smi
rocm_agent_enumerator
hipinfo  # Windows

# Alternative
offload-arch  # Available in some ROCm installations
```

**Available targets** (see `cmake/therock_amdgpu_targets.cmake`):
- **CDNA**: gfx90a (MI210/250/250X), gfx940/941/942 (MI300A/X)
- **RDNA3**: gfx1100/1101/1102/1103 (RX 7000 series)
- **RDNA2**: gfx1030 (RX 6800/6900)
- **Vega**: gfx900/906/908 (Vega 10/20, MI100)

### Component Selection

```cmake
# Build everything (default)
cmake -B build -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Build subset
cmake -B build \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_ENABLE_HIP_CLR=ON \
  -DTHEROCK_ENABLE_ROCBLAS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Enable entire artifact group
cmake -B build \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_MATH_LIBS=ON \  # All math libraries
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

**Available groups** (from `BUILD_TOPOLOGY.toml`):
- `THIRD_PARTY_SYSDEPS` - System dependencies
- `THIRD_PARTY_LIBS` - Development libraries
- `BASE` - rocm-cmake, half
- `COMPILER` - amd-llvm, hipify
- `CORE_RUNTIME` - hip-clr, rocr-runtime, OpenCL
- `CORE_AMDSMI` - AMD SMI
- `MATH_LIBS` - rocBLAS, rocFFT, rocSPARSE, rocRAND, etc.
- `ML_LIBS` - MIOpen, Composable Kernel
- `COMM_LIBS` - RCCL, rocSHMEM
- `PROFILER_CORE` - rocprofiler-sdk
- `DEBUG_TOOLS` - rocgdb, amd-dbgapi

### Build Types

```cmake
# Global build type (default: Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Per-component override
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Drocblas_BUILD_TYPE=RelWithDebInfo \
  -Dhip_clr_BUILD_TYPE=Debug \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

**Build types:**
- `Release` - Optimized, no debug info
- `RelWithDebInfo` - Optimized with debug info (recommended for profiling)
- `Debug` - No optimization, full debug info
- `MinSizeRel` - Optimize for size

### Optimization Options

```cmake
# Use ccache for faster rebuilds
cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Enable sanitizers (AddressSanitizer, UBSan, etc.)
cmake -B build \
  -DTHEROCK_USE_SANITIZERS=address \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Parallel job control (Ninja job pools)
cmake -B build \
  -DTHEROCK_LLVM_JOB_POOL_SIZE=8 \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

### Testing Configuration

```cmake
# Enable tests (on by default when TheRock is top-level project)
cmake -B build \
  -DTHEROCK_BUILD_TESTING=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Disable tests
cmake -B build \
  -DTHEROCK_BUILD_TESTING=OFF \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

## Build Directory Layout

```
build/
├── component-name/
│   ├── build/              # CMake build tree (ExternalProject)
│   ├── stage/              # Component install (isolated)
│   ├── dist/               # Component + dependencies merged
│   └── stamp/              # ExternalProject stamp files
├── dist/rocm/              # **Final unified ROCm installation**
│   ├── bin/                # Executables (hipcc, rocblas-bench, etc.)
│   ├── lib/                # Libraries (librocblas.so, etc.)
│   ├── include/            # Headers
│   └── share/              # Data files, CMake configs
├── cmake/
│   └── therock_topology.cmake  # Auto-generated topology targets
└── CMakeCache.txt          # CMake cache
```

**Key output:** `build/dist/rocm/` contains the complete ROCm installation.

## Build Phases

Each component follows a 4-phase build:

### 1. **Configure Phase**
- CMake configures component's build tree
- `pre_hook_*.cmake` scripts run before configuration
- Component's `CMakeLists.txt` is processed
- Dependencies are resolved

```bash
# Triggered by: ninja component
# Or reconfigure: cmake --build build --target component-configure
```

### 2. **Build Phase**
- Source code is compiled
- Libraries and executables are built
- Tests are built (if `THEROCK_BUILD_TESTING=ON`)

```bash
# Triggered by: ninja component+build
```

### 3. **Stage Phase**
- Artifacts installed to `build/component/stage/`
- Isolated install tree (this component only)
- `CMAKE_INSTALL_PREFIX` points to stage directory

```bash
# Triggered by: ninja component+stage
```

### 4. **Dist Phase**
- Artifacts copied to `build/component/dist/`
- Dependencies merged from other components' dist directories
- `post_hook_*.cmake` scripts run
- Final output: `build/dist/rocm/` (unified installation)

```bash
# Triggered by: ninja component+dist
# Or: ninja component (full build includes dist)
```

## Test Commands

### Running Tests

```bash
# Run all tests via CTest
ctest --test-dir build

# Run tests for specific component
ctest --test-dir build -R rocblas
ctest --test-dir build -R hip

# Verbose output
ctest --test-dir build -V

# Run specific test
ctest --test-dir build -R rocblas-level3-gemm

# Run in parallel
ctest --test-dir build -j 8
```

### Running Tests Directly

```bash
# Most test binaries are in build/dist/rocm/bin/
export LD_LIBRARY_PATH=build/dist/rocm/lib

# HIP tests
build/dist/rocm/bin/test_hip_*

# rocBLAS tests
build/dist/rocm/bin/rocblas-test

# rocFFT tests
build/dist/rocm/bin/rocfft-test

# rocRAND tests
build/dist/rocm/bin/rocrand-test
```

### Building Tests Only

```bash
# Build test binaries without running
ninja -C build build-tests

# Then run via CTest
ctest --test-dir build
```

### Test Categories

Tests are organized by:
- **Unit tests**: Test individual functions/APIs
- **Integration tests**: Test component interaction
- **Performance tests**: Benchmarks (separate from unit tests)
- **Conformance tests**: Validate spec compliance

## Dependencies

### System Dependencies (Linux)

**Ubuntu/Debian:**
```bash
sudo apt install \
    gfortran git ninja-build cmake g++ pkg-config xxd \
    automake libtool python3-venv python3-dev \
    libegl1-mesa-dev texinfo bison flex
```

**RHEL/CentOS:**
```bash
sudo dnf install \
    gcc gcc-c++ gcc-gfortran git ninja-build cmake pkgconfig \
    automake libtool python3-devel mesa-libEGL-devel \
    texinfo bison flex
```

**SUSE:**
```bash
sudo zypper install \
    gcc gcc-c++ gcc-fortran git ninja cmake pkg-config \
    automake libtool python3-devel Mesa-libEGL-devel \
    texinfo bison flex
```

### Python Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

**Key Python packages:**
- `tomli` - TOML parsing (Python <3.11)
- `jinja2` - Template generation
- `requests` - HTTP requests for artifact fetching
- `pytest` - Testing framework
- `dvc[s3]` - Data version control (optional, for MIOpen kernels)

### Build-Time Dependencies

Provided by TheRock's `third-party/` directory:
- **LLVM** (merged into amd-llvm)
- **googletest** (C++ testing)
- **grpc** (RPC framework)
- **fmt** (C++ formatting)
- **boost** (C++ libraries)
- **eigen** (Linear algebra)
- **fftw3** (CPU FFT reference)

### Runtime Dependencies

- **AMD GPU Driver**: amdgpu kernel module (Linux) or AMD Windows driver
- **ROCm Kernel Driver**: Loaded by ROCr runtime
- **System Libraries**: libnuma, libdrm, zlib, libelf

## Configuration Examples

### Minimal Build (HIP only)

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_ENABLE_HIP_CLR=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build
```

### Math Libraries Build

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_MATH_LIBS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build
# Builds: rocBLAS, rocFFT, rocSPARSE, rocRAND, rocPRIM, rocWMMA, etc.
```

### Multi-Architecture Build

```bash
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES="gfx90a;gfx942;gfx1100" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

ninja -C build
# Takes longer, produces artifacts for all 3 architectures
```

### Debug Build with Tests

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTHEROCK_BUILD_TESTING=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build
ctest --test-dir build -V
```

### Faster Development Build (subset)

```bash
# Build only what you're working on
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_ROCBLAS=ON \
  -DTHEROCK_BUILD_TESTING=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build rocblas
```

## Common Workflows

### Iterating on a Component

```bash
# 1. Make changes to component source
vim math-libs/BLAS/rocBLAS/library/src/blas3/rocblas_gemm.cpp

# 2. Rebuild just that component
ninja -C build rocblas+build

# 3. Test changes
build/dist/rocm/bin/rocblas-test --gtest_filter=*gemm*

# 4. If tests pass, commit
cd math-libs/BLAS/rocBLAS
git add -u
git commit -m "fix(rocblas): Improve GEMM performance for gfx1100"
```

### Testing a Built Component

```bash
# Build component
ninja -C build rocblas

# Set library path
export LD_LIBRARY_PATH=build/dist/rocm/lib

# Run tests
build/dist/rocm/bin/rocblas-test

# Run benchmarks
build/dist/rocm/bin/rocblas-bench --function gemm --precision f64_r

# Or via CTest
ctest --test-dir build -R rocblas -V
```

### Force Rebuild

```bash
# Rebuild specific component from scratch
ninja -C build rocblas+expunge
ninja -C build rocblas

# Rebuild everything from scratch
rm -rf build
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1100
ninja -C build
```

### Update Submodules

```bash
# WARNING: fetch_sources.py is DESTRUCTIVE
# It resets all submodules and reapplies patches

# 1. Commit any changes in submodules first
cd math-libs/BLAS/rocBLAS
git add -u && git commit -m "My changes"
cd ../../..

# 2. Fetch sources (resets submodules)
python3 ./build_tools/fetch_sources.py

# 3. If you lost work, check reflog
cd math-libs/BLAS/rocBLAS
git reflog  # Find lost commit
git cherry-pick <commit-hash>
```

### Using Pre-built Artifacts

```bash
# Fetch pre-built artifacts instead of building
python3 ./build_tools/fetch_artifacts.py \
  --artifact rocblas \
  --commit abc123 \
  --output-dir build/dist/rocm/

# Then configure to skip building that component
cmake -B build \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_ROCFFT=ON \  # Only build rocFFT
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

## Performance Tips

1. **Use Ninja**: Faster than Make
   ```bash
   cmake -B build -GNinja  # Instead of -G"Unix Makefiles"
   ```

2. **Use ccache**: Significantly faster rebuilds
   ```bash
   cmake -B build \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
   ```

3. **Parallel builds**: Ninja auto-detects cores
   ```bash
   ninja -C build  # Uses all cores by default
   ninja -C build -j 16  # Limit to 16 jobs
   ```

4. **Build subset**: Only build what you need
   ```bash
   cmake -B build -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_ROCBLAS=ON
   ```

5. **Use DVC for kernels**: Avoid recompiling MIOpen kernels
   ```bash
   snap install --classic dvc  # Ubuntu
   # DVC automatically pulls pre-compiled kernels
   ```

6. **Incremental builds**: Use `+build` targets
   ```bash
   ninja -C build rocblas+build  # Faster than full rebuild
   ```

## Troubleshooting

### Build Fails

```bash
# Check CMake configuration
cmake -B build -DTHEROCK_AMDGPU_FAMILIES=gfx1100 -L  # List cache variables

# Verbose build output
ninja -C build -v

# Build single component to isolate issue
ninja -C build rocblas
```

### Out of Memory

```bash
# Limit parallel jobs
ninja -C build -j 4

# Or set job pool size for LLVM (biggest memory user)
cmake -B build -DTHEROCK_LLVM_JOB_POOL_SIZE=4
```

### Disk Space

```bash
# LLVM build tree is largest (~10+ GB)
# Use ccache and clean build directories after

# Check sizes
du -sh build/compiler/amd-llvm/build

# Clean specific component
ninja -C build amd-llvm+expunge
```

### Test Failures

```bash
# Run failing test with verbose output
ctest --test-dir build -R failing-test -V

# Run test directly for more control
export LD_LIBRARY_PATH=build/dist/rocm/lib
build/dist/rocm/bin/rocblas-test --gtest_filter=*gemm*
```

## Environment Variables

### Build Environment

```bash
# Compiler selection
export CC=clang
export CXX=clang++

# ccache
export CCACHE_DIR=$HOME/.ccache
export CCACHE_MAXSIZE=50G

# Parallel build
export CMAKE_BUILD_PARALLEL_LEVEL=16
```

### Runtime Environment

```bash
# Library path
export LD_LIBRARY_PATH=build/dist/rocm/lib:$LD_LIBRARY_PATH

# HIP device selection
export HIP_VISIBLE_DEVICES=0,1

# rocBLAS logging
export ROCBLAS_LAYER=2

# rocFFT logging
export ROCFFT_LOG_TRACE_PATH=/tmp/rocfft.log
```

## Additional Resources

- **Build System Docs**: `docs/development/build_system.md`
- **Development Guide**: `docs/development/development_guide.md`
- **Environment Setup**: `docs/environment_setup_guide.md`
- **FAQ**: `docs/faq.md`
- **Windows Support**: `docs/development/windows_support.md`
- **CI Documentation**: `docs/continuous-integration.md`
