# Compiler Toolchain Components

## Purpose

The compiler toolchain provides LLVM/Clang-based compilation for AMD GPUs, including the AMD LLVM compiler, CUDA-to-HIP source translation (hipify), and SPIR-V support. This enables C++/HIP code compilation targeting AMD GPUs.

## Location

- **Primary directory**: `compiler/`
- **Artifact group**: `compiler`
- **Submodules**:
  - `amd-llvm/` (llvm-project) - LLVM/Clang with AMD GPU backend
  - `HIPIFY/` - CUDA→HIP source translator
  - `spirv-llvm-translator/` - SPIR-V↔LLVM translator

## Dependencies

### Depends On
- **base**: rocm-cmake, half
- **third-party**: LLVM (merged with amd-llvm), googletest

### Used By
- **core-runtime**: hip-clr uses compiler for device code
- **math-libs**: Compile kernels for GPU libraries
- **ml-libs**: MIOpen kernel compilation
- **All GPU code**: Any component with device kernels

## Artifacts (2 major)

### 1. **amd-llvm** (target-neutral)

AMD's fork of LLVM/Clang with GPU backend enhancements:

**Components:**
- **Clang**: C/C++ frontend with HIP support
- **LLD**: Linker with GPU code object linking
- **LLVM**: Optimizer with AMD GPU backend
- **Device Libraries**: GPU runtime libraries (math, atomics, etc.)
- **AMDComgr**: Code Object Manager library

**Features:**
- AMD GPU code generation (AMDGPU backend)
- HIP language support (C++ with GPU extensions)
- OpenMP offloading to AMD GPUs
- OpenCL C support
- Code object v4/v5 generation
- GPU architecture: gfx90a, gfx942, gfx1100, gfx1103, etc.

**Installed Binaries:**
```
bin/clang++          # C++ compiler
bin/clang            # C compiler
bin/lld              # Linker
bin/llvm-ar          # Archive tool
bin/llvm-objdump     # Object dumper
bin/llvm-readobj     # Object reader
bin/opt              # LLVM optimizer
bin/llc              # LLVM compiler
```

**Installed Libraries:**
```
lib/libamd_comgr.so       # Code Object Manager
lib/libLLVM-*.so          # LLVM libraries
lib/clang/*/lib/amdgcn/   # GPU device libraries
```

### 2. **hipify** (target-neutral)

Source-to-source translator converting CUDA to HIP:

**Tools:**
- **hipify-clang**: Clang-based translator (recommended)
- **hipify-perl**: Perl-based pattern matcher (legacy)

**Translation Examples:**
```cpp
// CUDA → HIP
cudaMalloc() → hipMalloc()
cudaMemcpy() → hipMemcpy()
cudaDeviceSynchronize() → hipDeviceSynchronize()
__global__ void kernel() → __global__ void kernel()  # Same syntax
```

**Installed Binaries:**
```
bin/hipify-clang     # Main translator
bin/hipify-perl      # Legacy translator
```

### 3. **spirv-llvm-translator** (target-neutral)

Bidirectional translator between SPIR-V and LLVM IR:

**Purpose:**
- Enable OpenCL SPIR-V kernels
- Support for Vulkan compute
- SYCL backend support

**Installed Binaries:**
```
bin/llvm-spirv       # Translator tool
```

## Entry Points

### Using AMD Clang

```bash
# Compile HIP code
clang++ -x hip --offload-arch=gfx1100 mykernel.hip -o myapp

# Via hipcc wrapper (recommended)
hipcc mykernel.hip -o myapp

# OpenMP GPU offloading
clang++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
        -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1100 \
        mycode.cpp -o myapp
```

### Using hipify

```bash
# Translate CUDA to HIP
hipify-clang mycudafile.cu > myhipfile.hip

# Batch translate
find . -name "*.cu" -exec hipify-clang {} -o {}.hip \;

# With Clang database
hipify-clang --cuda-path=/usr/local/cuda \
             -I/path/to/includes \
             mycudafile.cu
```

### Using SPIR-V Translator

```bash
# LLVM IR → SPIR-V
llvm-spirv mykernel.bc -o mykernel.spv

# SPIR-V → LLVM IR
llvm-spirv -r mykernel.spv -o mykernel.bc
```

### Code Object Manager API

```cpp
#include <amd_comgr.h>

// Create data set
amd_comgr_data_set_t input;
amd_comgr_create_data_set(&input);

// Add source code
amd_comgr_data_t source;
amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &source);
amd_comgr_set_data(source, code_size, code);
amd_comgr_data_set_add(input, source);

// Compile to executable
amd_comgr_data_set_t output;
amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                    &info, input, &output);
```

## Key Files

### AMD LLVM Structure
- `llvm/lib/Target/AMDGPU/` - AMD GPU backend
- `clang/lib/Driver/ToolChains/AMDGPU.cpp` - GPU toolchain
- `clang/lib/CodeGen/CGCUDAHIP.cpp` - HIP code generation
- `amd/comgr/` - Code Object Manager
- `amd/device-libs/` - GPU device libraries (math, builtins)

### HIPIFY Structure
- `src/` - Main translator source
- `include/` - Headers
- `tests/` - Translation tests

### SPIR-V Translator Structure
- `lib/SPIRV/` - Translation logic
- `tools/llvm-spirv/` - CLI tool

## Build Hooks

### Pre-hook for amd-llvm (`pre_hook_amd-llvm.cmake`)

```cmake
function(amd_llvm_pre_hook)
  # Enable AMDGPU backend
  set(LLVM_TARGETS_TO_BUILD "AMDGPU;X86" PARENT_SCOPE)
  
  # Build device libraries
  set(LLVM_BUILD_AMDGPU_DEVICE_LIBS ON PARENT_SCOPE)
  
  # Enable LLD
  set(LLVM_ENABLE_PROJECTS "clang;lld;compiler-rt" PARENT_SCOPE)
  
  # Code Object Manager
  set(LLVM_BUILD_AMD_COMGR ON PARENT_SCOPE)
endfunction()
```

### Pre-hook for hipify (`pre_hook_hipify.cmake`)

```cmake
function(hipify_pre_hook)
  # Link against AMD LLVM
  set(HIPIFY_CLANG_RES "${AMDLLVM_INSTALL_DIR}/lib/clang" PARENT_SCOPE)
endfunction()
```

## Patterns

### GPU Code Compilation Flow

```
HIP Source (.hip)
    ↓
Clang Frontend (parse C++ + GPU extensions)
    ↓
LLVM IR (host code)  +  LLVM IR (device code)
    ↓                    ↓
X86 Backend          AMDGPU Backend
    ↓                    ↓
Host Object (.o)     GPU Code Object (.co)
    ↓                    ↓
    Link (LLD)
    ↓
Executable with embedded GPU code
```

### Offload Bundler

Clang bundles host and device code:

```bash
# Compile to bundled object
clang++ -x hip --offload-arch=gfx1100 -c mykernel.hip -o mykernel.o

# Extract device code
clang-offload-bundler --type=o --targets=hip-amdgcn-amd-amdhsa--gfx1100 \
                      --inputs=mykernel.o --outputs=device.o --unbundle
```

### Device Libraries

GPU math implemented in device libraries:

```
lib/clang/*/lib/amdgcn/
├── oclc_*           # OpenCL C libraries
├── ocml.bc          # Math library (sin, cos, exp, etc.)
├── ockl.bc          # Kernel library (printf, atomics)
├── opencl.bc        # OpenCL built-ins
└── hip.bc           # HIP built-ins
```

Linked automatically during compilation:

```bash
# Implicit linking
clang++ -x hip mykernel.hip
# Automatically links: ocml.bc, ockl.bc, hip.bc
```

## CUDA Compatibility

### hipify Translation Coverage

**Fully Supported:**
- Runtime API: `cuda*` → `hip*`
- Driver API: `cu*` → `hip*`
- Math functions: `__cosf()` → `__cosf()`
- Kernel syntax: `__global__`, `__device__`, `__host__`
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`

**Partially Supported:**
- cuBLAS → hipBLAS (requires manual verification)
- cuFFT → rocFFT (API differences)
- Thrust → rocThrust (some algorithms differ)

**Not Supported:**
- CUDA-specific features: CUDA graphs (partially), cudaMemAdvise (limited)
- Warp-level primitives: Some require manual porting
- Texture memory: Limited HIP support

### HIP vs CUDA Syntax

```cpp
// Same in both
__global__ void kernel(float* data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  data[idx] = idx;
}

// Launch (same syntax, different namespace)
kernel<<<blocks, threads>>>(data);  // CUDA
hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, data);  // HIP
// Or just: kernel<<<blocks, threads>>>(data);  // HIP also supports
```

## Testing

### Test Organization

```bash
# LLVM tests (in amd-llvm submodule)
ninja -C build/compiler/amd-llvm/build check-llvm
ninja -C build/compiler/amd-llvm/build check-clang

# Device library tests
ninja -C build/compiler/amd-llvm/build check-amd-device-libs

# hipify tests
build/dist/rocm/bin/hipify-clang-test

# Via CTest
ctest --test-dir build -R llvm
ctest --test-dir build -R hipify
```

### Compiler Validation

```bash
# Compile simple HIP program
echo '__global__ void test() {}' > test.hip
clang++ -x hip --offload-arch=gfx1100 test.hip -c

# Check code object
llvm-objdump -d test.o

# Verify GPU ISA generated
clang++ -x hip --offload-arch=gfx1100 -S test.hip
# Produces: test.s with AMDGPU assembly
```

## Configuration

### CMake Options

```cmake
# AMD LLVM
-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86"
-DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt"
-DLLVM_BUILD_AMDGPU_DEVICE_LIBS=ON
-DLLVM_BUILD_AMD_COMGR=ON

# Build types
-DCMAKE_BUILD_TYPE=Release        # Optimized compiler
-DLLVM_ENABLE_ASSERTIONS=ON       # Debug assertions

# hipify
-DHIPIFY_CLANG_RES=/path/to/clang/resources
```

### Environment Variables

```bash
# Select compiler
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_COMPILER=clang  # vs "nvcc" for CUDA

# Device libraries
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode

# Debugging
export AMD_COMGR_SAVE_TEMPS=1      # Save intermediate files
export AMD_COMGR_REDIRECT_LOGS=stdout  # Verbose logging
```

## GPU Architectures

### Supported Targets

From `therock_amdgpu_targets.cmake`:

**CDNA (Compute):**
- gfx90a - MI200 series (MI210, MI250, MI250X)
- gfx940, gfx941, gfx942 - MI300 series (MI300A, MI300X)

**RDNA (Graphics):**
- gfx1030 - Navi 21 (RX 6800/6900)
- gfx1100, gfx1101, gfx1102, gfx1103 - RDNA3 (RX 7000 series)

**Vega:**
- gfx900, gfx906, gfx908 - Vega 10/20, MI100

### Target Features

```bash
# Compile for specific GPU
clang++ -x hip --offload-arch=gfx90a mykernel.hip

# Multiple targets
clang++ -x hip --offload-arch=gfx90a --offload-arch=gfx1100 mykernel.hip

# With features
clang++ -x hip --offload-arch=gfx90a:sramecc+:xnack- mykernel.hip
# sramecc+: Enable SRAM ECC
# xnack-: Disable XNACK (page migration)
```

## Known Issues

From hooks and development docs:

1. **Build time**: LLVM build takes 1-2 hours (use ccache)
2. **Disk space**: LLVM build tree can be 10+ GB
3. **hipify limitations**: Some CUDA idioms require manual porting
4. **Compiler flags**: Some GCC flags not supported in Clang

## Platform Support

### Linux
- **Full support**: Complete toolchain
- **Distributions**: Ubuntu, RHEL, SUSE
- **Default**: Recommended platform

### Windows
- **Partial support**: Clang works, some features disabled
- **Limitations**: Device libraries may have gaps
- **Build**: Requires Visual Studio 2022

## Future Work

From RFCs and LLVM development:

1. **C++20/23 GPU**: Better C++ standard library on GPU
2. **Improved diagnostics**: Better error messages for GPU code
3. **Compilation speed**: Faster device code generation
4. **SYCL support**: Full SYCL frontend
5. **Heterogeneous debugging**: Better CPU+GPU debugging
