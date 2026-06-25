# Core Runtime Components

## Purpose

The core runtime provides the fundamental HIP (Heterogeneous-compute Interface for Portability) runtime, ROCm low-level runtime (ROCr), OpenCL implementation, and system management interfaces. These are the essential components that enable GPU programming and execution on AMD hardware.

## Location

- **Primary directory**: `core/`
- **Artifact group**: `core-runtime`, `core-amdsmi`, `hip-runtime`, `opencl-runtime`, `rocrtst`
- **Key submodules**:
  - `hip-clr/` - HIP runtime on Common Language Runtime
  - `rocr-runtime/` - ROCm Runtime (thunk and runtime)
  - `ocl/` - OpenCL implementation
  - `ocl-icd/` - OpenCL ICD loader
  - `amdsmi/` - AMD System Management Interface
  - `hipInfo/` - HIP info utility

## Dependencies

### Depends On
- **compiler**: amd-llvm, hipify (for HIP compilation)
- **base**: rocm-cmake, half
- **third-party**: LLVM, googletest, grpc

### Used By
- **math-libs**: All BLAS/FFT/RAND libraries
- **ml-libs**: MIOpen, Composable Kernel
- **comm-libs**: RCCL, rocSHMEM
- **profiler**: rocprofiler tools
- **debug-tools**: rocgdb, amd-dbgapi
- **external-builds**: PyTorch, JAX

## Artifacts (11 total)

### HIP Runtime
1. **hip-clr** (target-neutral)
   - HIP runtime implementation on CLR backend
   - Provides HIP API for GPU programming
   - Includes hipcc compiler driver
   - Location: Build output from hip-clr submodule

### ROCr Runtime
2. **rocr-runtime** (target-neutral)
   - Low-level ROCm runtime and HSA implementation
   - GPU kernel dispatch and memory management
   - Thunk library for kernel driver communication

3. **rocrtst** (target-neutral)
   - ROCr runtime tests and validation suite

### OpenCL
4. **ocl** (target-neutral)
   - OpenCL runtime implementation
   - Enables OpenCL GPU programming

5. **ocl-icd** (target-neutral)
   - OpenCL ICD (Installable Client Driver) loader
   - Allows multiple OpenCL implementations

### System Management
6. **amdsmi** (target-neutral)
   - AMD System Management Interface library
   - Query GPU status, temperature, usage
   - Device enumeration and monitoring

### HIP Tools
7. **hipinfo** (target-neutral)
   - HIP information utility
   - Displays GPU capabilities and configuration
   - Location: `core/hipInfo/`

### Kernel Packaging
8. **kpack** (target-specific)
   - Kernel package management
   - Pre-compiled GPU kernels

### Debug Support
9. **rocr-debug-agent** (target-neutral)
   - ROCr debugging agent for GPU debugging
   - Enables source-level debugging

10. **rocr-debug-agent-tests** (target-neutral)
    - Debug agent test suite

## Entry Points

### HIP API
```cpp
// HIP runtime header
#include <hip/hip_runtime.h>

// Device query
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);

// Kernel launch
hipLaunchKernelGGL(myKernel, blocks, threads, 0, 0, args);
```

### ROCr Runtime
```cpp
// HSA runtime
#include <hsa/hsa.h>

// Initialize runtime
hsa_init();

// Iterate agents (GPUs)
hsa_iterate_agents(callback, nullptr);
```

### OpenCL
```cpp
// OpenCL API
#include <CL/cl.h>

// Platform query
clGetPlatformIDs(1, &platform, nullptr);

// Create context
cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
```

### System Management
```cpp
// AMD SMI
#include <amd_smi/amdsmi.h>

// Initialize SMI
amdsmi_init();

// Get device count
uint32_t count;
amdsmi_get_socket_handles(&count, nullptr);
```

### Command-Line Tools
```bash
# HIP information
hipinfo

# HIP compiler
hipcc mykernel.cpp -o myapp

# AMD SMI
amd-smi static
amd-smi monitor
```

## Key Files

### HIP Runtime (hip-clr)
- `hipamd/src/hip_context.cpp` - HIP context management
- `hipamd/src/hip_device.cpp` - Device management
- `hipamd/src/hip_memory.cpp` - Memory operations
- `hipamd/src/hip_stream.cpp` - Stream management
- `hipamd/include/hip/hip_runtime.h` - Main HIP header
- `bin/hipcc` - HIP compiler driver

### ROCr Runtime
- `src/core/runtime/hsa.cpp` - HSA runtime implementation
- `src/core/runtime/runtime.cpp` - Core runtime logic
- `src/core/runtime/queue.cpp` - Queue management
- `src/inc/hsa.h` - HSA API header
- `src/libhsakmt/` - Kernel thunk library

### OpenCL
- `opencl/amdocl/cl_*.cpp` - OpenCL API implementations
- `api/opencl/khronos/headers/opencl2.2/CL/` - OpenCL headers

### AMD SMI
- `src/amd_smi.cc` - Main SMI implementation
- `include/amd_smi/amdsmi.h` - SMI API header
- `cli/` - Command-line interface

## Patterns

### Build Hooks

**Pre-configuration hook** (`pre_hook_ROCR-Runtime.cmake`):
```cmake
function(rocr_pre_hook)
  # Set ROCr-specific options
  set(BUILD_SHARED_LIBS ON PARENT_SCOPE)
endfunction()
```

**Post-distribution hook** (`post_hook_hip-clr.cmake`):
```cmake
function(hip_clr_post_hook)
  # Install hipcc wrapper scripts
  # Set up HIP environment
endfunction()
```

### Runtime Initialization Pattern

Most components follow this pattern:
```cpp
// Initialize
hip_init() / hsa_init() / clGetPlatformIDs()

// Query devices
hip_get_device_count() / hsa_iterate_agents() / clGetDeviceIDs()

// Select device
hipSetDevice() / hsa_agent_get_info() / clCreateContext()

// Execute work
hipLaunchKernel() / hsa_queue_create() / clEnqueueNDRangeKernel()

// Cleanup
hipDeviceReset() / hsa_shut_down() / clReleaseContext()
```

### Error Handling

```cpp
// HIP error checking
hipError_t err = hipMalloc(&ptr, size);
if (err != hipSuccess) {
  const char* msg = hipGetErrorString(err);
  // Handle error
}

// HSA status checking
hsa_status_t status = hsa_init();
if (status != HSA_STATUS_SUCCESS) {
  const char* msg;
  hsa_status_string(status, &msg);
  // Handle error
}
```

## Component Integration

### HIP → ROCr Stack

```
Application (HIP API)
    ↓
hip-clr (HIP runtime implementation)
    ↓
ROCr Runtime (HSA implementation)
    ↓
Thunk (libhsakmt)
    ↓
Kernel Driver (amdgpu)
    ↓
AMD GPU Hardware
```

### Compiler Integration

```
Source Code (.cpp, .hip)
    ↓
hipcc (compiler driver)
    ↓
amd-llvm (LLVM/Clang with AMD backend)
    ↓
Device Code → GPU Binary (code object)
Host Code → CPU Binary
    ↓
Linked Application
    ↓
HIP Runtime (execution)
```

### OpenCL vs HIP

Both use the same underlying runtime:
```
OpenCL API → OpenCL Runtime → ROCr → Hardware
HIP API → HIP Runtime → ROCr → Hardware
```

## Testing

### Test Locations

```bash
# HIP tests (in hip-clr submodule)
build/dist/rocm/bin/test_hip_*

# ROCr tests
build/dist/rocm/bin/rocrtst

# Run via CTest
ctest --test-dir build -R hip
ctest --test-dir build -R rocr
```

### Test Categories

1. **Unit tests**: Test individual APIs
2. **Integration tests**: Test component interaction
3. **Performance tests**: Benchmark runtime operations
4. **Conformance tests**: Validate spec compliance

## Configuration

### CMake Options

```cmake
# HIP backend selection
-DHIP_PLATFORM=amd  # or nvidia for CUDA backend

# ROCr debug builds
-DROCR_BUILD_TYPE=Debug

# OpenCL features
-DBUILD_ICD=ON  # Build ICD loader

# Tests
-DTHEROCK_BUILD_TESTING=ON
```

### Environment Variables

```bash
# HIP device selection
export HIP_VISIBLE_DEVICES=0,1

# Enable HIP tracing
export HIP_TRACE_API=1

# ROCr debug logging
export HSA_ENABLE_DEBUG=1

# OpenCL platform selection
export OCL_ICD_VENDORS=/opt/rocm/etc/OpenCL/vendors
```

## Platform Support

### Linux
- **Full support**: Ubuntu, RHEL, SUSE
- **ROCr runtime**: Native Linux implementation
- **Kernel driver**: amdgpu (open-source)

### Windows
- **HIP support**: Native Windows implementation
- **ROCr runtime**: Windows port
- **Kernel driver**: AMD Windows driver

## Known Issues

From `post_hook_hip-clr.cmake` and development docs:

1. **hipcc wrapper**: Needs careful PATH setup to find LLVM
2. **Multi-GPU**: Device enumeration can be non-deterministic
3. **Memory pools**: Requires kernel driver support
4. **Windows builds**: Some features Linux-only

## Future Work

From RFCs:

1. **Unified runtime**: Merge HIP and ROCr runtime improvements
2. **SYCL support**: Add SYCL frontend to runtime
3. **Heterogeneous compute**: Better CPU+GPU integration
4. **Memory management**: Improved unified memory support
