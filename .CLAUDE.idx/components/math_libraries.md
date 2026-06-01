# Math Libraries Components

## Purpose

Math libraries provide GPU-accelerated mathematical operations including BLAS (Basic Linear Algebra Subprograms), sparse linear algebra, FFT (Fast Fourier Transform), random number generation, and parallel primitives. These are performance-critical libraries used by scientific computing, machine learning, and HPC applications.

## Location

- **Primary directory**: `math-libs/`
- **Subdirectories**:
  - `BLAS/` - BLAS-family libraries (rocBLAS, rocSPARSE, hipBLAS, hipSPARSE, hipBLASLt, hipSPARSELt)
  - `support/` - Support libraries (rocPRIM, rocWMMA, libhipcxx)
- **Artifact group**: `math-libs`
- **Type**: Mostly `target-specific` (built per GPU architecture)

## Dependencies

### Depends On
- **core-runtime**: hip-clr, rocr-runtime (for GPU execution)
- **compiler**: amd-llvm (for device code compilation)
- **base**: rocm-cmake, half
- **third-party**: googletest, boost, fftw3, host-blas

### Used By
- **ml-libs**: MIOpen uses rocBLAS
- **comm-libs**: RCCL uses math primitives
- **external-builds**: PyTorch, JAX depend heavily on math libs

## Artifacts (8 major libraries)

### BLAS Libraries

1. **rocBLAS** (target-specific)
   - AMD's GPU-accelerated BLAS implementation
   - Dense linear algebra: GEMM, GEMV, TRSM, etc.
   - Optimized kernels for AMD architectures
   - Includes Tensile kernel generator
   - Pre-compiled kernel database (via DVC)

2. **hipBLAS** (target-neutral)
   - Portable BLAS API (works on AMD and NVIDIA)
   - Wrapper around rocBLAS (AMD) or cuBLAS (NVIDIA)
   - Drop-in replacement for cuBLAS

3. **rocSPARSE** (target-specific)
   - Sparse BLAS operations
   - Sparse matrix formats: CSR, COO, ELL, etc.
   - SpMV, SpMM, triangular solve

4. **hipSPARSE** (target-neutral)
   - Portable sparse BLAS API
   - Wrapper around rocSPARSE/cuSPARSE

5. **hipBLASLt** (target-neutral)
   - BLAS-like operations with flexible layouts
   - Matrix multiplication with custom strides
   - Wrapper around rocBLASLt/cuBLASLt

6. **hipSPARSELt** (target-neutral)
   - Sparse matrix operations with structured sparsity
   - 2:4 structured sparsity support

### FFT Library

7. **rocFFT** (target-specific)
   - GPU-accelerated Fast Fourier Transform
   - 1D, 2D, 3D transforms
   - Complex-to-complex, real-to-complex
   - Batched transforms
   - Optimized kernels per architecture

### Random Number Generation

8. **rocRAND** (target-specific)
   - GPU random number generation
   - Distributions: uniform, normal, log-normal, Poisson
   - Pseudo-random: XORWOW, MRG32k3a, Philox, LFSR113
   - Quasi-random: Sobol, Scrambled Sobol

### Support Libraries

9. **rocPRIM** (target-neutral, header-only)
   - GPU parallel primitives
   - Block-level and device-level algorithms
   - Reductions, scans, sorts, radix sort
   - Used as foundation by other libraries

10. **rocWMMA** (target-neutral, header-only)
    - Wave Matrix Multiply-Accumulate
    - Low-level interface to matrix core instructions
    - Used by GEMM kernels

11. **libhipcxx** (target-neutral, header-only)
    - HIP C++ Standard Library
    - GPU-compatible std:: algorithms
    - Atomic operations, synchronization primitives

## Entry Points

### rocBLAS

```cpp
#include <rocblas/rocblas.h>

// Initialize
rocblas_handle handle;
rocblas_create_handle(&handle);

// Matrix multiplication: C = alpha*A*B + beta*C
rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

// Cleanup
rocblas_destroy_handle(handle);
```

### rocSPARSE

```cpp
#include <rocsparse/rocsparse.h>

// Initialize
rocsparse_handle handle;
rocsparse_create_handle(&handle);

// Sparse matrix-vector: y = alpha*A*x + beta*y
rocsparse_dcsrmv(handle, rocsparse_operation_none,
                 m, n, nnz, &alpha, descr, csr_val, csr_row_ptr, csr_col_ind,
                 x, &beta, y);

// Cleanup
rocsparse_destroy_handle(handle);
```

### rocFFT

```cpp
#include <rocfft/rocfft.h>

// Create plan
rocfft_plan plan;
rocfft_plan_create(&plan, rocfft_placement_notinplace,
                   rocfft_transform_type_complex_forward,
                   rocfft_precision_double, 1, &length, 1, nullptr);

// Execute
rocfft_execute(plan, in_buffers, out_buffers, nullptr);

// Cleanup
rocfft_plan_destroy(plan);
```

### rocRAND

```cpp
#include <rocrand/rocrand.h>

// Create generator
rocrand_generator gen;
rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_XORWOW);

// Generate random numbers
rocrand_generate_uniform_double(gen, data, n);

// Cleanup
rocrand_destroy_generator(gen);
```

### rocPRIM (header-only)

```cpp
#include <rocprim/rocprim.hpp>

// Device-level reduction
hipMalloc(&d_temp, temp_storage_bytes);
rocprim::reduce(d_temp, temp_storage_bytes,
                d_input, d_output, size,
                rocprim::plus<int>(), stream);
```

## Key Files

### rocBLAS Structure
- `library/src/blas*/` - BLAS level 1/2/3 implementations
- `library/src/blas3/Tensile/` - Tensile kernel generator
- `library/include/rocblas/rocblas.h` - Main API header
- `library/src/handle.cpp` - Handle management
- `clients/benchmarks/` - Performance benchmarks
- `clients/tests/` - Unit tests

### rocSPARSE Structure
- `library/src/level*/` - Sparse BLAS levels
- `library/src/conversion/` - Format conversion
- `library/include/rocsparse/rocsparse.h` - API header

### rocFFT Structure
- `library/src/device/` - GPU kernels
- `library/src/plan.cpp` - FFT plan creation
- `library/include/rocfft/rocfft.h` - API header

### rocRAND Structure
- `library/src/rng/` - RNG implementations
- `library/include/rocrand/rocrand.h` - API header

## Build Hooks

### rocBLAS Pre-hook (`pre_hook_rocBLAS.cmake`)

```cmake
function(rocblas_pre_hook)
  # Set Tensile options
  set(Tensile_LOGIC "asm_full" PARENT_SCOPE)
  set(Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" PARENT_SCOPE)
  
  # Use pre-built kernel database if available
  if(EXISTS "${DVC_KERNELS_PATH}/rocBLAS")
    set(Tensile_LIBRARY_FORMAT "msgpack" PARENT_SCOPE)
  endif()
endfunction()
```

### hipBLASLt Post-hook (`post_hook_hipblasltprovider.cmake`)

```cmake
function(hipblasltprovider_post_hook)
  # Install provider configuration
  install(FILES hipblaslt_providers.json
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipblaslt)
endfunction()
```

## Patterns

### Tensile Kernel Generation (rocBLAS)

rocBLAS uses Tensile to generate optimized GEMM kernels:

```python
# Tensile configuration
GlobalParameters:
  - MinimumRequiredVersion: 4.4.0
  - PrintLevel: 1
  
BenchmarkProblems:
  - ProblemType:
      DataType: d  # Double precision
      OperationType: GEMM
    Sizes:
      - Range: [[64, 4096, 64], [64, 4096, 64], [1]]
```

Tensile generates:
1. Assembly kernels (`.s` files)
2. Kernel selection logic (decision trees)
3. Msgpack database (serialized kernel metadata)

### Pre-compiled Kernel Databases

Large libraries (rocBLAS, hipBLASLt) use DVC for kernel databases:

```bash
# DVC pulls pre-compiled kernels
dvc pull library/src/blas3/Tensile/Logic/asm_full/

# Without DVC, kernels compile from source (hours)
```

### API Versioning

Libraries use semantic versioning in headers:

```cpp
#define ROCBLAS_VERSION_MAJOR 4
#define ROCBLAS_VERSION_MINOR 2
#define ROCBLAS_VERSION_PATCH 0
```

## Performance Optimization

### Kernel Selection

Libraries select kernels based on:
1. **Matrix dimensions**: Small, medium, large
2. **GPU architecture**: gfx90a, gfx942, gfx1100
3. **Data layout**: Row-major vs column-major
4. **Data type**: FP64, FP32, FP16, INT8

Example (rocBLAS):
```cpp
// Kernel selection logic
if (m < 64 && n < 64) {
  // Small kernel
  return gemm_small_kernel;
} else if (arch == gfx90a) {
  // gfx90a optimized
  return gemm_gfx90a_large_kernel;
} else {
  // Fallback
  return gemm_generic_kernel;
}
```

### Memory Access Patterns

Optimized for:
- **Coalesced access**: Threads access contiguous memory
- **Shared memory**: Reduce global memory traffic
- **Register blocking**: Keep data in registers

### Wave-level Primitives

Use GPU wave instructions:
```cpp
// rocPRIM wave reduction
__device__ int wave_reduce(int value) {
  return rocprim::warp_reduce<int>().reduce(value, rocprim::plus<int>());
}
```

## Testing

### Test Organization

```bash
# Unit tests
build/dist/rocm/bin/rocblas-test
build/dist/rocm/bin/rocsparse-test
build/dist/rocm/bin/rocfft-test

# Benchmarks
build/dist/rocm/bin/rocblas-bench --function gemm --precision f64_r
build/dist/rocm/bin/rocfft-bench --length 1024 --batch 100

# Via CTest
ctest --test-dir build -R rocblas
```

### Performance Validation

Benchmarks compare against reference:
- **CPU BLAS**: Intel MKL, OpenBLAS
- **CPU FFT**: FFTW
- Target: ≥90% of theoretical peak performance

## Configuration

### CMake Options

```cmake
# rocBLAS
-DBUILD_WITH_TENSILE=ON  # Enable Tensile (required for performance)
-DTensile_LOGIC=asm_full  # Full assembly kernels
-DTensile_LIBRARY_FORMAT=msgpack  # Use msgpack database

# rocFFT
-DROCFFT_DEVICE_KERNELS_PATH=/path/to/prebuilt  # Use pre-compiled kernels

# All math libs
-DBUILD_CLIENTS_TESTS=ON  # Build tests
-DBUILD_CLIENTS_BENCHMARKS=ON  # Build benchmarks
```

### Environment Variables

```bash
# rocBLAS
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
export ROCBLAS_LAYER=2  # Enable logging

# rocFFT
export ROCFFT_LOG_TRACE_PATH=/tmp/rocfft_trace.log
export ROCFFT_LAYER=3  # Verbose logging

# rocRAND
export ROCRAND_LOG_TRACE_PATH=/tmp/rocrand.log
```

## Artifact Types

### Target-Specific Libraries
Built separately per GPU architecture:
- **rocBLAS**: Each GPU family gets its own kernel database
- **rocFFT**: Architecture-specific kernels
- **rocRAND**: Optimized per GPU

Distributed as:
```
rocblas-gfx90a-6.5.0.tar.xz
rocblas-gfx942-6.5.0.tar.xz
rocblas-gfx1100-6.5.0.tar.xz
```

### Target-Neutral Libraries
Single build with all architectures:
- **hipBLAS**: Wrapper library (runtime GPU detection)
- **rocPRIM**: Header-only
- **libhipcxx**: Header-only

## Known Issues

From build hooks and RFCs:

1. **Tensile build time**: Assembly kernel generation is slow (hours)
   - **Workaround**: Use DVC to pull pre-built kernels
   
2. **Kernel database size**: Can be several GB per architecture
   - **Solution**: Split databases in kpack artifact
   
3. **hipBLAS provider selection**: Runtime overhead
   - **Future**: Compile-time provider selection

4. **Memory allocation**: Large working sets can cause OOM
   - **Workaround**: Use rocBLAS workspace APIs

## Platform Support

### Linux
- **Full support**: All libraries
- **Optimizations**: Native assembly kernels

### Windows  
- **Partial support**: Most libraries work
- **Limitations**: Some Tensile features disabled
- **Performance**: May be lower than Linux

## Future Work

From RFCs:

1. **Unified BLAS stack**: Consolidate rocBLAS/hipBLAS/hipBLASLt
2. **Mixed precision**: Better FP16/BF16/TF32 support
3. **Sparse-dense operations**: Hybrid sparse/dense kernels
4. **Auto-tuning**: Runtime kernel selection based on profiling
5. **libhipcxx expansion**: More C++17/20 features on GPU
