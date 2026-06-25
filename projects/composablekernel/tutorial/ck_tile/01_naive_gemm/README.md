# CK Tile Practice GEMM Example

This is a practice implementation of a GEMM (General Matrix Multiplication) kernel using the CK Tile API. It demonstrates the fundamental concepts of GPU kernel development using CK Tile's hierarchical tile system.

## CK Tile API Structure

In the composable_kernel library's ck_tile API, **A Kernel is composed of a Problem, a Policy and an Epilogue**:

1. **Problem** describes the shape, data type, data layout, precision of our GEMM matrices
2. **Policy** describes how the data in the matrix (or tile) is mapped to the threads
3. **Epilogue** describes additional computation work performed after the gemm computations (this example does not have an epilogue)

## Overview

This example implements a complete GEMM kernel `C = A × B` using the CK Tile framework, showcasing:

- **Problem Setup** - Setting up the problem (input/output shapes, data types, mathematical operations), composing a kernel (pipeline, policy, epilogue), kernel launch
- **Block-level Pipelining** - creating tensor views, dispatching to block-level GEMM
- **Block-level GEMM Computation** - Block tiles, tile window creation, loading/storing to DRAM and Register memory
- **Warp-level GEMM Computation** - Warp tiles, MFMA level computation

## Problem Setup and Data Flow

### Problem Size Configuration
We set the problem size using the M, N and K variables:
```cpp
ck_tile::index_t M = 1024;   // Number of rows in A and C
ck_tile::index_t N = 512;  // Number of columns in B and C
ck_tile::index_t K = 256;  // Number of columns in A, rows in B
```

### Host Matrix Creation
Three host matrices A (M×K), B (N×K) and C (M×N) are created, initialized on the CPU and copied over to the GPU global/DRAM memory:
```cpp
// Host tensors with proper strides
ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);  // M × K
ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);  // N × K
ck_tile::HostTensor<CDataType> c_host(c_lengths, c_strides);  // M × N

// Initialize with random data
ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

// Allocate device memory and transfer data
ck_tile::DeviceMem a_device(a_host);
a_device.ToDevice(a_host.data());
```

### PracticeGemmShape Configuration
A PracticeGemmShape struct holds the dimension of each BlockTile and WaveTile:

```cpp
using BlockTile = ck_tile::sequence<256, 128, 32>;  // M, N, K per block
using WaveTile  = ck_tile::sequence<16, 16, 16>;   // M, N, K per wave
```
- A BlockTile of size MxK (256x32) on A matrix and NxK (128x32) on B matrix. A WaveTile of size MxN (16x16) on C matrix.


- BlockTiles iterate in K dimension to fetch data required for computing region of C covered by C's block tile.
- BlockTiles are further subdivided into WarpTiles.
- WarpTiles over A and B similarly work together to calculate the WarpTile of C.

### Problem and Policy Composition
```cpp
// A Problem is composed from Shape and info about the data
using PracticeGemmHostProblem = ck_tile::
    PracticeGemmHostProblem<ADataType, BDataType, CDataType, AccDataType, PracticeGemmShape>;

// A Policy is created describing data-to-thread mapping
using PracticeGemmHostPolicy = ck_tile::PracticeGemmHostPolicy;

// A Kernel is then composed of Problem and Policy
using gemm_kernel = ck_tile::PracticeGemmKernel<PracticeGemmHostProblem, PracticeGemmHostPolicy>;
```

### Kernel Launch
`ck_tile::launch_kernel()` is used to launch the kernel on device. It calls the `operator()` function of `PracticeGemmKernel{}`:
```cpp
float ave_time = ck_tile::launch_kernel(
    ck_tile::stream_config{nullptr, true, 0, 0, 1},
    ck_tile::make_kernel<kBlockSize, kBlockPerCU>(
        gemm_kernel{},  // Kernel composed of Problem + Policy
        kGridSize,      // Grid dimensions
        kBlockSize,     // Block dimensions
        0,              // Dynamic shared memory
        // Kernel arguments: device buffers and problem dimensions
        a_device.GetDeviceBuffer(), b_device.GetDeviceBuffer(), c_device.GetDeviceBuffer(),
        M, N, K, stride_a, stride_b, stride_c));
```

### Result Verification
The results from the kernel are compared with results from CPU based computation function:
```cpp
// CPU reference implementation
ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
reference_basic_gemm<ADataType, BDataType, AccDataType, CDataType>(a_host, b_host, c_host_ref);

// Device results
ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);

// Verify correctness
bool pass = ck_tile::check_err(c_host_dev, c_host_ref);
```

### Runtime Flow

The main program (`practice_gemm.cpp`) is the entry point for the runtime flow:

```cpp
int main()
{
    // 1. Define data types and problem sizes
    using ADataType = ck_tile::half_t;
    ck_tile::index_t M = 2048, N = 1024, K = 512;

    // 2. Create host tensors and initialize
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);

    // 3. Allocate device memory and transfer data
    ck_tile::DeviceMem a_device(a_host);

    // 4. Configure tile shapes
    using BlockTile = ck_tile::sequence<256, 128, 32>;
    using WaveTile  = ck_tile::sequence<16, 16, 16>;

    // 5. Launch kernel
    using gemm_kernel = ck_tile::PracticeGemmKernel<Problem, Policy>;
    float ave_time = ck_tile::launch_kernel(/*...*/);

    // 6. Verify results
    bool pass = verify_results(a_host, b_host, c_host);

    // 7. Print performance metrics
    print_performance_metrics(ave_time, M, N, K);
}
```

## Building and Running

```bash
# From composable_kernel root directory
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_practice_gemm -j

# Run with sample sizes
./bin/tile_example_practice_gemm
```
This example serves as a foundation for understanding more complex GEMM implementations and optimization strategies in the CK Tile framework.
