# Practice GEMM: Step-by-Step Code Walkthrough

This document provides a detailed walkthrough of `practice_gemm.cpp`, explaining each step of implementing a GEMM (General Matrix Multiplication) kernel using the CK Tile API.

## Overview

We'll implement `C = A × B` where:
- `A` is an `M × K` matrix
- `B` is an `N × K` matrix (note: transposed layout)
- `C` is an `M × N` matrix

The implementation uses a hierarchical tiling strategy with two levels:
1. **Block Tiles**: Processed by thread blocks
2. **Wave Tiles**: Processed by warps (wavefronts) within blocks

---

## Step 1: Define Data Types

```cpp
using ADataType   = ck_tile::half_t;
using BDataType   = ck_tile::half_t;
using CDataType   = float;
using AccDataType = float;
```

**What's happening:**
- We use `half_t` (FP16) for input matrices A and B.
- We use `float` (FP32) for output matrix C and accumulation for numerical accuracy
- In typical CK examples, this information is part of a `GemmConfig` struct, but here we define it directly for simplicity
---

## Step 2: Define Problem Size

```cpp
ck_tile::index_t M = 512;
ck_tile::index_t N = 256;
ck_tile::index_t K = 64;
ck_tile::index_t verification = 1;

ck_tile::index_t stride_a = K;
ck_tile::index_t stride_b = K;
ck_tile::index_t stride_c = N;
```

**What's happening:**
- `M = 512`: Number of rows in A and C
- `N = 256`: Number of columns in B and C
- `K = 64`: Inner dimension (columns of A, rows of B)
- Strides define memory layout (row-major for A and C, transposed for B)

**Memory Layout:**
```
Matrix A (M×K):        Matrix B (N×K):        Matrix C (M×N):
[512 rows]             [256 rows]             [512 rows]
[64 cols]              [64 cols]              [256 cols]
stride = K             stride = K             stride = N
```

---

## Step 3: Create Host Tensors

```cpp
auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};

auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};
auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};
auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};

ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
ck_tile::HostTensor<CDataType> c_host(c_lengths, c_strides);
```

**What's happening:**
- We create three tensors on the host (CPU) memory
- Each tensor is defined by its shape (`lengths`) and memory layout (`strides`)
- `HostTensor` is a CK Tile utility class that manages CPU memory

**Stride explanation:**
- For A: `stride_a = K` means moving to the next row requires skipping K elements
- For B: `stride_b = K` means B is stored in transposed format
- For C: `stride_c = N` means row-major layout

---

## Step 4: Initialize Tensors with Random Data

```cpp
ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_host);
ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_host);
c_host.SetZero();
```

**What's happening:**
- A and B are filled with random values in the range [-5.0, 5.0]
- C is initialized to zero (will store the output)

**Optional: Print Tensor Contents**
```cpp
// Commented out in the code, but available for debugging:
// a_host.print_first_n(10);  // Print first 10 elements of A
```

The `print_first_n()` helper function can display tensor contents for debugging purposes.

---

## Step 5: Allocate Device Memory and Transfer Data

```cpp
ck_tile::DeviceMem a_device(a_host);
ck_tile::DeviceMem b_device(b_host);
ck_tile::DeviceMem c_device(c_host);
```

**What's happening:**
- `DeviceMem` allocates GPU memory matching the size of host tensors
- The constructor **automatically transfers data from host to device**
- This is a convenience wrapper around `hipMalloc` and `hipMemcpy`

**Memory Flow:**
```
CPU (Host)              GPU (Device)
┌─────────┐            ┌─────────┐
│ a_host  │ ────────>  │a_device │
│ b_host  │ ────────>  │b_device │
│ c_host  │ ────────>  │c_device │
└─────────┘            └─────────┘
```

---

## Step 6: Configure Hierarchical Tiling

```cpp
using BlockTile = ck_tile::sequence<256, 128, 32>;
using WaveTile  = ck_tile::sequence<16, 16, 16>;
```

**What's happening:**
- We define a two-level tiling hierarchy for the GEMM computation

### Block Tile (256 × 128 × 32)
- **256**: M dimension per block (rows of A and C)
- **128**: N dimension per block (columns of B and C)
- **32**: K dimension per block (inner dimension)
- Each block tile is processed by one **thread block** (256 threads)

### Wave Tile (16 × 16 × 16)
- **16 × 16**: Output tile dimensions (M × N) per warp iteration
- **16**: K dimension per warp iteration
- Each wave tile is processed by one **warp** (64 threads on AMD GPUs)

**Important:** The WaveTile (16×16×16) is NOT the same as the MFMA instruction size (32×32×8). The WaveTile represents the work done per warp per iteration, while MFMA is the underlying hardware instruction. Multiple MFMA operations may be needed to compute one wave tile

**Important Note:**
In this example, the problem size (256 × 128 × 32) is **identical** to the block tile size, so only **one thread block** is needed to compute the entire problem.

### Tiling Visualization:

#### Matrix A (M × K = 256 × 32):
```
┌─────────────────────────────────────┐
│  One Block Tile (256 × 32)          │
│  ┌────┬────┐                        │
│  │16×│16× │  ← Wave tiles (16×16)   │
│  │ 16│ 16 │     in M×K space        │
│  ├────┼────┤                        │
│  │    │    │                        │
│  ├────┼────┤                        │
│  │ .. │ .. │  16 tiles in M         │
│  ├────┼────┤  2 tiles in K          │
│  │    │    │                        │
│  └────┴────┘                        │
│                                     │
└─────────────────────────────────────┘
```

#### Matrix B (N × K = 128 × 32):
```
┌──────────────────────────────┐
│  One Block Tile (128 × 32)   │
│  ┌────┬────┐                 │
│  │16×│16× │  ← Wave tiles    │
│  │ 16│ 16 │     (16×16)      │
│  ├────┼────┤                 │
│  │    │    │                 │
│  ├────┼────┤  8 tiles in N   │
│  │ .. │ .. │  2 tiles in K   │
│  ├────┼────┤                 │
│  │    │    │                 │
│  └────┴────┘                 │
└──────────────────────────────┘
```

#### Matrix C (M × N = 256 × 128) - Output:
```
┌─────────────────────────────────────────────────┐
│  One Block Tile (256 × 128)                     │
│                                                  │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐     │
│  │16× │    │    │    │    │    │    │    │     │
│  │ 16 │    │    │    │    │    │    │    │     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤     │
│  │    │    │    │    │    │    │    │    │     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤     │
│  │    │    │    │    │    │    │    │    │     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤     │
│  │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤     │
│  │    │    │    │    │    │    │    │    │     │
│  └────┴────┴────┴────┴────┴────┴────┴────┘     │
│                                                  │
│  16 wave tiles in M direction                   │
│  8 wave tiles in N direction                    │
│  Total: 128 wave tiles (16×16 each)             │
└─────────────────────────────────────────────────┘
```

#### How Wave Tiles Combine (C = A × B):
```
Matrix A          Matrix B (stored transposed N×K)          Matrix C
(256×32)          (128×32)                                  (256×128)

Row of A tiles:   Row of B tiles:                One wave tile in C:
┌────┬────┐      ┌────┬────┐                    ┌────┐
│ A₀ │ A₁ │  ×   │ B₀ │ B₁ │                =   │ C  │ (16×16)
└────┴────┘      └────┴────┘                    └────┘
  16×16 each       16×16 each

Computation: C = A₀×B₀ᵀ + A₁×B₁ᵀ
             ↑             ↑
          K=0..15      K=16..31
          
Each wave tile in C is computed by:
- Taking one row of wave tiles from A (2 tiles along K)
- Taking one row of wave tiles from B (2 tiles along K)
  Note: B is stored transposed (N×K), so a "row" in storage corresponds 
  to a "column" in the logical B^T matrix used in computation
- Performing dot product: Σ(A_k × B_k^T) for k=0,1
```

**Key Insight:**
- Each **wave tile in C** (16×16) requires a **dot product** of 2 wave tiles from A and 2 wave tiles from B
- Since B is stored transposed (N×K layout), we access **rows** of B tiles in memory
- This is the fundamental operation repeated across all 128 wave tiles in C
- Each warp computes one wave tile using MFMA instructions

---

## Step 7: Create Shape, Problem, and Policy Structs

```cpp
using PracticeGemmShape = ck_tile::PracticeGemmShape<BlockTile, WaveTile>;
std::cout << "PracticeGemmShape: " << PracticeGemmShape::GetName() << std::endl;

using PracticeGemmHostProblem = ck_tile::
    PracticeGemmHostProblem<ADataType, BDataType, CDataType, AccDataType, PracticeGemmShape>;

using PracticeGemmHostPolicy = ck_tile::PracticeGemmHostPolicy;
```

**What's happening:**

### 1. **Shape Struct**
Encapsulates all tile shape information (BlockTile and WaveTile dimensions).

### 2. **Problem Struct**
Holds complete problem description:
- Data types (ADataType, BDataType, CDataType, AccDataType)
- Shape information (BlockTile, WaveTile)

In more complex examples, this would also include:
- Data layouts (row-major, column-major)
- Mathematical operations (e.g., transposed GEMM)

### 3. **Policy Struct**
Describes data movement and thread-to-data mapping:
- Currently contains `MakeBlock2TileMap()`: Maps thread block IDs to tile positions
- In more complex kernels, includes:
  - DRAM access patterns
  - LDS (Local Data Share) usage strategies
  - Thread distribution within blocks

**CK Tile Design Pattern:**
```
Kernel = Problem + Policy + Epilogue
         ↑         ↑        ↑
      (What)    (How)   (Post-processing)
```

---

## Step 8: Calculate Grid and Block Dimensions

```cpp
ck_tile::index_t kGridSize = ck_tile::integer_divide_ceil(M, PracticeGemmShape::BlockTile_M) *
                             ck_tile::integer_divide_ceil(N, PracticeGemmShape::BlockTile_N);

std::cout << "kGridSize: " << kGridSize << std::endl;

constexpr ck_tile::index_t kBlockSize = 256;
constexpr ck_tile::index_t kBlockPerCU = 1;
```

**What's happening:**

### Grid Size Calculation
```cpp
kGridSize = ceil(M / BlockTile_M) × ceil(N / BlockTile_N)
          = ceil(512 / 256) × ceil(256 / 128)
          = 2 × 2
          = 4 thread blocks
```

Our problem requires **4 thread blocks** to cover the entire output matrix C (2 blocks in M direction, 2 blocks in N direction).

### Block Configuration
- `kBlockSize = 256`: Each thread block has 256 threads
  - 256 threads / 64 threads per warp = **4 warps per block**
- `kBlockPerCU = 1`: Launch 1 block per Compute Unit (for simplicity)

**Thread Hierarchy:**
```
GPU
└── 1 Thread Block (Grid)
    └── 256 Threads
        ├── Warp 0 (threads 0-63)
        ├── Warp 1 (threads 64-127)
        ├── Warp 2 (threads 128-191)
        └── Warp 3 (threads 192-255)
```

---

## Step 9: Create and Launch the Kernel

```cpp
using gemm_kernel =
    ck_tile::PracticeGemmKernel<PracticeGemmHostProblem, PracticeGemmHostPolicy>;

float ave_time = ck_tile::launch_kernel(
    ck_tile::stream_config{nullptr, true, 0, 0, 1},
    ck_tile::make_kernel<kBlockPerCU>(gemm_kernel{},
                                      kGridSize,
                                      kBlockSize,
                                      0,
                                      static_cast<ADataType*>(a_device.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      stride_a,
                                      stride_b,
                                      stride_c));
```

**What's happening:**

### 1. Kernel Composition
```cpp
using gemm_kernel = ck_tile::PracticeGemmKernel<Problem, Policy>;
```
The kernel is composed from Problem and Policy structs, following the CK Tile design pattern.

### 2. Kernel Launch
`launch_kernel()` is a CK Tile utility that:
- Launches the GPU kernel using HIP runtime
- Measures execution time
- Returns average execution time in milliseconds

### 3. Launch Parameters
- **Stream config**: `{nullptr, true, 0, 0, 1}` - default stream, timing enabled
- **Grid size**: `kGridSize = 1` - number of thread blocks
- **Block size**: `kBlockSize = 256` - threads per block
- **Shared memory**: `0` - no dynamic shared memory in this example
- **Kernel arguments**: Device pointers and problem dimensions

### 4. Kernel Execution Flow
```
launch_kernel() calls gemm_kernel.operator()()
    ↓
PracticeGemmKernel::operator()
    ↓
Creates tensor views over device memory
    ↓
Calls block-level pipeline
    ↓
Block pipeline calls warp-level pipeline
    ↓
Warp pipeline calls MFMA instructions
    ↓
Results written back to C matrix
```

---

## Step 10: Verify Results

```cpp
auto pass = true;

if(verification)
{
    // Reference gemm on CPU
    ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
    reference_basic_gemm<ADataType, BDataType, AccDataType, CDataType>(
        a_host, b_host, c_host_ref);
    
    // Copy GPU results back to host
    ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);
    c_device.FromDevice(c_host_dev.mData.data());
    
    // Compare results
    pass &= ck_tile::check_err(c_host_dev, c_host_ref, "Error: Incorrect results!", 1e-3, 1e-3);
    std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
}
```

**What's happening:**

### 1. CPU Reference Implementation
```cpp
reference_basic_gemm<...>(a_host, b_host, c_host_ref);
```
Computes GEMM on CPU using a simple nested loop implementation (ground truth).

### 2. Copy GPU Results to Host
```cpp
c_device.FromDevice(c_host_dev.mData.data());
```
Transfers the computed result from GPU memory back to CPU for comparison.

### 3. Error Checking
```cpp
ck_tile::check_err(c_host_dev, c_host_ref, "Error: Incorrect results!", 1e-3, 1e-3);
```
Compares GPU and CPU results element-wise with tolerance:
- **Relative error**: 1e-3 (0.1%)
- **Absolute error**: 1e-3

**Verification Flow:**
```
CPU                     GPU
┌─────────┐            ┌─────────┐
│ a_host  │ ────────>  │a_device │
│ b_host  │ ────────>  │b_device │
└─────────┘            └─────────┘
     │                      │
     ↓                      ↓
reference_gemm()       GPU kernel
     │                      │
     ↓                      ↓
┌──────────┐          ┌──────────┐
│c_host_ref│          │c_device  │
└──────────┘          └──────────┘
     │                      │
     │                      ↓
     │                 FromDevice()
     │                      │
     ↓                      ↓
     └────> check_err() <───┘
                 │
                 ↓
            Pass/Fail
```

---

## Complete Execution Flow Summary

```
1. Define data types (FP16 inputs, FP32 output)
   ↓
2. Set problem size (M=256, N=128, K=32)
   ↓
3. Create host tensors and initialize with random data
   ↓
4. Allocate device memory and transfer data (CPU → GPU)
   ↓
5. Configure hierarchical tiling (BlockTile, WaveTile)
   ↓
6. Create Shape, Problem, and Policy structs
   ↓
7. Calculate grid/block dimensions (1 block, 256 threads)
   ↓
8. Compose and launch kernel (Problem + Policy)
   ↓
9. Execute GEMM on GPU
   │  ├─ Block-level pipeline
   │  ├─ Warp-level pipeline
   │  └─ MFMA instructions
   ↓
10. Verify results (compare GPU vs CPU reference)
    ↓
11. Calculate and print performance metrics
    ↓
12. Return success/failure
```

---