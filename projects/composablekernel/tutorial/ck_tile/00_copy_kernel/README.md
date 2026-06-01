# CK Tile Framework: Getting Started with Tile Copy Operations

## Overview

### Copy Kernel
A minimal CK_Tile memory copy implementation demonstrating the basic setup required to write a kernel in CK Tile.
This experimental kernel is intended for novice CK developers. It introduces the building blocks of CK Tile and provides a sandbox for experimenting with kernel parameters.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture 
# (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Make the copy kernel executable
make tile_example_copy -j
```
This will result in an executable `build/bin/test_copy_basic`

## example
```
args:
          -m        input matrix rows. (default 64)
          -n        input matrix cols. (default 8)
          -id       wave to use for computation. (default 0)
          -v        validation flag to check device results. (default 1)
          -prec     datatype precision to use. (default fp16)
          -warmup   no. of warmup iterations. (default 50)
          -repeat   no. of iterations for kernel execution time. (default 100)
```

## CK Tile Architecture Components

The CK Tile framework is built around four key architectural components that work together to define and execute GPU kernels: shape, policy, problem, and pipeline.

### **1. Shape**
Defines the **hierarchical tile structure** and **memory layout** of the kernel:

```cpp
using Shape = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, ThreadTile>;
```

**Components:**
- **BlockWaves**: Number of concurrent waves per block (e.g., `seq<4, 1>` for 4 waves along M, 1 along N)
- **BlockTile**: Total elements processed by one block (e.g., `seq<512, 8>`)
- **WaveTile**: Elements processed by one wave (e.g., `seq<32, 8>`)
- **ThreadTile**: Elements processed by one thread (e.g., `seq<1, 4>` for 4 contiguous elements)

**Purpose**: Defines the **work distribution hierarchy** from threads → waves → blocks.

### **2. Problem**
Defines the **data types** and **kernel configuration**:

```cpp
using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
```

**Components:**
- **XDataType**: Input/output data type (e.g., `float`, `half`)
- **Shape**: The tile shape defined above

**Purpose**: Encapsulates **what** the kernel operates on and **how** it's configured.

### **3. Policy**
Defines the **memory access patterns** and **distribution strategies**:

```cpp
using Policy = ck_tile::TileCopyPolicy<Problem>;
```

**Key Functions:**
- **MakeDRAMDistribution()**: Defines how threads access DRAM memory.

**Purpose**: Defines **how** data is accessed and distributed across threads.

### **4. Pipeline**
Defines the **execution flow** and **memory movement patterns**:

```cpp
// Example pipeline stages:
// 1. DRAM → Registers (load_tile)
// 2. Registers → LDS (store_tile)
// 3. LDS → Registers (load_tile with distribution)
// 4. Registers → DRAM (store_tile)
```

**Purpose**: Defines the **sequence of operations** and **memory movement strategy**.

### **Component Interaction**

```cpp
// Complete kernel definition
using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, ThreadTile>;
using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
using Policy  = ck_tile::TileCopyPolicy<Problem>;
using Kernel  = ck_tile::TileCopyKernel<Problem, Policy>;
```

**Flow:**
1. **Shape** defines the tile structure and work distribution
2. **Problem** combines data types with the shape
3. **Policy** defines memory access patterns for the problem
4. **Kernel** implements the actual computation using all components

### **Why This Architecture?**

#### **Separation of Concerns**
- **Shape**: Focuses on **work distribution** and **tile structure**
- **Problem**: Focuses on **data types** and **configuration**
- **Policy**: Focuses on **memory access** and **optimization**
- **Pipeline**: Focuses on **execution flow** and **synchronization**

#### **Reusability**
- Same **Shape** can be used with different **Problems**
- Same **Policy** can be applied to different **Problems**
- **Pipelines** can be reused across different kernels

#### **Performance Optimization**
- **Shape** enables optimal work distribution
- **Policy** enables optimal memory access patterns
- **Pipeline** enables optimal execution flow

## Core Concepts

### Hierarchical Tile Structure

The CK Tile framework organizes work in a hierarchical manner:

1. **ThreadTile**: Number of contiguous elements processed by a single thread
   - Enables vectorized memory loads/stores.
   - Example: `ThreadTile = seq<1, 4>` means each thread loads 4 contiguous elements along the N dimension
   - A ThreadTile can be imagined as a thread-level tile

2. **WaveTile**: Number of elements covered by a single wave (64 threads on CDNA, 32 threads on RDNA)
   - Must satisfy: `Wave_Tile_M / ThreadTile_M * Wave_Tile_N / ThreadTile_N == WaveSize`
   - This ensures the number of threads needed equals the wave size
   - Example: `WaveTile = seq<64, 4>` with `ThreadTile = seq<1, 4>` means:
     - Each thread handles 4 elements (ThreadTile_N = 4)
     - Wave needs 64×4/4 = 64 threads to cover 64×4 = 256 elements
     - Total elements = 256, which requires WaveSize = 64 threads

3. **BlockTile**: Number of elements covered by one block (typically mapped to one CU)
   - Example: `BlockTile = seq<256, 64>` means each block processes 256×64 elements

4. **BlockWaves**: Number of concurrent waves active in a block
   - Typical: 4 waves for heavy workloads (e.g., GEMM)
   - Limit: up to 1024 threads per block → up to 16 waves (CDNA) or 32 waves (RDNA)
   - Example: `BlockWaves = seq<4, 1>` means 4 waves along M, 1 along N

### Wave Repetition

In many scenarios, the total work (BlockTile) is larger than what the available waves can cover in a single iteration. This requires **wave repetition**:

```cpp
// Calculate how many times a wave needs to repeat to cover the entire block tile
static constexpr index_t WaveRepetitionPerBlock_M =
    Block_Tile_M / (Waves_Per_Block_M * Wave_Tile_M);
static constexpr index_t WaveRepetitionPerBlock_N =
    Block_Tile_N / (Waves_Per_Block_N * Wave_Tile_N);
```

**Key Insight**: When waves repeat, the effective work per thread becomes `ThreadTile * Repeat`, not just `ThreadTile`.

## Tile Distribution Encoding

The tile distribution encoding specifies how work is distributed across threads:

```cpp
constexpr auto outer_encoding =
    tile_distribution_encoding<sequence<1>, // replication
                               tuple<sequence<M0, M1, M2>, sequence<N0, N1>>, // hierarchy
                               tuple<sequence<1>, sequence<1, 2>>, // parallelism
                               tuple<sequence<1>, sequence<2, 0>>,  // paralleism
                               sequence<1, 2>, // yield
                               sequence<0, 1>>{}; // yield
```

### Encoding Parameters Explained

- **M0, M1, M2**: Hierarchical distribution along M dimension
  - M0: Number of wave iterations along M
  - M1: Number of waves along M  
  - M2: Number of threads per wave along M
- **N0, N1**: Distribution along N dimension
  - N0: Number of threads along N
  - N1: ThreadTile size (elements per thread)
- **Order and layout**: The inner-most (rightmost) dimension is the fastest-changing. Choosing `N1 = ThreadTile_N` maps vector width to contiguous addresses, i.e., row-major access in this example.
- **YIELD arguments**: Both `Repeat` and `ThreadTile` because effective work per thread is `ThreadTile * Repeat`

## Tensor Abstractions

### Tensor Descriptor
Defines the logical structure of a tensor:
```cpp
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, N),           // tensor dimensions
    make_tuple(N, 1),           // strides
    number<ThreadTile_N>{},     // per-thread vector length
    number<1>{}                 // guaranteed last dimension vector stride
);
```

### Tensor View
Combines memory buffer with tensor descriptor:
```cpp
auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
    p_x,                        // memory buffer
    make_tuple(M, N),           // dimensions
    make_tuple(N, 1),           // strides  
    number<S::ThreadTile_N>{},  // per-thread vector length
    number<1>{}                 // guaranteed last dimension vector stride
);
```

### Tile Window
A view into a specific tile of the tensor with thread distribution:
```cpp
auto x_window = make_tile_window(
    x_m_n,                      // tensor view
    make_tuple(Block_Tile_M, Block_Tile_N),  // tile size
    {iM, 0},                    // tile origin
    tile_distribution           // how work is distributed among threads
);
```

## The test_copy_basic Kernel

### Kernel Structure

The `TileCopyKernel` implements a basic copy operation from input tensor `x` to output tensor `y`:

```cpp
template <typename Problem_, typename Policy_>
struct TileCopyKernel
{
    CK_TILE_DEVICE void operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N) const
    {
        // 1. Create tensor views
        // 2. Create tile windows  
        // 3. Iterate over N dimension tiles
        // 4. Load, copy, and store data
    }
};
```

### Step-by-Step Execution

1. **Tensor View Creation**:
   ```cpp
   const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
       p_x, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});
   ```
   - Creates views for both input and output tensors
   - Specifies vectorized access with `ThreadTile_N` elements per load

2. **Tile Window Creation**:
   ```cpp
   auto x_window = make_tile_window(x_m_n,
                                   make_tuple(number<S::Block_Tile_M>{}, number<S::Block_Tile_N>{}),
                                   {iM, 0},
                                   Policy::template MakeDRAMDistribution<Problem>());
   ```
   - Creates windows into specific tiles of the tensors
   - Each block processes one tile starting at `{iM, 0}`
   - Tile distribution determines how threads access data

3. **N-Dimension Iteration**:
   ```cpp
   index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_Tile_N));
   for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
   ```
   - If tensor N dimension > Block_Tile_N, multiple iterations are needed
   - Each iteration processes one tile along N dimension

4. **Load-Store Operations**:
   ```cpp
   dram_reg_tile dram_tile;
   load_tile(dram_tile, x_window);      // Load from global memory to registers
   store_tile(y_window, dram_tile);     // Store from registers to global memory
   move_tile_window(x_window, {0, S::Block_Tile_N});  // Move to next N tile
   move_tile_window(y_window, {0, S::Block_Tile_N});
   ```

### How Load/Store Works

1. **Load Tile**: 
   - Each thread loads its assigned elements based on tile distribution
   - Vectorized loads enable efficient memory bandwidth utilization
   - Data is distributed to per-thread register buffers

2. **Store Tile**:
   - Each thread writes its assigned elements back to global memory
   - Maintains the same distribution pattern as load

3. **Tile Window Movement**:
   - Moves the window to the next tile along N dimension
   - Enables processing of large tensors that don't fit in one tile

## Memory Access Patterns

### Vectorized Access
- Enabled by specifying vector length in tensor views
- Each thread loads/stores multiple contiguous elements in one operation
- Improves memory bandwidth utilization

### Thread Distribution
- Tile distribution encoding determines which threads access which elements
- Ensures all threads participate and no data is missed
- Enables memory coalescing for optimal performance

### Coordinate Transform (Embed)
- Maps multi-dimensional tensor indices to linear memory addresses
- Handles stride calculations automatically
- Enables efficient access to non-contiguous memory layouts
