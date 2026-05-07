# Host-Level Pipeline: Orchestrating Block-Level GEMM

This document explains the **host-level pipeline** (`PracticeGemmHostPipeline`), which orchestrates the distribution of work across thread blocks and manages the high-level flow of the GEMM computation.

## Overview

The host-level pipeline is responsible for:
1. **Calculating tile coverage**: How many tiles are needed to cover matrices A, B, and C
2. **Block-to-tile mapping**: Assigning each thread block to a specific tile
3. **Creating tile windows**: Establishing sliding windows over tensor views
4. **Delegating computation**: Calling the block-level pipeline to perform actual GEMM
5. **Storing results**: Writing computed tiles from registers (VGPRs) back to DRAM

```cpp
template <typename Problem_, typename Policy_ = PracticeGemmHostPolicy>
struct PracticeGemmHostPipeline
{
    template <typename ADRAMTensorView, typename BDRAMTensorView, typename CDRAMTensorView>
    CK_TILE_DEVICE void operator()(const ADRAMTensorView& a_dram,
                                   const BDRAMTensorView& b_dram,
                                   CDRAMTensorView& c_dram) const
    {
        // 1. Calculate problem dimensions and tile coverage
        // 2. Map thread block to tile coordinates
        // 3. Create tile windows over A and B
        // 4. Call block-level pipeline to compute
        // 5. Store result to C
    }
};
```

---

## Step 1: Calculate Problem Dimensions and Tile Coverage

```cpp
// Size of the entire problem
const auto M = a_dram.get_tensor_descriptor().get_length(number<0>{}); // M x K
const auto N = c_dram.get_tensor_descriptor().get_length(number<1>{}); // M x N
const auto K = a_dram.get_tensor_descriptor().get_length(number<1>{}); // M x K

// Size of the block tile
const auto MPerBlock = BlockTile::at(number<0>{});  // 256
const auto NPerBlock = BlockTile::at(number<1>{});  // 128
const auto KPerBlock = BlockTile::at(number<2>{});  // 32

// Number of block tiles needed to cover C matrix
const auto num_tile_n = integer_divide_ceil(N, NPerBlock);  // ceil(256/128) = 2
const auto num_tile_m = integer_divide_ceil(M, MPerBlock);  // ceil(512/256) = 2
```

### What's Happening:

1. **Extract problem dimensions** from tensor descriptors:
   - `M = 512`: Rows in A and C
   - `N = 256`: Columns in B and C
   - `K = 64`: Inner dimension (columns of A, rows of B)

2. **Get block tile sizes** from the `BlockTile` configuration:
   - `MPerBlock = 256`: Each block processes 256 rows
   - `NPerBlock = 128`: Each block processes 128 columns
   - `KPerBlock = 32`: Each block processes 32 elements in K dimension per iteration

3. **Calculate tile coverage**:
   - `num_tile_m = ceil(M / MPerBlock) = ceil(512/256) = 2` tiles in M direction
   - `num_tile_n = ceil(N / NPerBlock) = ceil(256/128) = 2` tiles in N direction
   - **Total tiles = 2 × 2 = 4 tiles** → We need **4 thread blocks**!

### Visual Representation:

```
Matrix C (512 × 256):
┌──────────────────────┬──────────────────────┐
│   Tile (0,0)         │   Tile (0,1)         │  ← num_tile_n = 2
│   256×128            │   256×128            │
│   Block 0            │   Block 1            │
│                      │                      │
├──────────────────────┼──────────────────────┤
│   Tile (1,0)         │   Tile (1,1)         │
│   256×128            │   256×128            │
│   Block 2            │   Block 3            │
│                      │                      │
└──────────────────────┴──────────────────────┘
         ↑
    num_tile_m = 2

Total blocks needed = 2 × 2 = 4 blocks

Each block computes one 256×128 tile of the output matrix C.
```

### How Blocks Cover Matrices A and B:

```
Matrix A (512 × 64):                Matrix B (256 × 64):
┌─────────────┬──────┐             ┌─────────────┬──────┐
│ Block 0,2   │  K   │             │ Block 0,1   │  K   │
│ uses rows   │  →   │             │ uses rows   │  →   │
│ 0-255       │      │             │ 0-127       │      │
├─────────────┼──────┤             ├─────────────┼──────┤
│ Block 1,3   │  K   │             │ Block 2,3   │  K   │
│ uses rows   │  →   │             │ uses rows   │  →   │
│ 256-511     │      │             │ 128-255     │      │
└─────────────┴──────┘             └─────────────┴──────┘
   256 rows    64 cols                128 rows    64 cols
   
Each block needs to iterate over K dimension (64/32 = 2 iterations)
```

---

## Step 2: Map Thread Block to Tile Coordinates

```cpp
// Get block id (0 to total_blocks - 1)
const auto id_block = get_block_id();

// Map block id to 2D tile coordinates
const auto block2tile = Policy::MakeBlock2TileMap(num_tile_m, num_tile_n);
const auto tile_id = block2tile(id_block);

const auto tile_id_m = tile_id.at(number<0>{});  // M coordinate
const auto tile_id_n = tile_id.at(number<1>{});  // N coordinate
```

### What's Happening:

Each thread block needs to know **which tile of the output matrix C it should compute**. The `MakeBlock2TileMap` function creates a mapping from linear block ID to 2D tile coordinates.

### The `MakeBlock2TileMap` Function:

```cpp
CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
{
    // Create a merge transform: (N0, M0) → linear index
    const auto unmerge = make_merge_transform(make_tuple(N0, M0));

    return [unmerge](index_t block_id) {
        multi_index<2> unmerged;
        // Convert linear block_id back to 2D coordinates
        unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

        // Return (m_idx, n_idx) - note the swap!
        return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));
    };
}
```

### In Our Example (2×2 Grid):

```cpp
// Block 0:
id_block = 0
tile_id = block2tile(0) = (0, 0)  // Top-left tile
tile_id_m = 0, tile_id_n = 0

// Block 1:
id_block = 1
tile_id = block2tile(1) = (1, 0)  // Bottom-left tile
tile_id_m = 1, tile_id_n = 0

// Block 2:
id_block = 2
tile_id = block2tile(2) = (0, 1)  // Top-right tile
tile_id_m = 0, tile_id_n = 1

// Block 3:
id_block = 3
tile_id = block2tile(3) = (1, 1)  // Bottom-right tile
tile_id_m = 1, tile_id_n = 1
```

**Key Point**: Each of the 4 blocks knows exactly which 256×128 tile of C it's responsible for computing!

---

## Step 3: Calculate Tile Origin and Create Tile Windows

```cpp
// Calculate the starting position of this tile in the global matrix
const auto tile_origin_m = tile_id_m * MPerBlock;  // e.g., Block 1: 1 * 256 = 256
const auto tile_origin_n = tile_id_n * NPerBlock;  // e.g., Block 2: 1 * 128 = 128

// Create tile windows over A and B tensor views
const auto a_block_window = make_tile_window(
    a_dram,                                      // Tensor view over A
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),  // Window size: 256×32
    {tile_origin_m, 0}                          // Origin: varies by block
);

const auto b_block_window = make_tile_window(
    b_dram,                                      // Tensor view over B
    make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),  // Window size: 128×32
    {tile_origin_n, 0}                          // Origin: varies by block
);
```

### Tile Origins for Each Block:

```cpp
// Block 0 (Tile 0,0):
tile_origin_m = 0 * 256 = 0
tile_origin_n = 0 * 128 = 0
a_block_window origin: (0, 0)    → covers A rows 0-255
b_block_window origin: (0, 0)    → covers B rows 0-127

// Block 1 (Tile 1,0):
tile_origin_m = 1 * 256 = 256
tile_origin_n = 0 * 128 = 0
a_block_window origin: (256, 0)  → covers A rows 256-511
b_block_window origin: (0, 0)    → covers B rows 0-127

// Block 2 (Tile 0,1):
tile_origin_m = 0 * 256 = 0
tile_origin_n = 1 * 128 = 128
a_block_window origin: (0, 0)    → covers A rows 0-255
b_block_window origin: (128, 0)  → covers B rows 128-255

// Block 3 (Tile 1,1):
tile_origin_m = 1 * 256 = 256
tile_origin_n = 1 * 128 = 128
a_block_window origin: (256, 0)  → covers A rows 256-511
b_block_window origin: (128, 0)  → covers B rows 128-255
```

### What are Tile Windows?

A **tile window** is a **sliding window** over a larger tensor view. It:
- Defines a **rectangular region** within the tensor
- Has a **fixed size** (e.g., 256×32 for A)
- Has an **origin** (starting position)
- Can be **moved** to access different regions
### Visual Representation (Block 0 Example):

```
Matrix A (512 × 64):                    Matrix B (256 × 64):
┌─────────────┬─────────────┐          ┌─────────────┬─────────────┐
│ ┏━━━━━━━━━┓ │             │          │ ┏━━━━━━━━━┓ │             │
│ ┃ Window  ┃ │             │          │ ┃ Window  ┃ │             │
│ ┃ 256×32  ┃ │             │          │ ┃ 128×32  ┃ │             │
│ ┃ K=0-31  ┃ │             │          │ ┃ K=0-31  ┃ │             │
│ ┗━━━━━━━━━┛ │             │          │ ┗━━━━━━━━━┛ │             │
│             │             │          ├─────────────┼─────────────┤
├─────────────┼─────────────┤          │             │             │
│             │             │          │             │             │
│             │             │          │             │             │
│             │             │          │             │             │
└─────────────┴─────────────┘          └─────────────┴─────────────┘
  Origin: (0, 0)                         Origin: (0, 0)
  Covers rows 0-255                      Covers rows 0-127
  Covers cols 0-31 (first K iteration)   Covers cols 0-31 (first K iteration)
```

**Note**: The window initially covers K columns 0-31. It will move to cover K columns 32-63 in the next iteration.

### Tile Window Properties:

```cpp
// Tile window structure (conceptual):
struct tile_window {
    TensorView& tensor_view;     // Reference to underlying tensor
    Tuple window_lengths;         // Size of the window (256, 32)
    MultiIndex window_origin;     // Starting position (0, 0)
    
    // Can move the window:
    void move(MultiIndex step);   // Shift window by step
    
    // Access data through the window:
    auto load();                  // Load data from windowed region
};
```


### Tile Window Movement: Iterating Over K Dimension

In our example, **K=64** but **KPerBlock=32**, so we need **2 iterations** over the K dimension:

```
Matrix A (512 × 64) - Block 0's view:
┌─────────────┬─────────────┐
│ ┏━━━━━━━━━┓ │ ╔═══════════╗ │
│ ┃ Iter 0  ┃ │ ║  Iter 1   ║ │  ← Window slides along K
│ ┃ 256×32  ┃ │ ║  256×32   ║ │
│ ┃ K=0-31  ┃ │ ║  K=32-63  ║ │
│ ┗━━━━━━━━━┛ │ ╚═══════════╝ │
├─────────────┼─────────────┤
│             │             │
│  Block 1's  │             │
│  region     │             │
└─────────────┴─────────────┘

Matrix B (256 × 64) - Block 0's view:
┌─────────────┬─────────────┐
│ ┏━━━━━━━━━┓ │ ╔═══════════╗ │
│ ┃ Iter 0  ┃ │ ║  Iter 1   ║ │
│ ┃ 128×32  ┃ │ ║  128×32   ║ │
│ ┃ K=0-31  ┃ │ ║  K=32-63  ║ │
│ ┗━━━━━━━━━┛ │ ╚═══════════╝ │
├─────────────┼─────────────┤
│  Block 2's  │             │
│  region     │             │
└─────────────┴─────────────┘
```

### How Windows Move (Conceptual - handled by block pipeline):

```cpp
// Iteration 0:
a_block_window origin: (tile_origin_m, 0)     // K columns 0-31
b_block_window origin: (tile_origin_n, 0)     // K columns 0-31
// Compute: C_partial_0 = A[:, 0:31] × B[:, 0:31]

// Move windows to next K position:
move_tile_window(a_block_window, {0, 32});
move_tile_window(b_block_window, {0, 32});

// Iteration 1:
a_block_window origin: (tile_origin_m, 32)    // K columns 32-63
b_block_window origin: (tile_origin_n, 32)    // K columns 32-63
// Compute: C_partial_1 = A[:, 32:63] × B[:, 32:63]

// Final result:
// C_tile = C_partial_0 + C_partial_1
```

**Key Insight**: The tile windows **slide along the K dimension** to cover the full inner product. Each block accumulates partial results across K iterations to compute its final tile of C.

---

## Step 4: Delegate to Block-Level Pipeline

```cpp
// Get the block-level pipeline from policy
constexpr auto block_gemm_pipeline =
    Policy::template GetPracticeGemmBlockPipeline<Problem>();

// Calculate number of K iterations needed
int num_loops_k = integer_divide_ceil(K, KPerBlock);  // ceil(64/32) = 2

// Allocate shared memory (LDS) for block-level computation
__shared__ char p_smem_char[block_gemm_pipeline.GetStaticLDSSize()];

// Call block-level pipeline to compute C tile
const auto c_block_tile =
    block_gemm_pipeline(a_block_window, b_block_window, num_loops_k, p_smem_char);
```

### What's Happening:

1. **Retrieve block pipeline**: The policy provides the block-level GEMM implementation
2. **Calculate K iterations**: How many times to iterate over the K dimension
   - In our example: `K=64, KPerBlock=32` → **2 iterations**
   - Each iteration processes 32 elements of the K dimension
   - Results are accumulated across iterations

3. **Allocate shared memory**: 
   - `__shared__` declares memory shared by all threads in the block
   - `GetStaticLDSSize()` returns the required size in bytes
   - This memory is used for:
     - Staging data from DRAM → LDS
     - Cooperative loading by threads
     - Fast access during computation

4. **Execute block pipeline**:
   - Takes A and B tile windows as input
   - Performs the GEMM computation: `C_tile = A_tile × B_tile`
   - Returns result in `c_block_tile` (stored in VGPRs - registers)

### Memory Hierarchy During Computation:

```
┌─────────────────────────────────────────────────────────────┐
│ DRAM (Global Memory) - Slowest, Largest                     │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │   A matrix  │  │   B matrix  │  │   C matrix  │         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
         ↓ load                ↓ load              ↑ store
┌─────────────────────────────────────────────────────────────┐
│ LDS (Shared Memory) - Fast, Limited Size (~64KB)           │
│ ┌─────────────┐  ┌─────────────┐                           │
│ │  A_tile     │  │  B_tile     │  ← Staged here            │
│ │  (p_smem)   │  │  (p_smem)   │                           │
│ └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
         ↓ load                ↓ load
┌─────────────────────────────────────────────────────────────┐
│ VGPRs (Registers) - Fastest, Smallest (~256 regs/thread)   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  c_block_tile (accumulated result)                      │ │
│ │  Computation happens here using MFMA instructions       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Block Pipeline Responsibilities:

The block pipeline (called here) will:
1. Load A and B tiles from DRAM → LDS (cooperative loading)
2. Distribute work among warps
3. Each warp loads its portion from LDS → VGPRs
4. Perform MFMA operations: `C += A × B`
5. Accumulate results in VGPRs
6. Return final `c_block_tile` in registers

---

## Step 5: Store Results to DRAM

```cpp
// Create a tile window over C for writing results
auto c_window = make_tile_window(
    c_dram,                                      // Tensor view over C
    make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),  // Window size: 256×128
    {tile_origin_m, tile_origin_n}              // Origin: varies by block
);

// Store computed tile from VGPRs to DRAM
store_tile(c_window, c_block_tile);
```

### C Window Origins for Each Block:

```cpp
// Block 0: Writes to top-left tile
c_window origin: (0, 0)      → writes to C[0:255, 0:127]

// Block 1: Writes to bottom-left tile
c_window origin: (256, 0)    → writes to C[256:511, 0:127]

// Block 2: Writes to top-right tile
c_window origin: (0, 128)    → writes to C[0:255, 128:255]

// Block 3: Writes to bottom-right tile
c_window origin: (256, 128)  → writes to C[256:511, 128:255]
```

### What's Happening:

1. **Create C tile window**: 
   - Size: 256×128 (matches our block tile size)
   - Origin: Varies by block - each block writes to its assigned region
   - This window defines **where** to write the results

2. **Store tile to DRAM**:
   - `c_block_tile`: Computed results in VGPRs (registers)
   - `c_window`: Destination window in DRAM
   - `store_tile()`: Efficiently writes data from registers → DRAM

### The `store_tile` Function:

Recall from our earlier discussion, `store_tile` does:

```cpp
template <typename TileWindow, typename DistributedTensor>
void store_tile(TileWindow& tile_window_tmp,
                const DistributedTensor& dstr_tensor)
{
    // 1. Extract tile distribution from distributed tensor
    using TileDstr = typename DistributedTensor::TileDistribution;
    
    // 2. Upgrade simple tile window to one with distribution
    auto tile_window = make_tile_window(
        tile_window_tmp.get_bottom_tensor_view(),
        tile_window_tmp.get_window_lengths(),
        tile_window_tmp.get_window_origin(),
        TileDstr{}  // Add distribution info
    );
    
    // 3. Store using vectorized writes
    tile_window.store(dstr_tensor);
}
```

### Memory Flow:

```
VGPRs (Registers)                    DRAM (Global Memory)
┌─────────────────────┐              ┌─────────────────────┐
│  c_block_tile       │              │  C matrix           │
│  ┌───┬───┬───┬───┐  │              │  ┌───────────────┐  │
│  │W0 │W1 │W2 │W3 │  │  store_tile  │  │               │  │
│  ├───┼───┼───┼───┤  │  ==========> │  │  c_window     │  │
│  │...│...│...│...│  │  vectorized  │  │  (256×128)    │  │
│  └───┴───┴───┴───┘  │              │  │               │  │
│  Distributed across  │              │  └───────────────┘  │
│  threads/warps       │              │  Origin: (0, 0)     │
└─────────────────────┘              └─────────────────────┘

Each thread writes its portion using vector stores (e.g., float4)
```

### Store Optimization:

The `store_tile` function:
- Uses **vectorized stores** (write multiple elements at once)
- Ensures **coalesced memory access** (adjacent threads write adjacent memory)
- Respects **tile distribution** (each thread knows what data it owns)
- Handles **out-of-bounds** checking (for partial tiles at boundaries)

---

## Complete Flow Visualization

Let's trace the complete flow for **Block 0** (other blocks follow the same pattern):

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Calculate Tile Coverage                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ M=512, N=256, K=64                                          │ │
│ │ MPerBlock=256, NPerBlock=128, KPerBlock=32                  │ │
│ │ num_tile_m = ceil(512/256) = 2                              │ │
│ │ num_tile_n = ceil(256/128) = 2                              │ │
│ │ Total blocks needed = 2 × 2 = 4 blocks                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Map Block to Tile (Block 0 example)                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Block ID: 0                                                 │ │
│ │ Tile coordinates: (0, 0) - top-left tile                   │ │
│ │ Tile origin: (0, 0)                                         │ │
│ │                                                             │ │
│ │ (Blocks 1,2,3 get different tile coordinates)              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Create Tile Windows                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ a_block_window: 256×32 starting at (0,0) over A            │ │
│ │ b_block_window: 128×32 starting at (0,0) over B            │ │
│ │ Windows initially cover K columns 0-31                      │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Execute Block Pipeline (2 K iterations)                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Allocate shared memory (LDS)                                │ │
│ │ Call block_gemm_pipeline(a_window, b_window, 2, p_smem)    │ │
│ │                                                             │ │
│ │ K Iteration 0 (K=0-31):                                     │ │
│ │   ├─ Load A tile: DRAM → LDS → VGPRs                       │ │
│ │   ├─ Load B tile: DRAM → LDS → VGPRs                       │ │
│ │   ├─ Compute: C_partial_0 = A[:, 0:31] × B[:, 0:31]        │ │
│ │   └─ Move windows: {0, 32}                                  │ │
│ │                                                             │ │
│ │ K Iteration 1 (K=32-63):                                    │ │
│ │   ├─ Load A tile: DRAM → LDS → VGPRs                       │ │
│ │   ├─ Load B tile: DRAM → LDS → VGPRs                       │ │
│ │   ├─ Compute: C_partial_1 = A[:, 32:63] × B[:, 32:63]      │ │
│ │   └─ Accumulate: C_tile = C_partial_0 + C_partial_1        │ │
│ │                                                             │ │
│ │ Return c_block_tile in VGPRs (256×128 accumulated result)  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Store Results                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Create c_window: 256×128 starting at (0,0) over C          │ │
│ │ store_tile(c_window, c_block_tile)                          │ │
│ │   └─ Write from VGPRs → DRAM (vectorized stores)            │ │
│ │                                                             │ │
│ │ Block 0 writes to C[0:255, 0:127]                          │ │
│ │ (Other blocks write to their respective regions)           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

All 4 blocks execute in parallel, each computing its assigned 256×128 tile!
```

---

## Key Concepts Summary

### 1. **Tile Coverage**
- Determines how many thread blocks are needed
- Each block processes one tile of the output matrix C
- Calculated as `ceil(dimension / tile_size)`

### 2. **Block-to-Tile Mapping**
- Maps linear block ID to 2D tile coordinates
- Uses column-major ordering for better memory coalescing
- Each block knows which tile it's responsible for

### 3. **Tile Windows**
- **Sliding windows** over larger tensor views
- Define a rectangular region with fixed size and movable origin
- Provide efficient, structured access to tensor data
- Can be moved to access different regions (e.g., for K iterations)

### 4. **Memory Hierarchy**
- **DRAM (Global)**: Largest, slowest - stores full matrices
- **LDS (Shared)**: Medium, fast - stages tiles for cooperative access
- **VGPRs (Registers)**: Smallest, fastest - performs computation

### 5. **Data Flow**
```
DRAM → Tile Windows → LDS → VGPRs → Computation → VGPRs → DRAM
  ↑                                                           ↓
  A, B matrices                                         C matrix
```

---

## Next Steps

The host-level pipeline has set up the work and delegated to the block-level pipeline. Next, we'll explore:
- **Block-level pipeline**: How tiles are loaded, distributed to warps, and computed
- **Warp-level pipeline**: How warps perform MFMA operations
- **Memory optimization**: LDS usage, bank conflicts, coalescing

The host level provides the **orchestration**, while the block and warp levels provide the **execution**!

