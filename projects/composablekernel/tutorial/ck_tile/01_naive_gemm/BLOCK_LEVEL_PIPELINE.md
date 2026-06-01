# Block-Level Pipeline: PracticeGemmBlockPipelineAGmemBGmemCreg

## Overview

The **Block-Level Pipeline** is where the actual GEMM computation happens for one block tile. It orchestrates:
1. **Data movement** from DRAM → Registers → LDS
2. **GEMM computation** using data in LDS
3. **Iteration** over the K dimension when needed

This pipeline is called by the host-level pipeline for each block tile that covers a portion of the output matrix C.

---

## Architecture: Problem and Policy

Like other components in CK Tile, the block pipeline follows the **Problem/Policy** pattern:

### Problem: `PracticeGemmBlockPipelineProblem`
Contains:
- **Data types**: `ADataType`, `BDataType`, `CDataType`, `AccDataType`
- **Shape information**: `BlockTile` and `WaveTile` dimensions

### Policy: `PracticeGemmBlockPolicy`
Contains strategies for:
1. **Tile Distribution** (`MakeADramTileDistribution`, `MakeBDramTileDistribution`)
   - Defines how 256 threads in a block map to elements of a block tile
   - Each thread knows which elements to load/store from DRAM to its registers
   - We'll cover tile distribution construction in detail later

2. **LDS Layout** (`MakeALdsBlockDescriptor`, `MakeBLdsBlockDescriptor`)
   - Describes how data is logically organized in Local Data Share (LDS)
   - Optimizes for bank conflict avoidance and efficient access patterns
   - We'll cover LDS descriptor construction in detail later

3. **Warp Pipeline** (`GetPracticeWaveGemmPipeline`)
   - Returns the warp-level GEMM implementation

---

## Inputs and Outputs

```cpp
template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                index_t num_loop,
                                void* p_smem) const
```

### Inputs:
- `a_dram_block_window_tmp`: Tile window over A in DRAM (size: MPerBlock × KPerBlock)
- `b_dram_block_window_tmp`: Tile window over B in DRAM (size: NPerBlock × KPerBlock)
- `num_loop`: Number of iterations along K dimension
- `p_smem`: Pointer to shared memory (LDS)

### Output:
- `c_block_tile`: A `static_distributed_tensor` containing the computed C tile in registers (VGPRs)

---

## Step-by-Step Walkthrough

### Step 1: Create LDS Tensor Views

```cpp
// A tile in LDS
ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

// B tile in LDS (placed after A in shared memory)
BDataType* p_b_lds = static_cast<BDataType*>(
    static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
```

**What's happening:**
- We partition the shared memory (`p_smem`) into two regions: one for A, one for B
- We create **tensor views** over these LDS regions using descriptors from the policy
- `a_lds_block` and `b_lds_block` are logical views over raw LDS memory

**Memory Layout:**
```
Shared Memory (LDS):
┌─────────────────────┬─────────────────────┐
│   A Block Tile      │   B Block Tile      │
│   (256×32 fp16)     │   (128×32 fp16)     │
└─────────────────────┴─────────────────────┘
↑                     ↑
p_a_lds               p_b_lds
```

---

### Step 2: Create Tile Windows for Data Movement

We create **6 tile windows** for different purposes:

#### 2a. DRAM → Registers (Load from DRAM)

```cpp
auto a_copy_dram_window = make_tile_window(
    a_dram_block_window_tmp.get_bottom_tensor_view(),
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),  // 256×32
    a_dram_block_window_tmp.get_window_origin(),
    Policy::template MakeADramTileDistribution<Problem>());  // ← Tile distribution!
```

**Key Points:**
- `a_copy_dram_window` is a `tile_window_with_static_distribution`
- The **tile distribution** tells each thread which elements to load from DRAM
- This window will **slide along the K dimension** in the loop

#### 2b. Registers → LDS (Store to LDS)

```cpp
auto a_copy_lds_window = make_tile_window(
    a_lds_block,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),  // 256×32
    {0, 0},  // Origin at (0, 0) in LDS
    a_copy_dram_window.get_tile_distribution());  // ← Same distribution as DRAM!
```

**Key Points:**
- Uses the **same tile distribution** as `a_copy_dram_window`
- This ensures each thread stores to LDS in the same pattern it loaded from DRAM
- Origin is always `{0, 0}` because LDS is reused for each K iteration

#### 2c. LDS → Registers (GEMM Input)

```cpp
auto a_lds_gemm_window = make_tile_window(
    a_lds_block,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    {0, 0});  // No tile distribution!
```

**Key Points:**
- This is a `tile_window_with_static_lengths` (no explicit distribution)
- Used as input to the warp-level GEMM
- The warp GEMM will handle its own thread mapping internally

**Similar windows are created for B:**
- `b_copy_dram_window`: Load B from DRAM
- `b_copy_lds_window`: Store B to LDS
- `b_lds_gemm_window`: Read B from LDS for GEMM

---

### Step 3: Create Distributed Tensors (VGPRs)

```cpp
using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

ABlockTile a_block_tile;  // Per-thread registers for A
BBlockTile b_block_tile;  // Per-thread registers for B
```

#### What is `make_static_distributed_tensor`?

**`make_static_distributed_tensor`** creates a **`static_distributed_tensor`**, which is a compile-time abstraction for **distributed per-thread register storage**.

**Key Properties:**
1. **Per-thread VGPRs**: Each thread owns a **different slice** of the tile in its registers
2. **Compile-time sized**: Buffer size determined by tile distribution at compile time
3. **Zero-overhead**: All indexing and layout transformations happen at compile time

**How it works:**

```cpp
template <typename DataType_, typename StaticTileDistribution_>
struct static_distributed_tensor
{
    using DataType = remove_cvref_t<DataType_>;
    using StaticTileDistribution = remove_cvref_t<StaticTileDistribution_>;
    
    // Calculate per-thread storage size from tile distribution
    using ThreadTensorDesc = 
        remove_cvref_t<decltype(StaticTileDistribution{}.get_ys_to_d_descriptor())>;
    
    static constexpr index_t kThreadElementSpaceSize = 
        ThreadTensorDesc{}.get_element_space_size();
    
    // Per-thread register array (VGPRs)
    thread_buffer<DataType, get_thread_buffer_size()> thread_buf_;
};
```

**The tile distribution defines:**
- **Which elements each thread owns** in the tile
- **How many elements** each thread stores (buffer size)
- **How elements are laid out** in each thread's registers

**Concrete Example for 256×32 tile with 256 threads:**

```
Thread 0:  a_block_tile.thread_buf_ = [A[0,0], A[0,1], ..., A[0,31]]   (32 fp16 values)
Thread 1:  a_block_tile.thread_buf_ = [A[1,0], A[1,1], ..., A[1,31]]   (32 fp16 values)
Thread 2:  a_block_tile.thread_buf_ = [A[2,0], A[2,1], ..., A[2,31]]   (32 fp16 values)
...
Thread 255: a_block_tile.thread_buf_ = [A[255,0], A[255,1], ..., A[255,31]] (32 fp16 values)
```

**Collectively:**
- All 256 threads together hold the **entire 256×32 tile** (8192 elements)
- Each thread's buffer lives in its **own VGPRs**
- No two threads own the same element

**Distributed Ownership Analogy:**
Think of a tile as a **jigsaw puzzle**:
- The **tile distribution** is the cutting pattern
- Each **thread** gets one puzzle piece (its slice)
- Each **`static_distributed_tensor`** is a box holding all pieces
- Each thread's **`thread_buf_`** is its individual piece in its own registers

---

### Step 4: The GEMM Loop

```cpp
// Initialize C accumulator to zero
auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};
tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

index_t iCounter = num_loop;  // Number of K iterations

while(iCounter > 0)
{
    // 1. Load from DRAM to registers
    a_block_tile = load_tile(a_copy_dram_window);  // DRAM → VGPRs
    b_block_tile = load_tile(b_copy_dram_window);  // DRAM → VGPRs
    
    // 2. Move windows for next iteration
    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);  // Step by (0, 32)
    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);  // Step by (0, 32)
    
    // 3. Store from registers to LDS
    store_tile(a_copy_lds_window, a_block_tile);  // VGPRs → LDS
    store_tile(b_copy_lds_window, b_block_tile);  // VGPRs → LDS
    
    // 4. Synchronize threads (ensure all data is in LDS)
    block_sync_lds();
    
    // 5. Compute GEMM using data in LDS
    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
    
    // 6. Synchronize threads (before overwriting LDS in next iteration)
    block_sync_lds();
    
    iCounter--;
}

return c_block_tile;  // Return accumulated result in registers
```

---

## Detailed Loop Breakdown

### Phase 1: Load (DRAM → VGPRs)

```cpp
a_block_tile = load_tile(a_copy_dram_window);
```

**What happens:**
1. Each thread reads **its assigned elements** from DRAM (determined by tile distribution)
2. Data is loaded into **per-thread registers** (VGPRs)
3. Uses **vectorized loads** for efficiency (e.g., loading 8 fp16 values at once)

**Example for Thread 0:**
```
Thread 0 loads:
  A[0,0:7]   (8 fp16 values, one vector load)
  A[1,0:7]   (8 fp16 values, one vector load)
  ...
```

### Phase 2: Move Windows

```cpp
constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, KPerBlock);
move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
```

**What happens:**
- The tile window **slides along the K dimension** by `KPerBlock` (32 in our example)
- This prepares for the next K iteration
- The window origin moves from `(0, 0)` → `(0, 32)` → `(0, 64)` → ...

**Visualization for Problem Size 512×256×64:**
```
Matrix A (512×64):
┌─────────────────────────────────────┐
│ Block 0: rows 0-255                 │
│ ┌──────────┬──────────┐             │
│ │ K=0:31   │ K=32:63  │             │  ← Window slides right
│ │ Iter 0   │ Iter 1   │             │
│ └──────────┴──────────┘             │
└─────────────────────────────────────┘
```

### Phase 3: Store (VGPRs → LDS)

```cpp
store_tile(a_copy_lds_window, a_block_tile);
```

**What happens:**
1. Each thread writes **its elements** from registers to LDS
2. Uses the **same distribution** as the DRAM load
3. Data is now in **shared memory**, accessible to all threads in the block

**Why this step?**
- GEMM computation needs **all threads** to access **all data**
- Registers are per-thread; LDS is shared across the block
- LDS acts as a "staging area" for collaborative computation

### Phase 4: Synchronize

```cpp
block_sync_lds();
```

**What happens:**
- All threads in the block **wait** until everyone has finished storing to LDS
- Ensures no thread starts reading from LDS before all writes are complete
- Critical for correctness!

### Phase 5: GEMM Computation

```cpp
block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
```

**What happens:**
1. The warp-level GEMM reads data from LDS
2. Performs matrix multiplication using MFMA instructions
3. Accumulates results into `c_block_tile` (in registers)

**Note:** `c_block_tile` stays in registers throughout all K iterations, accumulating results.

### Phase 6: Synchronize Again

```cpp
block_sync_lds();
```

**What happens:**
- Ensures all threads have finished reading from LDS
- Safe to overwrite LDS in the next iteration

---

## Memory Flow Diagram

```
Iteration 0 (K=0:31):
┌─────────┐   load_tile   ┌──────────┐   store_tile   ┌─────────┐
│  DRAM   │ ────────────> │  VGPRs   │ ─────────────> │   LDS   │
│ A[0:255,│               │ (per-    │                │ A_block │
│   0:31] │               │  thread) │                │         │
└─────────┘               └──────────┘                └─────────┘
                                                            │
                                                            │ block_gemm
                                                            ↓
                                                      ┌──────────┐
                                                      │ c_block_ │
                                                      │   tile   │
                                                      │ (VGPRs)  │
                                                      └──────────┘

Iteration 1 (K=32:63):
┌─────────┐   load_tile   ┌──────────┐   store_tile   ┌─────────┐
│  DRAM   │ ────────────> │  VGPRs   │ ─────────────> │   LDS   │
│ A[0:255,│               │ (per-    │                │ A_block │
│  32:63] │               │  thread) │                │ (reused)│
└─────────┘               └──────────┘                └─────────┘
                                                            │
                                                            │ block_gemm
                                                            ↓
                                                      ┌──────────┐
                                                      │ c_block_ │
                                                      │   tile   │
                                                      │ (accum.) │
                                                      └──────────┘
```

---

## Example: Problem Size 512×256×64

### Block 0 Computation

**Input:**
- `a_dram_block_window_tmp`: Covers A[0:255, 0:31] initially
- `b_dram_block_window_tmp`: Covers B[0:127, 0:31] initially (B is transposed)
- `num_loop`: 2 (since K=64, KPerBlock=32)

**Iteration 0:**
1. Load A[0:255, 0:31] and B[0:127, 0:31] from DRAM to VGPRs
2. Move windows: A → [0:255, 32:63], B → [0:127, 32:63]
3. Store to LDS
4. Compute: `C[0:255, 0:127] += A[0:255, 0:31] × B[0:127, 0:31]^T`

**Iteration 1:**
1. Load A[0:255, 32:63] and B[0:127, 32:63] from DRAM to VGPRs
2. Move windows: A → [0:255, 64:95], B → [0:127, 64:95] (out of bounds, but loop ends)
3. Store to LDS
4. Compute: `C[0:255, 0:127] += A[0:255, 32:63] × B[0:127, 32:63]^T`

**Output:**
- `c_block_tile`: Contains C[0:255, 0:127] in distributed registers

---

## Key Concepts Summary

### 1. Tile Distribution
- **Maps threads to data elements** for load/store operations
- Each thread knows exactly which elements it's responsible for
- Enables **parallel, vectorized** memory access
- **Same distribution** used for DRAM load and LDS store

### 2. Static Distributed Tensor
- **Per-thread register storage** (VGPRs)
- Each thread owns a **different slice** of the tile
- **Compile-time sized** for zero-overhead abstraction
- Used for: `a_block_tile`, `b_block_tile`, `c_block_tile`

### 3. Tile Window Movement
- Windows **slide** over larger tensors
- Enables iteration over the K dimension
- `move_tile_window(window, step)` updates the origin

### 4. LDS as Staging Area
- **Shared memory** accessible to all threads in a block
- Required because GEMM needs all threads to access all data
- **Reused** across K iterations (same LDS buffer)

### 5. Synchronization
- `block_sync_lds()` ensures memory consistency
- **Before GEMM**: All stores to LDS are complete
- **After GEMM**: All reads from LDS are complete

---

## Deep Dive: `static_distributed_tensor` Mechanics

### How Tile Distribution Creates Per-Thread Storage

When you call:
```cpp
using ABlockTile = decltype(make_static_distributed_tensor<fp16_t>(ABlockTileDistr{}));
ABlockTile a_block_tile;
```

**Step 1: Extract Thread Tensor Descriptor**

The tile distribution contains a `ys_to_d_descriptor` that maps:
- **Y dimensions** (logical tile coordinates, e.g., M, K)
- **D dimension** (per-thread register index, linearized)

```cpp
using ThreadTensorDesc = 
    decltype(StaticTileDistribution{}.get_ys_to_d_descriptor());
```

**Step 2: Calculate Per-Thread Buffer Size**

```cpp
static constexpr index_t kThreadElementSpaceSize = 
    ThreadTensorDesc{}.get_element_space_size();

static constexpr index_t get_thread_buffer_size()
{
    return kThreadElementSpaceSize / PackedSize;
}
```

**Example:**
- 256×32 tile distributed across 256 threads
- Each thread owns 32 elements (one row)
- `thread_buffer_size = 32` (for PackedSize=1)

**Step 3: Allocate Thread Buffer**

```cpp
thread_buffer<DataType, get_thread_buffer_size()> thread_buf_;
```

This is essentially:
```cpp
fp16_t data[32];  // Per-thread register array (VGPRs)
```

### Usage in Load/Store Operations

**Load from DRAM:**
```cpp
a_block_tile = load_tile(a_copy_dram_window);
```

What happens internally:
1. Each thread queries the tile distribution: "Which elements do I own?"
2. Thread 0 learns it owns A[0,0:31]
3. Thread 0 loads those elements from DRAM into `a_block_tile.thread_buf_[0:31]`
4. All 256 threads do this **in parallel**

**Store to LDS:**
```cpp
store_tile(a_copy_lds_window, a_block_tile);
```

What happens internally:
1. Each thread reads from its `a_block_tile.thread_buf_`
2. Thread 0 writes A[0,0:31] from its registers to LDS
3. All 256 threads do this **in parallel**
4. After `block_sync_lds()`, the entire tile is in shared LDS

### Distributed Indexing

The `static_distributed_tensor` supports compile-time indexing:

```cpp
// Access using distributed indices
auto value = a_block_tile(tile_distributed_index<i, j>{});
```

Internally:
1. Convert distributed index → Y index (logical tile coordinates)
2. Calculate buffer offset using `ThreadTensorDesc`
3. Access `thread_buf_[offset]`

All of this happens **at compile time** with zero runtime overhead!

### Why This Design?

**Benefits:**
1. **Parallel Memory Access**: All threads load/store simultaneously
2. **Vectorization**: Each thread can use vector loads (e.g., 8×fp16 at once)
3. **Zero Overhead**: All indexing resolved at compile time
4. **Type Safety**: Distribution mismatch caught at compile time
5. **Register Pressure**: Compiler knows exact VGPR usage

**Trade-offs:**
- Requires compile-time tile sizes
- Distribution must be static
- More complex type system

### Memory Hierarchy Summary

```
┌─────────────────────────────────────────────────────────────┐
│                         DRAM (Global Memory)                 │
│                    Full matrices A, B, C                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ load_tile (parallel, vectorized)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    VGPRs (Per-Thread Registers)              │
│  Thread 0: a_block_tile.thread_buf_ = [A[0,0:31]]          │
│  Thread 1: a_block_tile.thread_buf_ = [A[1,0:31]]          │
│  ...                                                         │
│  Thread 255: a_block_tile.thread_buf_ = [A[255,0:31]]      │
│                                                              │
│  ← static_distributed_tensor manages this distribution      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ store_tile (parallel, vectorized)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LDS (Shared Memory)                       │
│              Entire block tile (256×32)                      │
│           Accessible to all threads in block                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:**
`static_distributed_tensor` is the abstraction that enables efficient, parallel data movement between DRAM and LDS through per-thread VGPRs, with all coordination happening at compile time.



