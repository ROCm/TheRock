# Tile Distribution: Mapping Threads to Data

## Overview

**Tile Distribution** describes how each thread in a thread block maps to elements of a block tile. It defines the hierarchical pattern of data distribution across threads, warps, and thread blocks.

## The Problem

Given a block tile of size `MPerBlock × KPerBlock` (e.g., 256×32), we need to determine:
- Which threads load which elements.
- How the threads are organized into warps.
- The number of times each warp repeats its pattern.
- The number of elements each thread can load in a single vector instruction.

---

## Bottom-Up Construction Approach

### Step 1: Determine K Dimension Layout

**Start with the innermost dimension (K) for memory coalescing:**

```cpp
constexpr index_t K1 = 16 / sizeof(ADataType);  // Elements per thread (vector load)
constexpr index_t K0 = kKPerBlock / K1;          // Threads needed in K dimension
```

**Example (with fp16):**
- `K1 = 16 / 2 = 8` → Each thread loads 8 fp16 elements in a single vector instruction
- `kKPerBlock = 32` 
- `K0 = 32 / 8 = 4` → We need 4 threads along K to cover the entire K dimension

**Visual:**
```
K dimension (32 elements):
Thread 0: [0-7]  Thread 1: [8-15]  Thread 2: [16-23]  Thread 3: [24-31]
   K1=8              K1=8               K1=8                K1=8
├──────────────────────────────────────────────────────────────┤
                    K0=4 threads
```

---

### Step 2: Determine M Dimension Layout

**Now partition the M dimension hierarchically:**

#### Level 1: Threads per Warp in M (M2)

```cpp
constexpr index_t M2 = get_warp_size() / K0;
```

- Warp size = 64 threads
- K dimension already uses `K0 = 4` threads per row
- `M2 = 64 / 4 = 16` → Each warp can have 16 threads in M dimension

**Visual (Single Warp):**
```
         K dimension (4 threads)
      ┌─────┬─────┬─────┬─────┐
   0  │ T0  │ T1  │ T2  │ T3  │
   1  │ T4  │ T5  │ T6  │ T7  │
   2  │ T8  │ T9  │ T10 │ T11 │
M  3  │ T12 │ T13 │ T14 │ T15 │  ← 16 rows
   ...│ ... │ ... │ ... │ ... │    (M2=16)
  15  │ T60 │ T61 │ T62 │ T63 │
      └─────┴─────┴─────┴─────┘
     One Warp = 64 threads
```

#### Level 2: Warps per Block (M1)

```cpp
constexpr index_t M1 = kBlockSize / get_warp_size();
```

- `kBlockSize = 256` threads per block
- `M1 = 256 / 64 = 4` → We have 4 warps per block

**Visual (4 Warps):**
```
       Warp 0 (rows 0-15)
       Warp 1 (rows 16-31)
       Warp 2 (rows 32-47)
       Warp 3 (rows 48-63)
       ↑
    M1 = 4 warps cover 64 rows total
```

#### Level 3: Repetitions (M0)

```cpp
constexpr index_t M0 = kMPerBlock / (M2 * M1);
```

- `kMPerBlock = 256` rows to cover
- `M2 * M1 = 16 * 4 = 64` rows covered by all warps
- `M0 = 256 / 64 = 4` → Each warp must repeat its pattern 4 times

**Visual (Complete Block):**
```
┌──────────────┐
│ Iteration 0  │ ← Warp 0: rows 0-15,   Warp 1: rows 16-31,  ...
│ (rows 0-63)  │
├──────────────┤
│ Iteration 1  │ ← Warp 0: rows 64-79,  Warp 1: rows 80-95,  ...
│ (rows 64-127)│
├──────────────┤
│ Iteration 2  │ ← Warp 0: rows 128-143, Warp 1: rows 144-159, ...
│(rows 128-191)│
├──────────────┤
│ Iteration 3  │ ← Warp 0: rows 192-207, Warp 1: rows 208-223, ...
│(rows 192-255)│
└──────────────┘
  M0 = 4 iterations
```

---

## The Tile Distribution Encoding

Now we can construct the distribution:

```cpp
tile_distribution_encoding<
    sequence<1>,                                      // [1] Replication
    tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,   // [2] Hierarchy
    tuple<sequence<1>, sequence<1, 2>>,              // [3] Parallelism: 
    tuple<sequence<1>, sequence<2, 0>>,              // [3] Parallelism
    sequence<1, 2>,                                   // [4] Yield 
    sequence<0, 1>                                    // [4] Yield 
>
```

### [1] Replication: `sequence<1>`

Defines how many times warp patterns are replicated:
- `1` = Each warp has a unique pattern (no replication)
- `2` = Warp 0 and Warp 1 do the same thing, Warp 2 and Warp 3 do the same thing
- `4` = All warps do the same thing

In our case: `1` means no replication (each warp is independent).

---

### [2] Hierarchy: The Multi-Level Structure

```cpp
tuple<sequence<M0, M1, M2>, sequence<K0, K1>>
     └───────┬──────────┘  └──────┬────────┘
           M dimension         K dimension
```

**Concrete values:**
- M hierarchy: `sequence<4, 4, 16>` = (4 repetitions, 4 warps, 16 threads/warp)
- K hierarchy: `sequence<4, 8>` = (4 threads, 8 elements/thread)

---

### [3] Parallelism: Addressing the Hierarchy

**The key insight:** Read the tuples **vertically** to understand indexing!

```cpp
tuple<sequence<1>, sequence<1, 2>>  
tuple<sequence<1>, sequence<2, 0>>  
```

#### Reading Pattern

**Column 1 (Dimension 0 = M):**
```
sequence<1>     → Address hierarchy index 1,1 → M1 (warps/block in M dimension)
sequence<1>     
```

**Column 2 (Dimension 1 = K):**
```
sequence<1, 2>  
sequence<2, 0>  
```
[1,2]  M2=threads/warp in M dimension
[2,0]  K0=threads/warp in K dimension

---

### [4] Yield Sequences: Output Ordering

```cpp
sequence<1, 2>  
sequence<0, 1>      

[1,0] means M0=repetitions/warp in M dimension
[2,1] means K1=elements/thread in K dimension
```
---

## Complete Example: Thread 25 in Warp 0

Let's trace where **Thread 25** in **Warp 0** reads data:

### Thread Coordinates
- Thread ID in warp: 25
- Warp ID in block: 0

### Decompose Thread 25
```
Thread 25 in a 2D layout (M2=16, K0=4):
Row index: 25 / 4 = 6
Col index: 25 % 4 = 1
```

### M Position (Row)
```
M0 iteration: 0 (first iteration)
M1 warp:      0 (warp 0)
M2 thread:    6 (6th row in warp)
→ M position = 0*64 + 0*16 + 6 = 6
```

### K Position (Column)  
```
K0 thread: 1 (column group 1)
K1 elements: 8 (will load 8 consecutive elements)
→ K position = 1*8 + [0-7] = elements 8-15
```

**Result:** Thread 25 in Warp 0 loads **row 6, columns 8-15** (8 elements).

---

## Why This Matters

### 1. **Memory Coalescing**
- Consecutive threads access consecutive memory → efficient global memory access
- K dimension uses K1=8 for vectorized loads

### 2. **Warp Efficiency**  
- All 64 threads in a warp are utilized
- Natural 2D layout: 16 threads (M) × 4 threads (K) = 64 threads

### 3. **Scalability**
- M0 repetitions allow handling larger tiles
- Same pattern scales to different sizes

### 4. **Register Allocation**
- Each thread knows exactly how many elements it will hold
- Compiler can allocate registers optimally

---

## Summary Table

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **K1** | 8 | Elements per thread (vector width) |
| **K0** | 4 | Threads along K per row |
| **M2** | 16 | Threads along M per warp |
| **M1** | 4 | Warps per block |
| **M0** | 4 | Repetitions of warp pattern |
| **Total Threads** | 256 | M0×M1×M2 = 4×4×16 (actually M1×64) |
| **Total Elements** | 8192 | 256×32 (MPerBlock × KPerBlock) |
| **Elements/Thread** | 32 | M0×K1 = 4×8 |

---

## Visualization: Complete Thread Block

```
Block Tile: 256×32

      K dimension (32 elements)
      ├─────────────────────┤
  0   ┌──────────────────────┐  ┐
  16  │      Warp 0          │  │
  32  │      Warp 1          │  │ Iteration 0
  48  │      Warp 2          │  │ (M0=0)
  64  │      Warp 3          │  ┘
  80  ├──────────────────────┤  ┐
  96  │      Warp 0          │  │
 112  │      Warp 1          │  │ Iteration 1
 128  │      Warp 2          │  │ (M0=1)
 144  │      Warp 3          │  ┘
 160  ├──────────────────────┤  ┐
 176  │      Warp 0          │  │
 192  │      Warp 1          │  │ Iteration 2
 208  │      Warp 2          │  │ (M0=2)
 224  │      Warp 3          │  ┘
 240  ├──────────────────────┤  ┐
 256  │      Warp 0          │  │
      │      Warp 1          │  │ Iteration 3
      │      Warp 2          │  │ (M0=3)
      │      Warp 3          │  ┘
      └──────────────────────┘

Each warp processes 16 rows × 32 cols = 512 elements
Each iteration processes 64 rows × 32 cols = 2048 elements
Total: 4 iterations × 2048 = 8192 elements ✓
```

---

## Key Takeaways

1. **Bottom-up construction**: Start from vector width (K1), build up through thread/warp/block hierarchy
2. **Vertical reading**: The repeat and elements tuples are read column-wise to address hierarchy levels
3. **Replication controls redundancy**: How many warps share the same pattern
4. **Hierarchy encodes structure**: The multi-level sequence defines the complete mapping

This design enables CK to achieve maximum GPU performance through optimal thread-to-data mapping!

