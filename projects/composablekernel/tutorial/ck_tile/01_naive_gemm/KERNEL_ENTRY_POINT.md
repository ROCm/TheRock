# PracticeGemmKernel: Understanding the Kernel Entry Point

This document explains the `PracticeGemmKernel` structure, which serves as the **entry point** for our GEMM GPU kernel. We'll dive deep into how raw memory is transformed into structured tensor views.

## Overview

The `PracticeGemmKernel` is a templated struct that:
1. Takes raw device memory pointers for matrices A, B, and C
2. Wraps them into **tensor views** - logical, structured views over physical memory
3. Dispatches to the host-level pipeline for computation

```cpp
template <typename Problem_, typename Policy_>
struct PracticeGemmKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(const typename Problem::ADataType* p_a,
                                   const typename Problem::BDataType* p_b,
                                   typename Problem::CDataType* p_c,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t stride_a,
                                   const index_t stride_b,
                                   const index_t stride_c) const
    {
        // Step 1: Create tensor views over raw memory
        auto a_dram = make_naive_tensor_view<address_space_enum::global>(
            p_a, make_tuple(M, K), make_tuple(stride_a, 1), number<8>{}, number<1>{});

        auto b_dram = make_naive_tensor_view<address_space_enum::global>(
            p_b, make_tuple(N, K), make_tuple(stride_b, 1), number<8>{}, number<1>{});

        const auto c_dram = make_naive_tensor_view<address_space_enum::global>(
            p_c, make_tuple(M, N), make_tuple(stride_c, 1), number<8>{}, number<1>{});

        // Step 2: Dispatch to host-level pipeline
        PracticeGemmHostPipeline<Problem, Policy>{}(a_dram, b_dram, c_dram);
    }
};
```

---

## What are Tensor Views?

A **tensor view** is a **logical, structured view over raw physical memory**. It doesn't own or allocate memory—it simply provides a way to interpret and access existing memory as a multi-dimensional tensor.

### Key Components of a Tensor View:

1. **Memory Type**: Where the data lives (global/DRAM, LDS/shared, registers)
2. **Raw Pointer**: Points to the actual data in memory
3. **Shape**: Dimensions of the tensor (e.g., M×K for matrix A)
4. **Strides**: How to navigate through memory to access elements
5. **Guaranteed Vector Length**: How many consecutive elements can be loaded in one vector instruction
6. **Guaranteed Vector Stride**: The stride of those vectorizable elements

---

## The Memory Abstraction Hierarchy

CK Tile uses a three-layer abstraction to go from raw memory to structured tensors:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: TENSOR VIEW                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ • Logical multi-dimensional structure                   │ │
│ │ • Shape: (M, K) = (256, 32)                            │ │
│ │ • Strides: (32, 1) for row-major layout                │ │
│ │ • Provides: operator[], coordinate-based access         │ │
│ │ • Knows: How to map (i,j) → linear offset              │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           ↓ wraps                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Layer 2: BUFFER VIEW                                    │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ • Linear view of memory                             │ │ │
│ │ │ • Pointer: p_data_ → device memory                  │ │ │
│ │ │ • Size: Total number of elements                    │ │ │
│ │ │ • Address space: global/LDS/generic                 │ │ │
│ │ │ • Provides: Vectorized loads/stores, bounds checking│ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           ↓ wraps                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Layer 1: RAW PHYSICAL MEMORY                            │ │
│ │ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐ │ │
│ │ │ 0.0 │ 1.0 │ 2.0 │ 3.0 │ 4.0 │ 5.0 │ 6.0 │ 7.0 │ ... │ │ │
│ │ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘ │ │
│ │   ↑                                                       │ │
│ │   p_a (raw pointer from hipMalloc)                       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: `make_naive_tensor_view`

Let's break down the function call for matrix A:

```cpp
auto a_dram = make_naive_tensor_view<address_space_enum::global>(
    p_a,                    // Raw pointer to device memory
    make_tuple(M, K),       // Shape: (256, 32)
    make_tuple(stride_a, 1), // Strides: (32, 1) - row-major
    number<8>{},            // Guaranteed vector length
    number<1>{}             // Guaranteed vector stride
);
```

### Function Signature:

```cpp
template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          memory_operation_enum DstInMemOp      = memory_operation_enum::set,
          amd_buffer_coherence_enum Coherence   = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength = -1,
          index_t GuaranteedLastDimensionVectorStride = -1>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_view(DataType* __restrict__ p,
                       const tuple<Lengths...>& lengths,
                       const tuple<Strides...>& strides,
                       number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                       number<GuaranteedLastDimensionVectorStride> = number<-1>{})
{
    // Step 1: Create tensor descriptor (shape + stride information)
    auto desc = make_naive_tensor_descriptor(lengths,
                                             strides,
                                             number<GuaranteedLastDimensionVectorLength>{},
                                             number<GuaranteedLastDimensionVectorStride>{});

    // Step 2: Create buffer view (pointer + size + address space)
    auto buffer_view =
        make_buffer_view<BufferAddressSpace, Coherence>(p, desc.get_element_space_size());

    // Step 3: Combine into tensor view
    return tensor_view<decltype(buffer_view), decltype(desc), DstInMemOp>{buffer_view, desc};
}
```

---

## Parameter Breakdown

### 1. **Template Parameter: `address_space_enum::global`**

Specifies where the memory lives:
- `global`: GPU global memory (DRAM) - slowest but largest
- `lds`: Local Data Share (shared memory) - fast, limited size
- `generic`: Generic address space
- `vgpr`: Vector General Purpose Registers - fastest, smallest

In our case, `global` means the data is in GPU DRAM.

### 2. **`p_a` - Raw Pointer**

The raw device memory pointer returned by `hipMalloc`. Points to the start of the matrix data.

### 3. **`make_tuple(M, K)` - Shape/Lengths**

Defines the logical dimensions of the tensor:
- For matrix A: `(256, 32)` means 256 rows, 32 columns
- This is the **logical view**, independent of how data is physically laid out

### 4. **`make_tuple(stride_a, 1)` - Strides**

Defines how to navigate through memory:
- **Stride for dimension 0 (rows)**: `stride_a = K = 32`
  - To move to the next row, skip 32 elements
- **Stride for dimension 1 (columns)**: `1`
  - To move to the next column, skip 1 element

**Row-major layout example:**
```
Memory:  [a₀₀, a₀₁, a₀₂, ..., a₀₃₁, a₁₀, a₁₁, a₁₂, ..., a₁₃₁, ...]
          ↑                         ↑
          Row 0 starts here         Row 1 starts here (offset = 32)

To access element A[i][j]:
    offset = i * stride_a + j * 1
           = i * 32 + j
```

### 5. **`number<8>{}` - Guaranteed Last Dimension Vector Length**

This tells the tensor view: **"The last dimension (K) is guaranteed to have at least 8 consecutive elements that can be loaded together in a single vector instruction."**

#### Why is this important?

Modern GPUs can load multiple elements in one instruction (vectorized loads):
- `float4`: Load 4 floats at once
- `float8`: Load 8 floats at once (if supported)

By specifying `number<8>{}`, we're telling the system:
- "You can safely use vector loads of up to 8 elements"
- "The memory alignment and layout support this"

**Example:**
```cpp
// Without vectorization (slow):
for (int j = 0; j < 8; j++) {
    data[j] = memory[offset + j];  // 8 separate loads
}

// With vectorization (fast):
float8 vec = *reinterpret_cast<float8*>(&memory[offset]);  // 1 load!
```

### 6. **`number<1>{}` - Guaranteed Last Dimension Vector Stride**

This specifies the **stride between consecutive vectorizable elements** in the last dimension.

- `number<1>{}` means: "Consecutive elements in the last dimension are contiguous in memory (stride = 1)"
- This confirms that elements `A[i][0], A[i][1], A[i][2], ..., A[i][7]` are stored consecutively

**Why does this matter?**

For efficient vectorized loads, elements must be:
1. **Contiguous** (stride = 1) ✓
2. **Aligned** properly in memory
3. **Within the same cache line** (ideally)

If the stride were `2`, it would mean:
```
A[i][0] is at offset 0
A[i][1] is at offset 2  (not 1!)
A[i][2] is at offset 4
```
This would prevent efficient vectorization.

---

## What is a Buffer View?

A **buffer view** is the middle layer between raw memory and tensor view. It provides:

### Core Responsibilities:

1. **Memory Management**
   - Holds the raw pointer: `T* p_data_`
   - Tracks buffer size: `BufferSizeType buffer_size_`
   - Knows the address space: `global`, `lds`, etc.

2. **Vectorized Access**
   ```cpp
   template <typename VectorType>
   CK_TILE_DEVICE VectorType get(index_t offset);
   ```
   - Provides efficient vector loads/stores
   - Handles alignment requirements

3. **Bounds Checking** (optional)
   ```cpp
   template <bool oob_conditional_check = true>
   CK_TILE_DEVICE auto get(index_t i, index_t linear_offset);
   ```
   - Can optionally check if access is within bounds
   - Returns invalid value (default 0) for out-of-bounds access

4. **Address Space Awareness**
   - Uses different load/store instructions based on address space
   - Global memory: `global_load`, `global_store`
   - LDS: `ds_read`, `ds_write`

### Buffer View Structure:

```cpp
template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
struct buffer_view
{
    T* p_data_;                              // Raw pointer
    BufferSizeType buffer_size_;             // Total elements
    remove_cvref_t<T> invalid_element_value_; // Value for OOB access

    // Access operators
    const T& operator[](index_t i) const;    // Read
    T& operator()(index_t i);                // Write
    
    // Vectorized access
    template <typename VectorType>
    VectorType get(index_t offset);
};
```

---

## Visual Example: Matrix A Memory Layout

Let's visualize how matrix A (256×32, fp16) is organized:

### Raw Physical Memory (Linear):
```
GPU DRAM Address Space:
┌─────────────────────────────────────────────────────────────────┐
│ Byte 0                                                          │
│ ↓                                                               │
│ [a₀₀][a₀₁][a₀₂]...[a₀₃₁][a₁₀][a₁₁][a₁₂]...[a₁₃₁][a₂₀]...     │
│  ↑                        ↑                                     │
│  Row 0 (32 elements)      Row 1 (32 elements)                  │
│                                                                 │
│  Total: 256 rows × 32 cols × 2 bytes/element = 16,384 bytes   │
└─────────────────────────────────────────────────────────────────┘
         ↑
         p_a (raw pointer)
```

### Buffer View Layer:
```
buffer_view<address_space_enum::global, fp16_t, ...>
┌─────────────────────────────────────────────────────────────────┐
│ p_data_ = p_a                                                   │
│ buffer_size_ = 256 × 32 = 8,192 elements                       │
│ address_space = global (DRAM)                                   │
│                                                                 │
│ Provides:                                                       │
│ • Linear indexing: buffer_view[i] → element at offset i        │
│ • Vectorized loads: get<float4>(offset) → load 4 fp16s at once│
│ • Bounds checking: is offset < buffer_size_?                   │
└─────────────────────────────────────────────────────────────────┘
```

### Tensor View Layer:
```
tensor_view<buffer_view, tensor_descriptor>
┌─────────────────────────────────────────────────────────────────┐
│ Shape: (256, 32)                                                │
│ Strides: (32, 1)                                                │
│ Guaranteed vector length: 8                                     │
│ Guaranteed vector stride: 1                                     │
│                                                                 │
│ Logical 2D View:                                                │
│     Col:  0    1    2   ...  31                                │
│   Row 0: [a₀₀][a₀₁][a₀₂] ... [a₀₃₁]  ← Can vector load 8 at once│
│   Row 1: [a₁₀][a₁₁][a₁₂] ... [a₁₃₁]                           │
│   Row 2: [a₂₀][a₂₁][a₂₂] ... [a₂₃₁]                           │
│   ...                                                           │
│   Row 255: [a₂₅₅,₀] ... [a₂₅₅,₃₁]                             │
│                                                                 │
│ Provides:                                                       │
│ • Multi-dimensional indexing: A[i][j]                          │
│ • Coordinate transformation: (i,j) → linear offset = i*32 + j  │
│ • Tile window creation: Extract sub-tensors                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Flow: Raw Memory → Tensor View

Let's trace the complete transformation for matrix A:

### Step 1: Kernel Launch (Host Side)
```cpp
// On host: Allocate device memory
hipMalloc(&p_a, M * K * sizeof(fp16_t));  // Returns raw pointer

// Launch kernel
kernel<<<grid, block>>>(p_a, p_b, p_c, M, N, K, ...);
```

### Step 2: Inside Kernel (Device Side)
```cpp
// Receive raw pointer
const fp16_t* p_a;  // Points to GPU DRAM

// Step 2a: Create tensor descriptor
auto desc = make_naive_tensor_descriptor(
    make_tuple(256, 32),    // Shape
    make_tuple(32, 1),      // Strides
    number<8>{},            // Vector length
    number<1>{}             // Vector stride
);
// desc now knows: "This is a 256×32 tensor, row-major, vectorizable by 8"

// Step 2b: Create buffer view
auto buffer_view = make_buffer_view<address_space_enum::global>(
    p_a,                    // Raw pointer
    256 * 32                // Total elements
);
// buffer_view now wraps p_a with size and address space info

// Step 2c: Create tensor view
auto a_dram = tensor_view{buffer_view, desc};
// a_dram now provides structured, multi-dimensional access to p_a
```

### Step 3: Using the Tensor View
```cpp
// Access element A[i][j]
auto value = a_dram[make_tuple(i, j)];

// Create a tile window (sub-tensor)
auto tile = make_tile_window(
    a_dram,
    make_tuple(16, 16),  // 16×16 tile
    make_tuple(0, 0)     // Starting at origin
);

// Load tile into registers with vectorization
auto tile_data = load_tile(tile);  // Uses vector loads internally!
```

---

## Why This Abstraction?

### Benefits:

1. **Type Safety**: Can't accidentally access wrong dimensions
2. **Performance**: Compiler knows about vectorization opportunities
3. **Flexibility**: Same code works for different memory spaces (DRAM, LDS, registers)
4. **Maintainability**: Logical structure separate from physical layout
5. **Optimization**: Guaranteed vector properties enable aggressive optimizations

### Example: Without Tensor Views (Manual Indexing)
```cpp
// Ugly, error-prone, hard to optimize:
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
        float val = p_a[tile_offset_i * stride_a + tile_offset_j + i * stride_a + j];
        // Hope the compiler vectorizes this? 🤞
    }
}
```

### Example: With Tensor Views (Clean, Optimized)
```cpp
// Clean, safe, automatically vectorized:
auto tile = make_tile_window(a_dram, make_tuple(16, 16), origin);
auto tile_data = load_tile(tile);  // Vectorized loads guaranteed!
```

---

## Summary

The `PracticeGemmKernel` entry point transforms raw GPU memory into structured, multi-dimensional tensors through a three-layer abstraction:

1. **Raw Memory**: Linear array of bytes in GPU DRAM
2. **Buffer View**: Adds size, address space, and vectorized access
3. **Tensor View**: Adds shape, strides, and multi-dimensional indexing

This abstraction enables:
- ✅ Clean, readable code
- ✅ Type-safe multi-dimensional access
- ✅ Automatic vectorization
- ✅ Flexible memory space handling
- ✅ Efficient tile-based computation

The tensor views created here are then passed to the host-level pipeline, which orchestrates the block-level GEMM computation!

