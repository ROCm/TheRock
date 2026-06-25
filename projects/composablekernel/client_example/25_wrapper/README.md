[Back to the main page](../../README.md)

# Composable Kernel Wrapper GEMM Tutorial

This tutorial demonstrates how to implement matrix multiplication (GEMM) using the Composable Kernel wrapper. The three examples show both basic and optimized GEMM implementations, as well as how to use the wrapper for tensor transformations such as im2col.

---

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/25_wrapper
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (basic GEMM)
./wrapper_basic_gemm

# Example run (optimized GEMM)
./wrapper_optimized_gemm

# Example run (im2col transformation)
./wrapper_img2col

# Example run (tensor transform using wrapper)
./tensor_transform_using_wrapper
```

---

## Source Code Structure

### Directory Layout
```
client_example/25_wrapper/
├── wrapper_basic_gemm.cpp         # Basic GEMM using CK wrapper
├── wrapper_optimized_gemm.cpp     # Optimized GEMM using CK wrapper
├── wrapper_img2col.cpp            # im2col transformation using CK wrapper
├── tensor_transform_using_wrapper.cpp # General tensor transform example
├── CMakeLists.txt                 # Build configuration for the example
├── README.md                      # This tutorial and reference
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures wrapper parameters, launches the kernel, and verifies the result.
- **CK wrapper API usage**:  
  Demonstrates how to create layouts, tensors, and launch GEMM or tensor transforms using the wrapper.

---

## Additional Details

## Overview

The CK wrapper provides a flexible interface for launching GEMM kernels and tensor operations. This tutorial presents:
- A base GEMM implementation (minimal optimizations)
- An optimized GEMM using `gridwise_gemm_xdlops_v2r3`
- Examples of tensor transformations (e.g., im2col)

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__ DeviceGemm(const void* p_a,
                                                        const void* p_b,
                                                        void* p_c,
                                                        const ck::index_t M,
                                                        const ck::index_t N,
                                                        const ck::index_t K,
                                                        const BlockShape tile_shape,
                                                        const ThreadLayout thread_layout)
```

We pass pointers to global memory and matrix dimensions via arguments. Additionally, we pass
selected lengths of processed data through each block (`tile_shape`) and thread layout
(`thread_layout`). For compilation time parameters, we define the data type,
[traits for the GEMM operation](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/wrapper/traits/blockwise_gemm_xdl_traits.hpp)
and scalar per vector value during copy.

Step 1: Create layouts for global and LDS memory.

```cpp
    // Specify layouts for global memory.
    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));

    // Specify layouts for tiles.
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(NPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto c_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, NPerBlock), ck::make_tuple(NPerBlock, ck::Number<1>{}));

    // Apply padding for global memory.
    auto a_global_layout_padded = ck::wrapper::pad(a_global_layout, shape(a_tile_layout));
    auto b_global_layout_padded = ck::wrapper::pad(b_global_layout, shape(b_tile_layout));
    auto c_global_layout_padded = ck::wrapper::pad(c_global_layout, shape(c_tile_layout));
```

We pad layouts for global tensors in case M, N, and K are not divisible by `MPerBlock`, `NPerBlock`, or
`KPerBlock`.

Step 2: Create tensors for global and LDS memory.

```cpp
    // Make tensors for global memory.
    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_global_layout_padded);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_global_layout_padded);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_global_layout_padded);

    // Allocate LDS memory.
    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout)];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout)];

    // Make tensors for lds memory.
    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);
```

We must specify parameters for copy and convert block indexes to tuple:

```cpp
    // Specify block index as tuple.
    const auto block_idxs = ck::make_tuple(static_cast<ck::index_t>(blockIdx.x),
                                           static_cast<ck::index_t>(blockIdx.y),
                                           ck::wrapper::slice());
    // Specify access parameters for copy.
    using DimAccessOrder             = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim = 1;
```

We create a local tile (per block) and local partitions (per thread) for the global memory (`C`). We also
define and clear an output register (`c_vgpr_reg`) for the accumulation.

```cpp
    auto c_global_local_tile = ck::wrapper::make_local_tile(
        c_global_tensor,
        tile_shape,
        block_idxs,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(KPerBlock)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    // Create C vgpr to accumulate results.
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    // Clear C vgpr.
    ck::wrapper::clear(c_vgpr_reg);
```

We use two specific functions for `blockwise_gemm`: `make_blockwise_gemm_xdl_c_local_partition` and
`make_blockwise_gemm_xdl_c_vgpr`. This helps to choose the appropriate partition for the `C` output
and define tensors with specific layouts for `blockwise_gemm`. In the following step, we use only
generic functions for the CK wrapper.

Step 3: Create the compute loop.

```cpp
    const ck::index_t num_loop = ck::math::integer_divide_ceil(K, KPerBlock);
    ck::index_t i              = 0;
    do
    {
        // Get KPerBlock slice.
        const auto k_slice           = ck::wrapper::slice(i * KPerBlock, (i + 1) * KPerBlock);
        auto a_global_tensor_k_slice = a_global_tensor(ck::wrapper::slice(), k_slice);
        auto b_global_tensor_k_slice = b_global_tensor(ck::wrapper::slice(), k_slice);
        // Create local tiles for A and B.
        auto a_global_local_tile = ck::wrapper::make_local_tile(
            a_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}));
        auto b_global_local_tile = ck::wrapper::make_local_tile(
            b_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}));
        // Copy from global to LDS.
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            a_global_local_tile, a_lds_tensor, thread_layout);
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            b_global_local_tile, b_lds_tensor, thread_layout);
        // Synchronize lds.
        ck::block_sync_lds();
        // Execute blockwise GEMM.
        ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
            a_lds_tensor, b_lds_tensor, c_vgpr_reg);

        ++i;
    } while(i < num_loop);
```

Loop iterate over `K / KPerBlock`. Each time a local tile is created for A and B tensors (tensor per block),
data is copied from global memory to LDS. The `blockwise_gemm` function performs the GEMM
operation on `a_lds_tensor` and `b_lds_tensor`, and stores results in `c_vgpr_reg`.

The end result from `c_vgpr_reg` is stored in the `C` local partition (tensor per thread):

```cpp
    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
```

---

## Related Examples

- [01_gemm](../01_gemm/README.md): Basic GEMM client example
- [27_im2col_col2im](../27_im2col_col2im/README.md): im2col/col2im transformations
- [25_gemm_bias_e_permute](../../example/25_gemm_bias_e_permute/README.md): GEMM with bias and permutation in the main example directory

---
[Back to Client Examples](../README.md)
