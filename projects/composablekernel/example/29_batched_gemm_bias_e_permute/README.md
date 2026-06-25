# Batched GEMM with Bias, Elementwise Operation, and Permutation

This example demonstrates a **Batched GEMM** where each individual GEMM operation is fused with a bias addition, a second elementwise operation, and a final permutation of the output. This kernel is designed to accelerate layers that have a batch-parallel structure, such as the dense layers in a Transformer's feed-forward network, when they are part of a larger fused computational graph.

## Mathematical Formulation

This operation performs `B` independent fused GEMM operations in parallel, where `B` is the batch count. For each batch item `b` from `0` to `B-1`:

1.  **GEMM Stage**: A standard matrix multiplication.
    $C_{temp1[b]} = A_{[b]} \times B_{[b]}$

2.  **Bias Addition Stage**: A bias vector `D_[b]` is broadcast and added.
    $C_{temp2[b]} = C_{temp1[b]} + D_{[b]}$

3.  **Elementwise Stage**: A second elementwise operation is performed with tensor `E_[b]`.
    $C_{temp3[b]} = C_{temp2[b]} \odot E_{[b]}$

4.  **Permutation Stage**: The final result for the batch item is permuted.
    $F_{[b]} = \text{permute}(C_{temp3[b]})$

All four stages for all `B` batch items are executed within a single kernel launch. The intermediate results are kept in registers and never written to global memory.

**Distinction from Grouped Version**:
-   In this **Batched** version, all `B` problems are uniform. They share the same dimensions (M, N, K), layouts, and permutations. The input/output tensors are accessed with a constant batch stride.
-   In the **Grouped** version (`28_grouped_gemm_bias_e_permute`), each of the `G` problems can have different dimensions, layouts, and strides, offering more flexibility.

## Algorithmic Strategy: Batch-Parallel GEMM with Fused Epilogue

The implementation combines the scheduling strategy of Batched GEMM with the multi-stage fused epilogue.

1.  **Batch Scheduling**: The `B` independent problems are distributed across the GPU's thread blocks. The grid-wise kernel is designed such that each thread block is assigned to compute one of the `B` fused operations.

2.  **Fused GEMM Execution**: Once a thread block is assigned a batch item `b`, it executes a complete fused GEMM for that item's specific data. This involves:
    -   Calculating the base memory addresses for $A_{[b]}, B_{[b]}, D_{[b]}, E_{[b]}$, and $F_{[b]}$ using the batch index and the constant batch stride.
    -   Executing a standard tiled GEMM for $A_{[b]} \times B_{[b]}$, accumulating the result in registers.
    -   Executing the fused epilogue:
        -   Load the bias `D_[b]` and add it.
        -   Load the elementwise tensor `E_[b]` and apply the operation.
        -   Calculate the permuted destination coordinates and write the final result to `F_{[b]`.

This approach is extremely efficient when the batch size `B` is large enough to saturate the GPU's parallelism.

## Source Code Organization

-   [`batched_gemm_bias_e_permute_xdl.cpp`](./batched_gemm_bias_e_permute_xdl.cpp): The main example file. It sets up the batched problem, defining the batch size, strides, and the single permutation rule that applies to all batch items. It then instantiates the `DeviceBatchedGemmBiasEPermute` operation.
-   [`../../include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_bias_e_permute_impl.hpp`](../../include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_bias_e_permute_impl.hpp): The high-level device interface for this specific fused operation.
-   The underlying grid-wise kernel contains the logic to map thread blocks to batch items (`block_to_batch`) and then execute the full fused GEMM pipeline for the assigned item.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/29_batched_gemm_bias_e_permute
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the example with default settings
./batched_gemm_bias_e_permute_xdl

# Run with verification, data initialization, and timing
./batched_gemm_bias_e_permute_xdl 1 2 1
```

## Applications

This kernel is ideal for optimizing the feed-forward network (FFN) block in a Transformer, especially when layout transformations are needed between layers.

A typical Transformer FFN block is:
`FFN(X) = Linear_2(ReLU(Linear_1(X)))`

-   `Linear_1` is a GEMM.
-   `ReLU` is an elementwise activation.
-   `Linear_2` is another GEMM.

Sometimes, for performance reasons (e.g., to align with a subsequent layer's expected input layout), the output of the FFN needs to be permuted. This kernel could fuse the `Linear_2` GEMM with its bias, a subsequent elementwise operation (if any), and the final permutation, all while operating on a batch of input sequences. This avoids multiple kernel launches and saves significant memory bandwidth, leading to faster model execution.
