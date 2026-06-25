# Batched GEMM with Reduction

This example demonstrates a Batched General Matrix-Matrix Multiplication (Batched GEMM) where the result of each individual GEMM in the batch is then reduced along one of its dimensions. This is a specialized fusion pattern that combines a compute-intensive operation (GEMM) with a memory-intensive one (reduction), offering significant performance benefits for specific workloads.

## Mathematical Formulation

The operation performs a standard GEMM for each item in a batch, and then reduces the resulting matrix to a vector. For each batch item `b` from `0` to `BatchCount-1`:

1.  **GEMM Stage**: A standard matrix multiplication is performed.
    $C_{[b]} = A_{[b]} \times B_{[b]}$

2.  **Reduction Stage**: The resulting matrix $C_{[b]}$ is reduced along one of its dimensions (e.g., the M dimension) to produce an output vector $D_{[b]}$.
    $D_{[b], j} = \bigoplus_{i=0}^{M-1} C_{[b], i, j}$

Where:
-   $A_{[b]}$ is an $M \times K$ matrix.
-   $B_{[b]}$ is a $K \times N$ matrix.
-   $C_{[b]}$ is the intermediate $M \times N$ result matrix for batch `b`.
-   $D_{[b]}$ is the final $1 \times N$ output vector for batch `b`.
-   $\bigoplus$ is a binary, associative reduction operator like sum, max, or min.

The key optimization is that the intermediate matrix $C_{[b]}$ is never written to global memory. The reduction is fused directly into the GEMM kernel.

## Algorithmic Strategy: Fused GEMM and Reduction

The implementation fuses the reduction into the epilogue of a batched GEMM kernel. The batch dimension provides a natural axis for parallelism.

1.  **Batch Scheduling**: The `BatchCount` GEMM problems are distributed across the GPU's thread blocks. Each block is assigned one or more GEMMs from the batch to compute.

2.  **Tiled GEMM Core**: For each assigned GEMM, the thread block runs a standard tiled GEMM algorithm to compute the product $A_{[b]} \times B_{[b]}$. The result for each tile of $C_{[b]}$ is accumulated in the private registers of the threads.

3.  **Fused Reduction Epilogue**: This is where the fusion occurs. Instead of writing the computed tile of $C_{[b]}$ to global memory, the threads use it as input for a parallel reduction.
    -   **Intra-Block Reduction**: The threads within a block, which collectively hold the values for a tile of $C_{[b]}$, perform a local reduction. For example, to reduce along the M dimension, threads responsible for different M-rows but the same N-column will cooperate, using fast shared memory to sum their partial results.
    -   **Inter-Block Reduction**: Since multiple thread blocks may be working on different M-tiles for the same batch item, their partial reduction results must be combined. Each block writes its partial sum to a designated location in the output vector `D`, using atomic operations (like `atomicAdd`) to safely accumulate the final result.

This strategy completely eliminates the global memory traffic associated with the intermediate matrix `C`, which is often the largest tensor in the operation. This leads to substantial savings in memory bandwidth and improved performance.

## Source Code Organization

-   [`batched_gemm_reduce_xdl.cpp`](./batched_gemm_reduce_xdl.cpp): The main example file. It sets up the batched GEMM problem and instantiates the `DeviceBatchedGemmReduce` operation, specifying the reduction dimension and operator.
-   [`../../include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce.hpp`](../../include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce.hpp): The high-level device interface for this fused operation.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_reduce_xdl_cshuffle.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_reduce_xdl_cshuffle.hpp): The grid-wise kernel that implements the fused logic. It handles the batch scheduling, the tiled GEMM, and the fused reduction epilogue with atomic operations for inter-block communication.

## Build and Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build the Example
```bash
cd /path/to/composable_kernel/example/18_batched_gemm_reduce
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
./batched_gemm_reduce_xdl

# Run with verification, data initialization, and timing
./batched_gemm_reduce_xdl 1 2 1
```

## Applications

This fused pattern is less common than simple GEMM+Bias but is highly effective for specific algorithms.

-   **Gradient Computations**: In some complex neural network layers, the gradient calculation might involve a matrix product followed by a summation. For example, computing the gradient with respect to a bias term often involves summing the output gradients over the batch and spatial dimensions. If the output gradient itself is the result of a GEMM, this fused kernel could be applicable.
-   **Custom Attention Mechanisms**: While standard attention involves a `softmax`, some research explores attention-like mechanisms that might use a simple sum or max reduction instead. If the query-key interaction is formulated as a batched GEMM, this kernel could compute the attention weights in a single, fused step.
-   **Scientific Computing**: Certain numerical methods, particularly in physics or signal processing, may involve performing a linear transform (GEMM) on a set of signals (a batch) and then integrating the result (a reduction).
