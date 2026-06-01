# Grouped GEMM with Multiple A, B, and D Tensors

This example demonstrates a **Grouped GEMM operation with multiple A, B, and D tensors**. This is an advanced fusion pattern that extends the basic grouped GEMM to handle multiple input matrices and auxiliary tensors simultaneously, enabling complex multi-input computational graphs to be executed in a single kernel launch.

## Mathematical Formulation

This operation performs `G` independent GEMM operations in parallel, where each group can have multiple input matrices and auxiliary tensors.

For each group `g` from `0` to `G-1`:
1.  **Multiple Input GEMMs**: Compute products from multiple A and B tensor pairs.
    $C_{temp0[g]} = A_{0[g]} \times B_{0[g]}$
    $C_{temp1[g]} = A_{1[g]} \times B_{1[g]}$
    $\vdots$
    $C_{tempK[g]} = A_{K[g]} \times B_{K[g]}$

2.  **Combination with Auxiliary Tensors**: Apply a user-defined function that combines the GEMM results with multiple D tensors.
    $E_{[g]} = f(C_{temp0[g]}, C_{temp1[g]}, \ldots, C_{tempK[g]}, D_{0[g]}, D_{1[g]}, \ldots, D_{M[g]})$

The key optimization is that all intermediate tensors are **never written to global memory**. The multiple GEMMs and auxiliary tensor operations are fused into a single kernel.

## Algorithmic Strategy: Multi-Input Grouped GEMM with Complex Epilogue

This kernel represents one of the most complex fusion patterns, combining multiple matrix multiplications with auxiliary tensor operations.

1.  **Group Scheduling**: The `G` independent problems are distributed across thread blocks, with each block assigned to one group.

2.  **Multi-GEMM Computation**: Within each thread block:
    -   **Parallel GEMM Execution**: Multiple GEMM operations are computed simultaneously, with each using different portions of the available registers and compute resources.
    -   **Register Management**: Careful orchestration of register usage to accommodate multiple accumulation buffers for the different GEMM operations.
    -   **Memory Interleaving**: Coordinated loading of multiple A and B matrix tiles to maximize memory bandwidth utilization.

3.  **Complex Fused Epilogue**: After computing all GEMMs for a group:
    -   **Load Auxiliary Tensors**: Read the corresponding D tensor values for the group.
    -   **Apply Fusion Function**: Execute the user-defined function `f` that combines all GEMM results and auxiliary tensors.
    -   **Store Result**: Write the final fused result to the output tensor.

This approach enables extremely complex computational patterns while maintaining the memory bandwidth efficiency of deep fusion.

## Source Code Organization

-   [`grouped_gemm_multi_ABD_xdl.cpp`](./grouped_gemm_multi_ABD_xdl.cpp): The main example file. It sets up multiple sets of A and B matrices for each group, multiple D tensors, and instantiates the highly complex device operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp): The device interface for this advanced fusion pattern.
-   The underlying kernel manages multiple simultaneous matrix multiplications with extremely complex register allocation and memory access patterns.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/59_grouped_gemm_multi_ABD
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
./grouped_gemm_multi_ABD_xdl

# Run with verification, data initialization, and timing
./grouped_gemm_multi_ABD_xdl 1 2 1
```

## Applications

This highly specialized kernel is valuable for complex computational patterns found in advanced neural network architectures.

-   **Multi-Branch Networks**: Architectures that compute multiple parallel paths that are later combined, such as Inception modules or complex residual blocks.
-   **Multi-Head Attention Variants**: Advanced attention mechanisms that compute multiple different attention patterns simultaneously and combine them.
-   **Ensemble Methods**: When multiple model predictions need to be computed and combined in a single operation.
-   **Complex Gating Mechanisms**: Advanced neural network layers that use multiple matrix operations for different gating or routing decisions.
-   **Multi-Modal Fusion**: Combining features from different modalities (e.g., vision and text) where each modality requires different linear transformations.

## Performance Considerations

This kernel pushes the boundaries of GPU computation complexity:

-   **Register Pressure**: Managing multiple simultaneous GEMM operations requires careful register allocation
-   **Memory Bandwidth**: Coordinating multiple data streams while maintaining coalesced access patterns
-   **Instruction Scheduling**: Balancing multiple computational streams to maximize throughput
-   **Complexity vs. Performance**: The benefits of fusion must outweigh the increased kernel complexity

This example showcases the extreme flexibility of the Composable Kernel framework, demonstrating how highly specialized computational patterns can be implemented efficiently on modern GPU architectures.
