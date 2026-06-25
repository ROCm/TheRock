# Grouped GEMM with Bias, Elementwise Operation, and Permutation

This example demonstrates a highly complex and specialized fusion: a **Grouped GEMM** where each individual GEMM operation is fused with a bias addition, a second elementwise operation, and a final permutation of the output. This kernel is designed to accelerate layers that have a group-parallel structure, such as depthwise separable convolutions or multi-head attention, when they are part of a larger fused computational graph.

## Mathematical Formulation

This operation performs `G` independent fused GEMM operations in parallel, where `G` is the group count. For each group `g` from `0` to `G-1`:

1.  **GEMM Stage**: A standard matrix multiplication.
    $C_{temp1[g]} = A_{[g]} \times B_{[g]}$

2.  **Bias Addition Stage**: A bias vector `D_[g]` is broadcast and added.
    $C_{temp2[g]} = C_{temp1[g]} + D_{[g]}$

3.  **Elementwise Stage**: A second elementwise operation is performed with tensor `E_[g]`.
    $C_{temp3[g]} = C_{temp2[g]} \odot E_{[g]}$

4.  **Permutation Stage**: The final result for the group is permuted.
    $F_{[g]} = \text{permute}(C_{temp3[g]})$

All four stages for all `G` groups are executed within a single kernel launch. The intermediate results are kept in registers and never written to global memory.

## Algorithmic Strategy: Group-Parallel GEMM with Fused Epilogue

The implementation combines the scheduling strategy of Grouped GEMM with the multi-stage fused epilogue seen in `25_gemm_bias_e_permute`.

1.  **Group Scheduling**: The `G` independent problems are distributed across the GPU's thread blocks. The grid-wise kernel is designed such that each thread block is assigned to compute one of the `G` fused operations.

2.  **Fused GEMM Execution**: Once a thread block is assigned a group `g`, it executes a complete fused GEMM for that group's specific data. This involves:
    -   Calculating the base memory addresses for $A_{[g]}, B_{[g]}, D_{[g]}, E_{[g]}$, and $F_{[g]}$ using the group index and the problem description for that group.
    -   Executing a standard tiled GEMM for $A_{[g]} \times B_{[g]}$, accumulating the result in registers.
    -   Executing the fused epilogue:
        -   Load the bias `D_[g]` and add it.
        -   Load the elementwise tensor `E_[g]` and apply the operation.
        -   Calculate the permuted destination coordinates and write the final result to `F_[g]`.

This approach maximizes parallelism at two levels: the coarse-grained parallelism across the `G` groups, and the fine-grained data parallelism within each individual GEMM operation.

## Source Code Organization

-   [`grouped_gemm_bias_e_permute_xdl.cpp`](./grouped_gemm_bias_e_permute_xdl.cpp): The main example file. It demonstrates the complex setup for a grouped problem, defining the `G` sets of input tensors and the permutation. It then instantiates the `DeviceGroupedGemmBiasEPermute` operation.
-   [`../../include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_bias_e_permute_impl.hpp`](../../include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_bias_e_permute_impl.hpp): The high-level device interface for this specific fused operation. It takes arrays of tensor descriptors, one for each group.
-   The underlying grid-wise kernel contains the logic to map thread blocks to groups and then execute the full fused GEMM pipeline for the assigned group.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/28_grouped_gemm_bias_e_permute
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
./grouped_gemm_bias_e_permute_xdl

# Run with verification, data initialization, and timing
./grouped_gemm_bias_e_permute_xdl 1 2 1
```

## Applications

This highly specialized kernel is valuable for optimizing specific patterns in modern neural networks:

-   **Multi-Head Attention (MHA)**: The computation for each head in MHA is independent. The entire MHA block can be viewed as a Grouped GEMM where the number of groups `G` is the number of attention heads. If the Q, K, or V projections involve fusions with bias, other elementwise ops, and permutations to prepare the data for the batched GEMM, this kernel could potentially fuse a large part of that logic.
-   **Depthwise Separable Convolutions**: The depthwise part of this convolution is a Grouped GEMM with `G` equal to the number of channels. If this is followed by a fused activation function (e.g., a gated activation) and a permutation, this kernel could be a perfect match.
-   **Mixture-of-Experts (MoE) Models**: In MoE layers, an input is routed to one of several "expert" sub-networks. If these experts have identical structure, their execution can be formulated as a Grouped GEMM, where `G` is the number of experts. Any fusions within the expert network could be captured by this kernel.

This example showcases the extreme composability of the library, allowing for the creation of highly tailored, high-performance kernels for complex, group-parallel computational graphs.
