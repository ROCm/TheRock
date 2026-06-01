# Split-K GEMM with Bias, Elementwise Operation, and Permutation

This example demonstrates a highly complex fusion: a **Split-K GEMM** where the final result is fused with a bias addition, a second elementwise operation, and a final permutation. This kernel combines the parallelism-enhancing Split-K strategy with a multi-stage epilogue, making it suitable for accelerating very large or "skinny" GEMMs that are part of a more complex computational graph.

## Mathematical Formulation

The operation first computes a GEMM using the Split-K algorithm and then applies a sequence of fused operations.

1.  **Split-K GEMM Stage**: The matrix multiplication $C_{temp1} = A \times B$ is computed by splitting the `K` dimension into `S` chunks and summing the partial products.
    $C_{temp1} = \sum_{s=0}^{S-1} (A_s \times B_s)$

2.  **Bias Addition Stage**: A bias vector `D` is broadcast and added.
    $C_{temp2} = C_{temp1} + D$

3.  **Elementwise Stage**: A second elementwise operation is performed with tensor `E`.
    $C_{temp3} = C_{temp2} \odot E$

4.  **Permutation Stage**: The final result is permuted.
    $F = \text{permute}(C_{temp3})$

The key is that the reduction (summation) of the partial GEMM products is fused with the entire epilogue chain (Bias, E-wise, Permute).

## Algorithmic Strategy: Split-K with a Fused Reduction Epilogue

The implementation combines the Split-K algorithm with the multi-stage fused epilogue seen in previous examples.

1.  **Splitting the K-Dimension**: The `K` dimension is logically split into `S` parts to create `S` parallel partial GEMM problems.

2.  **Parallel Partial GEMMs**: The `S` partial GEMMs are executed in parallel across the GPU's thread blocks. A thread block is assigned to compute a tile of a *partial* product $C_s$.

3.  **Fused Reduction and Epilogue**: The method for reducing the partial sums and applying the epilogue is critical.
    -   **Workspace Approach**: A common strategy is to use a temporary workspace in global memory.
        -   **Stage 1 (Partial Products)**: Each of the `S` parallel GEMMs computes its partial product $C_s$ and writes it to a unique slice of a temporary workspace tensor.
        -   **Stage 2 (Reduce + Epilogue)**: A second, specialized kernel is launched. This kernel reads the `S` partial products from the workspace, reduces (sums) them on-the-fly, and then immediately applies the full Bias-E-Permute epilogue before writing the final result `F` to memory.
    -   **Atomic-based Approach**: For some data types and operations, it's possible to perform the reduction using atomic operations. The first block to arrive at an output element would compute its partial result, apply the epilogue, and write it out. Subsequent blocks would compute their partial results, read the intermediate value from the output buffer, add their contribution, and then atomically write the new sum back. This is more complex and often less performant due to atomic contention.

Composable Kernel's implementation abstracts this complexity, providing a single device-level operation that manages the workspace, the two stages, and the complex epilogue.

## Source Code Organization

-   [`splitk_gemm_bias_e_permute_xdl.cpp`](./splitk_gemm_bias_e_permute_xdl.cpp): The main example file. It sets up the GEMM problem, the bias and elementwise tensors, the permutation, and instantiates the `DeviceSplitkGemmBiasEPermute` operation.
-   The device-level interface and underlying kernels are highly specialized. They manage the Split-K parameter, the workspace allocation (if needed), and the two-stage execution process, combining the logic from `DeviceGemmSplitK` and `DeviceGemmBiasEPermute`.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/43_splitk_gemm_bias_e_permute
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
./splitk_gemm_bias_e_permute_xdl

# Run with verification, data initialization, and timing
./splitk_gemm_bias_e_permute_xdl 1 2 1
```

## Applications

This highly specialized kernel is useful when a very large GEMM (that would benefit from Split-K) is immediately followed by a series of operations that can be fused.

-   **Large Feed-Forward Networks**: In a Transformer with a very large hidden dimension, the GEMMs in the FFN block might become "skinny" (large K, smaller M/N). If this FFN is also fused with residual connections (bias/add) and layout permutations, this kernel could be a perfect fit, offering both the parallelism benefits of Split-K and the memory bandwidth savings of the fused epilogue.
-   **Final Classifier Layers**: The final layer of a large classification model is often a very large GEMM. If this layer's output needs to be reshaped or post-processed, this kernel could fuse those operations directly into the Split-K GEMM.

This example showcases the extreme composability of the library, allowing for the creation of highly tailored, high-performance kernels that combine different algorithmic strategies (like Split-K) with deep fusion.
