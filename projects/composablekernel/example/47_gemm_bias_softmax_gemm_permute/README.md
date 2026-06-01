# GEMM-Bias-Softmax-GEMM-Permute Fusion

This example demonstrates an extremely complex and highly specialized fusion: **GEMM → Bias → Softmax → GEMM → Permute**. This pattern represents a complete, optimized attention mechanism with additional layout transformation, making it ideal for Transformer models that require specific output formats.

## Mathematical Formulation

The operation performs a complete attention calculation with an additional permutation at the end.

1.  **First GEMM (QK^T)**: Compute attention scores.
    $S_{temp} = Q \times K^T$

2.  **Bias Addition**: Add attention bias (e.g., positional bias or causal mask).
    $S'_{temp} = S_{temp} + \text{Bias}$

3.  **Softmax**: Apply softmax to get attention weights.
    $P = \text{softmax}(S'_{temp})$

4.  **Second GEMM (PV)**: Apply attention weights to values.
    $O_{temp} = P \times V$

5.  **Permutation**: Reorder dimensions for subsequent processing.
    $O = \text{permute}(O_{temp})$

The key optimization is that all intermediate tensors (`S_temp`, `S'_temp`, `P`, `O_temp`) are **never written to global memory**. The entire attention calculation and permutation are performed in a single, monolithic kernel.

## Algorithmic Strategy: Extended Tiled Attention with Permuted Output

This kernel extends the fused attention algorithm with bias addition and output permutation.

1.  **Batch Scheduling**: The attention problems are distributed across thread blocks, with each block handling one attention head for one batch item.

2.  **Extended Tiled Computation**: The tiled attention algorithm is enhanced to include bias and permutation.
    -   **Load Q tile**: A tile of the Query matrix is loaded into registers.
    -   **Inner Loop over K/V tiles**:
        -   Load tiles of Key matrix `K` and Value matrix `V`.
        -   **Compute Score Tile (GEMM0)**: Compute QK^T and keep in registers.
        -   **Bias Addition**: Load and add the corresponding bias tile.
        -   **Online Softmax**: Apply the numerically stable online softmax algorithm.
        -   **Compute Output Tile (GEMM1)**: Multiply attention weights with V tile.
    -   **Permuted Store**: Instead of writing directly to the output, calculate the permuted destination coordinates and write the final result to the correct permuted location.

This approach maintains the memory efficiency of fused attention while adding the computational benefits of bias fusion and the layout flexibility of permutation.

## Source Code Organization

-   [`gemm_bias_softmax_gemm_permute_xdl.cpp`](./gemm_bias_softmax_gemm_permute_xdl.cpp): The main example file. It sets up the Q, K, V matrices, bias tensor, and permutation specification, then instantiates the highly specialized operation.
-   The device-level interface for this operation is extremely complex, combining attention computation with bias handling and permutation logic.
-   The underlying kernel represents one of the most sophisticated fusion patterns in the library, managing multiple computational stages and complex memory access patterns.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/47_gemm_bias_softmax_gemm_permute
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
./gemm_bias_softmax_gemm_permute_xdl

# Run with verification, data initialization, and timing
./gemm_bias_softmax_gemm_permute_xdl 1 2 1
```

## Applications in Advanced Transformer Architectures

This kernel is designed for advanced Transformer implementations that require specialized attention patterns and output formats.

-   **Relative Position Encoding**: Many modern Transformers use relative positional encodings that require adding learned bias terms to attention scores. This kernel can fuse these bias additions directly into the attention computation.
-   **Multi-Head Attention with Layout Optimization**: After computing attention for multiple heads, the output often needs to be permuted to optimize memory layout for subsequent layers. This kernel can perform the attention computation and layout transformation in a single pass.
-   **Causal Attention with Masking**: In autoregressive models, causal masking is applied as a bias term (typically large negative values) to prevent attending to future positions. This kernel can efficiently apply such masking.
-   **Custom Attention Variants**: Research architectures often require modified attention mechanisms with additional bias terms or specific output layouts. This kernel provides a high-performance foundation for such implementations.

This example represents the pinnacle of computational fusion, demonstrating how complex multi-stage algorithms can be optimized through deep kernel fusion.
