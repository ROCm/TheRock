# Fused Batched GEMM-Add-Add-ReLU-GEMM-Add

This example demonstrates an exceptionally deep and complex fusion, chaining two GEMMs with multiple elementwise additions and a ReLU activation. This pattern is designed to fuse a significant portion of a residual block, such as the feed-forward network (FFN) in a Transformer, into a single, highly optimized kernel.

## Mathematical Formulation

The operation computes a complex chain of operations, batched `B` times. For each batch item `b` from `0` to `B-1`:

1.  **First GEMM (GEMM0)**:
    $C_{temp1[b]} = A_{[b]} \times B_{[b]}$

2.  **First Add (Add0)**: An elementwise addition with tensor `D0`.
    $C_{temp2[b]} = C_{temp1[b]} + D0_{[b]}$

3.  **Second Add (Add1)**: Another elementwise addition with tensor `D1`.
    $C_{temp3[b]} = C_{temp2[b]} + D1_{[b]}$

4.  **Activation (ReLU)**: A Rectified Linear Unit activation is applied.
    $C_{temp4[b]} = \text{ReLU}(C_{temp3[b]})$

5.  **Second GEMM (GEMM1)**: The result is fed into a second GEMM.
    $E_{temp[b]} = C_{temp4[b]} \times C_{[b]}$

6.  **Third Add (Add2)**: A final elementwise addition with tensor `D2`.
    $E_{[b]} = E_{temp[b]} + D2_{[b]}$

The key optimization is that all intermediate tensors ($C_{temp1}$ through $E_{temp}$) are **never written to global memory**. They are produced and consumed entirely within the GPU's on-chip memory (registers and LDS/shared memory).

## Algorithmic Strategy: Deeply Fused Producer-Consumer Chain

This kernel represents a pinnacle of fusion capability. It chains two "producer-consumer" GEMMs together, with a series of elementwise operations fused into the epilogue of the first GEMM.

1.  **Batch Scheduling**: The `B` independent problems are distributed across the GPU's thread blocks. Each thread block is assigned to compute the full chain for one batch item `b`.

2.  **Fused Execution within a Thread Block**:
    -   **Compute GEMM0 Tile**: The thread block computes a tile of the first GEMM, $A_{[b]} \times B_{[b]}$. The result is held in registers.
    -   **Fused Epilogue (Add-Add-ReLU)**: Before this intermediate result is stored anywhere, the epilogue operations are applied directly to the data in registers.
        -   Load corresponding elements from `D0` and `D1`.
        -   Perform the two additions.
        -   Apply the ReLU activation.
    -   **Store to Shared Memory**: The result of this entire fused chain ($C_{temp4}$) is written to a designated region of **shared memory (LDS)**.
    -   **Synchronization**: A block-wide synchronization (`__syncthreads()`) ensures the intermediate result in LDS is visible to all threads in the block.
    -   **Compute GEMM1 Tile**: The threads immediately start the second GEMM, using the tile in shared memory as the input, multiplying it with tiles of `C`. The result is accumulated in registers.
    -   **Final Fused Epilogue (Add)**: Before the final result is stored, the last addition is fused.
        -   Load corresponding elements from `D2`.
        -   Perform the final addition in registers.
    -   **Store Final Result**: The final result `E` is written to global memory.

This deep fusion avoids five separate kernel launches and the associated read/write traffic for four large intermediate tensors, resulting in a massive performance improvement.

## Source Code Organization

-   [`batched_gemm_add_add_relu_gemm_add_xdl.cpp`](./batched_gemm_add_add_relu_gemm_add_xdl.cpp): The main example file. It sets up the numerous input tensors (A, B, C, D0, D1, D2) and instantiates the highly specialized device-level operation.
-   The device-level interface and underlying grid-wise kernel for this operation are extremely complex, templated on the multiple elementwise operations and managing the intricate data flow between registers, shared memory, and global memory.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/37_batched_gemm_add_add_relu_gemm_add
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
./batched_gemm_add_add_relu_gemm_add_xdl

# Run with verification, data initialization, and timing
./batched_gemm_add_add_relu_gemm_add_xdl 1 2 1
```

## Application to Transformer FFN Block

This kernel can fuse almost the entire Feed-Forward Network (FFN) block of a standard Transformer, including the residual connections.

A typical FFN block with pre-layer-normalization looks like this:
`Z = LayerNorm(X)`
`Y = Linear_2(ReLU(Linear_1(Z)))`
`Output = X + Y`

This kernel can compute `Y` and the final residual addition:
-   `A`: The normalized input `Z`.
-   `B`: The weight matrix for `Linear_1`.
-   `D0`: The bias for `Linear_1`.
-   `D1`: Not used in this specific mapping (can be zero).
-   `C`: The weight matrix for `Linear_2`.
-   `D2`: The bias for `Linear_2` plus the original input `X` for the residual connection.

By mapping the components of a Transformer FFN block to this kernel, a developer can achieve performance far beyond what is possible with a sequence of standard library calls. This demonstrates the power of Composable Kernel to create highly domain-specific, performance-leading fused operations.
