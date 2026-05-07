# Fused Batched GEMM-GEMM

This example demonstrates a **Batched GEMM-GEMM** operation, where two sequential General Matrix-Matrix Multiplications are fused into a single high-performance kernel. This pattern is common in multi-layer perceptrons (MLPs) and is a core component of the feed-forward network (FFN) block in Transformer models.

## Mathematical Formulation

The operation computes a chain of two matrix multiplications, batched `B` times. For each batch item `b` from `0` to `B-1`:

1.  **First GEMM (GEMM0)**:
    $D_{temp[b]} = A_{[b]} \times B_{[b]}$
    Where `A` has shape `[B, M, K0]`, `B` has shape `[B, K0, N]`. The intermediate result `D_temp` has shape `[B, M, N]`.

2.  **Second GEMM (GEMM1)**:
    $E_{[b]} = D_{temp[b]} \times C_{[b]}$
    Where `D_temp` (the output of GEMM0) has shape `[B, M, N]` and `C` has shape `[B, N, K1]`. The final output `E` has shape `[B, M, K1]`.

The critical optimization is that the intermediate tensor `D_temp` is **never written to global memory**. It is produced and consumed entirely within the GPU's on-chip memory (registers and LDS/shared memory), saving a massive amount of memory bandwidth.

## Algorithmic Strategy: Fused GEMM-GEMM via Shared Memory

The implementation uses a batch-parallel approach where each thread block is assigned a single batch item. Within the block, the two GEMMs are fused using shared memory as a buffer.

1.  **Batch Scheduling**: The `B` independent GEMM-GEMM problems are distributed across the GPU's thread blocks. Each thread block is assigned to compute the full chain for one batch item `b`.

2.  **Fused Execution within a Thread Block**:
    -   **Compute GEMM0 Tile**: The thread block first computes a tile of the intermediate tensor, $D_{temp[b]}$, using a standard tiled GEMM algorithm. The result of this computation is stored directly into a designated region of **shared memory (LDS)**.
    -   **Synchronization**: A block-wide synchronization (`__syncthreads()`) is performed. This is a critical step that ensures the *entire* tile of $D_{temp[b]}$ is visible to all threads in the block before the second GEMM begins.
    -   **Compute GEMM1 Tile**: The threads then immediately start computing the second GEMM. They use the intermediate tile stored in shared memory as the "A" matrix for this second GEMM, multiplying it with tiles of the `C` matrix. The result is accumulated in registers.
    -   **Store Final Result**: Once a tile of the final output `E` is computed, it is written to global memory.

This "producer-consumer" pattern within a thread block is highly efficient. It treats shared memory as a fast, programmable cache for the intermediate tensor, completely avoiding the slow round-trip to global HBM memory.

## Source Code Organization

-   [`batched_gemm_gemm_xdl.cpp`](./batched_gemm_gemm_xdl.cpp): The main example file. It sets up the three input tensors (A, B, C) for the batched operation and instantiates the `DeviceBatchedGemmGemm` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp`](../../include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp): The high-level device interface for the fused Batched GEMM-GEMM operation.
-   The underlying grid-wise kernel implements the complex fusion logic, managing the register usage for GEMM0, the write to shared memory, the synchronization, and the subsequent computation of GEMM1 using the data from shared memory.

## Build and Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build the Example
```bash
cd /path/to/composable_kernel/example/31_batched_gemm_gemm
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
./batched_gemm_gemm_xdl

# Run with verification, data initialization, and timing
./batched_gemm_gemm_xdl 1 2 1
```

## Application to Transformer FFN

This kernel is perfectly suited to optimize the Feed-Forward Network (FFN) block found in every layer of a Transformer model. The FFN is typically defined as:

`FFN(X) = Linear_2(Activation(Linear_1(X)))`

Where `Linear_1` and `Linear_2` are dense layers (GEMMs). If the activation function can also be fused (e.g., ReLU or GeLU), an even more complex kernel can be used. However, this `GEMM-GEMM` kernel provides the core fusion for the two most computationally expensive parts of the FFN. By fusing `Linear_1` and `Linear_2`, this kernel can significantly reduce the latency and memory bandwidth of the FFN block, leading to faster end-to-end model training and inference.
