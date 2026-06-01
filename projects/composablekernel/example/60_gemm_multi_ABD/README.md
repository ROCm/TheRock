# GEMM with Multiple A, B, and D Tensors

This example demonstrates a **GEMM operation with multiple A, B, and D tensors**. This is a non-grouped version of the previous example, focusing on fusing multiple matrix multiplications and auxiliary tensor operations into a single kernel for a single problem instance rather than multiple grouped problems.

## Mathematical Formulation

This operation performs multiple GEMM operations simultaneously and combines them with auxiliary tensors.

1.  **Multiple Input GEMMs**: Compute products from multiple A and B tensor pairs.
    $C_{temp0} = A_0 \times B_0$
    $C_{temp1} = A_1 \times B_1$
    $\vdots$
    $C_{tempK} = A_K \times B_K$

2.  **Combination with Auxiliary Tensors**: Apply a user-defined function that combines all GEMM results with multiple D tensors.
    $E = f(C_{temp0}, C_{temp1}, \ldots, C_{tempK}, D_0, D_1, \ldots, D_M)$

The key optimization is that all intermediate tensors are **never written to global memory**. All matrix multiplications and the final combination operation are fused into a single kernel.

## Algorithmic Strategy: Multi-Input GEMM with Complex Epilogue

This kernel extends the basic GEMM algorithm to handle multiple simultaneous matrix multiplications.

1.  **Unified Grid Scheduling**: A single grid of thread blocks handles all matrix multiplications simultaneously. Each thread block computes corresponding tiles from all GEMM operations.

2.  **Multi-GEMM Tile Computation**: Within each thread block:
    -   **Parallel Accumulation**: Multiple accumulator arrays are maintained in registers, one for each GEMM operation.
    -   **Coordinated Memory Access**: Tiles from all A and B matrices are loaded in a coordinated fashion to maximize memory bandwidth.
    -   **Register Orchestration**: Careful management of register usage to accommodate multiple simultaneous accumulations.

3.  **Unified Fused Epilogue**: After computing tiles for all GEMMs:
    -   **Load All Auxiliary Tensors**: Read corresponding elements from all D tensors.
    -   **Apply Complex Fusion Function**: Execute the user-defined function `f` that operates on all GEMM results and auxiliary tensors.
    -   **Single Output Store**: Write the final combined result to the output tensor.

This approach maximizes computational density by performing multiple matrix operations simultaneously while maintaining the memory efficiency of fusion.

## Source Code Organization

-   [`gemm_multi_ABD_xdl.cpp`](./gemm_multi_ABD_xdl.cpp): The main example file. It sets up multiple A and B matrices, multiple D tensors, and instantiates the `DeviceGemmMultiABD` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_gemm_multi_abd.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm_multi_abd.hpp): The device interface for this multi-input fusion pattern.
-   The underlying kernel implements sophisticated register management and memory access coordination for multiple simultaneous GEMM operations.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/60_gemm_multi_ABD
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
./gemm_multi_ABD_xdl

# Run with verification, data initialization, and timing
./gemm_multi_ABD_xdl 1 2 1
```

## Applications

This kernel is useful for complex computational patterns that require multiple simultaneous matrix operations.

-   **Multi-Stream Processing**: Computing multiple different transformations of the same data simultaneously (e.g., different projections in attention mechanisms).
-   **Ensemble Linear Layers**: When multiple linear transformations need to be computed and combined, such as in ensemble methods or multi-expert systems.
-   **Complex Gating Mechanisms**: Advanced neural network layers like Mixture of Experts (MoE) that require multiple matrix operations for routing and computation.
-   **Multi-Objective Optimization**: When multiple loss functions require different linear transformations of the same input.
-   **Feature Fusion**: Combining multiple feature representations that each require different linear projections.

## Comparison with Grouped Version

| Aspect | Grouped Multi-ABD | Non-Grouped Multi-ABD |
|--------|-------------------|----------------------|
| **Problem Structure** | G independent problems | Single unified problem |
| **Memory Layout** | Separate tensors per group | Single tensors with multiple channels |
| **Scheduling** | Group-parallel | Unified parallel |
| **Use Cases** | Independent computations | Correlated computations |
| **Complexity** | Higher (group management) | Lower (unified computation) |

## Performance Characteristics

-   **Computational Intensity**: Very high, as multiple matrix operations are performed simultaneously
-   **Memory Bandwidth**: Efficiently utilized through coordinated access patterns
-   **Register Usage**: High due to multiple accumulator arrays
-   **Instruction Throughput**: Maximized through parallel execution of multiple GEMM streams

This kernel demonstrates the ability to achieve extreme computational density while maintaining the benefits of operation fusion, making it valuable for applications that require multiple related matrix computations.
