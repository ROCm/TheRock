# GEMM with Double Multiply Operations

This example demonstrates a **GEMM followed by two sequential elementwise multiplication operations**. This fusion pattern is useful for implementing layers that require matrix multiplication followed by multiple scaling or masking operations, such as certain attention mechanisms or gated neural network architectures.

## Mathematical Formulation

The operation performs a matrix multiplication followed by two sequential elementwise multiplications.

1.  **GEMM Stage**: A standard matrix multiplication.
    $C_{temp1} = A \times B$

2.  **First Multiplication**: Elementwise multiplication with tensor `D`.
    $C_{temp2} = C_{temp1} \odot D$

3.  **Second Multiplication**: Elementwise multiplication with tensor `E`.
    $F = C_{temp2} \odot E$

The key optimization is that the intermediate tensors `C_temp1` and `C_temp2` are **never written to global memory**. All operations are fused into the GEMM's epilogue, operating on data held in registers.

## Algorithmic Strategy: GEMM with Dual-Multiply Epilogue

The implementation uses a tiled GEMM algorithm with a multi-stage fused epilogue that performs two sequential multiplications.

1.  **Tiled GEMM Core**: The kernel begins with a standard tiled GEMM. A thread block computes a tile of the product $A \times B$, accumulating the result in registers.

2.  **Dual-Multiply Epilogue**: Before any data is written to global memory, the following sequence occurs for the tile of data held in registers:
    -   **Load First Multiplicand**: Threads load the corresponding elements of tensor `D`.
    -   **First Multiplication**: The elementwise multiplication is performed in registers: `result *= D`.
    -   **Load Second Multiplicand**: Threads load the corresponding elements of tensor `E`.
    -   **Second Multiplication**: The second elementwise multiplication is performed in registers: `result *= E`.
    -   **Store Final Result**: The final result `F` is written to global memory.

This deep fusion eliminates multiple kernel launches and the memory bandwidth required to write and re-read intermediate tensors.

## Source Code Organization

-   [`gemm_multiply_multiply_xdl.cpp`](./gemm_multiply_multiply_xdl.cpp): The main example file. It sets up the input matrices (A, B) and auxiliary tensors (D, E), and instantiates the `DeviceGemmMultiplyMultiply` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_gemm_multiply_multiply.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm_multiply_multiply.hpp): The high-level device interface for this fused operation.
-   The underlying kernel implements the dual-multiply epilogue that performs both multiplication operations on register data before storing.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/65_gemm_multiply_multiply
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
./gemm_multiply_multiply_xdl

# Run with verification, data initialization, and timing
./gemm_multiply_multiply_xdl 1 2 1
```

## Applications

This fusion pattern is useful for several types of neural network operations and advanced computational patterns.

-   **Multi-Scale Attention**: Some attention mechanisms apply multiple scaling factors sequentially, such as learned attention scales followed by positional scaling.
-   **Gated Mechanisms**: Advanced gating architectures that use multiple multiplicative gates in sequence, such as in some RNN variants or transformer modifications.
-   **Feature Modulation**: Computer vision models that apply multiple feature modulation operations, such as style-based generators or attention-based feature refinement.
-   **Masking Operations**: Applying multiple types of masks (e.g., attention mask followed by a dropout mask) in sequence.
-   **Custom Activations**: Implementing complex activation functions that involve multiple multiplicative terms.
-   **Mixture of Experts**: Some MoE architectures use multiple routing or gating multiplications in sequence.

## Performance Considerations

The performance benefits of this fusion depend on several factors:

-   **Memory Bandwidth Savings**: Eliminates two full tensor read/write cycles for intermediate results
-   **Cache Locality**: Maintains data in registers throughout the computation pipeline
-   **Instruction Scheduling**: Allows better interleaving of compute and memory operations
-   **Kernel Launch Overhead**: Reduces from three separate kernel launches to one

## Comparison with Sequential Operations

| Approach | Kernel Launches | Memory Bandwidth | Register Pressure | Implementation Complexity |
|----------|----------------|------------------|-------------------|---------------------------|
| **Sequential** | 3 kernels | 3× intermediate storage | Low | Simple |
| **Fused** | 1 kernel | No intermediate storage | Medium | Moderate |

## Extension Possibilities

This pattern can be extended in several ways:

-   **More Multiplications**: Additional sequential multiplications can be added to the epilogue
-   **Mixed Operations**: Combine multiplications with additions or other elementwise operations
-   **Conditional Operations**: Apply multiplications conditionally based on masks or thresholds
-   **Broadcasting**: Handle different broadcasting patterns for the multiplicand tensors

This example demonstrates the flexibility of the epilogue fusion approach, showing how multiple sequential operations can be efficiently combined with matrix multiplication.
