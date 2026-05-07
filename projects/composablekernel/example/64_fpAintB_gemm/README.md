# Mixed-Precision GEMM: FP16 A × INT8 B

This example demonstrates a **mixed-precision GEMM operation** where matrix `A` is in FP16 (half-precision floating-point) format and matrix `B` is in INT8 (8-bit integer) format. This is an important optimization technique for inference workloads that enables significant memory bandwidth reduction while maintaining acceptable numerical accuracy.

## Mathematical Formulation

The operation performs matrix multiplication with mixed data types:
$C = A_{fp16} \times B_{int8}$

Where:
- Matrix `A` has FP16 elements with shape `[M, K]`
- Matrix `B` has INT8 elements with shape `[K, N]`  
- Matrix `C` typically has FP16 or FP32 elements with shape `[M, N]`

The computation involves:
1.  **Type Conversion**: INT8 elements of `B` are converted to FP16 during computation
2.  **Scaling**: Optional scaling factors can be applied to account for the quantization of `B`
3.  **Accumulation**: Products are accumulated in higher precision (typically FP32) to maintain numerical accuracy
4.  **Output Conversion**: Final results are converted to the desired output precision

## Algorithmic Strategy: Mixed-Precision Tiled GEMM

The implementation extends the standard tiled GEMM algorithm to handle mixed data types efficiently.

1.  **Tiled Matrix Multiplication**: Standard tiling approach with type-specific optimizations:
    -   **A Matrix Loading**: FP16 elements are loaded directly from global memory
    -   **B Matrix Loading**: INT8 elements are loaded and converted to FP16 in registers
    -   **Scaling Application**: If quantization scales are provided, they are applied during the conversion
    -   **Mixed-Type Computation**: FP16 × FP16 multiplication with FP32 accumulation

2.  **Memory Access Optimization**:
    -   **Bandwidth Efficiency**: INT8 storage for `B` reduces memory bandwidth by 2× compared to FP16
    -   **Coalescing**: Both data types are accessed with coalesced memory patterns
    -   **Vectorization**: Use vectorized loads where possible for both FP16 and INT8 data

3.  **Computation Precision**:
    -   **Multiply-Accumulate**: Use FP32 accumulators to prevent overflow and maintain accuracy
    -   **Hardware Utilization**: Leverage mixed-precision matrix instructions where available
    -   **Numerical Stability**: Careful handling of type conversions to minimize precision loss

## Source Code Organization

-   [`fpAintB_gemm_xdl.cpp`](./fpAintB_gemm_xdl.cpp): The main example file. It sets up FP16 matrix A, INT8 matrix B with optional scaling factors, and instantiates the `DeviceFpAintBGemm` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_fpAintB_gemm.hpp`](../../include/ck/tensor_operation/gpu/device/device_fpAintB_gemm.hpp): The device interface for mixed-precision GEMM operations.
-   The underlying kernel implements optimized mixed-type arithmetic with efficient type conversion and scaling operations.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/64_fpAintB_gemm
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
./fpAintB_gemm_xdl

# Run with verification, data initialization, and timing
./fpAintB_gemm_xdl 1 2 1
```

## Applications in Model Optimization

Mixed-precision GEMM is crucial for efficient neural network inference:

-   **Quantized Inference**: Deploy models with quantized weights (INT8) while keeping activations in higher precision (FP16)
-   **Memory-Constrained Environments**: Reduce memory footprint for weight storage while maintaining computational accuracy
-   **Edge Deployment**: Enable deployment on devices with limited memory bandwidth
-   **Large Language Models**: Reduce memory requirements for transformer models while preserving quality
-   **Computer Vision Models**: Optimize CNN inference with quantized convolution layers

## Performance Benefits

This mixed-precision approach provides several advantages:

-   **Memory Bandwidth**: 2× reduction in bandwidth for matrix B compared to FP16×FP16
-   **Storage Efficiency**: 50% reduction in storage requirements for quantized matrices
-   **Cache Efficiency**: More data fits in cache due to reduced memory footprint
-   **Energy Efficiency**: Lower memory traffic reduces energy consumption

## Quantization Considerations

Effective use of INT8 quantization requires:

-   **Calibration**: Proper calibration to determine appropriate scaling factors
-   **Range Analysis**: Understanding the dynamic range of weights to maximize INT8 utilization
-   **Accuracy Trade-offs**: Balancing between compression ratio and numerical accuracy
-   **Hardware Support**: Leveraging hardware features for efficient mixed-precision computation

## Comparison with Other Precision Formats

| Configuration | A Precision | B Precision | Memory Bandwidth | Accuracy | Hardware Support |
|---------------|-------------|-------------|------------------|----------|------------------|
| **FP32×FP32** | FP32 | FP32 | 1.0× (baseline) | Highest | Universal |
| **FP16×FP16** | FP16 | FP16 | 0.5× | High | Modern GPUs |
| **FP16×INT8** | FP16 | INT8 | 0.375× | Medium-High | Specialized |
| **INT8×INT8** | INT8 | INT8 | 0.25× | Medium | Specialized |

The FP16×INT8 configuration provides an excellent balance between memory efficiency and numerical accuracy for many inference workloads.
