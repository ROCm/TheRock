# 2D Convolution Forward with Quantization

## Theory

This example demonstrates **2D convolution forward with quantized weights or activations**. Quantization is used to reduce memory and computation by representing values with lower-precision integer types (e.g., int8), enabling efficient inference in deep learning.

**Mathematical Formulation:**
- Quantized convolution: $Y = \text{dequant}(X_q) * \text{dequant}(W_q)$
- $X_q$, $W_q$: quantized input and weight tensors (e.g., int8)
- $\text{dequant}(x_q) = (x_q - z) \cdot s$ (scale $s$, zero-point $z$)
- $Y$: output tensor (often in higher precision, e.g., float32 or float16)

**Algorithmic Background:**
- Quantized values are dequantized on-the-fly during convolution.
- Accumulation is performed in higher precision for accuracy.
- Supports symmetric and asymmetric quantization.
- Convolution is implemented as implicit GEMM for efficiency.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/40_conv2d_fwd_quantization
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./conv2d_fwd_quantization_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/40_conv2d_fwd_quantization/
├── conv2d_fwd_quantization_xdl.cpp         # Main example: sets up, runs, and verifies quantized conv2d
include/ck/tensor_operation/gpu/device/
│   └── device_conv2d_fwd_quantization.hpp       # Device-level quantized conv2d API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_conv2d_fwd_quantization_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_conv2d_fwd_quantization.hpp     # Grid-level quantized conv2d kernel
include/ck/tensor_operation/gpu/element/
    └── quantization_operations.hpp              # Quantization/dequantization utilities
```

### Key Classes and Functions

- **DeviceConv2dFwdQuantization** (in `device_conv2d_fwd_quantization.hpp`):  
  Device API for quantized 2D convolution.
- **gridwise_conv2d_fwd_quantization** (in `gridwise_conv2d_fwd_quantization.hpp`):  
  Implements the tiled/blocking quantized conv2d kernel.
- **quantization_operations** (in `quantization_operations.hpp`):  
  Defines quantization and dequantization functions.

This example demonstrates how Composable Kernel supports efficient quantized convolution for deep learning inference.
