# GEMM with Quantization

## Theory

This example demonstrates **GEMM (General Matrix Multiplication) with quantized inputs or weights**. Quantization is a technique to reduce memory and computation by representing values with lower-precision integer types (e.g., int8), commonly used for efficient inference in deep learning.

**Mathematical Formulation:**
- Quantized GEMM: $C = \text{dequant}(A_q) \times \text{dequant}(B_q)$
- $A_q$, $B_q$: quantized matrices (e.g., int8)
- $\text{dequant}(x_q) = (x_q - z) \cdot s$ (scale $s$, zero-point $z$)
- $C$: output matrix (often in higher precision, e.g., float32 or float16)

**Algorithmic Background:**
- Quantized values are dequantized on-the-fly during GEMM computation.
- Accumulation is performed in higher precision for accuracy.
- Supports symmetric and asymmetric quantization.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/14_gemm_quantization
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_quantization_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/14_gemm_quantization/
├── gemm_quantization_xdl.cpp         # Main example: sets up, runs, and verifies quantized GEMM
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_quantized.hpp       # Device-level quantized GEMM API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_quantized_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_quantized.hpp     # Grid-level quantized GEMM kernel
include/ck/tensor_operation/gpu/element/
    └── quantization_operations.hpp     # Quantization/dequantization utilities
```

### Key Classes and Functions

- **DeviceGemmQuantized** (in `device_gemm_quantized.hpp`):  
  Device API for quantized GEMM.
- **gridwise_gemm_quantized** (in `gridwise_gemm_quantized.hpp`):  
  Implements the tiled/blocking quantized GEMM kernel.
- **quantization_operations** (in `quantization_operations.hpp`):  
  Defines quantization and dequantization functions.

This example demonstrates how Composable Kernel supports efficient quantized matrix multiplication for deep learning inference.
