# GEMM with Bias, Elementwise, and Permute Fusion

## Theory

This example demonstrates **GEMM fused with bias addition, elementwise operation, and permutation**. This pattern is used in transformer models and other neural architectures where a linear transformation is followed by bias, activation, and layout transformation.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
- Bias: $Z = Y + \text{bias}$
- Elementwise: $E = f(Z)$ (e.g., activation)
- Permute: $O = \text{permute}(E, \text{axes})$

**Algorithmic Background:**
- The GEMM result is kept in registers, bias and elementwise ops are fused in the epilogue, and permutation is applied before writing to global memory.
- Permutation changes the layout/order of tensor axes (e.g., NCHW to NHWC).
- This fusion reduces memory traffic and is common in transformer and CNN pipelines.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/25_gemm_bias_e_permute
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_bias_e_permute_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/25_gemm_bias_e_permute/
├── gemm_bias_e_permute_xdl.cpp         # Main example: sets up, runs, and verifies GEMM+Bias+Elementwise+Permute
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_bias_e_permute.hpp       # Device-level API for fused GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_bias_e_permute_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_gemm_bias_e_permute.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceGemmBiasEPermute** (in `device_gemm_bias_e_permute.hpp`):  
  Device API for GEMM fused with bias, elementwise, and permutation.
- **gridwise_gemm_bias_e_permute** (in `gridwise_gemm_bias_e_permute.hpp`):  
  Implements the tiled/blocking GEMM kernel with fused epilogue and permutation.

This example demonstrates how Composable Kernel supports efficient fusion of linear, bias, activation, and layout operations for deep learning models.
